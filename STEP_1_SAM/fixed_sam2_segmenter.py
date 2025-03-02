import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import logging
from matplotlib.widgets import Button
import shutil
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf
import sam2

# Check if we're using the correct sam2 package
if os.path.isdir(os.path.join(sam2.__path__[0], "sam2")):
    raise RuntimeError(
        "You're likely running Python from the parent directory of the sam2 repository. "
        "This is not supported. Please run Python from another directory."
    )

# Device selection
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire script
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

# HF model mapping
HF_MODEL_ID_TO_FILENAMES = {
    "facebook/sam2-hiera-tiny": (
        "configs/sam2/sam2_hiera_t.yaml",
        "sam2_hiera_tiny.pt",
    ),
    "facebook/sam2-hiera-small": (
        "configs/sam2/sam2_hiera_s.yaml",
        "sam2_hiera_small.pt",
    ),
    "facebook/sam2-hiera-base-plus": (
        "configs/sam2/sam2_hiera_b+.yaml",
        "sam2_hiera_base_plus.pt",
    ),
    "facebook/sam2-hiera-large": (
        "configs/sam2/sam2_hiera_l.yaml",
        "sam2_hiera_large.pt",
    ),
    "facebook/sam2.1-hiera-tiny": (
        "configs/sam2.1/sam2.1_hiera_t.yaml",
        "sam2.1_hiera_tiny.pt",
    ),
    "facebook/sam2.1-hiera-small": (
        "configs/sam2.1/sam2.1_hiera_s.yaml",
        "sam2.1_hiera_small.pt",
    ),
    "facebook/sam2.1-hiera-base-plus": (
        "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "sam2.1_hiera_base_plus.pt",
    ),
    "facebook/sam2.1-hiera-large": (
        "configs/sam2.1/sam2.1_hiera_l.yaml",
        "sam2.1_hiera_large.pt",
    ),
}

def build_sam2_video_predictor(
    config_file,
    ckpt_path=None,
    device="mps",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    vos_optimized=False,
    **kwargs,
):
    hydra_overrides = [
        "++model._target_=sam2.sam2_video_predictor.SAM2VideoPredictor",
    ]
    if vos_optimized:
        hydra_overrides = [
            "++model._target_=sam2.sam2_video_predictor.SAM2VideoPredictorVOS",
            "++model.compile_image_encoder=True",  # Let sam2_base handle this
        ]

    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            # sigmoid mask logits in memory encoder for clicked frames
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            # fill small holes in low-res masks up to `fill_hole_area` (before resizing)
            "++model.fill_hole_area=8",
        ]
    hydra_overrides.extend(hydra_overrides_extra)

    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model

def _hf_download(model_id):
    from huggingface_hub import hf_hub_download

    config_name, checkpoint_name = HF_MODEL_ID_TO_FILENAMES[model_id]
    ckpt_path = hf_hub_download(repo_id=model_id, filename=checkpoint_name)
    return config_name, ckpt_path

def build_sam2_video_predictor_hf(model_id, **kwargs):
    config_name, ckpt_path = _hf_download(model_id)
    return build_sam2_video_predictor(
        config_file=config_name, ckpt_path=ckpt_path, **kwargs
    )

def _load_checkpoint(model, ckpt_path):
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
        missing_keys, unexpected_keys = model.load_state_dict(sd)
        if missing_keys:
            logging.error(missing_keys)
            raise RuntimeError()
        if unexpected_keys:
            logging.error(unexpected_keys)
            raise RuntimeError()
        logging.info("Loaded checkpoint successfully")

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    if len(pos_points) > 0:
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    if len(neg_points) > 0:
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

class InteractiveSegmenter:
    def __init__(self, video_dir):
        self.video_dir = video_dir
        self.output_dir = os.path.join(video_dir, "segmentation_output")
        self.mask_dir = os.path.join(self.output_dir, "masks")
        self.overlay_dir = os.path.join(self.output_dir, "overlays")
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.mask_dir, exist_ok=True)
        os.makedirs(self.overlay_dir, exist_ok=True)
        
        # Get frame filenames
        self.frame_names = [
            p for p in os.listdir(video_dir)
            if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
        ]
        self.frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        
        # Initialize SAM2 model
        self.model_id = "facebook/sam2.1-hiera-large"
        self.predictor = build_sam2_video_predictor_hf(self.model_id, device=device)
        self.inference_state = self.predictor.init_state(video_path=video_dir)
        self.predictor.reset_state(self.inference_state)
        
        # Initialize interactive variables
        self.points = []  # List of (x, y) coordinates
        self.labels = []  # List of 0 or 1 (negative or positive)
        self.current_frame_idx = 0
        self.current_obj_id = 1
        self.point_mode = 1  # 1 for positive, 0 for negative
        self.video_segments = {}  # To store segmentation results
        
        # Figure setup
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(bottom=0.2)
        
        # Add interaction buttons
        self.add_mode_button = Button(plt.axes([0.1, 0.05, 0.15, 0.06]), 'Pos/Neg Mode')
        self.add_mode_button.on_clicked(self.toggle_point_mode)
        
        self.next_frame_button = Button(plt.axes([0.3, 0.05, 0.15, 0.06]), 'Next Frame')
        self.next_frame_button.on_clicked(self.next_frame)
        
        self.prev_frame_button = Button(plt.axes([0.5, 0.05, 0.15, 0.06]), 'Prev Frame')
        self.prev_frame_button.on_clicked(self.prev_frame)
        
        self.process_button = Button(plt.axes([0.7, 0.05, 0.15, 0.06]), 'Process Video')
        self.process_button.on_clicked(self.process_video)
        
        # Connect mouse event
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        # Load first frame
        self.load_frame(self.current_frame_idx)
        plt.show()
    
    def load_frame(self, frame_idx):
        self.ax.clear()
        self.current_frame_idx = frame_idx
        self.img = Image.open(os.path.join(self.video_dir, self.frame_names[frame_idx]))
        self.ax.imshow(self.img)
        
        # Update title with current mode information
        mode_text = "POSITIVE" if self.point_mode == 1 else "NEGATIVE"
        self.ax.set_title(f"Frame {frame_idx} - {mode_text} click mode")
        
        # Show existing points if any
        if self.points and self.labels:
            points_array = np.array(self.points, dtype=np.float32)
            labels_array = np.array(self.labels, dtype=np.int32)
            show_points(points_array, labels_array, self.ax)
        
        # Show mask if the frame has been processed
        if frame_idx in self.video_segments and self.current_obj_id in self.video_segments[frame_idx]:
            show_mask(self.video_segments[frame_idx][self.current_obj_id], self.ax, obj_id=self.current_obj_id)
        
        plt.draw()
    
    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        
        # Add point to the list
        x, y = event.xdata, event.ydata
        self.points.append([x, y])
        self.labels.append(self.point_mode)
        
        # Process points
        self.process_points()
        
        # Refresh display
        self.load_frame(self.current_frame_idx)
    
    def toggle_point_mode(self, event):
        self.point_mode = 1 - self.point_mode  # Toggle between 1 and 0
        mode_text = "POSITIVE" if self.point_mode == 1 else "NEGATIVE"
        self.ax.set_title(f"Frame {self.current_frame_idx} - {mode_text} click mode")
        plt.draw()
    
    def next_frame(self, event):
        if self.current_frame_idx < len(self.frame_names) - 1:
            self.current_frame_idx += 1
            self.points = []  # Reset points for new frame
            self.labels = []
            self.load_frame(self.current_frame_idx)
    
    def prev_frame(self, event):
        if self.current_frame_idx > 0:
            self.current_frame_idx -= 1
            self.points = []  # Reset points for new frame
            self.labels = []
            self.load_frame(self.current_frame_idx)
    
    def process_points(self):
        if not self.points:
            return
        
        # Convert to numpy arrays
        points_array = np.array(self.points, dtype=np.float32)
        labels_array = np.array(self.labels, dtype=np.int32)
        
        # Process with the model
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=self.current_frame_idx,
            obj_id=self.current_obj_id,
            points=points_array,
            labels=labels_array,
        )
        
        # Store the mask
        if self.current_frame_idx not in self.video_segments:
            self.video_segments[self.current_frame_idx] = {}
        
        self.video_segments[self.current_frame_idx][self.current_obj_id] = (out_mask_logits[0] > 0.0).cpu().numpy()
    
    def process_video(self, event):
        print("Processing the entire video. This may take a while...")
        
        # Run propagation throughout the video
        self.video_segments = {}  # Reset to store fresh results
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
            self.video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        
        # Save the results
        try:
            self.save_results()
            print(f"Processing complete! Results saved to {self.output_dir}")
        except Exception as e:
            print(f"Error saving results: {e}")
    
    def save_results(self):
        """Save segmentation masks and overlays"""
        for frame_idx in range(len(self.frame_names)):
            if frame_idx in self.video_segments:
                # Load original frame
                img = Image.open(os.path.join(self.video_dir, self.frame_names[frame_idx]))
                img_np = np.array(img)
                
                # Create overlay figure
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.imshow(img_np)
                
                # Add masks
                for obj_id, mask in self.video_segments[frame_idx].items():
                    # FIX: Ensure mask has the right shape and format for PIL
                    # Reshape mask if needed to ensure it's a 2D array
                    if mask.ndim != 2:
                        # If mask has shape like (1, 1, width) or any 3D shape, try to squeeze it to 2D
                        try:
                            mask = np.squeeze(mask)
                        except:
                            print(f"Warning: Could not reshape mask with shape {mask.shape} for frame {frame_idx}, object {obj_id}")
                            continue
                    
                    # Ensure mask has the expected dimensions
                    if mask.ndim != 2:
                        print(f"Warning: Mask has unexpected shape {mask.shape} for frame {frame_idx}, object {obj_id}")
                        continue
                        
                    # Save binary mask
                    mask_filename = f"frame_{frame_idx:04d}_obj_{obj_id}.png"
                    mask_path = os.path.join(self.mask_dir, mask_filename)
                    
                    # Use binary values (0 or 255) for the mask
                    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
                    mask_img.save(mask_path)
                    
                    # Add to overlay
                    show_mask(mask, ax, obj_id=obj_id)
                
                # Save overlay
                overlay_filename = f"frame_{frame_idx:04d}_overlay.png"
                overlay_path = os.path.join(self.overlay_dir, overlay_filename)
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(overlay_path, bbox_inches='tight', pad_inches=0)
                plt.close(fig)

if __name__ == "__main__":
    # Set the directory with the video frames
    video_dir = input("Enter the directory containing video frames (e.g., ./ceramics_frames): ")
    
    if not os.path.exists(video_dir):
        print(f"Error: Directory {video_dir} does not exist")
        exit(1)
    
    # Start the interactive segmenter
    print("Starting interactive segmentation. Click on the image to add points.")
    print("- Use the 'Pos/Neg Mode' button to toggle between positive and negative points.")
    print("- Use the 'Next Frame' and 'Prev Frame' buttons to navigate between frames.")
    print("- Use the 'Process Video' button to generate segmentations for all frames.")
    
    try:
        segmenter = InteractiveSegmenter(video_dir)
    except Exception as e:
        print(f"Error initializing segmenter: {e}")