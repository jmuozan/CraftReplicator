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
import gc
import traceback
import time  # For timing operations

# Device selection
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    # Set MPS memory optimization for Apple Silicon
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.8"
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Memory optimization settings
torch.set_grad_enabled(False)  # Disable gradient tracking completely

if device.type == "cuda":
    # Lower precision for efficiency while maintaining accuracy
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # Enable TF32 for Ampere GPUs
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    # Set more aggressive GPU memory management
    torch.cuda.empty_cache()
    # Set lower GPU memory fraction if experiencing OOM errors
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

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
    **kwargs,
):
    # Use standard SAM2VideoPredictor without VOS optimizations
    # to maintain compatibility with your version
    hydra_overrides = [
        "++model._target_=sam2.sam2_video_predictor.SAM2VideoPredictor",
    ]

    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # Use only supported parameters for your version
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.03",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.95",
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            "++model.fill_hole_area=12",
            # Remove unsupported parameters
            # "++model.memory_encoder_extra_args.frequency_factor=1.0",
            # "++model.memory_encoder_extra_args.num_scale_levels=16",
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
        map_location = torch.device('cpu')
        print("Loading checkpoint from disk...")
        
        try:
            sd = torch.load(ckpt_path, map_location=map_location, weights_only=True)["model"]
            print("Loaded checkpoint, applying to model...")
            
            missing_keys, unexpected_keys = model.load_state_dict(sd)
            
            del sd
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            if missing_keys:
                logging.error(missing_keys)
                raise RuntimeError()
            if unexpected_keys:
                logging.error(unexpected_keys)
                raise RuntimeError()
            logging.info("Loaded checkpoint successfully")
            
        except RuntimeError as e:
            print(f"Memory error loading checkpoint: {e}")
            print("Try running on a machine with more memory or using a smaller model.")
            raise

def show_mask(mask, ax, obj_id=None, random_color=False, alpha=0.6):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([alpha])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], alpha])
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
    def __init__(self, video_dir, batch_size=1, preload_frames=False, model_id=None, 
                 keyframe_interval=10, propagation_steps=3):
        self.video_dir = video_dir
        self.output_dir = os.path.join(video_dir, "segmentation_output")
        self.mask_dir = os.path.join(self.output_dir, "masks")
        self.overlay_dir = os.path.join(self.output_dir, "overlays")
        
        # Keyframe management for better tracking
        self.keyframe_interval = keyframe_interval  # Re-apply annotations every N frames
        self.propagation_steps = propagation_steps  # Number of propagation refinement steps
        
        # Memory optimization options
        self.batch_size = batch_size
        self.preload_frames = preload_frames
        
        self.model_id = model_id if model_id else "facebook/sam2.1-hiera-large"
        
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
        
        print(f"Found {len(self.frame_names)} frames in {video_dir}")
        
        # We'll only cache the current frame to save memory
        self.frame_cache = {}
        
        # Print memory usage info
        try:
            import psutil
            mem = psutil.virtual_memory()
            print(f"System memory: {mem.available/1024/1024/1024:.1f}GB available out of {mem.total/1024/1024/1024:.1f}GB total")
        except ImportError:
            print("Could not import psutil to check memory usage")
        
        # Initialize SAM2 model
        print(f"Loading SAM2 model ({self.model_id}) on {device} device...")
        gc.collect()
        
        try:
            # Use standard predictor without VOS optimizations for compatibility
            self.predictor = build_sam2_video_predictor_hf(
                self.model_id, 
                device=device
            )
            
            # Initialize with the first frame only to save memory
            print("Initializing model state...")
            self.inference_state = self.predictor.init_state(video_path=video_dir)
            self.predictor.reset_state(self.inference_state)
            print("Model loaded and state initialized successfully")
        except Exception as e:
            print(f"Error loading model on {device}: {e}")
            
            if device.type == "mps":
                try_cpu = input("\nMPS device error. Would you like to try using CPU instead? (y/n, default: y): ").lower().strip()
                if try_cpu != "n":
                    print("Falling back to CPU. This will be slower but may avoid memory issues.")
                    cpu_device = torch.device("cpu")
                    try:
                        self.predictor = build_sam2_video_predictor_hf(self.model_id, device=cpu_device)
                        print("Initializing model state on CPU...")
                        self.inference_state = self.predictor.init_state(video_path=video_dir)
                        self.predictor.reset_state(self.inference_state)
                        print("Model loaded and state initialized successfully on CPU")
                        return
                    except Exception as cpu_e:
                        print(f"Error loading model on CPU as well: {cpu_e}")
            
            print("\nTroubleshooting tips:")
            print("1. If you're getting out-of-memory errors, try:")
            print("   - Close other applications to free system memory")
            print("   - Try using a smaller model (sam2.1-hiera-tiny or sam2.1-hiera-small)")
            print("2. If you're getting invalid watermark ratio errors:")
            print("   - Your PyTorch MPS implementation may have different requirements")
            print("   - Try removing the PYTORCH_MPS_HIGH_WATERMARK_RATIO environment variable")
            print("3. If you're getting unexpected keyword arguments errors:")
            print("   - Your SAM2 version might be different from what this script expects")
            raise
        
        # Initialize interactive variables
        self.points = []  # List of (x, y) coordinates
        self.labels = []  # List of 0 or 1 (negative or positive)
        self.current_frame_idx = 0
        self.current_obj_id = 1
        self.point_mode = 1  # 1 for positive, 0 for negative
        self.video_segments = {}  # To store segmentation results
        
        # Store initial points and reference mask for re-application
        self.initial_points = None
        self.initial_labels = None
        self.reference_mask = None
        
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
        
        # Clear button to reset points
        self.clear_button = Button(plt.axes([0.1, 0.13, 0.15, 0.06]), 'Clear Points')
        self.clear_button.on_clicked(self.clear_points)
        
        # Connect mouse event
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        # Load first frame
        self.load_frame(self.current_frame_idx)
        plt.show()
    
    def clear_points(self, event):
        """Clear all points and reset"""
        self.points = []
        self.labels = []
        self.initial_points = None
        self.initial_labels = None
        self.reference_mask = None
        self.load_frame(self.current_frame_idx)
    
    def load_frame(self, frame_idx):
        self.ax.clear()
        self.current_frame_idx = frame_idx
        
        # Try to get frame from cache first
        if frame_idx in self.frame_cache:
            self.img = self.frame_cache[frame_idx]
        else:
            self.img = Image.open(os.path.join(self.video_dir, self.frame_names[frame_idx]))
            # Cache only current frame
            self.frame_cache = {frame_idx: self.img}
        
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
        
        # Store initial points for reapplication
        if self.initial_points is None:
            self.initial_points = self.points.copy()
            self.initial_labels = self.labels.copy()
        
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
            # Don't clear points when changing frames
            self.load_frame(self.current_frame_idx)
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    def prev_frame(self, event):
        if self.current_frame_idx > 0:
            self.current_frame_idx -= 1
            # Keep points when changing frames
            self.load_frame(self.current_frame_idx)
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    def process_points(self):
        if not self.points:
            return
        
        # Convert to numpy arrays
        points_array = np.array(self.points, dtype=np.float32)
        labels_array = np.array(self.labels, dtype=np.int32)
        
        # Process with the model
        with torch.no_grad():
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=self.current_frame_idx,
                obj_id=self.current_obj_id,
                points=points_array,
                labels=labels_array,
            )
        
        # Store the mask and convert to CPU + numpy immediately to free GPU memory
        if self.current_frame_idx not in self.video_segments:
            self.video_segments[self.current_frame_idx] = {}
        
        # Store the reference mask for the first frame
        mask = (out_mask_logits[0] > 0.0).cpu().numpy()
        self.video_segments[self.current_frame_idx][self.current_obj_id] = mask
        
        if self.reference_mask is None and self.initial_points is not None:
            self.reference_mask = mask.copy()
        
        # Clean up GPU memory
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
    
    def process_video(self, event):
        print("Processing the entire video. This may take a while...")
        
        # Check if we have any points to work with
        if not self.points or not self.labels:
            print("\nERROR: No segmentation points added yet. Please click on objects in the image before processing.")
            print("Click on objects you want to segment (positive points), then click 'Process Video'.")
            return
        
        # Store initial points and labels for keyframe re-application
        if self.initial_points is None:
            self.initial_points = self.points.copy()
            self.initial_labels = self.labels.copy()
        
        try:
            total_frames = len(self.frame_names)
            print(f"Total frames to process: {total_frames}")
            
            # Use improved keyframe-based processing
            self._process_video_with_keyframes()
            
            # Verify outputs exist
            output_files = os.listdir(self.mask_dir) + os.listdir(self.overlay_dir)
            if not output_files:
                print("\nWARNING: No output files were generated. Something went wrong during processing.")
                print("Please try again with different points or check the error messages above.")
            else:
                print(f"Processing complete! {len(output_files)} files saved to {self.output_dir}")
            
        except Exception as e:
            print(f"Error during video processing: {e}")
            traceback.print_exc()
        finally:
            # Clean up memory
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
    
    def _process_video_with_keyframes(self):
        """Process video using keyframes for better tracking"""
        try:
            total_frames = len(self.frame_names)
            
            # Convert initial points to numpy arrays
            points_array = np.array(self.initial_points, dtype=np.float32)
            labels_array = np.array(self.initial_labels, dtype=np.int32)
            
            # Reset the inference state to start fresh
            print("Resetting model state for processing...")
            self.predictor.reset_state(self.inference_state)
            
            # Set initial frame as keyframe
            print(f"Setting initial keyframe at frame 0 with {len(points_array)} points")
            keyframe_indices = [0]  # First frame is always a keyframe
            
            # Apply points to the first frame
            with torch.no_grad():
                _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                    inference_state=self.inference_state,
                    frame_idx=0,
                    obj_id=self.current_obj_id,
                    points=points_array,
                    labels=labels_array,
                )
            
            # Save the first frame's result
            current_frame_data = {
                self.current_obj_id: (out_mask_logits[0] > 0.0).cpu().numpy()
            }
            self.save_single_frame_results(0, current_frame_data)
            
            # Process all frames
            print("Processing frames...")
            for frame_idx in range(1, total_frames):
                # For keyframes, apply the original points
                if frame_idx % self.keyframe_interval == 0:
                    print(f"Setting keyframe at frame {frame_idx}")
                    keyframe_indices.append(frame_idx)
                    
                    # Reset state for this keyframe
                    self.predictor.reset_state(self.inference_state)
                    
                    # Apply points to this keyframe
                    with torch.no_grad():
                        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                            inference_state=self.inference_state,
                            frame_idx=frame_idx,
                            obj_id=self.current_obj_id,
                            points=points_array,
                            labels=labels_array,
                        )
                    
                    # Save this keyframe result
                    current_frame_data = {
                        self.current_obj_id: (out_mask_logits[0] > 0.0).cpu().numpy()
                    }
                    self.save_single_frame_results(frame_idx, current_frame_data)
                    
                    # Process frames between this keyframe and the next
                    start_idx = frame_idx
                    end_idx = min(frame_idx + self.keyframe_interval, total_frames)
                    
                    print(f"Propagating from frame {start_idx} to {end_idx-1}")
                    self._propagate_from_keyframe(start_idx, end_idx)
                
                # Clean up memory periodically
                if frame_idx % 10 == 0:
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
            
            print("Video processing complete!")
            
        except Exception as e:
            print(f"Error during keyframe-based processing: {e}")
            traceback.print_exc()
    
    def _propagate_from_keyframe(self, start_idx, end_idx):
        """Propagate from a keyframe to subsequent frames"""
        try:
            for _ in range(self.propagation_steps):  # Multiple propagation passes
                with torch.no_grad():
                    for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
                        if out_frame_idx > start_idx and out_frame_idx < end_idx:
                            # Save masks for this frame
                            current_frame_data = {
                                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                                for i, out_obj_id in enumerate(out_obj_ids)
                            }
                            self.save_single_frame_results(out_frame_idx, current_frame_data)
                            print(f"Processed frame {out_frame_idx}")
                        
                        # Stop after end_idx frames
                        if out_frame_idx >= end_idx - 1:
                            break
        except Exception as e:
            print(f"Error during propagation: {e}")
            traceback.print_exc()
    
    def save_single_frame_results(self, frame_idx, frame_data):
        """Save segmentation masks and overlays for a single frame"""
        try:
            # Load original frame
            img = Image.open(os.path.join(self.video_dir, self.frame_names[frame_idx]))
            img_np = np.array(img)
            
            # Create overlay figure
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(img_np)
            
            # Add masks
            for obj_id, mask in frame_data.items():
                # Ensure mask has the right shape and format for PIL
                if mask.ndim != 2:
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
            plt.close(fig)  # Close figure to free memory
            
            # Force garbage collection
            gc.collect()
        except Exception as e:
            print(f"Error saving results for frame {frame_idx}: {e}")

if __name__ == "__main__":
    # Set the directory with the video frames
    video_dir = input("Enter the directory containing video frames (e.g., ./ceramics_frames): ")
    
    if not os.path.exists(video_dir):
        print(f"Error: Directory {video_dir} does not exist")
        exit(1)
    
    # Default model selection
    model_id = "facebook/sam2.1-hiera-large"
    
    # Ask about MPS watermark ratio if on MPS device
    if device.type == "mps":
        print("\nYou're using an MPS device (Apple Silicon).")
        watermark_choice = input("Would you like to disable the MPS watermark ratio limit? This may help with memory errors. (y/n, default: y): ").lower().strip()
        if watermark_choice != "n":
            print("Disabling MPS watermark ratio limit.")
            os.environ.pop("PYTORCH_MPS_HIGH_WATERMARK_RATIO", None)
            print("PyTorch MPS will now allocate memory as needed.")
    
    # Check available memory and suggest model options
    try:
        import psutil
        mem = psutil.virtual_memory()
        available_gb = mem.available/1024/1024/1024
        total_gb = mem.total/1024/1024/1024
        print(f"System memory: {available_gb:.1f}GB available out of {total_gb:.1f}GB total")
        
        if available_gb < 8 and device.type in ["mps", "cuda"]:
            print("\nWarning: Relatively low memory available.")
            print("You may experience out-of-memory errors with the large model.")
            use_large = input("Do you want to proceed with the large model anyway? (y/n, default: n): ").lower().strip() == 'y'
            
            if not use_large:
                model_options = {
                    "1": "facebook/sam2.1-hiera-tiny",
                    "2": "facebook/sam2.1-hiera-small",
                    "3": "facebook/sam2.1-hiera-base-plus",
                    "4": "facebook/sam2.1-hiera-large",
                    "5": "Use CPU instead of GPU (slower but more memory available)"
                }
                
                print("\nPlease select a model size:")
                for key, value in model_options.items():
                    print(f"{key}: {value}")
                
                choice = input("Enter choice (1-5, default: 2): ").strip()
                if not choice:
                    choice = "2"
                
                if choice == "5":
                    print("Using CPU for processing instead of GPU")
                    device = torch.device("cpu")
                else:
                    model_id = model_options.get(choice, "facebook/sam2.1-hiera-small")
                    print(f"Selected model: {model_id}")
    except ImportError:
        # No psutil, proceed with default
        pass
    
    # Clear up memory before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    # Ask for keyframe interval
    keyframe_interval = 10  # Default
    try:
        keyframe_input = input("\nEnter keyframe interval (how often to re-apply annotations, default: 10): ")
        if keyframe_input.strip():
            keyframe_interval = int(keyframe_input)
    except ValueError:
        print("Invalid value, using default of 10")
    
    # Ask for propagation steps
    propagation_steps = 3  # Default
    try:
        propagation_input = input("\nEnter propagation refinement steps (higher values improve quality but are slower, default: 3): ")
        if propagation_input.strip():
            propagation_steps = int(propagation_input)
    except ValueError:
        print("Invalid value, using default of 3")
    
    # Start the interactive segmenter
    print("\n=== HOW TO USE THIS IMPROVED TOOL ===")
    print("1. When the image appears, CLICK ON THE OBJECT you want to segment")
    print("2. By default, your clicks will add positive points (green markers)")
    print("3. Use the 'Pos/Neg Mode' button to toggle to negative mode (red markers) to exclude areas")
    print("4. Once you've added at least one point, click 'Process Video' to generate masks for all frames")
    print("5. Use 'Clear Points' if you want to start over")
    print("6. Results will be saved to the 'segmentation_output' folder in your frames directory")
    print("\nIMPROVEMENTS:")
    print(f"- Your annotations will be re-applied every {keyframe_interval} frames for better tracking")
    print(f"- Each segment will be refined with {propagation_steps} propagation passes")
    print("- Better memory management and object tracking")
    print("\nIMPORTANT: You MUST click on the object you want to segment before processing the video!")
    print("=========================\n")
    
    try:
        # Use batch processing and don't preload frames to save memory
        segmenter = InteractiveSegmenter(
            video_dir, 
            batch_size=1, 
            preload_frames=False, 
            model_id=model_id,
            keyframe_interval=keyframe_interval,
            propagation_steps=propagation_steps
        )
    except Exception as e:
        print(f"Error initializing segmenter: {e}")
        traceback.print_exc()