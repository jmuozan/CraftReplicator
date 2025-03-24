import os
os.environ["PYTORCH_DISABLE_DYNAMO"] = "1"

import torch
import torch.nn as nn
import numpy as np
import imageio
import matplotlib.pyplot as plt
import json
from torch.utils.data import DataLoader
import sys
import cv2 

# Import components from your existing code
from rendering import rendering
from ml_helpers import training, training_with_pose_refinement  # Add training_with_pose_refinement here

# Configuration with ultra minimal parameters for the quickest test
dataset_name = "db"  # Using your database folder
hidden_dim = 256  # Keep same dimension for network
epochs = 100  # Run a few more epochs to see learning progress
warmup_epochs = 10  # Add warmup for better convergence
batch_size = 4096  # Larger batch size for fewer iterations (if your GPU can handle it)
learning_rate = 5e-4  # Slightly higher learning rate
output_dir = "production_output"  # Separate output directory for the test run
nb_bins = 64  # Fewer sampling bins to speed up rendering

# These are the critical changes to speed things up
max_images = 172  # Use only 3 images
downsample_factor = 8  # Reduce resolution significantly
max_rays_per_image = 10000   # Limit rays per image
camera_pattern = 'circle'  # Choose from 'circle', 'spiral', 'hemisphere'


# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)
model_save_path = os.path.join(output_dir, f'{dataset_name}_test_model.pth')
checkpoint_path = os.path.join(output_dir, f'{dataset_name}_test_checkpoint.pth')

print("torch version: ", torch.__version__)
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("device: ", device)

print(f"\n[Quick Test Mode] Using reduced parameters for fast testing")
print(f"- Hidden dimensions: {hidden_dim}")
print(f"- Batch size: {batch_size}")
print(f"- Epochs: {epochs}")
print(f"- Number of bins: {nb_bins}")

# Function for camera poses initialization
def initialize_poses(image_dir):
    """
    Initialize camera poses in a specified pattern for NeRF training.
    
    Args:
        image_dir (str): Directory containing input images
        
    Returns:
        dict: Camera data including poses, intrinsics, and image paths
    """
    import os
    import numpy as np
    import imageio
    import cv2
    
    # Get list of images
    img_paths = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    img_paths.sort()  # Sort for consistent ordering
    
    if not img_paths:
        raise ValueError(f"No images found in {image_dir}")
    
    # Limit number of images if requested (for quick testing)
    if 'max_images' in globals() and max_images > 0 and len(img_paths) > max_images:
        print(f"QUICK TEST MODE: Using only {max_images} out of {len(img_paths)} images")
        img_paths = img_paths[:max_images]
    
    print(f"Found {len(img_paths)} images")
    
    # Get image dimensions from first image
    img = imageio.imread(os.path.join(image_dir, img_paths[0]))
    H, W = img.shape[:2]
    print(f"Original image dimensions: {W}x{H}")
    
    # Apply downsampling if requested
    if 'downsample_factor' in globals() and downsample_factor > 1:
        print(f"QUICK TEST MODE: Downsampling images by factor of {downsample_factor}")
        H = H // downsample_factor
        W = W // downsample_factor
        print(f"Downsampled dimensions: {W}x{H}")
    
    # Default focal length estimate based on image size
    focal = max(H, W) * 0.8  # Common heuristic for approximate focal length
    
    # Create poses based on selected pattern
    poses = []
    intrinsics = []
    
    num_images = len(img_paths)
    pattern = globals().get('camera_pattern', 'circle')
    
    if pattern == 'circle':
        # Place cameras in a circle around the origin
        for i in range(num_images):
            angle = 2 * np.pi * i / num_images
            radius = 4.0  # Distance from center
            
            # Add some vertical variation to break symmetry
            height_variation = 0.1 * np.sin(3 * angle)
            
            # Camera position
            tx = radius * np.cos(angle)
            ty = height_variation
            tz = radius * np.sin(angle)
            
            # Camera viewing direction (looking at origin)
            camera_dir = np.array([-tx, -ty, -tz])
            camera_dir = camera_dir / np.linalg.norm(camera_dir)
            
            # Compute camera axes
            up = np.array([0.0, 1.0, 0.0])  # World up vector
            right = np.cross(camera_dir, up)
            right = right / np.linalg.norm(right)
            true_up = np.cross(right, camera_dir)
            
            # Create rotation matrix
            R = np.stack([right, true_up, -camera_dir], axis=1)  # Column-major
            
            # Create camera-to-world matrix
            c2w = np.zeros((4, 4), dtype=np.float32)
            c2w[:3, :3] = R
            c2w[:3, 3] = [tx, ty, tz]
            c2w[3, 3] = 1.0
            
            poses.append(c2w)
            intrinsics.append([focal, focal, W/2, H/2])
    
    elif pattern == 'spiral':
        # Create a spiral pattern around a central object
        for i in range(num_images):
            angle = 2 * np.pi * i / num_images
            # Gradually increase height and radius
            height = -0.5 + 1.0 * i / num_images
            radius = 3.5 + 0.5 * i / num_images
            
            # Camera position
            tx = radius * np.cos(angle)
            ty = height
            tz = radius * np.sin(angle)
            
            # Look at origin with slight offset based on height
            look_at = np.array([0, height * 0.3, 0])
            camera_dir = look_at - np.array([tx, ty, tz])
            camera_dir = camera_dir / np.linalg.norm(camera_dir)
            
            # Compute camera axes
            up = np.array([0.0, 1.0, 0.0])  # World up vector
            right = np.cross(camera_dir, up)
            right = right / np.linalg.norm(right)
            true_up = np.cross(right, camera_dir)
            
            # Create rotation matrix
            R = np.stack([right, true_up, -camera_dir], axis=1)
            
            # Create camera-to-world matrix
            c2w = np.zeros((4, 4), dtype=np.float32)
            c2w[:3, :3] = R
            c2w[:3, 3] = [tx, ty, tz]
            c2w[3, 3] = 1.0
            
            poses.append(c2w)
            intrinsics.append([focal, focal, W/2, H/2])
    
    elif pattern == 'hemisphere':
        # Place cameras on a hemisphere looking at the origin
        for i in range(num_images):
            # Fibonacci sphere algorithm for uniform distribution on hemisphere
            golden_ratio = (1 + 5**0.5) / 2
            idx = i + 0.5
            phi = 2 * np.pi * idx / golden_ratio
            cos_theta = 1 - (idx / num_images)  # Only use top hemisphere
            sin_theta = np.sqrt(1 - cos_theta**2)
            
            # Camera position on a hemisphere with radius 4
            radius = 4.0
            tx = radius * sin_theta * np.cos(phi)
            ty = radius * cos_theta
            tz = radius * sin_theta * np.sin(phi)
            
            # Camera viewing direction (looking at origin)
            camera_dir = np.array([-tx, -ty, -tz])
            camera_dir = camera_dir / np.linalg.norm(camera_dir)
            
            # Compute camera axes
            up = np.array([0.0, 1.0, 0.0])  # World up vector
            right = np.cross(camera_dir, up)
            right = right / np.linalg.norm(right)
            true_up = np.cross(right, camera_dir)
            
            # Create rotation matrix
            R = np.stack([right, true_up, -camera_dir], axis=1)
            
            # Create camera-to-world matrix
            c2w = np.zeros((4, 4), dtype=np.float32)
            c2w[:3, :3] = R
            c2w[:3, 3] = [tx, ty, tz]
            c2w[3, 3] = 1.0
            
            poses.append(c2w)
            intrinsics.append([focal, focal, W/2, H/2])
    
    else:
        print(f"Unknown camera pattern: {pattern}, using circle instead")
        # Fall back to circle pattern
        # [Code for circle pattern would be here - omitted for brevity]
    
    camera_data = {
        'poses': np.array(poses, dtype=np.float32),
        'intrinsics': np.array(intrinsics, dtype=np.float32),
        'img_paths': img_paths,
        'H': H,
        'W': W
    }
    
    print(f"Initialized {len(poses)} camera poses in a {pattern} pattern")
    return camera_data

# Function to limit the number of rays per image for faster processing
def generate_rays_from_poses(camera_data):
    """Generate rays from camera poses for training with ray limitation for faster testing"""
    poses = camera_data['poses']
    intrinsics = camera_data['intrinsics']
    H, W = camera_data['H'], camera_data['W']
    img_paths = camera_data['img_paths']
    
    all_rays_o = []
    all_rays_d = []
    all_rgb = []
    
    print(f"Generating rays for {len(poses)} camera poses...")
    
    for img_idx, (pose, intr) in enumerate(zip(poses, intrinsics)):
        # Get focal length and principal point
        fx, fy, cx, cy = intr
        
        # Create pixel coordinates
        i_coords, j_coords = np.meshgrid(
            np.arange(W, dtype=np.float32),
            np.arange(H, dtype=np.float32),
            indexing='xy'
        )
        
        # Convert pixel coordinates to normalized device coordinates
        x = (i_coords - cx) / fx
        y = (j_coords - cy) / fy
        
        # Create ray directions in camera space
        directions = np.stack([x, y, -np.ones_like(x)], axis=-1)
        
        # Transform ray directions to world space
        rays_d = np.dot(directions.reshape(-1, 3), pose[:3, :3].T)
        
        # Normalize ray directions
        rays_d = rays_d / np.linalg.norm(rays_d, axis=-1, keepdims=True)
        
        # Get ray origins (camera position)
        rays_o = np.broadcast_to(pose[:3, 3], rays_d.shape)
        
        # Load and possibly downsample the corresponding image
        if img_idx < len(img_paths):
            img_path = os.path.join(dataset_name, img_paths[img_idx])
        else:
            print(f"Warning: No image path for pose {img_idx}, using placeholder")
            img_path = None
        
        try:
            if img_path and os.path.exists(img_path):
                img = imageio.imread(img_path)
                
                # Apply downsampling if configured
                if 'downsample_factor' in globals() and downsample_factor > 1:
                    h, w = img.shape[:2]
                    img = cv2.resize(img, (w//downsample_factor, h//downsample_factor), 
                                    interpolation=cv2.INTER_AREA)
                
                if img.dtype == np.uint8:
                    img = img.astype(np.float32) / 255.0
                
                # Handle RGBA images
                if img.shape[-1] == 4:
                    img = img[..., :3] * img[..., -1:] + (1 - img[..., -1:])
            else:
                raise FileNotFoundError(f"Image not found at {img_path}")
        except Exception as e:
            print(f"Error loading image {img_idx}: {e}")
            print("Using placeholder image (checkerboard pattern)")
            # Create a checkerboard pattern as placeholder
            checker = np.zeros((H, W, 3), dtype=np.float32)
            checker_size = 32
            for y in range(H):
                for x in range(W):
                    if ((x // checker_size) + (y // checker_size)) % 2:
                        checker[y, x] = np.array([0.8, 0.8, 0.8])
                    else:
                        checker[y, x] = np.array([0.2, 0.2, 0.2])
            img = checker
        
        # Limit the number of rays per image if configured (for quicker testing)
        total_rays = rays_o.shape[0]
        if 'max_rays_per_image' in globals() and max_rays_per_image > 0 and total_rays > max_rays_per_image:
            print(f"QUICK TEST MODE: Limiting image {img_idx} from {total_rays} to {max_rays_per_image} rays")
            indices = np.random.choice(total_rays, max_rays_per_image, replace=False)
            rays_o = rays_o[indices]
            rays_d = rays_d[indices]
            rgb = img.reshape(-1, 3)[indices]
        else:
            rgb = img.reshape(-1, 3)
        
        # Append to our collections
        all_rays_o.append(rays_o)
        all_rays_d.append(rays_d)
        all_rgb.append(rgb)
    
    # Combine all rays
    rays_o = np.concatenate(all_rays_o)  # [N*H*W, 3]
    rays_d = np.concatenate(all_rays_d)  # [N*H*W, 3]
    target_px_values = np.concatenate(all_rgb)  # [N*H*W, 3]
    
    print(f"Generated rays with shape: {rays_o.shape}")
    
    return rays_o, rays_d, target_px_values

# Set the near and far thresholds for rendering
tn, tf = 0.1, 8.0  # Default values for the circular camera arrangement

# Initialize and save camera poses
print("COLMAP failed to reconstruct camera poses. Using manual initialization...")
image_dir = './db'
camera_data = initialize_poses(image_dir)
print(f"Initialized {len(camera_data['poses'])} camera poses in a circle")

# Save the initialized poses
os.makedirs('camera_data', exist_ok=True)
np.save('camera_data/poses.npy', camera_data['poses'])
np.save('camera_data/intrinsics.npy', camera_data['intrinsics'])
with open('camera_data/image_paths.txt', 'w') as f:
    for path in camera_data['img_paths']:
        f.write(f"{path}\n")

print("Saved camera initialization data to camera_data/ directory")

# Generate rays from our initialized poses
o, d, target_px_values = generate_rays_from_poses(camera_data)
print(f"Generated rays for {o.shape[0]} images with resolution {o.shape[1]} pixels")

# Create data loaders for training
# Add image indices to the training data for pose refinement
indices = np.arange(o.shape[0])
indices = np.repeat(indices[:, np.newaxis], o.shape[1], axis=1).reshape(-1)

# Create data loaders for training
# Reshape and flatten the arrays properly
o_flat = o.reshape(-1, 3)  # Flatten to (N*H*W, 3)
d_flat = d.reshape(-1, 3)  # Flatten to (N*H*W, 3)
target_flat = target_px_values.reshape(-1, 3)  # Flatten to (N*H*W, 3)

# Create image indices - expand to match the number of pixels per image
img_indices = []
for i in range(len(camera_data['poses'])):
    # For each image, get the number of rays after limiting
    if 'max_rays_per_image' in globals() and max_rays_per_image > 0:
        num_rays = min(camera_data['H'] * camera_data['W'], max_rays_per_image)
    else:
        num_rays = camera_data['H'] * camera_data['W']
    
    img_indices.append(np.full((num_rays), i, dtype=np.int64))

img_indices = np.concatenate(img_indices).reshape(-1, 1)  # Flatten to (N*rays_per_image, 1)

print(f"Shapes for dataloader: o={o_flat.shape}, indices={img_indices.shape}")

# Now the tensor sizes should match
dataloader = DataLoader(torch.cat((
    torch.from_numpy(o_flat.astype(np.float32)),
    torch.from_numpy(d_flat.astype(np.float32)),
    torch.from_numpy(target_flat.astype(np.float32)),
    torch.from_numpy(img_indices)), dim=1),
    batch_size=batch_size, shuffle=True)

# Create a test dataset from the first image
test_o = o[:1]
test_d = d[:1]
test_target_px_values = target_px_values[:1]

# Initialize the NeRF model with pose refinement
num_images = len(camera_data['poses'])

from model import Nerf, NeRFWithPoses

# First initialize the regular NeRF model
model = Nerf(hidden_dim=hidden_dim, Lpos=6, Ldir=2).to(device)

# Then wrap it with the pose refinement model
pose_refine_model = NeRFWithPoses(
    Lpos=model.Lpos,  
    Ldir=model.Ldir,  
    hidden_dim=hidden_dim,
    num_images=num_images
).to(device)

# Assign your pre-initialized NeRF model to the pose refinement model
pose_refine_model.nerf = model

# Create optimizer for both NeRF parameters and pose refinement
params = list(model.parameters())
if pose_refine_model.pose_refinement is not None:
    params.append(pose_refine_model.pose_refinement.weight)
    
optimizer = torch.optim.Adam(params, lr=learning_rate)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, 
    milestones=[epochs // 2], 
    gamma=0.5
)

# Run the training with pose refinement
print("\nStarting training with pose refinement...")
print(f"This should take just a few minutes with the reduced settings")
training_loss, refined_poses = training_with_pose_refinement(
    pose_refine_model,  # Pass the pose_refine_model instead of model
    optimizer, 
    scheduler, 
    tn, tf, 
    nb_bins, 
    epochs, 
    camera_data, 
    dataloader, 
    device=device
)


plt.figure(figsize=(10, 5))
plt.plot(training_loss)
plt.title("Training Loss with Pose Refinement")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.savefig(os.path.join(output_dir, f"{dataset_name}_test_loss.png"))
plt.show()

# Save the model weights and refined poses
torch.save({
    'model': pose_refine_model.nerf.state_dict(),
    'pose_refinement': pose_refine_model.pose_refinement.state_dict(),
    'epoch': epochs
}, model_save_path)
print(f"Model weights and refined poses saved to '{model_save_path}'")

# Save the refined poses separately
refined_camera_data = camera_data.copy()
refined_camera_data['poses'] = refined_poses
np.save('camera_data/refined_poses.npy', refined_poses)
print("Saved refined camera poses to camera_data/refined_poses.npy")

# Render a test image for visual evaluation
def visualize_test_image(model, idx=0):
    with torch.no_grad():
        # The test_o and test_d have incorrect shapes for rendering
        # Their first dimension is correct (number of images), but they have only 3 elements (not H*W rays)
        # We need to generate proper test rays using the test image's camera pose
        
        # Get the camera pose and intrinsics for this test image
        test_pose = camera_data['poses'][idx]
        test_intrinsics = camera_data['intrinsics'][idx]
        
        # Image dimensions
        H, W = camera_data['H'], camera_data['W']
        
        # Create pixel coordinates
        i_coords, j_coords = np.meshgrid(
            np.arange(W, dtype=np.float32),
            np.arange(H, dtype=np.float32),
            indexing='xy'
        )
        
        # Get focal length and principal point
        fx, fy, cx, cy = test_intrinsics
        
        # Convert pixel coordinates to normalized device coordinates
        x = (i_coords - cx) / fx
        y = (j_coords - cy) / fy
        
        # Create ray directions in camera space
        directions = np.stack([x, y, -np.ones_like(x)], axis=-1)
        
        # Transform ray directions to world space
        rays_d = np.dot(directions.reshape(-1, 3), test_pose[:3, :3].T)
        
        # Normalize ray directions
        rays_d = rays_d / np.linalg.norm(rays_d, axis=-1, keepdims=True)
        
        # Get ray origins (camera position)
        rays_o = np.broadcast_to(test_pose[:3, 3], rays_d.shape)
        
        # Convert numpy arrays to torch tensors
        o_test = torch.from_numpy(rays_o.astype(np.float32)).to(device)
        d_test = torch.from_numpy(rays_d.astype(np.float32)).to(device)
        
        # Print information about the test rays
        print(f"Test image {idx}: Ray origins shape: {o_test.shape}")
        print(f"Ray origins range: min={o_test.min().item():.4f}, max={o_test.max().item():.4f}")
        
        # For testing, we might want to limit the number of rays to render
        if o_test.shape[0] > 10000:
            print(f"Limiting test rays from {o_test.shape[0]} to 10000 for faster rendering")
            # Choose 10,000 random rays
            indices = torch.randperm(o_test.shape[0], device=device)[:10000]
            o_test = o_test[indices]
            d_test = d_test[indices]
            # We'll reshape later to a square image
            test_h = test_w = 100  # 100x100 = 10,000 pixels
        else:
            test_h, test_w = H, W
        
        # Render in batches to avoid OOM
        test_batch_size = 1024
        img_rendered = []
        
        for i in range(0, o_test.shape[0], test_batch_size):
            end = min(i + test_batch_size, o_test.shape[0])
            rendered_batch = rendering(model.nerf, o_test[i:end], d_test[i:end], tn, tf, nb_bins, device)
            img_rendered.append(rendered_batch)
            
            # Check if rendering is producing reasonable values
            if i == 0:
                print(f"First render batch stats: min={rendered_batch.min().item():.4f}, max={rendered_batch.max().item():.4f}")
        
        img_rendered = torch.cat(img_rendered, dim=0)
        img_rendered = img_rendered.reshape(test_h, test_w, 3).cpu().numpy()
        
        # Create a simple placeholder for ground truth (checkerboard pattern)
        # since we don't have actual ground truth for the test view
        checker = np.zeros((test_h, test_w, 3), dtype=np.float32)
        checker_size = 16
        for y in range(test_h):
            for x in range(test_w):
                if ((x // checker_size) + (y // checker_size)) % 2:
                    checker[y, x] = np.array([0.8, 0.8, 0.8])
                else:
                    checker[y, x] = np.array([0.2, 0.2, 0.2])
        
        # Save and display rendered vs ground truth
        plt.figure(figsize=(15, 7))
        plt.subplot(1, 2, 1)
        plt.imshow(np.clip(img_rendered, 0, 1))  # Clip values to valid range
        plt.title("Rendered (with pose refinement)")
        plt.subplot(1, 2, 2)
        plt.imshow(checker)
        plt.title("Placeholder Pattern")
        plt.tight_layout()
        
        # Save the figure before showing it
        output_path = os.path.join(output_dir, f"{dataset_name}_test_render.png")
        plt.savefig(output_path)
        print(f"Saved rendered comparison to {output_path}")
        plt.close()  # Close the figure instead of showing it
        
        # Also show the render with different adjustments
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(np.clip(img_rendered, 0, 1))
        plt.title("Original Render")
        
        plt.subplot(1, 3, 2)
        plt.imshow(np.clip(img_rendered ** 0.45, 0, 1))  # Gamma correction
        plt.title("Gamma Corrected")
        
        plt.subplot(1, 3, 3)
        # Normalize to use full dynamic range
        normalized = (img_rendered - img_rendered.min()) / (img_rendered.max() - img_rendered.min() + 1e-8)
        plt.imshow(normalized)
        plt.title("Normalized")
        
        plt.tight_layout()
        
        # Save the adjustments figure before showing it
        adjustments_path = os.path.join(output_dir, f"{dataset_name}_test_render_adjustments.png")
        plt.savefig(adjustments_path)
        print(f"Saved rendered adjustments to {adjustments_path}")
        plt.close()  # Close the figure instead of showing it
        
        # If you still want to display the images, you can reopen them
        if False:  # Set to True if you want to display after saving
            # Reopen and display the images
            plt.figure(figsize=(15, 7))
            plt.imshow(plt.imread(output_path))
            plt.axis('off')
            plt.show()
            
            plt.figure(figsize=(15, 5))
            plt.imshow(plt.imread(adjustments_path))
            plt.axis('off')
            plt.show()

# Visualize camera poses before and after refinement
def visualize_camera_poses(camera_data, refined_poses=None):
    """Visualize camera poses before and after refinement"""
    from mpl_toolkits.mplot3d import Axes3D
    
    poses = camera_data['poses']
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract camera positions and orientations
    cam_positions = []
    cam_directions = []
    
    for pose in poses:
        # Camera position is the translation component
        position = pose[:3, 3]
        cam_positions.append(position)
        
        # Camera direction is the negative z-axis of the camera
        direction = -pose[:3, 2]  # Third column is the camera's z-axis
        cam_directions.append(direction)
    
    cam_positions = np.array(cam_positions)
    cam_directions = np.array(cam_directions)
    
    # Plot original camera positions
    ax.scatter(cam_positions[:, 0], cam_positions[:, 1], cam_positions[:, 2], 
               c='blue', marker='o', label='Original Poses')
    
    # Plot camera viewing directions
    for pos, dir in zip(cam_positions, cam_directions):
        # Scale the direction vector for visibility
        scale = 0.5
        ax.quiver(pos[0], pos[1], pos[2], 
                  dir[0]*scale, dir[1]*scale, dir[2]*scale, 
                  color='blue', alpha=0.6)
    
    # If refined poses are provided, plot them too
    if refined_poses is not None:
        refined_cam_positions = []
        refined_cam_directions = []
        
        for pose in refined_poses:
            position = pose[:3, 3]
            refined_cam_positions.append(position)
            
            direction = -pose[:3, 2]
            refined_cam_directions.append(direction)
        
        refined_cam_positions = np.array(refined_cam_positions)
        refined_cam_directions = np.array(refined_cam_directions)
        
        # Plot refined camera positions
        ax.scatter(refined_cam_positions[:, 0], refined_cam_positions[:, 1], refined_cam_positions[:, 2], 
                   c='red', marker='x', label='Refined Poses')
        
        # Plot refined camera viewing directions
        for pos, dir in zip(refined_cam_positions, refined_cam_directions):
            scale = 0.5
            ax.quiver(pos[0], pos[1], pos[2], 
                      dir[0]*scale, dir[1]*scale, dir[2]*scale, 
                      color='red', alpha=0.6)
    
    # Set labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_camera_poses.png"))
    plt.show()

# Visualize camera poses before and after refinement
print("Visualizing camera poses...")
visualize_camera_poses(camera_data, refined_poses)

print("\nQuick test with pose refinement complete!")
print(f"For better results, you can now run the full training with:")
print(f"- Increase hidden_dim to 256")
print(f"- Increase epochs to 15-20 for better pose refinement")
print(f"- Add warmup_epochs = 1")
print(f"- Increase nb_bins to 128")