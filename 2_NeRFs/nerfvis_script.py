# Disable PyTorch dynamo to avoid circular import issues with Python 3.13
import os
os.environ["PYTORCH_DISABLE_DYNAMO"] = "1"

import torch
import numpy as np
import json
from nerfvis import scene
from model import Nerf

# Configuration - Change these values as needed
dataset_name = "porcelain"            # Dataset folder name
model_path = "test_output/porcelain_test_model.pth"  # Path to model weights
resolution = 128           # Resolution for sampling (lower for faster visualization)
threshold = 0.7          # Density threshold (lower to capture more points)
port = 8888                # Port for visualization server
output_dir = "nerfvis_output"  # Output directory

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Define device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Fix for PyTorch 2.x compatibility in Python 3.13
import sys
if sys.version_info >= (3, 13):
    import torch._dynamo
    if not hasattr(torch._dynamo, 'config'):
        torch._dynamo.config = type('', (), {})()
        torch._dynamo.config.suppress_errors = True

# Analyze scene bounds from transforms.json
def get_scene_bounds(dataset_name):
    transforms_path = os.path.join(dataset_name, 'transforms.json')
    if not os.path.exists(transforms_path):
        transforms_path = 'transforms.json'
        if not os.path.exists(transforms_path):
            print(f"Warning: Could not find transforms.json for {dataset_name}")
            return (-2, 2), (-2, 2), (-2, 2)  # Default bounds
    
    with open(transforms_path, 'r') as f:
        transforms = json.load(f)
    
    # Extract camera positions
    camera_positions = []
    for frame in transforms['frames']:
        c2w = np.array(frame['transform_matrix'], dtype=np.float32)
        pos = c2w[:3, 3]
        camera_positions.append(pos)
    
    camera_positions = np.array(camera_positions, dtype=np.float32)
    
    # Calculate bounds with padding
    min_pos = camera_positions.min(axis=0) - 2.0  # Add padding
    max_pos = camera_positions.max(axis=0) + 2.0
    
    # Return x, y, z bounds
    return (min_pos[0], max_pos[0]), (min_pos[1], max_pos[1]), (min_pos[2], max_pos[2])

# Get scene bounds
x_bounds, y_bounds, z_bounds = get_scene_bounds(dataset_name)
print(f"Scene bounds: X={x_bounds}, Y={y_bounds}, Z={z_bounds}")

# Function to detect model parameters from checkpoint
def detect_model_parameters(model_path):
    try:
        # Load the state dict without instantiating the model
        state_dict = torch.load(model_path, map_location='cpu')
        
        # Check the dimensions of key parameters to determine hidden_dim
        if 'block1.0.bias' in state_dict:
            hidden_dim = state_dict['block1.0.bias'].shape[0]
        else:
            hidden_dim = 256  # Default if not found
            
        # Try to determine Lpos and Ldir from input layer dimensions
        if 'block1.0.weight' in state_dict:
            input_dim = state_dict['block1.0.weight'].shape[1]
            # Input dim is 3 + Lpos * 6 for block1
            Lpos = (input_dim - 3) // 6
        else:
            Lpos = 10  # Default if not found
            
        # Try to determine Ldir from rgb_head input dimensions
        if 'rgb_head.0.weight' in state_dict:
            rgb_input_dim = state_dict['rgb_head.0.weight'].shape[1]
            # rgb input dim includes hidden_dim + Ldir * 6 + 3
            Ldir = (rgb_input_dim - hidden_dim - 3) // 6
        else:
            Ldir = 4  # Default if not found
            
        print(f"Detected model parameters: hidden_dim={hidden_dim}, Lpos={Lpos}, Ldir={Ldir}")
        return hidden_dim, Lpos, Ldir
    except Exception as e:
        print(f"Error detecting model parameters: {e}")
        print("Using default values: hidden_dim=256, Lpos=10, Ldir=4")
        return 256, 10, 4

# Detect model parameters
hidden_dim, Lpos, Ldir = detect_model_parameters(model_path)

# Initialize the model with the detected parameters
model = Nerf(hidden_dim=hidden_dim, Lpos=Lpos, Ldir=Ldir).to(device)

# Load the saved weights
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Successfully loaded model from {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    raise e
    
model.eval()

# Create a scene
scene.set_title(f"NeRF Visualization - {dataset_name}")

# Add coordinate axes for reference
scene.add_axes("axes", length=1.0)

# Define a 3D grid of points to sample your NeRF based on scene bounds
x = np.linspace(x_bounds[0], x_bounds[1], resolution)
y = np.linspace(y_bounds[0], y_bounds[1], resolution)
z = np.linspace(z_bounds[0], z_bounds[1], resolution)
xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
points = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=-1).astype(np.float32)

# For each point, we'll use multiple viewing directions
num_directions = 4  # Reduced from 6 for quicker testing
directions = []
for i in range(num_directions):
    # Create evenly distributed viewing directions
    theta = np.pi * i / num_directions
    phi = 2 * np.pi * i / num_directions
    dx = np.sin(theta) * np.cos(phi)
    dy = np.sin(theta) * np.sin(phi)
    dz = np.cos(theta)
    directions.append([dx, dy, dz])

directions = np.array(directions, dtype=np.float32)
directions_tensor = torch.tensor(directions, dtype=torch.float32, device=device)

# Evaluate NeRF at these points and directions
print(f"Evaluating NeRF at {points.shape[0]} grid points...")
valid_points = []
valid_colors = []

# Adjust batch size based on available memory
if device.type == 'mps':
    batch_size = 5000  # Smaller batch size for MPS
elif device.type == 'cuda':
    batch_size = 8000  # Larger batch size for CUDA
else:
    batch_size = 5000  # Medium batch size for CPU

num_batches = (points.shape[0] + batch_size - 1) // batch_size

with torch.no_grad():
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, points.shape[0])
        batch_points = torch.tensor(points[start_idx:end_idx], dtype=torch.float32, device=device)
        
        batch_colors_list = []
        batch_densities_list = []
        
        for direction in directions_tensor:
            # Expand direction to match the batch size
            batch_dirs = direction.expand(batch_points.size(0), -1)
            
            # Get model output
            try:
                colors, density = model.intersect(batch_points, batch_dirs)
                batch_colors_list.append(colors)
                batch_densities_list.append(density)
            except Exception as e:
                print(f"Error in model inference: {e}")
                continue
        
        if not batch_colors_list or not batch_densities_list:
            print(f"Skipping batch {i+1}/{num_batches} due to inference errors")
            continue
            
        # Average colors across different view directions
        avg_colors = torch.stack(batch_colors_list).mean(dim=0)
        # Use maximum density across different view directions
        max_densities = torch.stack(batch_densities_list).max(dim=0)[0]
        
        # Get flattened density values
        density_vals = max_densities.flatten()
        
        # Find points above threshold
        batch_valid_indices = torch.where(density_vals > threshold)[0].cpu().numpy()
        
        if len(batch_valid_indices) > 0:
            # Get valid points and colors for this batch
            batch_valid_points = points[start_idx:end_idx][batch_valid_indices]
            batch_valid_colors = avg_colors[batch_valid_indices].cpu().numpy()
            
            # Make sure colors are in valid range [0,1]
            batch_valid_colors = np.clip(batch_valid_colors, 0, 1)
            
            # Append to our valid points and colors
            valid_points.append(batch_valid_points)
            valid_colors.append(batch_valid_colors)
            
        print(f"Processed batch {i+1}/{num_batches} - Found {len(batch_valid_indices)} valid points")

# Combine all valid points and colors
if valid_points:
    all_valid_points = np.vstack(valid_points)
    all_valid_colors = np.vstack(valid_colors)
    
    print(f"Total valid points: {len(all_valid_points)}")
    
    # Visualize the points
    scene.add_points(
        name="nerf_high_density",
        points=all_valid_points,
        vert_color=all_valid_colors,
        point_size=2.0
    )
    
    # Add camera positions from transforms.json
    try:
        transforms_path = os.path.join(dataset_name, 'transforms.json')
        if not os.path.exists(transforms_path):
            transforms_path = 'transforms.json'
            
        with open(transforms_path, 'r') as f:
            transforms = json.load(f)
        
        camera_positions = []
        for frame in transforms['frames']:
            c2w = np.array(frame['transform_matrix'], dtype=np.float32)
            pos = c2w[:3, 3]
            camera_positions.append(pos)
        
        camera_positions = np.array(camera_positions, dtype=np.float32)
        
        # Add camera positions to the scene
        scene.add_points(
            name="cameras",
            points=camera_positions,
            vert_color=np.array([[0, 1, 0]] * len(camera_positions), dtype=np.float32),  # Green for cameras
            point_size=8.0
        )
        
        print(f"Added {len(camera_positions)} camera positions to the scene")
    except Exception as e:
        print(f"Could not add camera positions: {e}")
else:
    print(f"No high-density points found with threshold={threshold}. Try lowering the threshold.")
    
# Add a wireframe cube to show the volume we're sampling
scene.add_wireframe_cube("bounds", 
                         min_pt=[x_bounds[0], y_bounds[0], z_bounds[0]], 
                         max_pt=[x_bounds[1], y_bounds[1], z_bounds[1]], 
                         color=[0.5, 0.5, 0.5])

# Save scene information to file
scene_path = os.path.join(output_dir, f"{dataset_name}_scene.json")
try:
    scene.to_file(scene_path)
    print(f"Scene information saved to {scene_path}")
except Exception as e:
    print(f"Could not save scene to file: {e}")

# Display the scene
print(f"Starting visualization server on port {port}...")
scene.display(port=port, open_browser=True)