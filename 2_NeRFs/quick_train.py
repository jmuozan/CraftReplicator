# Disable PyTorch dynamo to avoid circular import issues with Python 3.13
import os
os.environ["PYTORCH_DISABLE_DYNAMO"] = "1"

import torch
import torch.nn as nn
import numpy as np
import imageio
import matplotlib.pyplot as plt
import json
from torch.utils.data import DataLoader
from dataset import get_rays
from rendering import rendering
from model import Voxels, Nerf
from ml_helpers import training
import sys


# Configuration with minimal parameters for a quick test
dataset_name = "pottery"  # Dataset folder name
hidden_dim = 64            # Reduced hidden dimension for faster training
epochs = 1                 # Just 1 epoch for testing
warmup_epochs = 0          # Skip warmup to save time
batch_size = 512           # Smaller batch size to reduce memory usage
learning_rate = 5e-4       # Standard learning rate
output_dir = "test_output" # Separate output directory for the test run
nb_bins = 64               # Fewer sampling bins to speed up rendering

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

# Analyze dataset to determine appropriate rendering parameters
def analyze_scene_scale(dataset_name):
    """Analyze the scene scale from transforms.json to set appropriate tn/tf values"""
    transforms_path = os.path.join(dataset_name, 'transforms.json')
    if not os.path.exists(transforms_path):
        transforms_path = 'transforms.json'
        if not os.path.exists(transforms_path):
            print(f"Warning: Could not find transforms.json for {dataset_name}")
            return 0.1, 6.0  # Default values
    
    with open(transforms_path, 'r') as f:
        transforms = json.load(f)
    
    # Extract camera positions
    camera_positions = []
    for frame in transforms['frames']:
        c2w = np.array(frame['transform_matrix'], dtype=np.float32)
        pos = c2w[:3, 3]
        camera_positions.append(pos)
    
    camera_positions = np.array(camera_positions, dtype=np.float32)
    
    # Calculate distances from origin
    origin_distances = np.linalg.norm(camera_positions, axis=1)
    
    # Set near/far thresholds based on scene scale
    tn = max(0.1, origin_distances.min() * 0.5)  # Near threshold
    tf = origin_distances.max() * 1.5  # Far threshold
    
    print(f"Dataset: {dataset_name}")
    print(f"Camera distance from origin - min: {origin_distances.min():.2f}, max: {origin_distances.max():.2f}")
    print(f"Using tn={tn:.2f}, tf={tf:.2f}")
    
    return tn, tf

# Set the near and far thresholds based on dataset analysis
tn, tf = analyze_scene_scale(dataset_name)

# Load training rays from transforms.json
print(f"Loading {dataset_name} dataset...")
# For even faster testing, only load a subset of the training data
try:
    import random
    random.seed(42)  # For reproducibility
    
    # First, get a list of image paths from transforms.json
    transforms_path = os.path.join(dataset_name, 'transforms.json')
    if not os.path.exists(transforms_path):
        transforms_path = 'transforms.json'
    
    with open(transforms_path, 'r') as f:
        transforms = json.load(f)
    
    # Get all training paths
    train_paths = [f for f in transforms['frames'] if 'train' in f['file_path']]
    
    # Limit to a maximum of 10 images for quick testing
    max_test_images = 10
    if len(train_paths) > max_test_images:
        print(f"Limiting to {max_test_images} training images for quick testing")
        # Randomly sample a subset for testing
        train_paths = random.sample(train_paths, max_test_images)
        
        # Create a new temporary transforms.json with only these frames
        temp_transforms = transforms.copy()
        temp_transforms['frames'] = train_paths
        
        with open('temp_transforms.json', 'w') as f:
            json.dump(temp_transforms, f)
        
        # Point the get_rays function to this temporary file
        o, d, target_px_values = get_rays('.', mode='train')  # Assumes temp_transforms.json is in current directory
    else:
        o, d, target_px_values = get_rays(dataset_name, mode='train')
except Exception as e:
    print(f"Failed to sample images, using all training data: {e}")
    o, d, target_px_values = get_rays(dataset_name, mode='train')

print(f"Loaded {o.shape[0]} training images with resolution {o.shape[1]} pixels")

# Create data loaders for training
# Force float32 to avoid MPS issues
dataloader = DataLoader(torch.cat((
    torch.from_numpy(o.astype(np.float32)).reshape(-1, 3),
    torch.from_numpy(d.astype(np.float32)).reshape(-1, 3),
    torch.from_numpy(target_px_values.astype(np.float32)).reshape(-1, 3)), dim=1),
    batch_size=batch_size, shuffle=True)

# Load a single test image for evaluation
try:
    test_o, test_d, test_target_px_values = get_rays(dataset_name, mode='test')
    if test_o.shape[0] > 1:
        # Just use the first test image to save time
        test_o = test_o[:1]
        test_d = test_d[:1]
        test_target_px_values = test_target_px_values[:1]
    print(f"Loaded {test_o.shape[0]} test image")
    has_test_data = True
except Exception as e:
    print(f"No test data found: {e}")
    print("Will use first training image for evaluation")
    # Use first training image for testing
    test_o = o[:1]
    test_d = d[:1]
    test_target_px_values = target_px_values[:1]
    has_test_data = False

# Initialize the NeRF model with reduced capacity for faster training
model = Nerf(hidden_dim=hidden_dim, Lpos=6, Ldir=2).to(device)  # Reduced positional encoding dimensions

# Import torch._dynamo here to ensure it exists
import sys
if sys.version_info >= (3, 13):
    import torch._dynamo
    if not hasattr(torch._dynamo, 'config'):
        torch._dynamo.config = type('', (), {})()
        torch._dynamo.config.suppress_errors = True

# Wrap model creation and optimizer in a try-except block to handle potential errors
try:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones=[epochs // 2], 
        gamma=0.5
    )
except Exception as e:
    print(f"Error creating optimizer: {e}")
    print("Trying alternative optimizer setup...")
    # Alternative approach without dynamic loading
    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs, gamma=0.5)

# Run the quick test training
print("\nStarting quick test training...")
print(f"This should take just a few minutes with the reduced settings")
training_loss = training(model, optimizer, scheduler, tn, tf, nb_bins, epochs, dataloader, device=device)

plt.figure(figsize=(10, 5))
plt.plot(training_loss)
plt.title("Training Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.savefig(os.path.join(output_dir, f"{dataset_name}_test_loss.png"))
plt.show()

# Save the model weights
torch.save(model.state_dict(), model_save_path)
print(f"Test model weights saved to '{model_save_path}'")

# Render a test image for visual evaluation
def visualize_test_image(idx=0):
    with torch.no_grad():
        test_batch_size = 4096
        test_idx = idx
        test_h = int(np.sqrt(test_o.shape[1]))
        test_w = test_h
        
        # Get rays for a single test image - explicitly cast to float32
        o_test = torch.from_numpy(test_o[test_idx].astype(np.float32)).to(device)
        d_test = torch.from_numpy(test_d[test_idx].astype(np.float32)).to(device)
        
        # Print information about the test rays
        print(f"Test image {idx}: Ray origins shape: {o_test.shape}")
        print(f"Ray origins range: min={o_test.min().item():.4f}, max={o_test.max().item():.4f}")
        
        # Render in batches to avoid OOM
        img_rendered = []
        for i in range(0, test_h * test_w, test_batch_size):
            end = min(i + test_batch_size, test_h * test_w)
            rendered_batch = rendering(model, o_test[i:end], d_test[i:end], tn, tf, nb_bins, device)
            img_rendered.append(rendered_batch)
            
            # Check if rendering is producing reasonable values
            if i == 0:
                print(f"First render batch stats: min={rendered_batch.min().item():.4f}, max={rendered_batch.max().item():.4f}")
            
        img_rendered = torch.cat(img_rendered, dim=0)
        img_rendered = img_rendered.reshape(test_h, test_w, 3).cpu().numpy()
        
        # Get the ground truth image
        img_truth = test_target_px_values[test_idx].reshape(test_h, test_w, 3)
        
        # Save and display rendered vs ground truth
        plt.figure(figsize=(15, 7))
        plt.subplot(1, 2, 1)
        plt.imshow(np.clip(img_rendered, 0, 1))  # Clip values to valid range
        plt.title("Rendered (Quick Test)")
        plt.subplot(1, 2, 2)
        plt.imshow(img_truth)
        plt.title("Ground Truth")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{dataset_name}_test_render.png"))
        plt.show()
        
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
        plt.savefig(os.path.join(output_dir, f"{dataset_name}_test_render_adjustments.png"))
        plt.show()

# Visualize a test image
print("Rendering test image...")
visualize_test_image(0)

print("\nQuick test complete!")
print(f"If everything looks good, you can now run the full training with:")
print(f"- Increase hidden_dim to 256")
print(f"- Increase epochs to 8-10")
print(f"- Add warmup_epochs = 1")
print(f"- Increase nb_bins to 128")
print(f"- Use all training images instead of the limited subset")