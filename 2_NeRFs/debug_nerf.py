import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from dataset import get_rays
from model import Nerf
import rendering

# Load the transforms.json file to examine scene scales
def examine_transforms_file(json_path='transforms.json'):
    """Examine the transforms.json file to determine appropriate scale parameters"""
    if not os.path.exists(json_path):
        json_path = os.path.join('fox', json_path)
    
    with open(json_path, 'r') as f:
        transforms = json.load(f)
    
    # Extract camera positions to find the scale of the scene
    camera_positions = []
    for frame in transforms['frames']:
        c2w = np.array(frame['transform_matrix'], dtype=np.float32)
        pos = c2w[:3, 3]
        camera_positions.append(pos)
    
    camera_positions = np.array(camera_positions, dtype=np.float32)
    
    # Calculate statistics about camera positions
    min_pos = camera_positions.min(axis=0)
    max_pos = camera_positions.max(axis=0)
    mean_pos = camera_positions.mean(axis=0)
    
    # Calculate distances between cameras
    camera_distances = []
    for i in range(len(camera_positions)):
        for j in range(i+1, len(camera_positions)):
            dist = np.linalg.norm(camera_positions[i] - camera_positions[j])
            camera_distances.append(dist)
    
    camera_distances = np.array(camera_distances, dtype=np.float32)
    
    # Calculate distances from camera to origin
    origin_distances = np.linalg.norm(camera_positions, axis=1)
    
    print(f"Camera position statistics:")
    print(f"Min: {min_pos}")
    print(f"Max: {max_pos}")
    print(f"Mean: {mean_pos}")
    print(f"Range: {max_pos - min_pos}")
    print(f"\nCamera distances statistics:")
    print(f"Min distance between cameras: {camera_distances.min():.4f}")
    print(f"Max distance between cameras: {camera_distances.max():.4f}")
    print(f"Mean distance between cameras: {camera_distances.mean():.4f}")
    print(f"\nDistances from origin:")
    print(f"Min distance from origin: {origin_distances.min():.4f}")
    print(f"Max distance from origin: {origin_distances.max():.4f}")
    print(f"Mean distance from origin: {origin_distances.mean():.4f}")
    
    # Based on the analysis, suggest near/far thresholds
    min_dist = max(0.1, origin_distances.min() * 0.8)
    max_dist = origin_distances.max() * 1.2
    
    print(f"\nSuggested near threshold (tn): {min_dist:.2f}")
    print(f"Suggested far threshold (tf): {max_dist:.2f}")
    
    # Plot camera positions
    plt.figure(figsize=(10, 8))
    plt.scatter(camera_positions[:, 0], camera_positions[:, 2], c='blue', alpha=0.7)
    plt.scatter([0], [0], c='red', marker='x', s=100)  # Origin
    plt.title('Camera Positions (Top-Down View)')
    plt.xlabel('X Axis')
    plt.ylabel('Z Axis')
    plt.axis('equal')
    plt.grid(True)
    plt.show()
    
    return min_dist, max_dist

# Test ray generation to verify it's working correctly
def test_ray_generation():
    """Test ray generation from transforms.json"""
    test_o, test_d, _ = get_rays('fox', mode='test')
    
    print(f"Ray origins shape: {test_o.shape}")
    print(f"Ray directions shape: {test_d.shape}")
    
    # Check if ray directions are normalized
    ray_lengths = np.linalg.norm(test_d.reshape(-1, 3), axis=1)
    print(f"Ray direction lengths (should be ~1.0): min={ray_lengths.min():.6f}, max={ray_lengths.max():.6f}")
    
    # Check for NaN or inf values
    print(f"NaN in ray origins: {np.isnan(test_o).any()}")
    print(f"NaN in ray directions: {np.isnan(test_d).any()}")
    print(f"Inf in ray origins: {np.isinf(test_o).any()}")
    print(f"Inf in ray directions: {np.isinf(test_d).any()}")
    
    # Visualize rays from first image
    test_idx = 0
    test_h = int(np.sqrt(test_o.shape[1]))
    test_w = test_h
    
    # Plot ray origins in 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Sample rays for visualization (too many points will slow things down)
    sample_rate = 20
    o_sample = test_o[test_idx, ::sample_rate]
    d_sample = test_d[test_idx, ::sample_rate]
    
    # Plot ray origins
    ax.scatter(o_sample[:, 0], o_sample[:, 1], o_sample[:, 2], c='blue', alpha=0.5, s=1)
    
    # Plot ray directions for a few points
    for i in range(0, len(o_sample), 5):
        ax.quiver(o_sample[i, 0], o_sample[i, 1], o_sample[i, 2],
                  d_sample[i, 0], d_sample[i, 1], d_sample[i, 2],
                  length=0.5, color='red', alpha=0.7)
    
    ax.set_title('Ray Origins and Directions')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    
    return test_o, test_d

# Test rendering with different tn/tf values
def test_rendering(model_path='nerf_model_weights.pth'):
    """Test rendering with different tn/tf values"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    # Get test data
    test_o, test_d, test_target = get_rays('fox', mode='test')
    test_idx = 0
    test_h = int(np.sqrt(test_o.shape[1]))
    test_w = test_h
    
    # Create model and load weights
    model = Nerf(hidden_dim=256, Lpos=10, Ldir=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Get sample rays for testing
    o_test = torch.from_numpy(test_o[test_idx].astype(np.float32)).to(device)
    d_test = torch.from_numpy(test_d[test_idx].astype(np.float32)).to(device)
    
    # Test different tn/tf combinations
    tn_values = [0.1, 0.5, 1.0, 2.0]
    tf_values = [4.0, 6.0, 8.0, 10.0]
    nb_bins = 128
    
    fig, axes = plt.subplots(len(tn_values), len(tf_values), figsize=(15, 12))
    
    for i, tn in enumerate(tn_values):
        for j, tf in enumerate(tf_values):
            print(f"Testing tn={tn}, tf={tf}")
            
            # Render a small patch to save time
            center_h, center_w = test_h // 2, test_w // 2
            patch_size = 50
            h_start, h_end = center_h - patch_size // 2, center_h + patch_size // 2
            w_start, w_end = center_w - patch_size // 2, center_w + patch_size // 2
            
            # Calculate indices in flattened array
            indices = []
            for h in range(h_start, h_end):
                for w in range(w_start, w_end):
                    indices.append(h * test_w + w)
            
            # Get rays for the patch
            o_patch = o_test[indices]
            d_patch = d_test[indices]
            
            # Render
            with torch.no_grad():
                rgb = rendering.rendering(model, o_patch, d_patch, tn, tf, nb_bins, device)
                
            # Reshape to patch
            rgb = rgb.reshape(patch_size, patch_size, 3).cpu().numpy()
            
            # Display
            axes[i, j].imshow(np.clip(rgb, 0, 1))
            axes[i, j].set_title(f"tn={tn}, tf={tf}")
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig('tn_tf_tests.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    print("Examining transforms.json file...")
    suggested_tn, suggested_tf = examine_transforms_file()
    
    print("\nTesting ray generation...")
    test_o, test_d = test_ray_generation()
    
    # Uncomment to test rendering with different tn/tf values
    # If you have a trained model, you can test rendering with different parameters
    print("\nTesting rendering with different tn/tf values...")
    test_rendering()
    
    print("\nDebug Summary:")
    print("-" * 50)
    print(f"1. Suggested near/far thresholds: tn={suggested_tn:.2f}, tf={suggested_tf:.2f}")
    print("2. Check ray generation visualization above")
    print("3. If you've run the rendering tests, check the 'tn_tf_tests.png' file")
    print("4. Common issues:")
    print("   - Incorrect ray generation (check normalization and direction)")
    print("   - Wrong tn/tf values (try the suggested values)")
    print("   - Not enough training steps or too high learning rate")
    print("   - Not enough sampling bins along rays (try nb_bins=128 or higher)")