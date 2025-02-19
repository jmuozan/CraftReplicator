import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from scipy.spatial import KDTree

class GaussianSplat:
    def __init__(self, position, scale, rotation, opacity, color):
        self.position = position  # (3,) center position
        self.scale = scale      # (3,) scale in xyz
        self.rotation = rotation  # (4,) quaternion rotation
        self.opacity = opacity  # (1,) opacity value
        self.color = color      # (3,) RGB color

def sample_nerf_density_field(nerf_model, resolution=64, bounds=(-1, 1), device='cpu'):
    """Sample the NeRF's density field in a regular grid."""
    model = nerf_model.to(device)
    model.eval()
    
    # Create sampling grid
    x = np.linspace(bounds[0], bounds[1], resolution)
    y = np.linspace(bounds[0], bounds[1], resolution)
    z = np.linspace(bounds[0], bounds[1], resolution)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    
    points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)
    densities = []
    colors = []
    
    # Sample in batches to avoid memory issues
    batch_size = 4096
    with torch.no_grad():
        for i in tqdm(range(0, len(points), batch_size)):
            batch_points = torch.FloatTensor(points[i:i+batch_size]).to(device)
            # Use a dummy direction for now since we mainly care about density
            dummy_dirs = torch.zeros_like(batch_points).to(device)
            batch_colors, batch_density = model.intersect(batch_points, dummy_dirs)
            
            densities.append(batch_density.cpu().numpy())
            colors.append(batch_colors.cpu().numpy())
    
    densities = np.concatenate(densities)
    colors = np.concatenate(colors)
    
    return points, densities, colors

def extract_high_density_points(points, densities, colors, threshold=0.5):
    """Extract points with density above threshold."""
    mask = densities > threshold
    high_density_points = points[mask]
    high_density_colors = colors[mask]
    return high_density_points, high_density_colors

def cluster_points(points, colors, n_gaussians=1000):
    """Cluster points into Gaussian primitives using K-means."""
    from sklearn.cluster import KMeans
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_gaussians, random_state=42)
    cluster_labels = kmeans.fit_predict(points)
    
    gaussians = []
    for i in range(n_gaussians):
        cluster_mask = cluster_labels == i
        if not np.any(cluster_mask):
            continue
            
        # Get cluster points
        cluster_points = points[cluster_mask]
        cluster_colors = colors[cluster_mask]
        
        # Calculate gaussian parameters
        position = np.mean(cluster_points, axis=0)
        
        # Compute covariance for scale and rotation
        if len(cluster_points) > 1:
            covariance = np.cov(cluster_points.T)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            scale = np.sqrt(np.maximum(eigenvalues, 0.0001))
            
            # Convert eigenvectors to quaternion
            rotation = vectors_to_quaternion(eigenvectors)
        else:
            scale = np.array([0.01, 0.01, 0.01])
            rotation = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Average color
        color = np.mean(cluster_colors, axis=0)
        
        # Set initial opacity based on density
        opacity = min(1.0, np.mean(cluster_mask.astype(float)))
        
        gaussians.append(GaussianSplat(position, scale, rotation, opacity, color))
    
    return gaussians

def vectors_to_quaternion(vectors):
    """Convert rotation matrix to quaternion."""
    R = vectors
    trace = np.trace(R)
    
    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2
        w = 0.25 * S
        x = (R[2,1] - R[1,2]) / S
        y = (R[0,2] - R[2,0]) / S
        z = (R[1,0] - R[0,1]) / S
    else:
        if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
            w = (R[2,1] - R[1,2]) / S
            x = 0.25 * S
            y = (R[0,1] + R[1,0]) / S
            z = (R[0,2] + R[2,0]) / S
        elif R[1,1] > R[2,2]:
            S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
            w = (R[0,2] - R[2,0]) / S
            x = (R[0,1] + R[1,0]) / S
            y = 0.25 * S
            z = (R[1,2] + R[2,1]) / S
        else:
            S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
            w = (R[1,0] - R[0,1]) / S
            x = (R[0,2] + R[2,0]) / S
            y = (R[1,2] + R[2,1]) / S
            z = 0.25 * S
            
    return np.array([w, x, y, z])

def save_gaussians(gaussians, filename):
    """Save Gaussian splats to a file."""
    data = {
        'positions': np.array([g.position for g in gaussians]),
        'scales': np.array([g.scale for g in gaussians]),
        'rotations': np.array([g.rotation for g in gaussians]),
        'opacities': np.array([g.opacity for g in gaussians]),
        'colors': np.array([g.color for g in gaussians])
    }
    np.savez(filename, **data)

def main():
    # Load the NeRF model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nerf_model = torch.load('/Users/jorgemuyo/Desktop/CraftReplicator/model_nerf_mps')
    
    # Sample the density field
    print("Sampling density field...")
    points, densities, colors = sample_nerf_density_field(nerf_model, resolution=64, device=device)
    
    # Extract high density points
    print("Extracting high density points...")
    high_density_points, high_density_colors = extract_high_density_points(points, densities, colors)
    
    # Convert to Gaussian splats
    print("Converting to Gaussian splats...")
    gaussians = cluster_points(high_density_points, high_density_colors, n_gaussians=2000)
    
    # Save the results
    print("Saving Gaussian splats...")
    save_gaussians(gaussians, 'gaussian_splats.npz')
    print("Conversion complete!")

if __name__ == "__main__":
    main()