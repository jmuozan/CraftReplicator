import torch
import numpy as np
from nerfvis import scene
import os
from tqdm import tqdm

# Check if MPS is available
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device")
else:
    device = torch.device("cpu")
    print("Using CPU device")

# Use the same model definition from your view_nerf.py file
class SimplifiedNGPNeRF(torch.nn.Module):
    # [Your model definition here - already in your script]
    # ...
    def __init__(self, T, Nl, L, device, aabb_scale, F=2):
        super(SimplifiedNGPNeRF, self).__init__()
        # Instant NGP components
        self.T = T  # Hash table size
        self.Nl = Nl  # Resolution levels
        self.F = F  # Feature dimensions per level
        self.L = L  # Directional encoding levels
        self.aabb_scale = aabb_scale
        self.lookup_tables = torch.nn.ParameterDict(
            {str(i): torch.nn.Parameter((torch.rand(
                (T, 2), device=device).float() * 2 - 1) * 1e-4) for i in range(len(Nl))})
        self.pi1, self.pi2, self.pi3 = 1, 2_654_435_761, 805_459_861
        
        # Network components
        # Density network
        self.density_MLP = torch.nn.Sequential(
            torch.nn.Linear(self.F * len(Nl), 64),
            torch.nn.ReLU(), 
            torch.nn.Linear(64, 16)
        ).to(device)
        
        # Color network
        self.color_MLP = torch.nn.Sequential(
            torch.nn.Linear(16 + 27, 64),  # features + dir_encoding
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 3),
            torch.nn.Sigmoid()
        ).to(device)

    def positional_encoding(self, x):
        """Apply positional encoding to input directions"""
        out = [x]
        for j in range(self.L):
            out.append(torch.sin(2 ** j * x))
            out.append(torch.cos(2 ** j * x))
        return torch.cat(out, dim=1)

    def compute_hash_features(self, x, mask):
        """Compute multi-resolution hash encoding features for 3D positions without using grid_sample"""
        features = torch.empty((x[mask].shape[0], self.F * len(self.Nl)), device=x.device).float()
        
        for i, N in enumerate(self.Nl):
            # Computing vertices
            floor = torch.floor(x[mask] * N)
            local_pos = (x[mask] * N) - floor  # Position within the grid cell [0,1]^3
            
            # Get the 8 corners of the grid cell
            vertices = torch.zeros((x[mask].shape[0], 8, 3), dtype=torch.int32, device=x.device)
            vertices[:, 0] = floor
            vertices[:, 1] = torch.cat((floor[:, 0:1] + 1, floor[:, 1:2], floor[:, 2:3]), dim=1)
            vertices[:, 2] = torch.cat((floor[:, 0:1], floor[:, 1:2] + 1, floor[:, 2:3]), dim=1)
            vertices[:, 3] = torch.cat((floor[:, 0:1] + 1, floor[:, 1:2] + 1, floor[:, 2:3]), dim=1)
            vertices[:, 4] = torch.cat((floor[:, 0:1], floor[:, 1:2], floor[:, 2:3] + 1), dim=1)
            vertices[:, 5] = torch.cat((floor[:, 0:1] + 1, floor[:, 1:2], floor[:, 2:3] + 1), dim=1)
            vertices[:, 6] = torch.cat((floor[:, 0:1], floor[:, 1:2] + 1, floor[:, 2:3] + 1), dim=1)
            vertices[:, 7] = floor + 1

            # Hashing
            a = vertices[:, :, 0] * self.pi1
            b = vertices[:, :, 1] * self.pi2
            c = vertices[:, :, 2] * self.pi3
            h_x = torch.remainder(torch.bitwise_xor(torch.bitwise_xor(a, b), c), self.T)
            
            # Get feature values at the 8 corners
            corner_features = torch.zeros((x[mask].shape[0], 8, self.F), device=x.device)
            for j in range(8):
                corner_features[:, j] = self.lookup_tables[str(i)][h_x[:, j]]
            
            # Manual trilinear interpolation
            for f in range(self.F):
                # Extract the feature values at the 8 corners of the cell
                c000 = corner_features[:, 0, f]
                c100 = corner_features[:, 1, f]
                c010 = corner_features[:, 2, f]
                c110 = corner_features[:, 3, f]
                c001 = corner_features[:, 4, f]
                c101 = corner_features[:, 5, f]
                c011 = corner_features[:, 6, f]
                c111 = corner_features[:, 7, f]
                
                # Get interpolation weights
                x_w = local_pos[:, 0]
                y_w = local_pos[:, 1]
                z_w = local_pos[:, 2]
                
                # Interpolate along x
                c00 = c000 * (1 - x_w) + c100 * x_w
                c01 = c001 * (1 - x_w) + c101 * x_w
                c10 = c010 * (1 - x_w) + c110 * x_w
                c11 = c011 * (1 - x_w) + c111 * x_w
                
                # Interpolate along y
                c0 = c00 * (1 - y_w) + c10 * y_w
                c1 = c01 * (1 - y_w) + c11 * y_w
                
                # Interpolate along z and store the result
                features[:, i*self.F + f] = c0 * (1 - z_w) + c1 * z_w
                
        return features

    def forward(self, x, d):
        """
        Forward pass of the simplified model.
        
        Args:
            x: 3D positions (batch_size, 3)
            d: View directions (batch_size, 3)
            
        Returns:
            Tuple containing RGB colors and densities
        """
        # Ensure inputs are float32
        x = x.float()
        d = d.float()
        
        # Normalize positions to [-0.5, 0.5]^3
        x_normalized = x / self.aabb_scale
        mask = (x_normalized[:, 0].abs() < .5) & (x_normalized[:, 1].abs() < .5) & (x_normalized[:, 2].abs() < .5)
        x_normalized += 0.5  # x in [0, 1]^3
        
        # Initialize outputs
        rgb = torch.zeros((x.shape[0], 3), device=x.device).float()
        sigma = torch.zeros((x.shape[0]), device=x.device).float() - 100000  # log space
        
        if torch.sum(mask) == 0:
            return rgb, torch.exp(sigma)
        
        # Get hash encoding features for positions
        features = self.compute_hash_features(x_normalized, mask)
        
        # Process directions
        d_encoded = self.positional_encoding(d[mask])
        
        # Get density features
        h = self.density_MLP(features)
        sigma[mask] = h[:, 0]  # First feature is density
        
        # Compute RGB
        rgb[mask] = self.color_MLP(torch.cat([h, d_encoded], dim=1))
        
        return rgb, torch.exp(sigma)

# NGP parameters
L = 16  # Number of levels
F = 2   # Features per level
T = 2**19  # Hash table size
N_min = 16
N_max = 2048
b = np.exp((np.log(N_max) - np.log(N_min)) / (L - 1))
Nl = [int(np.floor(N_min * b**l)) for l in range(L)]

# Model parameters
aabb_scale = 3  # Bounding box scale

# Create model instance
model = SimplifiedNGPNeRF(T, Nl, 4, device, aabb_scale)

# Load checkpoint
checkpoint = torch.load("nerf_model_weights.pth", map_location=device)
print("Checkpoint keys:", checkpoint.keys())

# Try loading the state dictionary
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
elif 'model' in checkpoint:
    model.load_state_dict(checkpoint['model'])
else:
    print("Couldn't find model state dict in checkpoint.")

model.eval()  # Set to evaluation mode

# Create a point cloud representation
def create_point_cloud(model, num_points=100000, density_threshold=0.1, device=device):
    """
    Sample points from the NeRF model to create a point cloud.
    
    Args:
        model: The NeRF model
        num_points: Number of points to sample
        density_threshold: Threshold for keeping points based on density
        device: Computation device
    
    Returns:
        points: 3D point positions
        colors: RGB colors for each point
    """
    print(f"Sampling {num_points} points from NeRF...")
    
    # Sample random points in the bounding box
    points = torch.rand(num_points, 3, device=device) * 2 - 1
    
    # Use a fixed view direction for all points
    directions = torch.tensor([0, 0, 1], device=device).expand_as(points)
    
    # Process in batches
    batch_size = 4096
    num_batches = (points.shape[0] + batch_size - 1) // batch_size
    
    all_points = []
    all_colors = []
    
    with torch.no_grad():
        for i in tqdm(range(num_batches)):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, points.shape[0])
            
            batch_points = points[start_idx:end_idx]
            batch_dirs = directions[start_idx:end_idx]
            
            rgb, sigma = model(batch_points, batch_dirs)
            
            # Keep only points with density above threshold
            mask = sigma > density_threshold
            if torch.sum(mask) > 0:
                all_points.append(batch_points[mask].cpu().numpy())
                all_colors.append(rgb[mask].cpu().numpy())
    
    # Combine results
    if all_points:
        final_points = np.vstack(all_points)
        final_colors = np.vstack(all_colors)
        print(f"Found {len(final_points)} points above density threshold")
        return final_points, final_colors
    else:
        print("No points found above density threshold")
        return np.zeros((0, 3)), np.zeros((0, 3))

# Set a reasonable density threshold based on your model
# You might need to adjust this value
density_threshold = 0.5

# Create point cloud
points, colors = create_point_cloud(model, num_points=500000, density_threshold=density_threshold)

# Create scene
scene.add_axes()
scene.set_title("NeRF Point Cloud Visualization")

# Add point cloud to scene
if len(points) > 0:
    scene.add_points("NeRF Points", points, vert_color=colors, point_size=0.01)
else:
    print("No points to visualize. Try lowering the density_threshold.")

# Display the scene
scene.display()