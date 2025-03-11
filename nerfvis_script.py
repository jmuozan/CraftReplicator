import torch
import numpy as np
from nerfvis import scene
import os
import argparse
import sys

# Define the classes from instant-ngp-nerfw_cuda directly in this script
# This avoids import issues
class CombinedNGPNeRFW(torch.nn.Module):
    """
    Combined model that uses Instant NGP's hash encoding for efficiency
    with NeRF-W's appearance and transient modeling for handling in-the-wild images.
    """
    def __init__(self, T, Nl, L, device, aabb_scale, N_vocab=100, N_a=48, N_tau=16, beta_min=0.1, F=2):
        super(CombinedNGPNeRFW, self).__init__()
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
        
        # NeRF-W components
        self.N_vocab = N_vocab
        self.N_a = N_a
        self.N_tau = N_tau
        self.beta_min = beta_min
        
        # Embeddings for appearance and transient modeling
        self.embedding_a = torch.nn.Embedding(N_vocab, N_a).to(device)
        self.embedding_t = torch.nn.Embedding(N_vocab, N_tau).to(device)
        
        # Network components
        # Static scene representation
        self.density_MLP = torch.nn.Sequential(
            torch.nn.Linear(self.F * len(Nl), 64),
            torch.nn.ReLU(), 
            torch.nn.Linear(64, 16)
        ).to(device)
        
        # Static RGB with appearance conditioning
        self.static_rgb_MLP = torch.nn.Sequential(
            torch.nn.Linear(16 + 27 + N_a, 64),  # features + dir_encoding + appearance
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 3),
            torch.nn.Sigmoid()
        ).to(device)
        
        # Transient components
        self.transient_MLP = torch.nn.Sequential(
            torch.nn.Linear(16 + N_tau, 64),  # features + transient embedding
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU()
        ).to(device)
        
        self.transient_density_MLP = torch.nn.Sequential(
            torch.nn.Linear(64, 1),
            torch.nn.Softplus()
        ).to(device)
        
        self.transient_rgb_MLP = torch.nn.Sequential(
            torch.nn.Linear(64, 3),
            torch.nn.Sigmoid()
        ).to(device)
        
        self.transient_beta_MLP = torch.nn.Sequential(
            torch.nn.Linear(64, 1),
            torch.nn.Softplus()
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
            # We'll do this for each feature dimension separately
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

    def forward(self, x, d, appearance_idx=None, transient_idx=None):
        """
        Forward pass of the combined model.
        
        Args:
            x: 3D positions (batch_size, 3)
            d: View directions (batch_size, 3)
            appearance_idx: Indices for appearance embedding (batch_size,)
            transient_idx: Indices for transient embedding (batch_size,)
            
        Returns:
            Tuple containing RGB colors and densities for both static and transient components
        """
        # Ensure inputs are float32
        x = x.float()
        d = d.float()
        
        # Normalize positions to [-0.5, 0.5]^3
        x_normalized = x / self.aabb_scale
        mask = (x_normalized[:, 0].abs() < .5) & (x_normalized[:, 1].abs() < .5) & (x_normalized[:, 2].abs() < .5)
        x_normalized += 0.5  # x in [0, 1]^3
        
        # Initialize outputs
        static_rgb = torch.zeros((x.shape[0], 3), device=x.device).float()
        static_sigma = torch.zeros((x.shape[0]), device=x.device).float() - 100000  # log space
        transient_rgb = torch.zeros((x.shape[0], 3), device=x.device).float()
        transient_sigma = torch.zeros((x.shape[0]), device=x.device).float() - 100000  # log space
        transient_beta = torch.ones((x.shape[0]), device=x.device).float() * self.beta_min
        
        if torch.sum(mask) == 0:
            return static_rgb, torch.exp(static_sigma), transient_rgb, torch.exp(transient_sigma), transient_beta
        
        # Get hash encoding features for positions
        features = self.compute_hash_features(x_normalized, mask)
        
        # Process directions
        d_encoded = self.positional_encoding(d[mask])
        
        # Get static density features
        h = self.density_MLP(features)
        static_sigma[mask] = h[:, 0]  # First feature is density
        
        # Get appearance and transient embeddings if provided
        if appearance_idx is not None:
            a_embedded = self.embedding_a(appearance_idx[mask])
        else:
            a_embedded = torch.zeros((torch.sum(mask), self.N_a), device=x.device)
            
        if transient_idx is not None:
            t_embedded = self.embedding_t(transient_idx[mask])
        else:
            t_embedded = torch.zeros((torch.sum(mask), self.N_tau), device=x.device)
        
        # Compute static RGB with appearance conditioning
        static_rgb[mask] = self.static_rgb_MLP(torch.cat([h, d_encoded, a_embedded], dim=1))
        
        # Compute transient features
        transient_features = self.transient_MLP(torch.cat([h, t_embedded], dim=1))
        
        # Get transient outputs
        transient_sigma[mask] = self.transient_density_MLP(transient_features).squeeze(-1)
        transient_rgb[mask] = self.transient_rgb_MLP(transient_features)
        transient_beta[mask] = self.transient_beta_MLP(transient_features).squeeze(-1) + self.beta_min
        
        return static_rgb, torch.exp(static_sigma), transient_rgb, torch.exp(transient_sigma), transient_beta

# Define device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

def load_model_from_checkpoint(checkpoint_path):
    """Load the CombinedNGPNeRFW model from a checkpoint file"""
    # Load the checkpoint first to inspect parameters
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract state dict
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    # Determine vocabulary size from the embedding weights
    N_vocab = state_dict['embedding_a.weight'].shape[0]
    N_a = state_dict['embedding_a.weight'].shape[1]
    N_tau = state_dict['embedding_t.weight'].shape[1]
    print(f"Detected vocabulary size: {N_vocab}")
    print(f"Detected appearance embedding size: {N_a}")
    print(f"Detected transient embedding size: {N_tau}")
    
    # NGP parameters (must match those used in training)
    L = 16  # Number of levels
    F = 2   # Features per level
    T = 2**19  # Hash table size
    N_min = 16
    N_max = 2048
    b = np.exp((np.log(N_max) - np.log(N_min)) / (L - 1))
    Nl = [int(np.floor(N_min * b**l)) for l in range(L)]
    beta_min = 0.1  # Minimum uncertainty
    
    # Create model with the same architecture as during training
    model = CombinedNGPNeRFW(T, Nl, 4, device, 3, N_vocab, N_a, N_tau, beta_min)
    
    # Load the state dict
    model.load_state_dict(state_dict)
    
    model.eval()  # Set to evaluation mode
    return model

def visualize_nerf(model, output_dir='nerfvis_output', threshold=0.01):
    """Visualize the NeRF model using nerfvis"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a scene
    scene.set_title("Instant-NGP-NeRFW Visualization")
    
    # Add coordinate axes for reference
    scene.add_axes("axes", length=1.0)
    
    # Define a 3D grid of points to sample the NeRF
    resolution = 64  # Lower resolution to start with
    
    # Scale should match the aabb_scale used in the model
    scale = model.aabb_scale
    
    # Let's focus on a smaller region first - front central portion
    x = np.linspace(-scale/4, scale/4, resolution)
    y = np.linspace(-scale/4, scale/4, resolution)
    z = np.linspace(-scale/3, scale/3, resolution)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    points = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=-1)
    
    # Try multiple viewing directions for better coverage
    num_directions = 4  
    directions = []
    for i in range(num_directions):
        # Create evenly distributed viewing directions
        theta = np.pi * i / num_directions
        phi = 2 * np.pi * i / num_directions
        dx = np.sin(theta) * np.cos(phi)
        dy = np.sin(theta) * np.sin(phi)
        dz = np.cos(theta)
        directions.append([dx, dy, dz])
    
    directions = np.array(directions)
    directions_tensor = torch.tensor(directions, dtype=torch.float32, device=device)
    
    # Evaluate NeRF at these points and directions
    print("Evaluating NeRF at grid points...")
    valid_points = []
    valid_colors = []
    
    batch_size = 1024  # Smaller batch size
    num_batches = (points.shape[0] + batch_size - 1) // batch_size
    
    # Try various appearance indices
    appearance_indices = [0, 1, 2]  # Try first few appearance embeddings
    
    with torch.no_grad():
        for appearance_idx_val in appearance_indices:
            print(f"Trying appearance index: {appearance_idx_val}")
            
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, points.shape[0])
                batch_points = torch.tensor(points[start_idx:end_idx], dtype=torch.float32, device=device)
                
                batch_static_rgb_list = []
                batch_transient_rgb_list = []
                batch_static_densities_list = []
                batch_transient_densities_list = []
                
                # Use the specific appearance/transient embedding index
                appearance_idx = torch.ones(batch_points.size(0), dtype=torch.long, device=device) * appearance_idx_val
                transient_idx = torch.ones(batch_points.size(0), dtype=torch.long, device=device) * appearance_idx_val
                
                for direction in directions_tensor:
                    # Expand direction to match the batch size
                    batch_dirs = direction.expand(batch_points.size(0), -1)
                    
                    # Get model output using the CombinedNGPNeRFW forward method
                    static_rgb, static_sigma, transient_rgb, transient_sigma, _ = model(
                        batch_points, 
                        batch_dirs, 
                        appearance_idx, 
                        transient_idx
                    )
                    
                    batch_static_rgb_list.append(static_rgb)
                    batch_transient_rgb_list.append(transient_rgb)
                    batch_static_densities_list.append(static_sigma)
                    batch_transient_densities_list.append(transient_sigma)
                
                # Average colors across different view directions
                avg_static_colors = torch.stack(batch_static_rgb_list).mean(dim=0)
                avg_transient_colors = torch.stack(batch_transient_rgb_list).mean(dim=0)
                
                # Combined color (static + transient)
                avg_combined_colors = torch.clamp(avg_static_colors + avg_transient_colors, 0, 1)
                
                # Use maximum density across different view directions
                max_static_densities = torch.stack(batch_static_densities_list).max(dim=0)[0]
                max_transient_densities = torch.stack(batch_transient_densities_list).max(dim=0)[0]
                
                # Combined density (static + transient)
                combined_densities = max_static_densities + max_transient_densities
                
                # Find points above threshold
                batch_valid_indices = torch.where(combined_densities > threshold)[0].cpu().numpy()
                
                if len(batch_valid_indices) > 0:
                    # Get valid points and colors for this batch
                    batch_valid_points = points[start_idx:end_idx][batch_valid_indices]
                    batch_valid_colors = avg_combined_colors[batch_valid_indices].cpu().numpy()
                    
                    # Append to our valid points and colors
                    valid_points.append(batch_valid_points)
                    valid_colors.append(batch_valid_colors)
                
                if i % 10 == 0:
                    print(f"Processed batch {i+1}/{num_batches} - Found {len(batch_valid_indices)} valid points")
                    
            # If we've found points with this appearance index, stop trying others
            if valid_points and len(valid_points) > 0:
                print(f"Found points with appearance index {appearance_idx_val}")
                break
    
    # Combine all valid points and colors
    if valid_points and len(valid_points) > 0:
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
    else:
        print("No high-density points found. Try using a different threshold or check the model.")
        
        # Add a small visible sphere at the origin for reference
        sphere_points = []
        sphere_colors = []
        radius = 0.1
        num_sphere_points = 1000
        
        for _ in range(num_sphere_points):
            # Generate random points on a sphere
            phi = np.random.uniform(0, 2 * np.pi)
            costheta = np.random.uniform(-1, 1)
            theta = np.arccos(costheta)
            
            x = radius * np.sin(theta) * np.cos(phi)
            y = radius * np.sin(theta) * np.sin(phi)
            z = radius * np.cos(theta)
            
            sphere_points.append([x, y, z])
            sphere_colors.append([1.0, 0.0, 0.0])  # Red sphere
        
        scene.add_points(
            name="origin_marker",
            points=np.array(sphere_points),
            vert_color=np.array(sphere_colors),
            point_size=2.0
        )
        
    # Add reference objects to visualize the bounds
    scene.add_wireframe_cube("bounds", scale=scale, color=[0.5, 0.5, 0.5])
    
    # Display the scene
    scene.display(port=8888, open_browser=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize NeRF-W with Instant NGP using nerfvis")
    parser.add_argument("--checkpoint", type=str, default="checkpoint_epoch_5.pt", 
                        help="Path to the checkpoint file")
    parser.add_argument("--output_dir", type=str, default="nerfvis_output", 
                        help="Output directory for visualizations")
    parser.add_argument("--threshold", type=float, default=0.01, 
                        help="Density threshold for visualization")
    
    args = parser.parse_args()
    
    # Load the model
    model = load_model_from_checkpoint(args.checkpoint)
    
    # Visualize the model
    visualize_nerf(model, output_dir=args.output_dir, threshold=args.threshold)