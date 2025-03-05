# instant-ngp-nerfw.py
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image

import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
import os
import argparse
from nerf_dataset import NeRFDataset  # Import our new dataset class

# The model classes remain the same as your original script
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
        self.density_MLP = nn.Sequential(
            nn.Linear(self.F * len(Nl), 64),
            nn.ReLU(), 
            nn.Linear(64, 16)
        ).to(device)
        
        # Static RGB with appearance conditioning
        self.static_rgb_MLP = nn.Sequential(
            nn.Linear(16 + 27 + N_a, 64),  # features + dir_encoding + appearance
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Sigmoid()
        ).to(device)
        
        # Transient components
        self.transient_MLP = nn.Sequential(
            nn.Linear(16 + N_tau, 64),  # features + transient embedding
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        ).to(device)
        
        self.transient_density_MLP = nn.Sequential(
            nn.Linear(64, 1),
            nn.Softplus()
        ).to(device)
        
        self.transient_rgb_MLP = nn.Sequential(
            nn.Linear(64, 3),
            nn.Sigmoid()
        ).to(device)
        
        self.transient_beta_MLP = nn.Sequential(
            nn.Linear(64, 1),
            nn.Softplus()
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

# Functions for rendering rays remain the same
def compute_accumulated_transmittance(alphas):
    """Compute accumulated transmittance along rays"""
    accumulated_transmittance = torch.cumprod(alphas, 1)
    return torch.cat((torch.ones((accumulated_transmittance.shape[0], 1), device=alphas.device, dtype=torch.float32),
                      accumulated_transmittance[:, :-1]), dim=-1)

def render_rays_combined(model, ray_origins, ray_directions, appearance_idx=None, transient_idx=None, 
                        hn=0, hf=0.5, nb_bins=192):
    """
    Render rays using combined NeRF-W and Instant NGP model
    
    Args:
        model: The neural network model
        ray_origins: Origins of rays (batch_size, 3)
        ray_directions: Directions of rays (batch_size, 3)
        appearance_idx: Indices for appearance embedding (batch_size,)
        transient_idx: Indices for transient embedding (batch_size,)
        hn: Near plane distance
        hf: Far plane distance
        nb_bins: Number of sample bins per ray
        
    Returns:
        Dictionary containing rendered outputs
    """
    device = ray_origins.device
    
    # Sample points along rays
    t = torch.linspace(hn, hf, nb_bins, device=device, dtype=torch.float32).expand(ray_origins.shape[0], nb_bins)
    # Perturb sampling along each ray
    mid = (t[:, :-1] + t[:, 1:]) / 2.
    lower = torch.cat((t[:, :1], mid), -1)
    upper = torch.cat((mid, t[:, -1:]), -1)
    u = torch.rand(t.shape, device=device, dtype=torch.float32)
    t = lower + (upper - lower) * u  # [batch_size, nb_bins]
    
    # Compute delta for transmittance calculation
    delta = torch.cat((t[:, 1:] - t[:, :-1], torch.tensor(
        [1e10], device=device, dtype=torch.float32).expand(ray_origins.shape[0], 1)), -1)

    # Compute the 3D points along each ray
    x = ray_origins.unsqueeze(1) + t.unsqueeze(2) * ray_directions.unsqueeze(1)  # [batch_size, nb_bins, 3]
    
    # Expand the ray_directions tensor to match the shape of x
    ray_directions_expanded = ray_directions.expand(nb_bins, ray_directions.shape[0], 3).transpose(0, 1)
    
    # Prepare appearance and transient indices if provided
    if appearance_idx is not None:
        appearance_idx_expanded = appearance_idx.expand(nb_bins, appearance_idx.shape[0]).transpose(0, 1).reshape(-1)
    else:
        appearance_idx_expanded = None
        
    if transient_idx is not None:
        transient_idx_expanded = transient_idx.expand(nb_bins, transient_idx.shape[0]).transpose(0, 1).reshape(-1)
    else:
        transient_idx_expanded = None
    
    # Forward pass through the model
    static_rgb, static_sigma, transient_rgb, transient_sigma, transient_beta = model(
        x.reshape(-1, 3), 
        ray_directions_expanded.reshape(-1, 3),
        appearance_idx_expanded,
        transient_idx_expanded
    )
    
    # Reshape outputs to [batch_size, nb_bins, channels]
    static_rgb = static_rgb.reshape(x.shape[0], x.shape[1], 3)
    static_sigma = static_sigma.reshape(x.shape[0], x.shape[1])
    transient_rgb = transient_rgb.reshape(x.shape[0], x.shape[1], 3)
    transient_sigma = transient_sigma.reshape(x.shape[0], x.shape[1])
    transient_beta = transient_beta.reshape(x.shape[0], x.shape[1])
    
    # Calculate alphas for static and transient components
    static_alphas = 1 - torch.exp(-static_sigma * delta)
    transient_alphas = 1 - torch.exp(-transient_sigma * delta)
    combined_alphas = 1 - torch.exp(-(static_sigma + transient_sigma) * delta)
    
    # Calculate weights
    transmittance = compute_accumulated_transmittance(1 - combined_alphas)
    weights = transmittance * combined_alphas
    weights_sum = weights.sum(dim=1)
    
    # Calculate static and transient weights based on respective densities
    static_weights = weights * static_sigma / (static_sigma + transient_sigma + 1e-10)
    transient_weights = weights * transient_sigma / (static_sigma + transient_sigma + 1e-10)
    
    # Calculate rendered colors
    static_rgb_map = torch.sum(static_weights.unsqueeze(-1) * static_rgb, dim=1)
    transient_rgb_map = torch.sum(transient_weights.unsqueeze(-1) * transient_rgb, dim=1)
    
    # Calculate depth
    depth_map = torch.sum(weights * t, dim=1)
    
    # Calculate beta (uncertainty)
    beta = torch.sum(transient_weights * transient_beta, dim=1) + model.beta_min
    
    # White background handling
    background = 1 - weights_sum.unsqueeze(-1)
    static_rgb_map = static_rgb_map + background
    combined_rgb_map = static_rgb_map + transient_rgb_map
    
    # Return results as dictionary
    results = {
        'rgb': combined_rgb_map,
        'static_rgb': static_rgb_map,
        'transient_rgb': transient_rgb_map,
        'depth': depth_map,
        'weights': weights,
        'opacity': weights_sum,
        'beta': beta
    }
    
    return results

def train_combined(model, optimizer, data_loader, device='mps', hn=0, hf=1, nb_epochs=1,
                  nb_bins=192, H=400, W=400, lambda_u=0.01):
    """
    Training function for the combined model
    
    Args:
        model: The neural network model
        optimizer: The optimizer
        data_loader: DataLoader for training data
        device: Device to use for computation
        hn: Near plane distance
        hf: Far plane distance
        nb_epochs: Number of training epochs
        nb_bins: Number of sample bins per ray
        H, W: Image height and width
        lambda_u: Weight for the transient density regularization
    """
    for epoch in range(nb_epochs):
        epoch_loss = 0
        for batch in tqdm(data_loader):
            # Extract data from batch
            # Format: [ray_origin(3), ray_direction(3), appearance_idx(1), transient_idx(1), rgb(3)]
            ray_origins = batch[:, :3].to(device).float()
            ray_directions = batch[:, 3:6].to(device).float()
            appearance_idx = batch[:, 6].long().to(device) if batch.shape[1] > 8 else None
            transient_idx = batch[:, 7].long().to(device) if batch.shape[1] > 8 else None
            gt_px_values = batch[:, -3:].to(device).float()
            
            # Render rays
            results = render_rays_combined(model, ray_origins, ray_directions, 
                                        appearance_idx, transient_idx,
                                        hn=hn, hf=hf, nb_bins=nb_bins)
            
            # Calculate loss (NeRF-W style)
            # RGB loss
            rgb_loss = ((results['rgb'] - gt_px_values) ** 2).mean()
            
            # Regularization for transient density (from NeRF-W)
            reg_loss = 0
            if 'beta' in results:
                beta_loss = 3 + torch.log(results['beta']).mean()  # +3 to make it positive
                transient_sparsity = lambda_u * results.get('transient_sigmas', torch.tensor(0.0)).mean()
                reg_loss = beta_loss + transient_sparsity
            
            # Total loss
            loss = rgb_loss + reg_loss
            
            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{nb_epochs}, Loss: {epoch_loss/len(data_loader):.6f}")

@torch.no_grad()
def test_combined(model, test_dataset, img_index, appearance_idx=None, transient_idx=None,
                 chunk_size=20, nb_bins=192, H=400, W=400, hn=0, hf=0.5, output_dir='novel_views'):
    """
    Test function for the combined model
    
    Args:
        model: The neural network model
        test_dataset: Dataset containing test data
        img_index: Index of the image to test
        appearance_idx: Index for appearance embedding
        transient_idx: Index for transient embedding
        chunk_size: Number of rays to process at once
        nb_bins: Number of sample bins per ray
        H, W: Image height and width
        hn, hf: Near and far plane distances
        output_dir: Directory to save output images
    """
    device = next(model.parameters()).device
    os.makedirs(output_dir, exist_ok=True)
    
    # Get rays for the entire image
    start_idx = img_index * H * W
    end_idx = (img_index + 1) * H * W
    
    # Check if the dataset is large enough
    if start_idx >= len(test_dataset) or end_idx > len(test_dataset):
        print(f"Warning: Image index {img_index} is out of bounds for dataset of size {len(test_dataset)}")
        # Use available rays instead
        start_idx = 0
        end_idx = min(H * W, len(test_dataset))
    
    ray_data = test_dataset[start_idx:end_idx]
    ray_origins = ray_data[:, :3].to(device)
    ray_directions = ray_data[:, 3:6].to(device)
    
    # Use fixed appearance and transient indices if provided
    if appearance_idx is not None:
        appearance_idx = torch.ones_like(ray_origins[:, 0], dtype=torch.long) * appearance_idx
    else:
        appearance_idx = ray_data[:, 6].long().to(device) if ray_data.shape[1] > 8 else None
        
    if transient_idx is not None:
        transient_idx = torch.ones_like(ray_origins[:, 0], dtype=torch.long) * transient_idx
    else:
        transient_idx = ray_data[:, 7].long().to(device) if ray_data.shape[1] > 8 else None
    
    # Render chunks of rays
    combined_rgb = []
    static_rgb = []
    transient_rgb = []
    depth = []
    beta = []
    
    for i in range(int(np.ceil(ray_origins.shape[0] / chunk_size))):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, ray_origins.shape[0])
        
        # Get chunk data
        rays_o = ray_origins[start:end]
        rays_d = ray_directions[start:end]
        
        # Get appearance and transient indices for this chunk if available
        a_idx = appearance_idx[start:end] if appearance_idx is not None else None
        t_idx = transient_idx[start:end] if transient_idx is not None else None
        
        # Render rays
        results = render_rays_combined(model, rays_o, rays_d, a_idx, t_idx, hn=hn, hf=hf, nb_bins=nb_bins)
        
        # Collect results
        combined_rgb.append(results['rgb'].cpu())
        static_rgb.append(results['static_rgb'].cpu())
        transient_rgb.append(results.get('transient_rgb', torch.zeros_like(results['rgb'])).cpu())
        depth.append(results['depth'].cpu())
        beta.append(results.get('beta', torch.ones_like(results['depth'])).cpu())
    
    # Concatenate results and reshape to image dimensions
    combined_rgb = torch.cat(combined_rgb, 0).reshape(H, W, 3).numpy()
    static_rgb = torch.cat(static_rgb, 0).reshape(H, W, 3).numpy()
    transient_rgb = torch.cat(transient_rgb, 0).reshape(H, W, 3).numpy()
    depth_map = torch.cat(depth, 0).reshape(H, W).numpy()
    beta_map = torch.cat(beta, 0).reshape(H, W).numpy()
    
    # Save images
    # Combined RGB
    combined_img = (combined_rgb.clip(0, 1) * 255).astype(np.uint8)
    Image.fromarray(combined_img).save(f'{output_dir}/combined_img_{img_index}.png')
    
    # Static RGB
    static_img = (static_rgb.clip(0, 1) * 255).astype(np.uint8)
    Image.fromarray(static_img).save(f'{output_dir}/static_img_{img_index}.png')
    
    # Transient RGB (may be very dark if no transient elements)
    transient_img = (transient_rgb.clip(0, 1) * 255).astype(np.uint8)
    Image.fromarray(transient_img).save(f'{output_dir}/transient_img_{img_index}.png')
    
    # Depth visualization
    depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
    depth_img = (depth_normalized * 255).astype(np.uint8)
    Image.fromarray(depth_img).save(f'{output_dir}/depth_img_{img_index}.png')
    
    # Beta/uncertainty visualization
    beta_normalized = (beta_map - beta_map.min()) / (beta_map.max() - beta_map.min() + 1e-8)
    beta_img = (beta_normalized * 255).astype(np.uint8)
    Image.fromarray(beta_img).save(f'{output_dir}/uncertainty_img_{img_index}.png')
    
    return combined_img, static_img, transient_img, depth_img, beta_img

# Example usage
if __name__ == "__main__":
    # Check if MPS is available
    if torch.backends.mps.is_available():
        device = 'mps'
        print("Using MPS device")
    else:
        device = 'cpu'
        print("MPS device not found, using CPU")
    
    # Load datasets and ensure float32 precision
    # For NeRF-W, we need additional appearance and transient indices
    # Format: [ray_origin(3), ray_direction(3), appearance_idx(1), transient_idx(1), rgb(3)]
    training_dataset = torch.from_numpy(np.load('training_data_800x800.pkl', allow_pickle=True)).float()
    testing_dataset = torch.from_numpy(np.load('testing_data_800x800.pkl', allow_pickle=True)).float()
    
    # NGP parameters
    L = 16  # Number of levels
    F = 2   # Features per level
    T = 2**19  # Hash table size
    N_min = 16
    N_max = 2048
    b = np.exp((np.log(N_max) - np.log(N_min)) / (L - 1))
    Nl = [int(np.floor(N_min * b**l)) for l in range(L)]
    
    # NeRF-W parameters
    N_vocab = 100  # Number of possible images/scenes
    N_a = 48      # Appearance embedding dimensions
    N_tau = 16    # Transient embedding dimensions
    beta_min = 0.1  # Minimum uncertainty
    
    # Create model
    model = CombinedNGPNeRFW(T, Nl, 4, device, 3, N_vocab, N_a, N_tau, beta_min)
    
    # Create optimizer with different parameter groups
    model_optimizer = torch.optim.Adam([
        {"params": model.lookup_tables.parameters(), "lr": 1e-2, "betas": (0.9, 0.99), "eps": 1e-15},
        {"params": model.density_MLP.parameters(), "lr": 1e-2, "betas": (0.9, 0.99), "eps": 1e-15},
        {"params": model.static_rgb_MLP.parameters(), "lr": 1e-2, "betas": (0.9, 0.99), "eps": 1e-15},
        {"params": model.transient_MLP.parameters(), "lr": 1e-2, "betas": (0.9, 0.99), "eps": 1e-15},
        {"params": model.transient_density_MLP.parameters(), "lr": 1e-2, "betas": (0.9, 0.99), "eps": 1e-15},
        {"params": model.transient_rgb_MLP.parameters(), "lr": 1e-2, "betas": (0.9, 0.99), "eps": 1e-15},
        {"params": model.transient_beta_MLP.parameters(), "lr": 1e-2, "betas": (0.9, 0.99), "eps": 1e-15},
        {"params": model.embedding_a.parameters(), "lr": 1e-3, "betas": (0.9, 0.99), "eps": 1e-15},
        {"params": model.embedding_t.parameters(), "lr": 1e-3, "betas": (0.9, 0.99), "eps": 1e-15}
    ])
    
    # DataLoader
    data_loader = DataLoader(training_dataset, batch_size=2**14, shuffle=True)
    
    # Train
    train_combined(model, model_optimizer, data_loader, device=device,
                  hn=2, hf=6, nb_epochs=10, nb_bins=192, H=800, W=800)
    
    # Test
    for img_index in range(10):  # Test first 10 images
        # Can specify appearance_idx to control the appearance style
        # Can specify transient_idx to control the transient objects
        test_combined(model, testing_dataset, img_index, 
                    appearance_idx=0,  # Use appearance from image 0
                    transient_idx=None,  # Don't include transient objects
                    nb_bins=192, H=800, W=800, hn=2, hf=6)