import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class PositionalEncoding(nn.Module):
    def __init__(self, num_frequencies=10, include_input=True):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.include_input = include_input
    
    def forward(self, x):
        """
        Args:
            x: Input coordinates (batch_size, ..., channel)
        """
        outputs = []
        if self.include_input:
            outputs.append(x)
        
        for i in range(self.num_frequencies):
            freq = 2.0 ** i
            for func in [torch.sin, torch.cos]:
                outputs.append(func(x * freq * np.pi))
        
        return torch.cat(outputs, dim=-1)

class NeRF(nn.Module):
   """
   Predicts RGB color and density
   """
   def __init__(self,
                pos_dim=3,
                dir_dim=3,
                hidden_dim=256,
                num_layers=8,
                skip_layers=[4],
                pos_frequencies=10,
                dir_frequencies=4):
       super().__init__()
       
       self.skip_layers = skip_layers
       
       # Position and direction encoders
       self.pos_encoder = PositionalEncoding(num_frequencies=pos_frequencies)
       self.dir_encoder = PositionalEncoding(num_frequencies=dir_frequencies)
       
       pos_encoded_dim = pos_dim * (2 * pos_frequencies + 1)
       dir_encoded_dim = dir_dim * (2 * dir_frequencies + 1)
       
       self.pos_layers = nn.ModuleList()
       input_dim = pos_encoded_dim
       for i in range(num_layers):
           if i in skip_layers:
               input_dim += pos_encoded_dim
           self.pos_layers.append(nn.Linear(input_dim, hidden_dim))
           input_dim = hidden_dim
       
       # Output layers for density and feature vector
       self.density_layer = nn.Linear(hidden_dim, 1)
       self.feature_layer = nn.Linear(hidden_dim, hidden_dim)
       
       self.dir_layer = nn.Linear(hidden_dim + dir_encoded_dim, hidden_dim // 2)
       self.color_layer = nn.Linear(hidden_dim // 2, 3)

       self.relu = nn.ReLU()
       self.sigmoid = nn.Sigmoid()
    

    def forward(self, pos, dir):
        """
        Args:
            pos: Position coordinates
            dir: View direction coordinates
            
        Returns:
            rgb: Predicted colors
            sigma: Predicted densities
        """
        pos_encoded = self.pos_encoder(pos)
        dir_encoded = self.dir_encoder(dir)
        
        # Process position features with skip connections
        x = pos_encoded
        for i, layer in enumerate(self.pos_layers):
            if i in self.skip_layers:
                x = torch.cat([x, pos_encoded], dim=-1)
            x = self.relu(layer(x))
        
        # Compute density and features
        sigma = self.relu(self.density_layer(x))
        features = self.feature_layer(x)
        
        dir_features = torch.cat([features, dir_encoded], dim=-1)
        x = self.relu(self.dir_layer(dir_features))
        rgb = self.sigmoid(self.color_layer(x))
        
        return rgb, sigma
   
    def render_rays(nerf_model, rays_o, rays_d, near, far, n_samples, rand=True):
        """
        Args:
            nerf_model: NeRF model instance
            rays_o: Ray origins
            rays_d: Ray directions
            near: Near plane distance
            far: Far plane distance
            n_samples: Number of samples per ray
            rand: Whether to add random noise to sample positions
        
        Returns:
            rgb: Rendered colors
            depth: Rendered depths
            weights: Sample weights
        """
        batch_size = rays_o.shape[0]
        
        # Sample points along each ray
        t_vals = torch.linspace(0., 1., n_samples, device=rays_o.device)
        z_vals = near * (1. - t_vals) + far * t_vals
        z_vals = z_vals.expand(batch_size, n_samples)
        
        if rand:
            # Add random noise to sample positions
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
            lower = torch.cat([z_vals[..., :1], mids], dim=-1)
            t_rand = torch.rand_like(z_vals)
            z_vals = lower + (upper - lower) * t_rand
        
        # Get sample positions along rays
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        
        pts_flat = pts.reshape(-1, 3)
        dirs_flat = rays_d[:, None].expand(-1, n_samples, -1).reshape(-1, 3)
        rgb, sigma = nerf_model(pts_flat, dirs_flat)
        
        rgb = rgb.reshape(batch_size, n_samples, 3)
        sigma = sigma.reshape(batch_size, n_samples)

        # Compute weights for volume rendering
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], dim=-1)
        alpha = 1. - torch.exp(-sigma * dists)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones_like(alpha[..., :1]), 1. - alpha + 1e-10], dim=-1),
            dim=-1
        )[..., :-1]
        
        rgb_final = (weights[..., None] * rgb).sum(dim=1)
        depth = (weights * z_vals).sum(dim=1)
        
        return rgb_final, depth, weights
   
   def test_nerf():
        model = NeRF()
        model.eval()
        
        print("forward pass")
        pos = torch.randn(4, 3)
        dir = torch.randn(4, 3)
        dir = dir / dir.norm(dim=-1, keepdim=True)
        rgb, sigma = model(pos, dir)
        print(f"RGB shape: {rgb.shape}, values range: [{rgb.min():.3f}, {rgb.max():.3f}]")
        print(f"Sigma shape: {sigma.shape}, values range: [{sigma.min():.3f}, {sigma.max():.3f}]")
        
        print("\nRay rendering")
        rays_o = torch.zeros(2, 3)
        rays_d = torch.tensor([[0., 0., 1.], [0., 1., 0.]])
        rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)
        
        rgb, depth, weights = render_rays(
            model, rays_o, rays_d,
            near=2., far=6.,
            n_samples=64, rand=True
        )
        
        print(f"Rendered RGB shape: {rgb.shape}")
        print(f"Rendered depth shape: {depth.shape}")
        print(f"Sample weights shape: {weights.shape}")

        # Multi-ray rendering
        print("\nMulti-ray rendering (simulated image)")
        n_rays = 32
        theta = torch.linspace(0, 2*np.pi, n_rays)
        rays_o = torch.zeros(n_rays, 3)
        rays_d = torch.stack([
            torch.cos(theta),
            torch.sin(theta),
            torch.ones_like(theta)
        ], dim=1)
        rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)
        
        rgb, depth, _ = render_rays(
            model, rays_o, rays_d,
            near=2., far=6.,
            n_samples=32, rand=True
        )
        
        # Visualize results
        plt.figure(figsize=(10, 4))
        
        plt.subplot(121)
        plt.scatter(theta.numpy(), depth.detach().numpy(), c=rgb.detach().numpy())
        plt.title('Depth vs Angle')
        plt.xlabel('Angle (radians)')
        plt.ylabel('Depth')
        
        plt.subplot(122)
        plt.scatter(rays_d[:, 0].numpy(), rays_d[:, 1].numpy(), c=rgb.detach().numpy())
        plt.title('Rendered Colors')
        plt.xlabel('Ray Direction X')
        plt.ylabel('Ray Direction Y')
        
        plt.tight_layout()
        # plt.show()

        return{
            "model": model,
            "test_rgb": rgb,
            "test_depth": depth,
            "test_rays_d": rays_d
        }
   

if __name__ == "__main__":
    test_results = test_nerf()