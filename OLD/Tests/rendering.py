import torch
from model import Nerf

def compute_accumulated_transmittance(betas):
    accumulated_transmittance = torch.cumprod(betas, 1)     
    return torch.cat((torch.ones(accumulated_transmittance.shape[0], 1, device=accumulated_transmittance.device),
                      accumulated_transmittance[:, :-1]), dim=1)

def rendering(model, rays_o, rays_d, tn, tf, nb_bins=100, device='mps', white_bckgr=True, batch_size=1024, debug=False):
    # Convert inputs to tensors on the correct device if they aren't already
    if not isinstance(rays_o, torch.Tensor):
        rays_o = torch.from_numpy(rays_o)
    if not isinstance(rays_d, torch.Tensor):
        rays_d = torch.from_numpy(rays_d)
    
    # Move to device
    rays_o = rays_o.to(device)
    rays_d = rays_d.to(device)
    
    # Get total number of rays
    n_rays = rays_o.shape[0]
    
    # Initialize output
    output = torch.zeros((n_rays, 3), device=device)
    
    # Process in batches
    for i in range(0, n_rays, batch_size):
        end_idx = min(i + batch_size, n_rays)
        rays_o_batch = rays_o[i:end_idx]
        rays_d_batch = rays_d[i:end_idx]
        
        t = torch.linspace(tn, tf, nb_bins).to(device)
        delta = torch.cat((t[1:] - t[:-1], torch.tensor([1e10], device=device)))
        
        if debug and i == 0:
            print(f"Sample points range: {t[0].item():.4f} to {t[-1].item():.4f}")
            print(f"Delta range: {delta[0].item():.4f} to {delta[-1].item():.4f}")
        
        x = rays_o_batch.unsqueeze(1) + t.unsqueeze(0).unsqueeze(-1) * rays_d_batch.unsqueeze(1)
        
        if debug and i == 0:
            print(f"Sample points (x) range: {x.min().item():.4f} to {x.max().item():.4f}")
        
        colors, density = model.intersect(x.reshape(-1, 3), 
                                        rays_d_batch.expand(x.shape[1], x.shape[0], 3).transpose(0, 1).reshape(-1, 3))
        
        if debug and i == 0:
            print(f"Raw colors range: {colors.min().item():.4f} to {colors.max().item():.4f}")
            print(f"Raw density range: {density.min().item():.4f} to {density.max().item():.4f}")
        
        colors = colors.reshape((x.shape[0], nb_bins, 3))
        density = density.reshape((x.shape[0], nb_bins))
        
        alpha = 1 - torch.exp(- density * delta.unsqueeze(0))
        
        if debug and i == 0:
            print(f"Alpha range: {alpha.min().item():.4f} to {alpha.max().item():.4f}")
        
        weights = compute_accumulated_transmittance(1 - alpha) * alpha
        
        if debug and i == 0:
            print(f"Weights range: {weights.min().item():.4f} to {weights.max().item():.4f}")
            print(f"Weights sum: {weights.sum(-1).mean().item():.4f}")
        
        if white_bckgr:
            c = (weights.unsqueeze(-1) * colors).sum(1)
            weight_sum = weights.sum(-1)
            output[i:end_idx] = c + 1 - weight_sum.unsqueeze(-1)
        else:
            c = (weights.unsqueeze(-1) * colors).sum(1)
            output[i:end_idx] = c
        
        if debug and i == 0:
            print(f"Batch output range: {output[i:end_idx].min().item():.4f} to {output[i:end_idx].max().item():.4f}")
    
    return output

def test_render(model, o, d, tn, tf, nb_bins=100, chunk_size=10, H=400, W=400, debug=True):
    """
    Memory efficient rendering function that processes data in chunks with debug info
    """
    # Ensure inputs are on the correct device
    device = next(model.parameters()).device
    
    if debug:
        # Print model device and sample parameter stats
        print(f"Model device: {device}")
        sample_param = next(model.parameters())
        print(f"Sample parameter range: {sample_param.min().item():.4f} to {sample_param.max().item():.4f}")
    
    # Convert to torch tensors if needed
    if not isinstance(o, torch.Tensor):
        o = torch.from_numpy(o)
    if not isinstance(d, torch.Tensor):
        d = torch.from_numpy(d)
        
    o = o.to(device)
    d = d.to(device)
    
    if debug:
        print(f"\nRay origins range: {o.min().item():.4f} to {o.max().item():.4f}")
        print(f"Ray directions range: {d.min().item():.4f} to {d.max().item():.4f}\n")
    
    # Split the rays into chunks
    total_pixels = H * W
    chunk_size = total_pixels // chunk_size  # Adjust chunk size based on total pixels
    
    # Initialize output tensor
    output = torch.zeros((total_pixels, 3), device=device)
    
    # Process each chunk
    for i in range(0, total_pixels, chunk_size):
        end_idx = min(i + chunk_size, total_pixels)
        
        # Get current chunk
        o_chunk = o[i:end_idx]
        d_chunk = d[i:end_idx]
        
        # Process chunk
        with torch.no_grad():  # Disable gradient computation for inference
            rendered_chunk = rendering(model, o_chunk, d_chunk, tn, tf, 
                                    nb_bins=nb_bins, device=device, 
                                    white_bckgr=False, debug=debug and i==0)
            output[i:end_idx] = rendered_chunk
    
    # Final output check
    if debug:
        print(f"\nFinal output range: {output.min().item():.4f} to {output.max().item():.4f}")
    
    # Reshape to final image dimensions
    return output.reshape(H, W, 3)