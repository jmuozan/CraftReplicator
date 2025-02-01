import torch
import numpy as np
from skimage.measure import marching_cubes
import trimesh
import trimesh.smoothing

def generate_detailed_mesh(model, device, N=200, scale=1.5, batch_size=50000, density_threshold=None):
    """
    Generate a detailed mesh from a NeRF model using batched processing.
    
    Args:
        model: The trained NeRF model
        device: torch device
        N: Number of points per dimension
        scale: Scene scale
        batch_size: Number of points to process at once
        density_threshold: Threshold for marching cubes (if None, uses mean-based threshold)
    """
    # Generate grid points
    x = torch.linspace(-scale, scale, N)
    y = torch.linspace(-scale, scale, N)
    z = torch.linspace(-scale, scale, N)
    x, y, z = torch.meshgrid((x, y, z), indexing='ij')
    xyz = torch.stack([x, y, z], dim=-1).reshape(-1, 3)
    
    # Process in batches to avoid memory issues
    densities = []
    for i in range(0, xyz.shape[0], batch_size):
        batch_xyz = xyz[i:i+batch_size].to(device)
        batch_dirs = torch.zeros_like(batch_xyz).to(device)
        
        with torch.no_grad():
            # Assuming model.forward returns (rgb, density)
            _, batch_density = model.forward(batch_xyz, batch_dirs)
            densities.append(batch_density.cpu())
    
    # Combine results
    density = torch.cat(densities, dim=0).numpy().reshape(N, N, N)
    
    # Determine threshold
    if density_threshold is None:
        density_threshold = 30 * np.mean(density)
    
    # Generate mesh using marching cubes
    vertices, faces, normals, values = marching_cubes(
        density,
        level=density_threshold,
        spacing=(scale*2/N, scale*2/N, scale*2/N)
    )
    
    # Create and clean mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, normals=normals)
    
    # Optional: Remove disconnected components
    components = mesh.split(only_watertight=False)
    if len(components) > 1:
        # Keep the largest component
        areas = np.array([c.area for c in components])
        mesh = components[np.argmax(areas)]
    
    # Smooth the mesh
    mesh = trimesh.smoothing.filter_laplacian(mesh)
    
    # Flip the normals by inverting faces
    mesh.invert()
    
    return mesh

# Usage example:
N = 250  # Increased resolution
scale = 1.5
mesh = generate_detailed_mesh(model, device, N=N, scale=scale)

# Display the mesh
mesh.show()


#mesh.export('mesh.obj')  # .stl, .ply, etc.