import torch
import os
import numpy as np
import cv2
import imageio
from PIL import Image
from tqdm import tqdm
import mcubes
import open3d as o3d
from plyfile import PlyData, PlyElement
from argparse import ArgumentParser

from rendering import rendering
from model import Nerf, NeRFWithPoses

# You might need to install these packages:
# pip install mcubes open3d plyfile

def get_opts():
    parser = ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True,
                        help='path to the trained model checkpoint')
    parser.add_argument('--output_mesh', type=str, default='output_mesh.ply',
                        help='output mesh filename')
    parser.add_argument('--grid_size', type=int, default=128,
                        help='size of the grid on 1 side, larger=higher resolution')
    parser.add_argument('--sigma_threshold', type=float, default=50.0,
                        help='threshold to consider a location is occupied')
    parser.add_argument('--x_range', nargs="+", type=float, default=[-2.0, 2.0],
                        help='x range of the object')
    parser.add_argument('--y_range', nargs="+", type=float, default=[-2.0, 2.0],
                        help='y range of the object')
    parser.add_argument('--z_range', nargs="+", type=float, default=[-2.0, 2.0],
                        help='z range of the object')
    parser.add_argument('--chunk_size', type=int, default=32*1024,
                        help='chunk size to split the input to avoid OOM')
    parser.add_argument('--device', type=str, default='cuda',
                        help='device to use')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='hidden dimension size used during training')
    parser.add_argument('--Lpos', type=int, default=6,
                        help='positional encoding parameter used during training')
    parser.add_argument('--Ldir', type=int, default=2,
                        help='directional encoding parameter used during training')
    
    return parser.parse_args()

def load_model(model_path, args, device='cuda'):
    """Load the NeRF model from checkpoint"""
    print(f"Loading model from {model_path}")
    
    # Create a model with the same architecture as used in training
    hidden_dim = args.hidden_dim
    Lpos = args.Lpos
    Ldir = args.Ldir
    
    print(f"Initializing model with hidden_dim={hidden_dim}, Lpos={Lpos}, Ldir={Ldir}")
    model = Nerf(hidden_dim=hidden_dim, Lpos=Lpos, Ldir=Ldir).to(device)
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                model_state = checkpoint['model']
                model.load_state_dict(model_state)
            elif 'nerf' in checkpoint:
                model_state = checkpoint['nerf']
                model.load_state_dict(model_state)
            else:
                # Try direct loading
                model.load_state_dict(checkpoint)
        else:
            # If checkpoint is the model itself
            if isinstance(checkpoint, Nerf):
                model = checkpoint
            elif isinstance(checkpoint, NeRFWithPoses):
                model = checkpoint.nerf
            else:
                raise ValueError("Loaded model is not a Nerf or NeRFWithPoses instance")
    
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    
    model.eval()  # Set to evaluation mode
    return model

def extract_mesh(model, grid_size, x_range, y_range, z_range, 
                 sigma_threshold, chunk_size, device='cuda'):
    """Extract mesh from NeRF model using marching cubes"""
    print("Extracting mesh...")
    
    # Define the dense grid
    xmin, xmax = x_range
    ymin, ymax = y_range
    zmin, zmax = z_range
    
    x = np.linspace(xmin, xmax, grid_size)
    y = np.linspace(ymin, ymax, grid_size)
    z = np.linspace(zmin, zmax, grid_size)
    
    # Reshape to grid points
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    points = np.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], -1)
    
    # Predict occupancy (density)
    sigma = np.zeros(len(points))
    
    with torch.no_grad():
        for i in tqdm(range(0, len(points), chunk_size)):
            # Get chunk of points
            chunk_points = torch.FloatTensor(points[i:i+chunk_size]).to(device)
            
            # Use arbitrary viewing directions (density is view-independent)
            # For simplicity, use the same direction for all points
            directions = torch.zeros_like(chunk_points).to(device)
            directions[:, 2] = 1.0  # looking along z-axis
            
            # Get model prediction (we only need density)
            colors, density = model.intersect(chunk_points, directions)
            
            # Store density values
            sigma[i:i+chunk_size] = density.cpu().numpy()
    
    # Reshape to grid
    sigma = sigma.reshape(grid_size, grid_size, grid_size)
    
    # Extract mesh with marching cubes
    print(f"Running marching cubes with threshold {sigma_threshold}...")
    vertices, triangles = mcubes.marching_cubes(sigma, sigma_threshold)
    
    # Convert vertices to actual coordinates
    vertices = vertices.astype(np.float32)
    vertices[:, 0] = vertices[:, 0] * (xmax - xmin) / grid_size + xmin
    vertices[:, 1] = vertices[:, 1] * (ymax - ymin) / grid_size + ymin
    vertices[:, 2] = vertices[:, 2] * (zmax - zmin) / grid_size + zmin
    
    return vertices, triangles

def main():
    args = get_opts()
    device = torch.device(args.device if torch.cuda.is_available() and args.device=='cuda' else "cpu")
    
    # Load model
    model = load_model(args.model_path, args, device)
    
    # Extract mesh
    vertices, triangles = extract_mesh(
        model,
        args.grid_size,
        args.x_range,
        args.y_range,
        args.z_range,
        args.sigma_threshold,
        args.chunk_size,
        device
    )
    
    print(f"Mesh has {len(vertices)} vertices and {len(triangles)} faces")
    
    # Convert to Open3D mesh for processing
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    
    # Compute vertex normals for better visualization
    mesh.compute_vertex_normals()
    print("Computing vertex normals done.")
    
    # Remove noise by keeping only the largest connected component
    try:
        print("Removing noise by keeping the largest component...")
        idxs, counts, _ = mesh.cluster_connected_triangles()
        if len(counts) > 0:
            largest_cluster = np.argmax(counts)
            triangles_to_remove = [i for i, cluster_idx in enumerate(idxs) if cluster_idx != largest_cluster]
            mesh.remove_triangles_by_index(triangles_to_remove)
            mesh.remove_unreferenced_vertices()
            print(f"After cleaning: {len(mesh.vertices)} vertices and {len(mesh.triangles)} faces")
    except Exception as e:
        print(f"Warning: Could not remove noise: {e}")
    
    # Save the mesh
    print(f"Saving mesh to {args.output_mesh}")
    o3d.io.write_triangle_mesh(args.output_mesh, mesh)
    
    print("Done!")

if __name__ == "__main__":
    main()