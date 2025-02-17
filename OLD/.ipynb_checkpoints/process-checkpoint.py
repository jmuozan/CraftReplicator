import numpy as np
import os
from pathlib import Path

def read_colmap_model(model_path):
    """Read COLMAP text format model"""
    cameras_path = os.path.join(model_path, "cameras.txt")
    images_path = os.path.join(model_path, "images.txt")
    
    # Read cameras
    cameras = {}
    with open(cameras_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line[0] == "#":  # Skip comments
                continue
            data = line.split()
            camera_id = int(data[0])
            model = data[1]
            width = int(data[2])
            height = int(data[3])
            params = np.array(data[4:], dtype=np.float64)
            cameras[camera_id] = {
                "model": model,
                "width": width,
                "height": height,
                "params": params
            }
    
    # Read images
    images = {}
    with open(images_path, "r") as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            if lines[i][0] == "#":  # Skip comments
                i += 1
                continue
            data = lines[i].split()
            image_id = int(data[0])
            qw, qx, qy, qz = map(float, data[1:5])
            tx, ty, tz = map(float, data[5:8])
            camera_id = int(data[8])
            name = data[9]
            
            # Convert quaternion to rotation matrix
            R = quaternion_to_rotation_matrix([qw, qx, qy, qz])
            t = np.array([tx, ty, tz])
            
            images[image_id] = {
                "R": R,
                "t": t,
                "camera_id": camera_id,
                "name": name
            }
            i += 2  # Skip points2D line
    
    return cameras, images

def quaternion_to_rotation_matrix(q):
    """Convert quaternion to rotation matrix"""
    w, x, y, z = q
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])
    return R

def convert_to_nerf_format(cameras, images, output_dir):
    """Convert COLMAP parameters to NeRF format"""
    os.makedirs(output_dir, exist_ok=True)
    
    for image_id, image_data in images.items():
        # Get camera parameters
        camera = cameras[image_data["camera_id"]]
        
        if camera["model"] == "SIMPLE_PINHOLE":
            f = camera["params"][0]
            cx, cy = camera["params"][1:]
            intrinsics = np.array([
                [f, 0, cx, 0],
                [0, f, cy, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
        else:
            print(f"Warning: Camera model {camera['model']} not fully supported")
            # Add handling for other camera models as needed
        
        # Create pose matrix
        R = image_data["R"]
        t = image_data["t"]
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = t
        
        # Save matrices
        base_name = Path(image_data["name"]).stem
        np.savetxt(os.path.join(output_dir, f"{base_name}_intrinsics.txt"), 
                  intrinsics.flatten(), fmt='%.16f')
        np.savetxt(os.path.join(output_dir, f"{base_name}_pose.txt"), 
                  pose.flatten(), fmt='%.16f')

if __name__ == "__main__":
    # Example usage
    colmap_model_path = "./sparse/0"  # Path to COLMAP output
    output_dir = "./nerf_params"      # Where to save converted parameters
    
    cameras, images = read_colmap_model(colmap_model_path)
    convert_to_nerf_format(cameras, images, output_dir)