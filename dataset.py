import numpy as np
import os
import imageio
import configparser
from pathlib import Path
import torch

def parse_colmap_ini(ini_path):
    """Parse COLMAP .ini configuration file"""
    config = configparser.ConfigParser()
    config.read(ini_path)
    
    # Extract camera parameters
    camera_params = {}
    for section in config.sections():
        if section.startswith('camera'):
            params = dict(config[section])
            camera_params[section] = {
                'model': params.get('model', ''),
                'width': int(params.get('width', 0)),
                'height': int(params.get('height', 0)),
                'params': [float(x) for x in params.get('params', '').split()]
            }
            
    # Extract image parameters
    image_params = {}
    for section in config.sections():
        if section.startswith('image'):
            params = dict(config[section])
            image_params[section] = {
                'camera_id': params.get('camera_id', ''),
                'qw': float(params.get('qw', 0)),
                'qx': float(params.get('qx', 0)), 
                'qy': float(params.get('qy', 0)),
                'qz': float(params.get('qz', 0)),
                'tx': float(params.get('tx', 0)),
                'ty': float(params.get('ty', 0)), 
                'tz': float(params.get('tz', 0)),
                'name': params.get('name', '')
            }
            
    return camera_params, image_params

def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    """Convert quaternion to rotation matrix"""
    R = np.zeros((3, 3))
    
    R[0,0] = 1 - 2*qy**2 - 2*qz**2
    R[0,1] = 2*qx*qy - 2*qz*qw
    R[0,2] = 2*qx*qz + 2*qy*qw
    
    R[1,0] = 2*qx*qy + 2*qz*qw
    R[1,1] = 1 - 2*qx**2 - 2*qz**2
    R[1,2] = 2*qy*qz - 2*qx*qw
    
    R[2,0] = 2*qx*qz - 2*qy*qw
    R[2,1] = 2*qy*qz + 2*qx*qw
    R[2,2] = 1 - 2*qx**2 - 2*qy**2
    
    return R

def get_rays_from_colmap(ini_path, images_dir, mode='train', train_val_split=0.8):
    """Generate camera rays from COLMAP data
    
    Args:
        ini_path: Path to COLMAP .ini file
        images_dir: Directory containing input images
        mode: Either 'train' or 'test' 
        train_val_split: Fraction of images to use for training
    
    Returns:
        rays_o: Ray origins
        rays_d: Ray directions  
        target_px_values: RGB values
    """
    
    camera_params, image_params = parse_colmap_ini(ini_path)
    
    # Sort images to ensure consistent train/test split
    image_names = sorted([p['name'] for p in image_params.values()])
    split_idx = int(len(image_names) * train_val_split)
    
    if mode == 'train':
        image_names = image_names[:split_idx]
    else:
        image_names = image_names[split_idx:]
        
    N = len(image_names)
    
    # Use parameters from first camera (assuming single camera)
    cam = list(camera_params.values())[0]
    H, W = cam['height'], cam['width']
    focal = cam['params'][0]  # Assuming SIMPLE_PINHOLE camera model
    
    images = []
    poses = []
    
    for img_name in image_names:
        # Find corresponding image parameters
        img_param = None
        for p in image_params.values():
            if p['name'] == img_name:
                img_param = p
                break
                
        if img_param is None:
            raise ValueError(f"Could not find parameters for image {img_name}")
            
        # Load image
        img_path = os.path.join(images_dir, img_name)
        img = imageio.imread(img_path) / 255.
        if img.shape[-1] == 4:  # RGBA -> RGB
            img = img[..., :3] * img[..., -1:] + (1 - img[..., -1:])
        images.append(img[None, ...])
        
        # Build camera pose
        R = quaternion_to_rotation_matrix(
            img_param['qw'], img_param['qx'], 
            img_param['qy'], img_param['qz']
        )
        t = np.array([img_param['tx'], img_param['ty'], img_param['tz']])
        
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = t
        poses.append(pose)
        
    images = np.concatenate(images)
    poses = np.stack(poses)
    
    rays_o = np.zeros((N, H*W, 3))
    rays_d = np.zeros((N, H*W, 3))
    target_px_values = images.reshape((N, H*W, 3))
    
    for i in range(N):
        c2w = poses[i]
        
        # Generate ray directions
        u = np.arange(W)
        v = np.arange(H)
        u, v = np.meshgrid(u, v)
        dirs = np.stack((u - W/2, -(v - H/2), -np.ones_like(u) * focal), axis=-1)
        
        # Transform directions to world space
        dirs = (c2w[:3, :3] @ dirs[..., None]).squeeze(-1)
        dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
        
        rays_d[i] = dirs.reshape(-1, 3)
        rays_o[i] += c2w[:3, 3]
        
    return rays_o, rays_d, target_px_values

# Example usage:
if __name__ == "__main__":
    ini_path = "fox/fox.ini"
    images_dir = "fox/imgs"
    
    # Get training rays
    rays_o, rays_d, target_px_values = get_rays_from_colmap(
        ini_path, images_dir, mode='train'
    )
    
    # Create data loader
    data = torch.cat((
        torch.from_numpy(rays_o).reshape(-1, 3).type(torch.float),
        torch.from_numpy(rays_d).reshape(-1, 3).type(torch.float),
        torch.from_numpy(target_px_values).reshape(-1, 3).type(torch.float)
    ), dim=1)
    
    batch_size = 1024
    data_loader = torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=True
    )