import numpy as np
import os
import imageio
import configparser
import sqlite3
import struct
from pathlib import Path

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file."""
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_cameras_binary(path_to_model_file):
    """
    Read COLMAP camera binary file.
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            width = camera_properties[2]
            height = camera_properties[3]
            
            num_params = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            params = read_next_bytes(fid, num_bytes=8*num_params,
                                   format_char_sequence="d" * num_params)
            
            cameras[camera_id] = {
                "model_id": model_id,
                "width": width,
                "height": height,
                "params": params
            }
        return cameras

def read_images_binary(path_to_model_file):
    """
    Read COLMAP images binary file.
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]

            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                         format_char_sequence="Q")[0]
            
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                     format_char_sequence="ddq"*num_points2D)
            
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                 tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            
            images[image_id] = {
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": camera_id,
                "name": image_name,
                "xys": xys,
                "point3D_ids": point3D_ids
            }
    return images

def qvec2rotmat(qvec):
    """Convert quaternion to rotation matrix."""
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def get_rays_from_colmap(base_dir, mode='train', split_ratio=0.9):
    """
    Get camera rays from COLMAP sparse reconstruction.
    
    Args:
        base_dir: Base directory containing COLMAP sparse reconstruction
        mode: Either 'train' or 'test'
        split_ratio: Ratio of images to use for training (default: 0.9)
    
    Returns:
        rays_o: Ray origins (N, H*W, 3)
        rays_d: Ray directions (N, H*W, 3)
        target_px_values: RGB values (N, H*W, 3)
    """
    sparse_dir = os.path.join(base_dir, 'sparse', '0')
    
    # Read COLMAP data
    cameras = read_cameras_binary(os.path.join(sparse_dir, 'cameras.bin'))
    images = read_images_binary(os.path.join(sparse_dir, 'images.bin'))
    
    # Sort images by name to ensure consistent order
    image_list = sorted(images.items(), key=lambda x: x[1]['name'])
    
    # Split into train/test
    split_idx = int(len(image_list) * split_ratio)
    if mode == 'train':
        image_list = image_list[:split_idx]
    else:
        image_list = image_list[split_idx:]
    
    rays_o = []
    rays_d = []
    target_px_values = []
    
    for image_id, image_data in image_list:
        # Get camera parameters
        camera = cameras[image_data['camera_id']]
        H, W = camera['height'], camera['width']
        f = camera['params'][0]  # Assuming SIMPLE_PINHOLE camera model
        
        # Load image
        img_path = os.path.join(base_dir, 'images', image_data['name'])
        img = imageio.imread(img_path).astype(np.float32) / 255.0
        
        # Convert quaternion to rotation matrix
        R = qvec2rotmat(image_data['qvec'])
        t = image_data['tvec']
        
        # Camera center (ray origin)
        C = -R.T @ t
        
        # Generate rays
        u = np.arange(W)
        v = np.arange(H)
        u, v = np.meshgrid(u, v)
        
        # Pixel coordinates to camera coordinates
        x = (u - W/2)
        y = -(v - H/2)
        z = -np.ones_like(u) * f
        
        # Convert to world coordinates
        dirs = np.stack([x, y, z], axis=-1)
        dirs = (R.T @ dirs.reshape(-1, 3).T).T
        dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
        
        rays_o.append(np.broadcast_to(C, dirs.shape))
        rays_d.append(dirs)
        target_px_values.append(img.reshape(-1, 3))
    
    rays_o = np.stack(rays_o)
    rays_d = np.stack(rays_d)
    target_px_values = np.stack(target_px_values)
    
    return rays_o, rays_d, target_px_values

def get_rays(datapath, mode='train'):
    """
    Wrapper function that supports both old text-based and new COLMAP-based loading.
    """
    # Check if COLMAP data exists
    if os.path.exists(os.path.join(datapath, 'sparse')):
        return get_rays_from_colmap(datapath, mode)
    else:
        # Original text-based loading code
        pose_file_names = [f for f in os.listdir(datapath + f'/{mode}/pose') if f.endswith('.txt')]
        intrisics_file_names = [f for f in os.listdir(datapath + f'/{mode}/intrinsics') if f.endswith('.txt')]
        img_file_names = [f for f in os.listdir(datapath + '/imgs') if mode in f]
        
        assert len(pose_file_names) == len(intrisics_file_names)
        assert len(img_file_names) == len(pose_file_names)
        
        N = len(pose_file_names)
        poses = np.zeros((N, 4, 4))
        intrinsics = np.zeros((N, 4, 4))
        images = []
        
        for i in range(N):
            name = pose_file_names[i]
            
            pose = open(datapath + f'/{mode}/pose/' + name).read().split()
            poses[i] = np.array(pose, dtype=float).reshape(4, 4)
            
            intrinsic = open(datapath + f'/{mode}/intrinsics/' + name).read().split()
            intrinsics[i] = np.array(intrinsic, dtype=float).reshape(4, 4)
            
            img = imageio.imread(datapath + '/imgs/' + name.replace('txt', 'png')) / 255.
            images.append(img[None, ...])
            
        images = np.concatenate(images)
        
        H = images.shape[1]
        W = images.shape[2]
        
        if images.shape[3] == 4:
            images = images[..., :3] * images[..., -1:] + (1 - images[..., -1:])
        
        rays_o = np.zeros((N, H*W, 3))
        rays_d = np.zeros((N, H*W, 3))
        target_px_values = images.reshape((N, H*W, 3))
        
        for i in range(N):
            c2w = poses[i]
            f = intrinsics[i, 0, 0]
            
            u = np.arange(W)
            v = np.arange(H)
            u, v = np.meshgrid(u, v)
            dirs = np.stack((u - W/2, -(v - H/2), -np.ones_like(u) * f), axis=-1)
            dirs = (c2w[:3, :3] @ dirs[..., None]).squeeze(-1)
            dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
            
            rays_d[i] = dirs.reshape(-1, 3)
            rays_o[i] += c2w[:3, 3]
        
        return rays_o, rays_d, target_px_values