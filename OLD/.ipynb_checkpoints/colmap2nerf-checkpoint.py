import numpy as np
import os
import shutil
from pathlib import Path
import collections

def parse_colmap_camera_file(camera_file):
    """Parse COLMAP camera.txt file."""
    cameras = {}
    with open(camera_file, "r") as f:
        lines = f.readlines()
        
    for line in lines:
        if line.startswith("#"):
            continue
        data = line.split()
        camera_id = int(data[0])
        model = data[1]
        width = int(data[2])
        height = int(data[3])
        params = np.array(data[4:], dtype=np.float64)
        
        cameras[camera_id] = {
            'model': model,
            'width': width,
            'height': height,
            'params': params
        }
    return cameras

def parse_colmap_images_file(images_file):
    """Parse COLMAP images.txt file."""
    images = {}
    with open(images_file, "r") as f:
        lines = f.readlines()

    line_idx = 0
    while line_idx < len(lines):
        line = lines[line_idx]
        if line.startswith("#"):
            line_idx += 1
            continue
            
        data = line.split()
        image_id = int(data[0])
        qw, qx, qy, qz = map(float, data[1:5])
        tx, ty, tz = map(float, data[5:8])
        camera_id = int(data[8])
        name = data[9]
        
        # Convert quaternion to rotation matrix
        R = quaternion_to_rotation_matrix([qw, qx, qy, qz])
        t = np.array([tx, ty, tz])
        
        images[image_id] = {
            'name': name,
            'camera_id': camera_id,
            'R': R,
            't': t
        }
        
        # Skip the next line (point correspondences)
        line_idx += 2
    return images

def quaternion_to_rotation_matrix(q):
    """Convert quaternion to rotation matrix."""
    w, x, y, z = q
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])
    return R

def create_intrinsics_matrix(camera):
    """Create 4x4 intrinsics matrix from COLMAP camera parameters."""
    if camera['model'] == 'SIMPLE_PINHOLE':
        f, cx, cy = camera['params']
        K = np.array([
            [f, 0, cx, 0],
            [0, f, cy, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    elif camera['model'] == 'PINHOLE':
        fx, fy, cx, cy = camera['params']
        K = np.array([
            [fx, 0, cx, 0],
            [0, fy, cy, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    elif camera['model'] == 'SIMPLE_RADIAL':
        # SIMPLE_RADIAL parameters: f, cx, cy, k
        # For NeRF, we'll ignore the radial distortion parameter k
        f, cx, cy, k = camera['params']
        K = np.array([
            [f, 0, cx, 0],
            [0, f, cy, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    else:
        raise ValueError(f"Unsupported camera model: {camera['model']}")
    return K

def create_pose_matrix(R, t):
    """Create 4x4 pose matrix from rotation matrix and translation vector."""
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = t
    return pose

def convert_colmap_to_nerf(colmap_dir, output_dir, images_dir, train_ratio=0.8):
    """Convert COLMAP output to NeRF training format.
    
    Args:
        colmap_dir: Directory containing COLMAP output (cameras.txt, images.txt)
        output_dir: Output directory for NeRF training data
        images_dir: Directory containing the source images
        train_ratio: Ratio of images to use for training
    """
    # Create output directories
    output_dir = Path(output_dir)
    for mode in ['train', 'test']:
        (output_dir / mode / 'pose').mkdir(parents=True, exist_ok=True)
        (output_dir / mode / 'intrinsics').mkdir(parents=True, exist_ok=True)
    (output_dir / 'imgs').mkdir(parents=True, exist_ok=True)
    
    # Parse COLMAP files
    cameras = parse_colmap_camera_file(os.path.join(colmap_dir, 'cameras.txt'))
    images = parse_colmap_images_file(os.path.join(colmap_dir, 'images.txt'))
    
    # Split into train/test
    image_ids = list(images.keys())
    np.random.shuffle(image_ids)
    n_train = int(len(image_ids) * train_ratio)
    train_ids = image_ids[:n_train]
    test_ids = image_ids[n_train:]
    
    # Process each image
    for mode, ids in [('train', train_ids), ('test', test_ids)]:
        for idx, image_id in enumerate(ids):
            image_data = images[image_id]
            camera = cameras[image_data['camera_id']]
            
            # Create pose matrix
            pose = create_pose_matrix(image_data['R'], image_data['t'])
            
            # Create intrinsics matrix
            K = create_intrinsics_matrix(camera)
            
            # Save matrices
            filename = f"{idx:06d}.txt"
            np.savetxt(output_dir / mode / 'pose' / filename, pose)
            np.savetxt(output_dir / mode / 'intrinsics' / filename, K)
            
            # Copy and rename image
            src_img = os.path.join(images_dir, image_data['name'])
            dst_img = output_dir / 'imgs' / f"{mode}_{idx:06d}.png"
            shutil.copy2(src_img, dst_img)
            
    print(f"Converted {len(train_ids)} training and {len(test_ids)} testing images")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert COLMAP output to NeRF training format")
    parser.add_argument("colmap_dir", help="Directory containing COLMAP output (cameras.txt, images.txt)")
    parser.add_argument("output_dir", help="Output directory for NeRF training data")
    parser.add_argument("--images-dir", help="Directory containing source images", default="imgs")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Ratio of images to use for training")
    args = parser.parse_args()
    
    convert_colmap_to_nerf(args.colmap_dir, args.output_dir, args.images_dir, args.train_ratio)