import os
import numpy as np
import collections
import struct

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file."""
    data = fid.read(num_bytes)
    if len(data) != num_bytes:
        raise EOFError(f"Expected {num_bytes} bytes but got {len(data)} bytes")
    return struct.unpack(endian_character + format_char_sequence, data)

def read_cameras_binary(path_to_model_file):
    """
    Read COLMAP camera parameters from binary file
    """
    cameras = {}
    Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
    
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_id = read_next_bytes(fid, 4, "i")[0]
            model_id = read_next_bytes(fid, 4, "i")[0]
            width = read_next_bytes(fid, 4, "i")[0]
            height = read_next_bytes(fid, 4, "i")[0]
            num_params = 4  # For PINHOLE camera model
            params = read_next_bytes(fid, num_params * 8, "d" * num_params)
            cameras[camera_id] = Camera(camera_id, model_id, width, height, params)
    return cameras

def read_images_binary(path_to_model_file):
    """
    Read COLMAP images from binary file with proper binary parsing
    """
    images = {}
    Image = collections.namedtuple(
        "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
    
    with open(path_to_model_file, "rb") as fid:
        num_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_images):
            image_id = read_next_bytes(fid, 4, "i")[0]
            qw, qx, qy, qz = read_next_bytes(fid, 4 * 8, "dddd")
            tx, ty, tz = read_next_bytes(fid, 3 * 8, "ddd")
            camera_id = read_next_bytes(fid, 4, "i")[0]
            
            # Read image name
            name = ""
            char_data = read_next_bytes(fid, 1, "c")[0]
            while char_data != b'\0':
                name += char_data.decode("utf-8")
                char_data = read_next_bytes(fid, 1, "c")[0]
            
            # Read points
            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            xys = np.zeros((num_points2D, 2))
            point3D_ids = np.zeros(num_points2D, dtype=np.int64)
            
            for point_idx in range(num_points2D):
                xy = read_next_bytes(fid, 2 * 8, "dd")
                xys[point_idx] = xy
                point3D_id = read_next_bytes(fid, 8, "q")[0]
                point3D_ids[point_idx] = point3D_id
                
            images[image_id] = Image(
                image_id, np.array([qw, qx, qy, qz]), np.array([tx, ty, tz]), 
                camera_id, name, xys, point3D_ids)
            
    return images

def qvec2rotmat(qvec):
    """Convert quaternion to rotation matrix"""
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

def convert_colmap_to_custom(base_dir, sparse_dir):
    """
    Convert COLMAP output to custom format matching the fox dataset structure
    """
    # Ensure directory structure exists
    for split in ['train', 'test']:
        for subdir in ['intrinsics', 'pose']:
            os.makedirs(os.path.join(base_dir, split, subdir), exist_ok=True)
    
    try:
        # Read COLMAP binary files
        cameras_file = os.path.join(sparse_dir, 'cameras.bin')
        images_file = os.path.join(sparse_dir, 'images.bin')
        
        if not os.path.exists(cameras_file) or not os.path.exists(images_file):
            print(f"COLMAP output files not found in {sparse_dir}")
            return
            
        cameras = read_cameras_binary(cameras_file)
        images = read_images_binary(images_file)
        
        for image_id, image in images.items():
            try:
                # Get camera parameters
                camera = cameras[image.camera_id]
                
                # Create full 4x4 intrinsics matrix in the expected format
                fx, fy, cx, cy = camera.params
                K = np.array([
                    [fx, 0.0, cx, 0.0],
                    [0.0, fy, cy, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]
                ])
                
                # Create pose matrix
                R = qvec2rotmat(image.qvec)
                t = image.tvec
                pose = np.eye(4)
                pose[:3, :3] = R
                pose[:3, 3] = t
                
                # Determine if train or test from image name
                split = 'train' if 'train' in image.name else 'test'
                img_num = int(image.name.split('_')[1].split('.')[0])
                
                # Save intrinsics with full 4x4 matrix
                intrinsics_path = os.path.join(base_dir, split, 'intrinsics', f'{split}_{img_num}.txt')
                np.savetxt(intrinsics_path, K.flatten(), fmt='%.16f')
                
                # Save pose
                pose_path = os.path.join(base_dir, split, 'pose', f'{split}_{img_num}.txt')
                np.savetxt(pose_path, pose.flatten(), fmt='%.16f')
                
            except Exception as e:
                print(f"Error processing image {image.name}: {str(e)}")
                continue
                
    except Exception as e:
        print(f"Error processing COLMAP files: {str(e)}")
        raise


def run_colmap(fox_dir):
    """
    Run COLMAP pipeline on the fox dataset with optimal parameters
    """
    # Create COLMAP workspace
    os.makedirs('colmap_workspace', exist_ok=True)
    
    # Define paths
    database_path = os.path.join('colmap_workspace', 'database.db')
    image_path = os.path.join(fox_dir, 'imgs')
    sparse_path = os.path.join('colmap_workspace', 'sparse')
    os.makedirs(sparse_path, exist_ok=True)
    
    # Run COLMAP commands with more robust parameters for challenging scenes
    os.system(f'colmap feature_extractor \
        --database_path {database_path} \
        --image_path {image_path} \
        --ImageReader.camera_model SIMPLE_PINHOLE \
        --SiftExtraction.use_gpu 1 \
        --SiftExtraction.max_num_features 16384 \
        --SiftExtraction.first_octave -1 \
        --SiftExtraction.num_octaves 4 \
        --SiftExtraction.peak_threshold 0.004 \
        --SiftExtraction.edge_threshold 20')
        
    os.system(f'colmap exhaustive_matcher \
        --database_path {database_path} \
        --SiftMatching.use_gpu 1 \
        --SiftMatching.max_ratio 0.9 \
        --SiftMatching.max_distance 0.8 \
        --SiftMatching.cross_check 1')
        
    os.system(f'colmap mapper \
        --database_path {database_path} \
        --image_path {image_path} \
        --output_path {sparse_path} \
        --Mapper.init_min_tri_angle 4 \
        --Mapper.multiple_models 1 \
        --Mapper.extract_colors 0 \
        --Mapper.ba_refine_focal_length 1 \
        --Mapper.ba_refine_extra_params 1 \
        --Mapper.min_num_matches 15 \
        --Mapper.abs_pose_min_num_inliers 10 \
        --Mapper.abs_pose_min_inlier_ratio 0.5')
    
    return os.path.join(sparse_path, '0')

if __name__ == "__main__":
    # First clean up any existing workspace
    if os.path.exists('colmap_workspace'):
        print("Removing old COLMAP workspace...")
        for root, dirs, files in os.walk('colmap_workspace', topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir('colmap_workspace')
    
    # Path to your fox dataset
    fox_dir = "fox"  # Change this to your fox directory path
    
    # Run COLMAP
    print("Running COLMAP pipeline...")
    sparse_dir = run_colmap(fox_dir)
    
    # Convert COLMAP output to your format
    print("Converting COLMAP output to custom format...")
    convert_colmap_to_custom(fox_dir, sparse_dir)
    print("Done!")