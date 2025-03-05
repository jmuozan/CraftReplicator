import os
import json
import numpy as np
from pathlib import Path
import struct
from PIL import Image
import argparse

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file."""
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_cameras_binary(path_to_model_file):
    """
    Read COLMAP's cameras.bin file
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        print(f"Number of cameras: {num_cameras}")
        for camera_line_index in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            width = camera_properties[2]
            height = camera_properties[3]
            
            # Different models have different numbers of parameters
            num_params = {
                0: 3,  # SIMPLE_PINHOLE
                1: 4,  # PINHOLE
                2: 4,  # SIMPLE_RADIAL
                3: 5,  # RADIAL
                4: 8,  # OPENCV
                5: 8,  # OPENCV_FISHEYE
                6: 12, # FULL_OPENCV
                7: 5,  # FOV
                8: 4,  # SIMPLE_RADIAL_FISHEYE
                9: 5,  # RADIAL_FISHEYE
                10: 12 # THIN_PRISM_FISHEYE
            }.get(model_id, 0)
            
            params = read_next_bytes(fid, num_bytes=8*num_params, format_char_sequence="d"*num_params)
            cameras[camera_id] = {
                "id": camera_id,
                "model_id": model_id,
                "width": width,
                "height": height,
                "params": list(params)
            }
    return cameras

def read_images_binary(path_to_model_file):
    """
    Read COLMAP's images.bin file
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        print(f"Number of images: {num_reg_images}")
        for image_index in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":  # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            
            # Skip the points
            num_points2D = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            # Skip 2D points
            fid.seek(num_points2D * 24, 1)
            
            images[image_id] = {
                "id": image_id,
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": camera_id,
                "name": image_name
            }
    return images

def qvec2rotmat(qvec):
    """Convert quaternion to rotation matrix"""
    w, x, y, z = qvec
    R = np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
        [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
        [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]
    ])
    return R

def prepare_nerf_data(colmap_dir, output_dir, image_dir, square_images=True):
    """Prepare NeRF dataset from COLMAP reconstruction"""
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    image_output_dir = os.path.join(output_dir, 'images')
    os.makedirs(image_output_dir, exist_ok=True)
    
    print(f"Reading COLMAP data from {colmap_dir}")
    cameras_path = os.path.join(colmap_dir, 'cameras.bin')
    images_path = os.path.join(colmap_dir, 'images.bin')
    
    if not os.path.exists(cameras_path) or not os.path.exists(images_path):
        print(f"Error: Missing COLMAP files in {colmap_dir}")
        return False
    
    # Read COLMAP output
    try:
        cameras = read_cameras_binary(cameras_path)
        images = read_images_binary(images_path)
        print(f"Successfully read {len(cameras)} cameras and {len(images)} images")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error reading COLMAP data: {e}")
        return False
    
    if not cameras or not images:
        print("Error: No cameras or images found in COLMAP output")
        return False
    
    # Get camera parameters from the first camera
    camera_id = next(iter(cameras))
    camera = cameras[camera_id]
    width, height = camera["width"], camera["height"]
    
    # For square images
    if square_images:
        size = max(width, height)
        width = height = size
    
    # Get focal length (depends on camera model)
    model_id = camera["model_id"]
    params = camera["params"]
    
    if model_id == 0:  # SIMPLE_PINHOLE
        focal = params[0]
    elif model_id == 1:  # PINHOLE
        focal = params[0]  # fx
    else:
        print(f"Warning: Using first parameter as focal length for camera model {model_id}")
        focal = params[0]
    
    # Camera angle
    camera_angle_x = 2 * np.arctan(width / (2 * focal))
    print(f"Camera parameters: {width}x{height}, focal={focal}, angle_x={np.degrees(camera_angle_x):.2f} degrees")
    
    # Prepare transforms
    frames = []
    for image_id, image in images.items():
        qvec = image["qvec"]
        tvec = image["tvec"]
        R = qvec2rotmat(qvec)
        
        # Create transformation matrix
        c2w = np.eye(4)
        c2w[:3, :3] = R
        c2w[:3, 3] = tvec
        
        # Get image name
        image_name = image["name"]
        # Extract stem (filename without extension)
        file_stem = Path(image_name).stem
        file_path = f"images/{file_stem}"
        
        # Find the image file
        src_path = os.path.join(image_dir, image_name)
        if not os.path.exists(src_path):
            # Try with different extensions
            found = False
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                alt_path = os.path.join(image_dir, file_stem + ext)
                if os.path.exists(alt_path):
                    src_path = alt_path
                    found = True
                    break
            
            if not found:
                print(f"Warning: Could not find image {image_name} in {image_dir}")
                continue
        
        # Copy and process the image
        dst_path = os.path.join(image_output_dir, f"{file_stem}.png")
        try:
            img = Image.open(src_path)
            
            if square_images:
                # Center crop to square
                min_dim = min(img.width, img.height)
                left = (img.width - min_dim) // 2
                top = (img.height - min_dim) // 2
                right = left + min_dim
                bottom = top + min_dim
                img = img.crop((left, top, right, bottom))
            
            img = img.resize((width, height), Image.LANCZOS)
            img.save(dst_path)
            print(f"Processed image: {src_path} -> {dst_path}")
        except Exception as e:
            print(f"Error processing image {src_path}: {e}")
            continue
        
        # Add frame to transforms
        frame = {
            "file_path": file_path,
            "transform_matrix": c2w.tolist()
        }
        frames.append(frame)
    
    if not frames:
        print("Error: No valid frames found")
        return False
    
    # Split into train/val/test
    num_frames = len(frames)
    i_train = max(1, int(num_frames * 0.8))
    i_val = max(i_train + 1, int(num_frames * 0.9))
    
    train_frames = frames[:i_train]
    val_frames = frames[i_train:i_val]
    test_frames = frames[i_val:]
    
    # Create transforms files
    transforms_train = {
        "camera_angle_x": float(camera_angle_x),
        "frames": train_frames
    }
    
    transforms_val = {
        "camera_angle_x": float(camera_angle_x),
        "frames": val_frames
    }
    
    transforms_test = {
        "camera_angle_x": float(camera_angle_x),
        "frames": test_frames
    }
    
    # Write transforms files
    with open(os.path.join(output_dir, 'transforms_train.json'), 'w') as f:
        json.dump(transforms_train, f, indent=2)
    
    with open(os.path.join(output_dir, 'transforms_val.json'), 'w') as f:
        json.dump(transforms_val, f, indent=2)
    
    with open(os.path.join(output_dir, 'transforms_test.json'), 'w') as f:
        json.dump(transforms_test, f, indent=2)
    
    print(f"Created transforms files with {len(train_frames)} training, {len(val_frames)} validation, and {len(test_frames)} test frames")
    print(f"NeRF dataset prepared successfully in {output_dir}")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare NeRF dataset from COLMAP reconstruction")
    parser.add_argument("--colmap_dir", required=True, help="Directory containing COLMAP reconstruction")
    parser.add_argument("--output_dir", required=True, help="Directory to save NeRF dataset")
    parser.add_argument("--image_dir", required=True, help="Directory containing original images")
    parser.add_argument("--square", action="store_true", help="Make images square (recommended for NeRF)")
    
    args = parser.parse_args()
    prepare_nerf_data(args.colmap_dir, args.output_dir, args.image_dir, args.square)