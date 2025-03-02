import os
import sys
import subprocess
import glob
import numpy as np
import cv2
import argparse
import shutil

# Make sure pycolmap is installed
try:
    import pycolmap
except ImportError:
    print("PyColmap is not installed. Installing it now...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pycolmap"])
    import pycolmap

def run_colmap(image_dir, db_path, output_dir):
    """
    Run COLMAP structure-from-motion on the images to get camera parameters.
    
    Args:
        image_dir: Directory containing the images
        db_path: Path to the COLMAP database
        output_dir: Directory to save COLMAP outputs
    """
    print("Running COLMAP reconstruction...")
    
    # Check if COLMAP executable is available (optional but helpful)
    try:
        colmap_path = subprocess.check_output(["which", "colmap"]).decode().strip()
        print(f"Using COLMAP binary at: {colmap_path}")
    except:
        print("COLMAP binary not found in PATH. We'll try to continue with pycolmap...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    sparse_dir = os.path.join(output_dir, "sparse")
    os.makedirs(sparse_dir, exist_ok=True)
    
    # Feature extraction - using the correct argument format
    print("Extracting features...")
    pycolmap.extract_features(
        database_path=db_path,
        image_path=image_dir,
        camera_model="SIMPLE_RADIAL"
    )
    
    # Feature matching
    print("Matching features...")
    pycolmap.match_exhaustive(database_path=db_path)
    
    # Triangulation
    print("Running mapper (this may take a while)...")
    reconstructions = pycolmap.incremental_mapping(
        database_path=db_path,
        image_path=image_dir,
        output_path=sparse_dir
    )
    
    if not reconstructions:
        print("COLMAP reconstruction failed! Check your images and try again.")
        sys.exit(1)
    
    print(f"COLMAP reconstruction created with {len(reconstructions[0].images)} images")
    return reconstructions[0]  # Return the first reconstruction

def create_nerf_dataset(reconstruction, image_dir, mask_dir, output_path, image_scale=1.0):
    """
    Create a NeRF-compatible dataset from COLMAP reconstruction.
    
    Args:
        reconstruction: COLMAP reconstruction object
        image_dir: Directory containing the original images
        mask_dir: Directory containing segmentation masks
        output_path: Path to save the .npz file
        image_scale: Scale factor to resize images (optional)
    """
    print("Creating NeRF dataset...")
    
    # Get all images in the reconstruction
    colmap_images = reconstruction.images
    
    # Get a list of image paths and sort them
    image_paths = []
    for img_id in colmap_images:
        img_path = os.path.join(image_dir, colmap_images[img_id].name)
        image_paths.append(img_path)
    
    image_paths = sorted(image_paths)
    
    if not image_paths:
        print("No images found in the reconstruction! Check COLMAP results.")
        return
    
    print(f"Processing {len(image_paths)} images...")
    
    # Read one image to get dimensions
    sample_img = cv2.imread(image_paths[0])
    if sample_img is None:
        print(f"Failed to read image {image_paths[0]}. Check file paths.")
        return
    
    h, w, _ = sample_img.shape
    h_target, w_target = int(h * image_scale), int(w * image_scale)
    
    # Prepare arrays for the dataset
    n_images = len(image_paths)
    images_array = np.zeros((n_images, h_target, w_target, 3), dtype=np.float32)  # RGB
    poses_array = np.zeros((n_images, 4, 4), dtype=np.float32)  # Camera poses
    
    # Check if mask directory exists and has files
    use_masks = os.path.exists(mask_dir) and len(os.listdir(mask_dir)) > 0
    if use_masks:
        print(f"Found mask directory: {mask_dir}")
        # For NeRF-W or masked training, we'll also create a masks array
        masks_array = np.zeros((n_images, h_target, w_target, 1), dtype=np.float32)
    else:
        print(f"No masks found in {mask_dir}. Creating dataset without masks.")
    
    # Process each image and its corresponding camera parameters
    for idx, image_path in enumerate(image_paths):
        print(f"Processing image {idx+1}/{n_images}: {os.path.basename(image_path)}")
        
        # Find the corresponding image in the COLMAP reconstruction
        image_name = os.path.basename(image_path)
        colmap_image = None
        img_id_val = None
        for img_id in colmap_images:
            if colmap_images[img_id].name == image_name:
                colmap_image = colmap_images[img_id]
                img_id_val = img_id
                break
        
        if colmap_image is None:
            print(f"Warning: Image {image_name} not found in COLMAP reconstruction. Skipping.")
            continue
        
        # Get camera parameters
        camera = reconstruction.cameras[colmap_image.camera_id]
        
        # Load the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to read image {image_path}. Skipping.")
            continue
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        
        # Resize if needed
        if image_scale != 1.0:
            img = cv2.resize(img, (w_target, h_target))
        
        # Store normalized image
        img_normalized = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
        images_array[idx] = img_normalized
        
        # Get camera pose (world-to-camera transformation)
        # For the current PyColmap version (3.11.1), we need to access the quaternion and convert manually
        
        # First, let's try to debug by printing available attributes
        image_attrs = dir(colmap_image)
        print(f"Available image attributes: {image_attrs}")
        
        # Access the quaternion (qvec) from the image
        # In current PyColmap, this is often represented as a property 'R'
        # Let's try various methods to get the rotation matrix
        
        if 'R' in image_attrs:
            # Directly access the rotation matrix if available
            rotation = np.array(colmap_image.R)
        elif 'rotmat' in image_attrs and callable(getattr(colmap_image, 'rotmat')):
            # Try the rotmat() method if available
            rotation = colmap_image.rotmat()
        elif 'qvec' in image_attrs:
            # If qvec is available but no conversion method
            qvec = colmap_image.qvec
            
            # Manual quaternion to rotation matrix conversion
            w, x, y, z = qvec
            rotation = np.array([
                [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
                [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
                [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
            ])
        else:
            # If all else fails, create a default identity matrix
            print(f"WARNING: Could not find rotation information for image {image_name}.")
            print(f"Using identity matrix as fallback. Results may be incorrect.")
            rotation = np.eye(3)
        
        # Get translation vector
        if hasattr(colmap_image, 'tvec'):
            translation = np.array(colmap_image.tvec).reshape(3, 1)
        elif hasattr(colmap_image, 't'):
            translation = np.array(colmap_image.t).reshape(3, 1)
        else:
            print(f"WARNING: Could not find translation vector for image {image_name}")
            translation = np.zeros((3, 1))  # Default to origin as fallback
        
        # Convert to camera-to-world transformation
        w2c = np.identity(4)
        w2c[:3, :3] = rotation
        w2c[:3, 3] = translation.flatten()
        c2w = np.linalg.inv(w2c)
        
        # Store the camera pose
        poses_array[idx] = c2w
    
    # Get focal length from the first camera
    camera_id = list(reconstruction.cameras.keys())[0]
    camera = reconstruction.cameras[camera_id]
    
    # Get focal length based on camera model
    fx = camera.params[0]  # First parameter is typically focal length
    if image_scale != 1.0:
        fx *= image_scale
    focal = fx
    
    # Save the dataset
    print(f"Saving dataset to {output_path}")
    
    if use_masks:
        np.savez(output_path, 
                images=images_array, 
                poses=poses_array, 
                focal=focal,
                masks=masks_array)
        
        print("Dataset creation complete!")
        print(f"Dataset info:")
        print(f"  - Number of images: {n_images}")
        print(f"  - Image dimensions: {h_target}x{w_target}")
        print(f"  - Focal length: {focal}")
        print(f"  - Masks included: Yes")
    else:
        np.savez(output_path, 
                images=images_array, 
                poses=poses_array, 
                focal=focal)
        
        print("Dataset creation complete!")
        print(f"Dataset info:")
        print(f"  - Number of images: {n_images}")
        print(f"  - Image dimensions: {h_target}x{w_target}")
        print(f"  - Focal length: {focal}")
        print(f"  - Masks included: No")

def main():
    parser = argparse.ArgumentParser(description='Create NeRF dataset from images using COLMAP')
    parser.add_argument('--image_dir', required=True, help='Directory containing input images')
    parser.add_argument('--mask_dir', required=True, help='Directory containing segmentation masks')
    parser.add_argument('--output_dir', required=True, help='Directory to save outputs')
    parser.add_argument('--image_scale', type=float, default=1.0, help='Scale factor for images')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Paths for COLMAP files
    db_path = os.path.join(args.output_dir, 'colmap.db')
    colmap_output_dir = os.path.join(args.output_dir, 'colmap_output')
    
    # Run COLMAP to get camera parameters
    reconstruction = run_colmap(args.image_dir, db_path, colmap_output_dir)
    
    # Create NeRF dataset
    npz_path = os.path.join(args.output_dir, 'nerf_dataset.npz')
    create_nerf_dataset(reconstruction, args.image_dir, args.mask_dir, npz_path, args.image_scale)

if __name__ == "__main__":
    main()