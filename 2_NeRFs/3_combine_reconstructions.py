import os
import subprocess
import json
import numpy as np
import cv2
import random
import shutil
from pathlib import Path

def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def sharpness(imagePath):
    image = cv2.imread(imagePath)
    if image is None:  # Check if image is loaded properly
        print(f"Warning: Could not load image {imagePath}")
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    return fm

def qvec2rotmat(qvec):
    return np.array([
        [
            1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
            2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
            2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]
        ], [
            2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
            1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
            2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]
        ], [
            2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
            2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
            1 - 2 * qvec[1]**2 - 2 * qvec[2]**2
        ]
    ])

def run_command(cmd):
    """Run a command and print its output"""
    print(f"Running: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    
    if stdout.strip():
        print("STDOUT:")
        print(stdout)
    
    if stderr.strip():
        print("STDERR:")
        print(stderr)
    
    return process.returncode

def combine_colmap_reconstructions(scenes_base_dir="glomap", out_path="transforms.json", aabb_scale=32, keep_colmap_coords=False, test_percentage=0.15):
    """
    Process and combine multiple COLMAP reconstructions.
    
    Args:
        scenes_base_dir: Base directory containing scene folders with reconstructions
        out_path: Output path for the combined transforms.json
        aabb_scale: The scene scale factor
        keep_colmap_coords: Whether to keep COLMAP's original coordinate system
        test_percentage: Percentage of images to use for testing
    """
    all_frames = []
    camera_info = None
    
    # Create output directory for final training/testing images
    output_dir = os.path.join("db", "final")
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all scene directories
    scene_dirs = [d for d in os.listdir(scenes_base_dir) 
                 if os.path.isdir(os.path.join(scenes_base_dir, d))]
    
    # Keep track of all images to split later
    all_images = []
    
    for scene_dir in scene_dirs:
        scene_path = os.path.join(scenes_base_dir, scene_dir)
        reconstruction_path = os.path.join(scene_path, "reconstruction", "0")
        
        # Check if this is a valid reconstruction folder
        if not os.path.exists(reconstruction_path):
            print(f"Warning: No reconstruction found in {reconstruction_path}, skipping...")
            continue
            
        print(f"Processing scene: {scene_dir}")
        
        # Create a text folder for this reconstruction
        text_folder = os.path.join(scene_path, "text")
        os.makedirs(text_folder, exist_ok=True)
        
        # Convert binary model to text format
        print(f"Converting binary model to text format...")
        run_command([
            'colmap', 'model_converter',
            '--input_path', reconstruction_path,
            '--output_path', text_folder,
            '--output_type', 'TXT'
        ])
        
        # Read cameras from text files
        cameras = {}
        with open(os.path.join(text_folder, "cameras.txt"), "r") as f:
            for line in f:
                if line[0] == "#":
                    continue
                els = line.split(" ")
                camera = {}
                camera_id = int(els[0])
                camera["w"] = float(els[2])
                camera["h"] = float(els[3])
                camera["fl_x"] = float(els[4])
                camera["fl_y"] = float(els[4])
                camera["k1"] = 0
                camera["k2"] = 0
                camera["k3"] = 0
                camera["k4"] = 0
                camera["p1"] = 0
                camera["p2"] = 0
                camera["cx"] = camera["w"] / 2
                camera["cy"] = camera["h"] / 2
                camera["is_fisheye"] = False
                
                if els[1] == "SIMPLE_PINHOLE":
                    camera["cx"] = float(els[5])
                    camera["cy"] = float(els[6])
                elif els[1] == "PINHOLE":
                    camera["fl_y"] = float(els[5])
                    camera["cx"] = float(els[6])
                    camera["cy"] = float(els[7])
                elif els[1] == "SIMPLE_RADIAL":
                    camera["cx"] = float(els[5])
                    camera["cy"] = float(els[6])
                    camera["k1"] = float(els[7])
                elif els[1] == "RADIAL":
                    camera["cx"] = float(els[5])
                    camera["cy"] = float(els[6])
                    camera["k1"] = float(els[7])
                    camera["k2"] = float(els[8])
                elif els[1] == "OPENCV":
                    camera["fl_y"] = float(els[5])
                    camera["cx"] = float(els[6])
                    camera["cy"] = float(els[7])
                    camera["k1"] = float(els[8])
                    camera["k2"] = float(els[9])
                    camera["p1"] = float(els[10])
                    camera["p2"] = float(els[11])
                
                # Calculate camera angles
                camera["camera_angle_x"] = np.arctan(camera["w"] / (camera["fl_x"] * 2)) * 2
                camera["camera_angle_y"] = np.arctan(camera["h"] / (camera["fl_y"] * 2)) * 2
                camera["fovx"] = camera["camera_angle_x"] * 180 / np.pi
                camera["fovy"] = camera["camera_angle_y"] * 180 / np.pi
                
                cameras[camera_id] = camera
                
                # Store camera info for the output JSON if not already set
                if camera_info is None:
                    camera_info = camera.copy()
        
        # Read images and their transformations
        with open(os.path.join(text_folder, "images.txt"), "r") as f:
            i = 0
            bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
            
            for line in f:
                line = line.strip()
                if line[0] == "#":
                    continue
                i = i + 1
                
                # Images are in pairs of lines in the file
                if i % 2 == 1:
                    elems = line.split(" ")
                    image_id = int(elems[0])
                    
                    # Extract image name
                    image_name = '_'.join(elems[9:])
                    
                    # Try to determine the path to the actual image
                    # Check in masked folder first, then regular locations
                    possible_locations = [
                        os.path.join("db", "masked", os.path.basename(image_name).replace('.jpg', '.png')),
                        os.path.join("db", "masked", image_name.replace('.jpg', '.png')),
                        os.path.join("db", image_name),
                        os.path.join("db", f"{scene_dir}_{image_name}"),
                        os.path.join("db", scene_dir, image_name),
                        # Add more possible patterns here
                    ]
                    
                    name = None
                    for loc in possible_locations:
                        if os.path.exists(loc):
                            name = loc
                            break
                    
                    if name is None:
                        print(f"Warning: Could not find image {image_name} in any of the expected locations")
                        
                        # Get a list of available images in the db directory
                        db_images = []
                        for root, dirs, files in os.walk("db"):
                            for file in files:
                                if file.endswith(('.jpg', '.png', '.jpeg')):
                                    db_images.append(os.path.join(root, file))
                        
                        # Try to find a matching image by substring
                        for db_image in db_images:
                            if os.path.basename(image_name).replace('.jpg', '') in db_image:
                                name = db_image
                                print(f"  Found matching image: {name}")
                                break
                        
                        if name is None:
                            print(f"  No matching image found, skipping...")
                            continue
                    
                    # Calculate image sharpness
                    b = sharpness(name)
                    print(f"{name}, sharpness={b}")
                    
                    # Get camera pose
                    qvec = np.array(tuple(map(float, elems[1:5])))
                    tvec = np.array(tuple(map(float, elems[5:8])))
                    R = qvec2rotmat(-qvec)
                    t = tvec.reshape([3, 1])
                    m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
                    c2w = np.linalg.inv(m)
                    
                    # Optional coordinate transformation
                    if not keep_colmap_coords:
                        c2w[0:3, 2] *= -1  # flip the y and z axis
                        c2w[0:3, 1] *= -1
                        c2w = c2w[[1, 0, 2, 3], :]
                        c2w[2, :] *= -1  # flip whole world upside down
                    
                    # Create the frame entry (without file_path for now)
                    frame = {
                        "original_path": name,  # Store original path temporarily
                        "sharpness": b,
                        "transform_matrix": c2w.tolist(),
                    }
                    
                    # Add camera parameters if we have multiple camera models
                    camera_id = int(elems[8])
                    if len(cameras) > 1:
                        frame.update(cameras[camera_id])
                    
                    all_frames.append(frame)
    
    if not all_frames:
        print("Error: No frames were generated from any scene.")
        return None
    
    # Now split images into train and test sets
    random.seed(42)  # For reproducibility
    num_test = max(1, int(len(all_frames) * test_percentage))
    test_indices = random.sample(range(len(all_frames)), num_test)
    test_indices_list = sorted(test_indices)  # Convert to sorted list for index lookup
    
    print(f"Splitting into {len(all_frames) - len(test_indices)} training and {len(test_indices)} testing images")
    
    # Rename and organize files according to train/test split
    train_count = 0
    test_count = 0
    
    for i, frame in enumerate(all_frames):
        original_path = frame.pop("original_path")  # Remove temporary field
        basename = os.path.basename(original_path)
        
        # Determine file extension (.png for masked images, keep original otherwise)
        file_ext = os.path.splitext(original_path)[1]
        if not file_ext:
            file_ext = ".png"  # Default to PNG if no extension
        
        if i in test_indices:
            # Test image
            dest_filename = f"test_{test_count:04d}{file_ext}"
            frame["file_path"] = f"db/final/{dest_filename}"
            dest_path = os.path.join(output_dir, dest_filename)
            test_count += 1
        else:
            # Train image
            dest_filename = f"train_{train_count:04d}{file_ext}"
            frame["file_path"] = f"db/final/{dest_filename}"
            dest_path = os.path.join(output_dir, dest_filename)
            train_count += 1
        
        # Copy the image to its final location
        try:
            shutil.copy2(original_path, dest_path)
            print(f"Copied {original_path} -> {dest_path}")
        except Exception as e:
            print(f"Error copying {original_path} to {dest_path}: {e}")
    
    # Create the output JSON structure
    if camera_info:
        out = {
            "camera_angle_x": camera_info["camera_angle_x"],
            "camera_angle_y": camera_info["camera_angle_y"],
            "fl_x": camera_info["fl_x"],
            "fl_y": camera_info["fl_y"],
            "k1": camera_info["k1"],
            "k2": camera_info["k2"],
            "k3": camera_info.get("k3", 0),
            "k4": camera_info.get("k4", 0),
            "p1": camera_info["p1"],
            "p2": camera_info["p2"],
            "is_fisheye": camera_info["is_fisheye"],
            "cx": camera_info["cx"],
            "cy": camera_info["cy"],
            "w": camera_info["w"],
            "h": camera_info["h"],
            "aabb_scale": aabb_scale,
            "frames": all_frames,
        }
    else:
        out = {
            "frames": all_frames,
            "aabb_scale": aabb_scale,
            "w": camera_info["w"],
            "h": camera_info["h"]
        }
    
    print(f"Generated {len(all_frames)} frames from {len(scene_dirs)} scenes")
    print(f"Writing output to {out_path}")
    
    with open(out_path, "w") as outfile:
        json.dump(out, outfile, indent=2)
    
    return out

if __name__ == "__main__":
    combine_colmap_reconstructions(
        scenes_base_dir="glomap",
        out_path="transforms.json",
        aabb_scale=32,
        keep_colmap_coords=False,
        test_percentage=0.15  # Set the percentage of images to use as test set
    )