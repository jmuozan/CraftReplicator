import os
import subprocess
import argparse
import json
import shutil
import glob
import sys
import numpy as np
import math
import cv2
from pathlib import Path, PurePosixPath

def load_scenes_from_json(segmentation_dir):
    """Load scene definitions from the scenes.json file created by 2_segmenter.py"""
    scenes_file = os.path.join(segmentation_dir, "scenes.json")
    
    if not os.path.exists(scenes_file):
        print(f"Error: Scenes file not found at {scenes_file}")
        print("Make sure you've run 2_segmenter.py first to define scenes.")
        return None, None
    
    try:
        with open(scenes_file, 'r') as f:
            scene_data = json.load(f)
            
        scenes = scene_data.get('scenes', [])
        annotations = scene_data.get('annotations', {})
        
        if not scenes:
            print("No scenes defined in the scenes.json file.")
            return None, None
            
        print(f"Loaded {len(scenes)} scenes from {scenes_file}")
        return scenes, annotations
    except Exception as e:
        print(f"Error loading scenes file: {e}")
        return None, None

def organize_images_by_scene(input_dir, scenes, output_base_dir):
    """Organize images into scene folders based on scene boundaries"""
    # Get all image files from the input directory
    image_files = []
    for ext in ['.png', '.jpg', '.jpeg']:
        image_files.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return None
    
    # Sort the image files by frame number
    # Expecting filenames to be like "extracted_0001.png" or have a number in them
    image_files.sort(key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)))))
    
    # Create scene directories
    scene_dirs = []
    for i in range(len(scenes)):
        scene_dir = os.path.join(output_base_dir, f"scene_{i}")
        os.makedirs(scene_dir, exist_ok=True)
        scene_dirs.append(scene_dir)
    
    # Determine scene for each image and copy to appropriate directory
    frames_processed = 0
    for image_path in image_files:
        # Extract the frame number from filename
        frame_num = int(''.join(filter(str.isdigit, os.path.basename(image_path))))
        
        # Find which scene this frame belongs to
        scene_idx = None
        for i in range(len(scenes)-1):
            if scenes[i] <= frame_num < scenes[i+1]:
                scene_idx = i
                break
        
        # If frame is after the last scene start, it belongs to the last scene
        if scene_idx is None and frame_num >= scenes[-1]:
            scene_idx = len(scenes) - 1
            
        if scene_idx is not None:
            # Copy the image to the appropriate scene directory
            dest_path = os.path.join(scene_dirs[scene_idx], os.path.basename(image_path))
            shutil.copy2(image_path, dest_path)
            frames_processed += 1
    
    print(f"Successfully organized {frames_processed} frames into {len(scene_dirs)} scenes")
    
    # Verify that scenes have images
    valid_scenes = []
    for i, scene_dir in enumerate(scene_dirs):
        num_images = len(os.listdir(scene_dir))
        if num_images > 0:
            print(f"Scene {i}: {num_images} images")
            valid_scenes.append(scene_dir)
        else:
            print(f"Scene {i}: No images, skipping")
    
    return valid_scenes

def run_glomap_on_scene(scene_path, output_base_dir):
    """Run GLOMAP on a single scene folder"""
    scene_name = os.path.basename(scene_path)
    output_dir = os.path.join(output_base_dir, scene_name)
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Database path for this scene
    db_path = os.path.join(output_dir, "database.db")
    
    # Create feature database
    print(f"\n--- Processing scene: {scene_name} ---")
    print("Extracting features...")
    cmd_feature = (
        f"colmap feature_extractor "
        f"--image_path {scene_path} "
        f"--database_path {db_path} "
        f"--ImageReader.single_camera_per_folder 1 "
        f"--SiftExtraction.max_num_features 10000 "
        f"--SiftExtraction.estimate_affine_shape true "
        f"--SiftExtraction.domain_size_pooling true"
    )
    subprocess.run(cmd_feature, shell=True, check=True)
    
    # Match features with exhaustive matcher and guided matching
    print("Matching features using exhaustive matcher...")
    cmd_match = (
        f"colmap exhaustive_matcher "
        f"--database_path {db_path} "
        f"--SiftMatching.guided_matching true"
    )
    subprocess.run(cmd_match, shell=True, check=True)
    
    # Run GLOMAP mapper with increased epipolar error for challenging scenes
    print("Running GLOMAP mapper...")
    sparse_dir = os.path.join(output_dir, "sparse")
    cmd_mapper = (
        f"glomap mapper "
        f"--database_path {db_path} "
        f"--image_path {scene_path} "
        f"--output_path {sparse_dir} "
        f"--TrackEstablishment.max_num_tracks 100000 "
        f"--RelPoseEstimation.max_epipolar_error 6"
    )
    subprocess.run(cmd_mapper, shell=True, check=True)
    
    # Return the path to the sparse model (or None if it doesn't exist)
    sparse_model_path = os.path.join(sparse_dir, "0")
    if os.path.exists(sparse_model_path):
        return sparse_model_path
    else:
        print(f"Warning: No sparse model created for {scene_name}")
        return None

def merge_models(model_paths, output_path):
    """Merge multiple COLMAP models into one"""
    if len(model_paths) < 2:
        print("Need at least two models to merge!")
        return None
    
    os.makedirs(output_path, exist_ok=True)
    
    # Start with the first model
    first_model = model_paths[0]
    merged_model = os.path.join(output_path, "merged")
    os.makedirs(merged_model, exist_ok=True)
    
    # Copy the first model to the merged location
    for file in glob.glob(os.path.join(first_model, "*")):
        shutil.copy(file, merged_model)
    
    # Merge each subsequent model
    for i, model_path in enumerate(model_paths[1:], 1):
        print(f"Merging model {i} of {len(model_paths)-1}...")
        
        # Create a temporary location for this step's merge
        temp_merged = os.path.join(output_path, f"temp_merged_{i}")
        os.makedirs(temp_merged, exist_ok=True)
        
        # Merge the current merged model with the next model
        cmd_merge = (
            f"colmap model_merger "
            f"--input_path1 {merged_model} "
            f"--input_path2 {model_path} "
            f"--output_path {temp_merged}"
        )
        subprocess.run(cmd_merge, shell=True, check=True)
        
        # Replace the merged model with the new merge
        shutil.rmtree(merged_model)
        shutil.move(temp_merged, merged_model)
    
    # Run a final bundle adjustment on the merged model
    final_model = os.path.join(output_path, "final")
    os.makedirs(final_model, exist_ok=True)
    
    cmd_bundle = (
        f"colmap bundle_adjuster "
        f"--input_path {merged_model} "
        f"--output_path {final_model}"
    )
    subprocess.run(cmd_bundle, shell=True, check=True)
    
    return final_model

def convert_to_text(model_path):
    """Convert binary model to text format for transforms.json generation"""
    text_path = os.path.join(os.path.dirname(model_path), "text")
    os.makedirs(text_path, exist_ok=True)
    
    cmd_convert = (
        f"colmap model_converter "
        f"--input_path {model_path} "
        f"--output_path {text_path} "
        f"--output_type TXT"
    )
    subprocess.run(cmd_convert, shell=True, check=True)
    
    return text_path

def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def sharpness(imagePath):
    image = cv2.imread(imagePath)
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

def rotmat(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    # handle exception for the opposite direction input
    if c < -1 + 1e-10:
        return rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))

def closest_point_2_lines(oa, da, ob, db): # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
    da = da / np.linalg.norm(da)
    db = db / np.linalg.norm(db)
    c = np.cross(da, db)
    denom = np.linalg.norm(c)**2
    t = ob - oa
    ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
    tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
    if ta > 0:
        ta = 0
    if tb > 0:
        tb = 0
    return (oa+ta*da+ob+tb*db) * 0.5, denom

def generate_transforms_json(text_folder, image_folder, output_path, aabb_scale=32, skip_early=0, keep_colmap_coords=False):
    """Generate transforms.json file similar to colmap2nerf.py"""
    print(f"Generating transforms.json file from {text_folder}...")
    
    # Read cameras
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
            elif els[1] == "SIMPLE_RADIAL_FISHEYE":
                camera["is_fisheye"] = True
                camera["cx"] = float(els[5])
                camera["cy"] = float(els[6])
                camera["k1"] = float(els[7])
            elif els[1] == "RADIAL_FISHEYE":
                camera["is_fisheye"] = True
                camera["cx"] = float(els[5])
                camera["cy"] = float(els[6])
                camera["k1"] = float(els[7])
                camera["k2"] = float(els[8])
            elif els[1] == "OPENCV_FISHEYE":
                camera["is_fisheye"] = True
                camera["fl_y"] = float(els[5])
                camera["cx"] = float(els[6])
                camera["cy"] = float(els[7])
                camera["k1"] = float(els[8])
                camera["k2"] = float(els[9])
                camera["k3"] = float(els[10])
                camera["k4"] = float(els[11])
            else:
                print("Unknown camera model ", els[1])
                
            camera["camera_angle_x"] = math.atan(camera["w"] / (camera["fl_x"] * 2)) * 2
            camera["camera_angle_y"] = math.atan(camera["h"] / (camera["fl_y"] * 2)) * 2
            camera["fovx"] = camera["camera_angle_x"] * 180 / math.pi
            camera["fovy"] = camera["camera_angle_y"] * 180 / math.pi

            print(f"camera {camera_id}:\n\tres={camera['w'],camera['h']}\n\tcenter={camera['cx'],camera['cy']}\n\tfocal={camera['fl_x'],camera['fl_y']}\n\tfov={camera['fovx'],camera['fovy']}\n\tk={camera['k1'],camera['k2']} p={camera['p1'],camera['p2']} ")
            cameras[camera_id] = camera
    
    if len(cameras) == 0:
        print("No cameras found!")
        return False
    
    # Read images and generate transforms
    with open(os.path.join(text_folder,"images.txt"), "r") as f:
        i = 0
        bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
        
        if len(cameras) == 1:
            camera = cameras[list(cameras.keys())[0]]
            out = {
                "camera_angle_x": camera["camera_angle_x"],
                "camera_angle_y": camera["camera_angle_y"],
                "fl_x": camera["fl_x"],
                "fl_y": camera["fl_y"],
                "k1": camera["k1"],
                "k2": camera["k2"],
                "k3": camera["k3"],
                "k4": camera["k4"],
                "p1": camera["p1"],
                "p2": camera["p2"],
                "is_fisheye": camera["is_fisheye"],
                "cx": camera["cx"],
                "cy": camera["cy"],
                "w": camera["w"],
                "h": camera["h"],
                "aabb_scale": aabb_scale,
                "frames": [],
            }
        else:
            out = {
                "frames": [],
                "aabb_scale": aabb_scale
            }

        up = np.zeros(3)
        frame_count = 0
        
        for line in f:
            line = line.strip()
            if line[0] == "#":
                continue
            i = i + 1
            if i < skip_early*2:
                continue
                
            if i % 2 == 1:
                elems = line.split(" ") # 1-4 is quat, 5-7 is trans, 9ff is filename (9, if filename contains no spaces)
                
                # Find the path to the corresponding image
                image_name = '_'.join(elems[9:])
                
                # Locate the image in the image folder
                image_paths = glob.glob(os.path.join(image_folder, "**", image_name), recursive=True)
                
                if not image_paths:
                    print(f"Warning: Image {image_name} not found in {image_folder}")
                    continue
                    
                # Use the first matching image path
                name = image_paths[0]
                # Make path relative to output directory (for NeRF compatibility)
                name = os.path.relpath(name, os.path.dirname(output_path))
                
                b = sharpness(image_paths[0])
                print(name, "sharpness=", b)
                
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                R = qvec2rotmat(-qvec)
                t = tvec.reshape([3,1])
                m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
                c2w = np.linalg.inv(m)
                
                if not keep_colmap_coords:
                    c2w[0:3,2] *= -1 # flip the y and z axis
                    c2w[0:3,1] *= -1
                    c2w = c2w[[1,0,2,3],:]
                    c2w[2,:] *= -1 # flip whole world upside down
                    up += c2w[0:3,1]

                frame = {"file_path": name, "sharpness": b, "transform_matrix": c2w}
                if len(cameras) != 1:
                    frame.update(cameras[int(elems[8])])
                    
                out["frames"].append(frame)
                frame_count += 1
    
    print(f"Found {frame_count} frames")
    
    if keep_colmap_coords:
        flip_mat = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])

        for f in out["frames"]:
            f["transform_matrix"] = np.matmul(f["transform_matrix"], flip_mat) # flip cameras
    else:
        # Reorient scene
        up = up / np.linalg.norm(up)
        print("up vector was", up)
        R = rotmat(up,[0,0,1]) # rotate up vector to [0,0,1]
        R = np.pad(R,[0,1])
        R[-1, -1] = 1

        for f in out["frames"]:
            f["transform_matrix"] = np.matmul(R, f["transform_matrix"]) # rotate up to be the z axis

        # Find a central point they are all looking at
        print("computing center of attention...")
        totw = 0.0
        totp = np.array([0.0, 0.0, 0.0])
        for f in out["frames"]:
            mf = f["transform_matrix"][0:3,:]
            for g in out["frames"]:
                mg = g["transform_matrix"][0:3,:]
                p, w = closest_point_2_lines(mf[:,3], mf[:,2], mg[:,3], mg[:,2])
                if w > 0.00001:
                    totp += p*w
                    totw += w
        if totw > 0.0:
            totp /= totw
        print(totp) # the cameras are looking at totp
        
        for f in out["frames"]:
            f["transform_matrix"][0:3,3] -= totp

        avglen = 0.
        for f in out["frames"]:
            avglen += np.linalg.norm(f["transform_matrix"][0:3,3])
        avglen /= frame_count
        print("avg camera distance from origin", avglen)
        for f in out["frames"]:
            f["transform_matrix"][0:3,3] *= 4.0 / avglen # scale to "nerf sized"

    # Convert numpy arrays to lists for JSON serialization
    for f in out["frames"]:
        f["transform_matrix"] = f["transform_matrix"].tolist()
    
    print(f"{frame_count} frames extracted and processed successfully")
    print(f"Writing transforms.json to {output_path}")
    
    with open(output_path, "w") as outfile:
        json.dump(out, outfile, indent=2)
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Run GLOMAP reconstruction using existing scene definitions and generate transforms.json")
    parser.add_argument("--input_dir", default="db", help="Directory containing the input images (default: db)")
    parser.add_argument("--segmentation_dir", default="vessel/segmentation_output", help="Directory containing the segmentation_output folder with scenes.json")
    parser.add_argument("--output_dir", default="", help="Output directory for the reconstructions (default: current directory)")
    parser.add_argument("--aabb_scale", default=32, type=int, help="NeRF scene scale (default: 32)")
    
    args = parser.parse_args()
    
    # Set default output directory to current directory if not specified
    if not args.output_dir:
        args.output_dir = os.getcwd()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load scene definitions
    scenes, annotations = load_scenes_from_json(args.segmentation_dir)
    if scenes is None:
        return
    
    # Create a directory for organizing the images by scene
    scene_images_dir = os.path.join(args.output_dir, "scene_images")
    os.makedirs(scene_images_dir, exist_ok=True)
    
    # Organize images by scene
    print("Organizing images by scene...")
    scene_dirs = organize_images_by_scene(args.input_dir, scenes, scene_images_dir)
    if not scene_dirs:
        print("No valid scenes with images found.")
        return
    
    # Process each scene
    print(f"Processing {len(scene_dirs)} scenes...")
    sparse_models = []
    
    for scene_dir in scene_dirs:
        scene_name = os.path.basename(scene_dir)
        try:
            sparse_model = run_glomap_on_scene(scene_dir, os.path.join(args.output_dir, "scene_reconstructions"))
            if sparse_model and os.path.exists(sparse_model):
                sparse_models.append(sparse_model)
                print(f"Successfully reconstructed scene: {scene_name}")
            else:
                print(f"Failed to reconstruct scene: {scene_name}")
        except Exception as e:
            print(f"Error processing scene {scene_name}: {e}")
    
    if len(sparse_models) < 1:
        print("No scenes were successfully reconstructed.")
        return
    
    final_model_dir = None
    
    # If only one scene was reconstructed, use it as the final model
    if len(sparse_models) == 1:
        print("Only one scene was successfully reconstructed, no need to merge.")
        final_model_dir = sparse_models[0]
    else:
        # Merge the models
        print("\n--- Merging Models ---")
        try:
            merged_model = merge_models(sparse_models, os.path.join(args.output_dir, "merged"))
            if merged_model and os.path.exists(merged_model):
                final_model_dir = merged_model
                print(f"Final merged model is at: {merged_model}")
            else:
                print("Model merging failed.")
                return
        except Exception as e:
            print(f"Error during model merging: {e}")
            return
    
    # Convert to text format
    text_dir = convert_to_text(final_model_dir)
    
    # Generate transforms.json
    transforms_path = os.path.join(args.output_dir, "transforms.json")
    if generate_transforms_json(text_dir, args.input_dir, transforms_path, aabb_scale=args.aabb_scale):
        print("\n--- Success! ---")
        print(f"Transforms JSON file generated at: {transforms_path}")
        print(f"Total number of cameras: {len(glob.glob(os.path.join(text_dir, 'images.txt')))//2}")
        print(f"You can use this transforms.json file with NeRF frameworks")
    else:
        print("Failed to generate transforms.json")

if __name__ == "__main__":
    main()