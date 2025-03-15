import os
import subprocess
import argparse
import time
import datetime
import re
from shutil import copy2, move

def rename_image_folder_if_needed(image_path):
    # Rename the image_path folder to "source" if it's named "input" or "images"
    parent_dir = os.path.abspath(os.path.join(image_path, os.pardir))
    current_folder_name = os.path.basename(os.path.normpath(image_path))
    
    if current_folder_name in ["input", "images"]:
        new_image_path = os.path.join(parent_dir, "source")
        os.rename(image_path, new_image_path)
        print(f"Renamed image folder from {current_folder_name} to: {new_image_path}")
        return new_image_path
    return image_path

def filter_images(image_path, interval):
    parent_dir = os.path.abspath(os.path.join(image_path, os.pardir))
    input_folder = os.path.join(parent_dir, 'input')

    if interval > 1:
        if not os.path.exists(input_folder):
            os.makedirs(input_folder)

        image_files = sorted([f for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))])
        filtered_files = image_files[::interval]

        for file in filtered_files:
            copy2(os.path.join(image_path, file), os.path.join(input_folder, file))

        return input_folder
    return image_path

def run_colmap(image_path, matcher_type, interval, model_type):
    # Rename the image_path folder if needed
    image_path = rename_image_folder_if_needed(image_path)

    parent_dir = os.path.abspath(os.path.join(image_path, os.pardir))
    image_path = filter_images(image_path, interval)

    # Count total input images
    total_input_images = len([f for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))])
    print(f"Total input images: {total_input_images}")

    distorted_folder = os.path.join(parent_dir, 'distorted')
    database_path = os.path.join(distorted_folder, 'database.db')
    sparse_folder = os.path.join(parent_dir, 'sparse')  # Top-level sparse folder
    sparse_zero_folder = os.path.join(sparse_folder, '0')  # The new subfolder we want to create

    os.makedirs(distorted_folder, exist_ok=True)
    os.makedirs(sparse_folder, exist_ok=True)

    log_file_path = os.path.join(parent_dir, "colmap_run.log")
    total_start_time = time.time()

    # Enhanced parameters for feature extraction and matching
    commands = [
        # Improved feature extraction with more features and better detection parameters
        f"colmap feature_extractor --image_path {image_path} --database_path {database_path} "
        f"--ImageReader.single_camera 1 --ImageReader.camera_model PINHOLE "
        f"--SiftExtraction.use_gpu 1 "
        f"--SiftExtraction.max_num_features 8000 "  # Increased from default
        f"--SiftExtraction.peak_threshold 0.004 "   # Lower threshold to detect more features
        f"--SiftExtraction.edge_threshold 10 "      # Less strict edge filtering
        f"--SiftExtraction.first_octave -1",        # Start at a smaller scale for more features

        # Matcher with optimized parameters
        f"colmap {matcher_type} --database_path {database_path} "
        f"--SiftMatching.use_gpu 1 "
        f"--SiftMatching.max_ratio 0.9 "            # Increase ratio test threshold
        f"--SiftMatching.max_distance 0.8 "         # More permissive distance threshold 
        f"--SiftMatching.cross_check 1",            # Enable cross-checking for better matches

        # Mapper with only TrackEstablishment parameter which should be compatible
        f"glomap mapper --database_path {database_path} --image_path {image_path} "
        f"--output_path {os.path.join(distorted_folder, 'sparse')} "
        f"--TrackEstablishment.max_num_tracks 10000"  # Increased from 5000
    ]

    with open(log_file_path, "w") as log_file:
        log_file.write(f"COLMAP run started at: {datetime.datetime.now()}\n")
        log_file.write(f"Total input images: {total_input_images}\n")

        for command in commands:
            command_start_time = time.time()
            log_file.write(f"Running command: {command}\n")
            subprocess.run(command, shell=True, check=True)
            command_end_time = time.time()
            command_elapsed_time = command_end_time - command_start_time
            log_file.write(f"Time taken for command: {command_elapsed_time:.2f} seconds\n")
            print(f"Time taken for command: {command_elapsed_time:.2f} seconds")

        if model_type == '3dgs':
            img_undist_cmd = (
                f"colmap image_undistorter "
                f"--image_path {image_path} "
                f"--input_path {os.path.join(distorted_folder, 'sparse/0')} "  # Use the sparse/0 in distorted
                f"--output_path {parent_dir} "  # Output undistorted results to the top-level folder
                f"--output_type COLMAP"
            )
            log_file.write(f"Running command: {img_undist_cmd}\n")
            undistort_start_time = time.time()
            exit_code = os.system(img_undist_cmd)
            undistort_end_time = time.time()
            undistort_elapsed_time = undistort_end_time - undistort_start_time

            if exit_code != 0:
                log_file.write(f"Undistortion failed with code {exit_code}. Exiting.\n")
                print(f"Undistortion failed with code {exit_code}. Exiting.")
                exit(exit_code)
            else:
                log_file.write(f"Time taken for undistortion: {undistort_elapsed_time:.2f} seconds\n")
                print(f"Time taken for undistortion: {undistort_elapsed_time:.2f} seconds")

        # Move the cameras.bin, images.bin, and points3D.bin files to sparse/0 in the top-level folder
        os.makedirs(sparse_zero_folder, exist_ok=True)
        for file_name in ['cameras.bin', 'images.bin', 'points3D.bin']:
            source_file = os.path.join(sparse_folder, file_name)
            dest_file = os.path.join(sparse_zero_folder, file_name)
            if os.path.exists(source_file):
                move(source_file, dest_file)
                log_file.write(f"Moved {file_name} to {sparse_zero_folder}\n")
                print(f"Moved {file_name} to {sparse_zero_folder}")

        # Print registered frames count
        registered_frames = get_registered_frames_count(sparse_zero_folder)
        log_file.write(f"Number of registered frames: {registered_frames} out of {total_input_images} input images\n")
        print(f"Number of registered frames: {registered_frames} out of {total_input_images} input images")
        
        if registered_frames > 0:
            registration_percentage = (registered_frames / total_input_images) * 100
            log_file.write(f"Registration success rate: {registration_percentage:.2f}%\n")
            print(f"Registration success rate: {registration_percentage:.2f}%")

        total_end_time = time.time()
        total_elapsed_time = total_end_time - total_start_time
        log_file.write(f"COLMAP run finished at: {datetime.datetime.now()}\n")
        log_file.write(f"Total time taken: {total_elapsed_time:.2f} seconds\n")
        print(f"Total time taken: {total_elapsed_time:.2f} seconds")

def get_registered_frames_count(sparse_zero_folder):
    """Count the number of registered frames/cameras from the images.bin file"""
    if not os.path.exists(sparse_zero_folder):
        return 0
        
    # Method 1: Use colmap model_info
    try:
        cmd = f"colmap model_info --path {sparse_zero_folder}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        output = result.stdout
        
        # Extract the number of registered images from the output
        match = re.search(r"Registered images:\s+(\d+)", output)
        if match:
            return int(match.group(1))
    except Exception as e:
        print(f"Error getting model info: {e}")
    
    # Method 2: Alternative approach if model_info fails
    try:
        images_file = os.path.join(sparse_zero_folder, 'images.bin')
        if os.path.exists(images_file):
            # Use colmap model_converter to convert binary to text
            text_path = os.path.join(os.path.dirname(sparse_zero_folder), 'temp_text')
            os.makedirs(text_path, exist_ok=True)
            convert_cmd = f"colmap model_converter --input_path {sparse_zero_folder} --output_path {text_path} --output_type TXT"
            subprocess.run(convert_cmd, shell=True, check=True)
            
            # Count lines in images.txt (excluding comments)
            images_txt = os.path.join(text_path, 'images.txt')
            if os.path.exists(images_txt):
                with open(images_txt, 'r') as f:
                    lines = [line for line in f if not line.startswith('#')]
                    # Each image takes 2 lines in the file
                    return len(lines) // 2
    except:
        pass
    
    # Method 3: Last resort - just check if the file exists
    images_file = os.path.join(sparse_zero_folder, 'images.bin')
    if os.path.exists(images_file):
        return 1  # We know at least 1 image was registered if the file exists
    
    return 0

def analyze_results(parent_dir):
    """Analyze the reconstruction results and print statistics"""
    sparse_zero_folder = os.path.join(parent_dir, 'sparse', '0')
    
    if not os.path.exists(sparse_zero_folder):
        print("Reconstruction folder not found. Process may have failed.")
        return
    
    # Get and print registered frames count
    registered_frames = get_registered_frames_count(sparse_zero_folder)
    print(f"\nAnalysis Results:")
    print(f"Number of registered frames/cameras: {registered_frames}")
    
    # Get more detailed information if possible
    try:
        cmd = f"colmap model_info --path {sparse_zero_folder}"
        print("\nDetailed model information:")
        subprocess.run(cmd, shell=True)
    except:
        print("Could not get detailed model information.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run COLMAP with optimized parameters to maximize registered frames.")
    parser.add_argument('--image_path', required=True, help="Path to the images folder.")
    parser.add_argument('--matcher_type', default='exhaustive_matcher', 
                        choices=['sequential_matcher', 'exhaustive_matcher'], 
                        help="Type of matcher to use (default: exhaustive_matcher for better results).")
    parser.add_argument('--interval', type=int, default=1, 
                        help="Interval of images to use (default: 1, meaning all images).")
    parser.add_argument('--model_type', default='3dgs', 
                        choices=['3dgs', 'nerfstudio'], 
                        help="Model type to run. '3dgs' includes undistortion, 'nerfstudio' skips undistortion.")
    parser.add_argument('--analyze', action='store_true',
                        help="Analyze reconstruction results after completion.")

    args = parser.parse_args()

    parent_dir = os.path.abspath(os.path.join(args.image_path, os.pardir))
    run_colmap(args.image_path, args.matcher_type, args.interval, args.model_type)
    
    if args.analyze:
        analyze_results(parent_dir)
    