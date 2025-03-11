import os
import random
import shutil
import argparse

def split_and_rename_images(input_folder, output_folder, test_percentage=20, file_extensions=None):
    """
    Renames all images in the input folder and separates them into test and train sets.
    
    Args:
        input_folder (str): Path to the folder containing images
        output_folder (str): Path to save the renamed images
        test_percentage (int): Percentage of images to be used as test set (0-100)
        file_extensions (list): List of file extensions to include (e.g., ['.png', '.jpg'])
    """
    # Default file extensions if none provided
    if file_extensions is None:
        file_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
    
    # Ensure file extensions start with a dot
    file_extensions = [ext if ext.startswith('.') else f'.{ext}' for ext in file_extensions]
    
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output directory: {output_folder}")
    
    # Get all image files
    image_files = []
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        if os.path.isfile(file_path):
            _, ext = os.path.splitext(filename)
            if ext.lower() in file_extensions:
                image_files.append(filename)
    
    # Check if any images were found
    if not image_files:
        print(f"No images with extensions {file_extensions} found in {input_folder}")
        return
    
    # Shuffle the file list for random selection
    random.shuffle(image_files)
    
    # Calculate split point
    test_count = int(len(image_files) * test_percentage / 100)
    
    # Split into test and train sets
    test_files = image_files[:test_count]
    train_files = image_files[test_count:]
    
    print(f"Total images: {len(image_files)}")
    print(f"Test set: {len(test_files)} images ({test_percentage}%)")
    print(f"Train set: {len(train_files)} images ({100-test_percentage}%)")
    
    # Rename and copy test files
    for i, filename in enumerate(test_files):
        src_path = os.path.join(input_folder, filename)
        _, ext = os.path.splitext(filename)
        dst_path = os.path.join(output_folder, f"test_{i}{ext}")
        shutil.copy2(src_path, dst_path)
        print(f"Created {dst_path}")
    
    # Rename and copy train files
    for i, filename in enumerate(train_files):
        src_path = os.path.join(input_folder, filename)
        _, ext = os.path.splitext(filename)
        dst_path = os.path.join(output_folder, f"train_{i}{ext}")
        shutil.copy2(src_path, dst_path)
        print(f"Created {dst_path}")
    
    print(f"Done! Renamed and split images saved to {output_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split and rename image files into test and train sets")
    parser.add_argument("--input", "-i", required=True, help="Input folder containing images")
    parser.add_argument("--output", "-o", required=True, help="Output folder for renamed images")
    parser.add_argument("--test", "-t", type=int, default=20, help="Percentage of images for test set (default: 20)")
    parser.add_argument("--extensions", "-e", nargs="+", help="File extensions to include (default: png jpg jpeg bmp tif tiff)")
    
    args = parser.parse_args()
    
    # Validate test percentage
    if args.test < 0 or args.test > 100:
        print("Error: Test percentage must be between 0 and 100")
        exit(1)
    
    split_and_rename_images(
        args.input, 
        args.output, 
        args.test, 
        args.extensions
    )