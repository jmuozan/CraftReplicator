import os
import random
import shutil
import argparse
import json

def split_and_rename_images_with_masks_and_transforms(input_folder, transforms_path=None, output_folder=None, 
                                                      masks_folder=None, output_masks_folder=None, 
                                                      test_percentage=20, file_extensions=None):
    """
    Renames images in the input folder, their masks, and updates the transforms.json file.
    
    Args:
        input_folder (str): Path to the folder containing original images
        transforms_path (str, optional): Path to the transforms.json file
        output_folder (str, optional): Path to save the renamed original images
        masks_folder (str, optional): Path to the folder containing mask images
        output_masks_folder (str, optional): Path to save the renamed mask images
        test_percentage (int): Percentage of images to be used as test set (0-100)
        file_extensions (list): List of file extensions to include
    """
    # Default output folder if not specified
    if output_folder is None:
        output_folder = os.path.join(input_folder, 'output')
    
    # Create output folders if they don't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output directory: {output_folder}")
    
    # If masks are provided, prepare masks output folder
    if masks_folder:
        # If no specific output masks folder is provided, create a 'masks' subfolder
        if output_masks_folder is None:
            output_masks_folder = os.path.join(output_folder, "masks")
        
        if not os.path.exists(output_masks_folder):
            os.makedirs(output_masks_folder)
            print(f"Created masks output directory: {output_masks_folder}")
    else:
        # Ensure output_masks_folder is None if no masks are provided
        output_masks_folder = None

    # Default file extensions if none provided
    if file_extensions is None:
        file_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
        
    # Ensure file extensions start with a dot
    file_extensions = [ext if ext.startswith('.') else f'.{ext}' for ext in file_extensions]
    
    # Get all image files
    image_files = []
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        if os.path.isfile(file_path):
            _, ext = os.path.splitext(filename)
            if ext.lower() in file_extensions:
                image_files.append(filename)
    
    # Filter images based on transforms.json if provided
    if transforms_path and os.path.exists(transforms_path):
        with open(transforms_path, 'r') as f:
            transforms_data = json.load(f)
        
        # Extract valid image filenames from transforms
        valid_images = [os.path.basename(frame['file_path']) for frame in transforms_data['frames']]
        image_files = [f for f in image_files if f in valid_images]
    
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
    
    # Create filename mapping
    filename_mapping = {}
    
    # Updated transforms data if transforms.json is provided
    updated_transforms_data = None
    if transforms_path and os.path.exists(transforms_path):
        updated_transforms_data = transforms_data.copy()
        updated_transforms_data['frames'] = []
    
    # Function to find the corresponding mask for an image (if masks are provided)
    def find_masks(image_filename):
        if not masks_folder:
            return []
        
        # Masks follow the format frame_XXXX_obj_Y.png
        # Extract the frame number from the original image filename
        basename, _ = os.path.splitext(image_filename)
        
        # Convert filenames like 102.jpg to frame numbers
        try:
            frame_number = int(basename)
            # Format with leading zeros (e.g., 0001, 0034)
            frame_prefix = f"frame_{frame_number:04d}"
            
            # Find all corresponding masks that start with this frame prefix
            matching_masks = []
            for mask_file in os.listdir(masks_folder):
                if mask_file.startswith(frame_prefix):
                    matching_masks.append(os.path.join(masks_folder, mask_file))
            
            return matching_masks
        except ValueError:
            # If filename isn't a number, try other pattern matching if needed
            return []
    
    # Rename and copy test files and their masks
    for i, filename in enumerate(test_files):
        # Handle original image
        src_path = os.path.join(input_folder, filename)
        _, ext = os.path.splitext(filename)
        dst_filename = f"test_{i}{ext}"
        dst_path = os.path.join(output_folder, dst_filename)
        shutil.copy2(src_path, dst_path)
        print(f"Created {dst_path}")
        
        # Store filename mapping
        filename_mapping[filename] = dst_filename
        
        # Handle corresponding masks
        mask_paths = find_masks(filename)
        if mask_paths and output_masks_folder:
            for i_mask, mask_path in enumerate(mask_paths):
                # Use the same naming convention as the renamed image, with _obj_N suffix if multiple objects
                _, ext = os.path.splitext(mask_path)
                mask_suffix = "" if len(mask_paths) == 1 else f"_obj_{i_mask+1}"
                mask_dst_path = os.path.join(output_masks_folder, f"test_{i}{mask_suffix}{ext}")
                shutil.copy2(mask_path, mask_dst_path)
                print(f"Created mask {mask_dst_path}")
    
    # Rename and copy train files and their masks
    for i, filename in enumerate(train_files):
        # Handle original image
        src_path = os.path.join(input_folder, filename)
        _, ext = os.path.splitext(filename)
        dst_filename = f"train_{i}{ext}"
        dst_path = os.path.join(output_folder, dst_filename)
        shutil.copy2(src_path, dst_path)
        print(f"Created {dst_path}")
        
        # Store filename mapping
        filename_mapping[filename] = dst_filename
        
        # Handle corresponding masks
        mask_paths = find_masks(filename)
        if mask_paths and output_masks_folder:
            for i_mask, mask_path in enumerate(mask_paths):
                # Use the same naming convention as the renamed image, with _obj_N suffix if multiple objects
                _, ext = os.path.splitext(mask_path)
                mask_suffix = "" if len(mask_paths) == 1 else f"_obj_{i_mask+1}"
                mask_dst_path = os.path.join(output_masks_folder, f"train_{i}{mask_suffix}{ext}")
                shutil.copy2(mask_path, mask_dst_path)
                print(f"Created mask {mask_dst_path}")
    
    # Save updated transforms.json if originally provided
    if updated_transforms_data is not None:
        # Update frames with new file paths
        for frame in transforms_data['frames']:
            old_filename = os.path.basename(frame['file_path'])
            if old_filename in filename_mapping:
                # Create a copy of the frame and update the file path
                updated_frame = frame.copy()
                updated_frame['file_path'] = os.path.join('db', filename_mapping[old_filename])
                updated_transforms_data['frames'].append(updated_frame)
        
        # Save updated transforms.json
        output_transforms_path = os.path.join(output_folder, 'transforms.json')
        with open(output_transforms_path, 'w') as f:
            json.dump(updated_transforms_data, f, indent=2)
        print(f"Updated transforms saved to {output_transforms_path}")
    
    print(f"Done! Renamed and split images saved to {output_folder}")
    if output_masks_folder:
        print(f"Matching masks saved to {output_masks_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split and rename image files, their masks, and update transforms.json")
    parser.add_argument("--input", "-i", required=True, help="Input folder containing original images")
    parser.add_argument("--masks", "-m", help="Input folder containing mask images (optional)")
    parser.add_argument("--transforms", "-t", help="Path to transforms.json file (optional)")
    parser.add_argument("--output", "-o", help="Output folder for renamed images (optional)")
    parser.add_argument("--output-masks", "-om", help="Output folder for renamed masks (optional)")
    parser.add_argument("--test-split", "-ts", type=int, default=20, help="Percentage of images for test set (default: 20)")
    parser.add_argument("--extensions", "-e", nargs="+", help="File extensions to include (default: png jpg jpeg bmp tif tiff)")
    
    args = parser.parse_args()
    
    # Validate test percentage
    if args.test_split < 0 or args.test_split > 100:
        print("Error: Test percentage must be between 0 and 100")
        exit(1)
    
    split_and_rename_images_with_masks_and_transforms(
        args.input,
        args.transforms,
        args.output,
        args.masks,
        args.output_masks,
        args.test_split,
        args.extensions
    )