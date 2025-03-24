import os
import cv2
import numpy as np
import re
import glob

def apply_masks_to_scene_images():
    # Paths
    masks_dir = "masks"
    output_dir = "db/masked"
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")
    
    # Get all mask files
    mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith(".png")])
    
    # Map frame numbers to image paths
    image_map = {}
    
    # Scan all scene folders for images
    scene_folders = glob.glob('db/scene*')
    for scene_folder in scene_folders:
        scene_images = glob.glob(f"{scene_folder}/*.jpg")
        for img_path in scene_images:
            # Extract the image number from filename
            img_number = int(os.path.basename(img_path).split('.')[0])
            image_map[img_number] = img_path
    
    print(f"Found {len(image_map)} images across all scene folders")
    
    # Process each mask
    processed_count = 0
    skipped_count = 0
    
    for mask_filename in mask_files:
        # Extract frame number from mask filename (frame_XXXX_obj_1.png)
        match = re.search(r'frame_(\d+)_obj', mask_filename)
        if not match:
            print(f"Couldn't parse frame number from {mask_filename}, skipping...")
            skipped_count += 1
            continue
        
        frame_num = int(match.group(1))
        
        # Find corresponding original image
        if frame_num not in image_map:
            print(f"No image found for frame {frame_num}, skipping...")
            skipped_count += 1
            continue
        
        orig_path = image_map[frame_num]
        
        # Read original image and mask
        original_img = cv2.imread(orig_path)
        if original_img is None:
            print(f"Failed to read original image {orig_path}, skipping...")
            skipped_count += 1
            continue
        
        mask_path = os.path.join(masks_dir, mask_filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Failed to read mask {mask_path}, skipping...")
            skipped_count += 1
            continue
        
        # Ensure mask is binary (0 or 255)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # Resize mask to match original image dimensions if they're different
        if mask.shape[:2] != original_img.shape[:2]:
            print(f"Resizing mask for {mask_filename} from {mask.shape[:2]} to {original_img.shape[:2]}")
            mask = cv2.resize(mask, (original_img.shape[1], original_img.shape[0]), 
                             interpolation=cv2.INTER_NEAREST)
        
        # Convert to BGRA (add alpha channel)
        output_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2BGRA)
        
        # Set alpha channel based on mask
        output_img[:, :, 3] = mask
        
        # Determine output filename, keeping original name but in masked folder
        output_filename = os.path.basename(orig_path).replace('.jpg', '.png')
        output_path = os.path.join(output_dir, output_filename)
        
        # Save the result
        cv2.imwrite(output_path, output_img)
        print(f"Processed {mask_filename} -> {output_path}")
        processed_count += 1
    
    print(f"Finished! Processed {processed_count} images, skipped {skipped_count}")
    print(f"Masked images saved to {output_dir}/")

if __name__ == "__main__":
    apply_masks_to_scene_images()