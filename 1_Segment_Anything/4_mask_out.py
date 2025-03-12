import os
import cv2
import numpy as np
import re

def extract_objects_with_masks(output_size=512):
    # Create output directory if it doesn't exist
    output_dir = "extracted_objects"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")
    
    # Paths
    frames_dir = "extracted_frames"
    masks_dir = "extracted_frames/segmentation_output/masks"
    
    # Get all mask files
    mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith(".png")])
    
    # Process each mask
    for mask_filename in mask_files:
        # Extract frame number from mask filename (frame_XXXX_obj_1.png)
        match = re.search(r'frame_(\d+)_obj', mask_filename)
        if not match:
            print(f"Couldn't parse frame number from {mask_filename}, skipping...")
            continue
        
        frame_num = int(match.group(1))
        
        # Find corresponding original image (looking for {frame_num}.jpg)
        orig_filename = f"{frame_num}.jpg"
        orig_path = os.path.join(frames_dir, orig_filename)
        
        # Check if original image exists
        if not os.path.exists(orig_path):
            print(f"Original image {orig_path} not found, skipping...")
            continue
        
        # Read original image and mask
        original_img = cv2.imread(orig_path)
        if original_img is None:
            print(f"Failed to read original image {orig_path}, skipping...")
            continue
        
        mask_path = os.path.join(masks_dir, mask_filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Failed to read mask {mask_path}, skipping...")
            continue
        
        # Ensure mask is binary (0 or 255)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # Find bounding box of the mask (non-zero pixels)
        non_zero_pixels = cv2.findNonZero(mask)
        if non_zero_pixels is None or len(non_zero_pixels) == 0:
            print(f"No object found in mask {mask_path}, skipping...")
            continue
            
        x, y, w, h = cv2.boundingRect(non_zero_pixels)
        
        # Calculate the size needed for a square that contains the entire object
        max_dim = max(w, h)
        
        # Calculate padding needed to center the object in the square
        pad_left = (max_dim - w) // 2
        pad_top = (max_dim - h) // 2
        
        # Create a square mask with the object centered
        square_mask = np.zeros((max_dim, max_dim), dtype=np.uint8)
        square_mask[pad_top:pad_top+h, pad_left:pad_left+w] = mask[y:y+h, x:x+w]
        
        # Create a square image with the original content
        square_img = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
        square_img[pad_top:pad_top+h, pad_left:pad_left+w] = original_img[y:y+h, x:x+w]
        
        # Convert to BGRA (add alpha channel)
        output_img = cv2.cvtColor(square_img, cv2.COLOR_BGR2BGRA)
        
        # Set alpha channel based on square mask
        output_img[:, :, 3] = square_mask
        
        # Resize to the desired output size if needed
        if output_size != max_dim:
            output_img = cv2.resize(output_img, (output_size, output_size), interpolation=cv2.INTER_AREA)
        
        # Save the result
        output_path = os.path.join(output_dir, f"extracted_{frame_num:04d}.png")
        cv2.imwrite(output_path, output_img)
        print(f"Processed {mask_filename} -> {output_path}")
    
    print(f"Finished! Extracted objects saved to {output_dir}/")

if __name__ == "__main__":
    # You can specify the desired square size here (default 512x512)
    extract_objects_with_masks(output_size=512)