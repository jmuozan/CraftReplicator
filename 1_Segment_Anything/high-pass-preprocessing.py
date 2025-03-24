import os
import cv2
import numpy as np
from tqdm import tqdm
import shutil

def create_clean_output_folder(output_dir):
    """Create a clean output folder, removing it if it exists already"""
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

def apply_high_pass_filter(img, kernel_size=11):
    """Apply a high-pass filter to the image to emphasize edges and details"""
    # Convert to grayscale if not already
    if len(img.shape) > 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    
    # Subtract blurred image from original to get high-pass filtered image
    high_pass = cv2.subtract(gray, blur)
    
    # Normalize to 0-255 range
    high_pass = cv2.normalize(high_pass, None, 0, 255, cv2.NORM_MINMAX)
    
    return high_pass

def preprocess_image(img, high_pass_strength=0.5, processing_mode="balanced"):
    """Apply multiple preprocessing techniques with different strategies based on mode"""
    # Convert to grayscale to work with structural elements
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE with different parameters based on mode
    if processing_mode == "detail":
        # Higher clip limit emphasizes more details
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    else:
        # More balanced approach
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    equalized = clahe.apply(gray)
    
    # Apply high-pass filter with different kernel size based on mode
    if processing_mode == "structure":
        # Larger kernel for bigger structural elements
        high_pass = apply_high_pass_filter(equalized, kernel_size=15)
    else:
        # Smaller kernel for finer details
        high_pass = apply_high_pass_filter(equalized, kernel_size=11)
    
    # Blend original equalized image with high-pass filter
    blended = cv2.addWeighted(equalized, 1 - high_pass_strength, high_pass, high_pass_strength, 0)
    
    # Apply additional edge enhancement (unsharp masking) if in detail mode
    if processing_mode == "detail":
        blurred = cv2.GaussianBlur(blended, (0, 0), 3.0)
        sharpened = cv2.addWeighted(blended, 1.5, blurred, -0.5, 0)
        blended = sharpened
    
    # Normalize to utilize full dynamic range
    normalized = cv2.normalize(blended, None, 0, 255, cv2.NORM_MINMAX)
    
    # Convert back to BGR for saving
    result = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
    
    return result

def create_multi_version_image(img, base_filename, output_dir):
    """Create multiple versions of the image with different preprocessing strategies"""
    # Create standard version with 0.5 strength (balanced)
    balanced = preprocess_image(img, high_pass_strength=0.5, processing_mode="balanced")
    cv2.imwrite(os.path.join(output_dir, base_filename), balanced)
    
    # Create detail-oriented version with higher strength
    detail = preprocess_image(img, high_pass_strength=0.6, processing_mode="detail")
    filename_parts = os.path.splitext(base_filename)
    detail_filename = f"{filename_parts[0]}_detail{filename_parts[1]}"
    cv2.imwrite(os.path.join(output_dir, detail_filename), detail)
    
    # Create structure-oriented version with lower strength
    structure = preprocess_image(img, high_pass_strength=0.4, processing_mode="structure")
    structure_filename = f"{filename_parts[0]}_structure{filename_parts[1]}"
    cv2.imwrite(os.path.join(output_dir, structure_filename), structure)

def process_all_images(input_dir, output_dir, multi_version=False):
    """Process all images in the input directory with various preprocessing parameters"""
    create_clean_output_folder(output_dir)
    
    # Get all image files
    image_files = [f for f in os.listdir(input_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
    
    print(f"Found {len(image_files)} images in {input_dir}")
    
    # Process each image
    for img_file in tqdm(image_files, desc="Processing images"):
        input_path = os.path.join(input_dir, img_file)
        
        # Read the image
        img = cv2.imread(input_path)
        
        if img is None:
            print(f"Warning: Could not read {input_path}, skipping...")
            continue
        
        if multi_version:
            # Create multiple versions with different parameters
            create_multi_version_image(img, img_file, output_dir)
        else:
            # Apply single version preprocessing with moderate high-pass filter
            processed_img = preprocess_image(img, high_pass_strength=0.5, processing_mode="balanced")
            output_path = os.path.join(output_dir, img_file)
            cv2.imwrite(output_path, processed_img)
    
    print(f"Processed images saved to {output_dir}")
    if multi_version:
        print(f"Created 3 versions of each image (standard, detail, structure)")

def main():
    # Set input and output directories
    input_dir = "db_split"
    output_dir = "db_split_highpass"
    
    # Clean up any previous database to avoid conflicts
    if os.path.exists('distorted/database.db'):
        os.remove('distorted/database.db')
    
    print("Applying improved preprocessing with balanced high-pass filter...")
    
    # Process images - set multi_version=True to create multiple versions of each image
    multi_version = False  # Change to True if you want multiple versions of each image
    process_all_images(input_dir, output_dir, multi_version)
    
    print("\nDone! You can now run COLMAP on the preprocessed images.")
    
    if multi_version:
        print("\nA multi-version approach was used. You can run COLMAP on all images together:")
        print(f"  python3 glomap.py --image_path {output_dir} --matcher_type exhaustive_matcher --SiftMatching.guided_matching 1")
    else:
        print("\nExample command with relaxed parameters for more matches:")
        print(f"  python3 glomap.py --image_path {output_dir} --matcher_type exhaustive_matcher --SiftMatching.guided_matching 1 --SiftMatching.max_num_matches 8192 --RelPoseFilter.min_inlier_num 15 --RelPoseFilter.min_inlier_ratio 0.15 --TrackFilter.min_triangulation_angle 2.0")
    
    print("\nTips for improving reconstruction:")
    print("  1. If results are still unsatisfactory, try with multi_version=True")
    print("  2. For incremental reconstruction, start with a subset of 10-20 good images")
    print("  3. For sequential videos, try --matcher_type sequential_matcher with a --SiftMatching.max_num_matches 10000")
    print("  4. You can set --TrackEstablishment.max_num_tracks to a higher value like 10000 for more points")

if __name__ == "__main__":
    main()