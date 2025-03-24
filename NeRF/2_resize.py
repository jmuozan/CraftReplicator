import os
import sys
from PIL import Image
import argparse

def resize_images(folder_path, max_dimension):
    """
    Resizes all images in the specified folder and deletes the originals.
    
    Args:
        folder_path (str): Path to the folder containing images
        max_dimension (int): Maximum dimension (width or height) in pixels
    """
    # Check if folder exists
    if not os.path.isdir(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist.")
        return
    
    # Get all files in the folder
    files = os.listdir(folder_path)
    
    # Counter for processed images
    processed_count = 0
    skipped_count = 0
    
    # Extensions to process
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
    
    print(f"Processing images in '{folder_path}'...")
    
    # Process each file
    for file in files:
        file_path = os.path.join(folder_path, file)
        
        # Skip directories
        if os.path.isdir(file_path):
            continue
        
        # Check if file is an image
        file_ext = os.path.splitext(file)[1].lower()
        if file_ext not in valid_extensions:
            skipped_count += 1
            continue
        
        try:
            # Open the image
            img = Image.open(file_path)
            
            # Get original dimensions
            width, height = img.size
            original_size = os.path.getsize(file_path) / 1024  # Size in KB
            
            # Skip if image is already smaller than max_dimension
            if width <= max_dimension and height <= max_dimension:
                print(f"Skipping '{file}' (already smaller than {max_dimension}px)")
                skipped_count += 1
                continue
            
            # Calculate new dimensions
            if width > height:
                new_width = max_dimension
                new_height = int(height * (max_dimension / width))
            else:
                new_height = max_dimension
                new_width = int(width * (max_dimension / height))
            
            # Resize image
            resized_img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Create a temporary filename for the resized image
            temp_file_path = os.path.join(folder_path, f"resized_{file}")
            
            # Save the resized image
            resized_img.save(temp_file_path, quality=95, optimize=True)
            
            # Get new file size
            new_size = os.path.getsize(temp_file_path) / 1024  # Size in KB
            
            # Close images
            img.close()
            resized_img.close()
            
            # Remove original file
            os.remove(file_path)
            
            # Rename the resized file to the original name
            os.rename(temp_file_path, file_path)
            
            print(f"Resized '{file}': {width}x{height} ({original_size:.1f} KB) -> {new_width}x{new_height} ({new_size:.1f} KB)")
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing '{file}': {str(e)}")
            skipped_count += 1
    
    print(f"\nResizing complete:")
    print(f"  - {processed_count} images resized")
    print(f"  - {skipped_count} files skipped")

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Resize images in a folder to a maximum dimension.')
    parser.add_argument('folder', help='Path to the folder containing images')
    parser.add_argument('--max-dimension', type=int, default=None, 
                        help='Maximum dimension (width or height) in pixels')
    
    # Parse arguments
    args = parser.parse_args()
    
    # If max_dimension wasn't provided, ask for it
    max_dimension = args.max_dimension
    if max_dimension is None:
        while True:
            try:
                max_dimension = int(input("Enter the maximum dimension in pixels: "))
                if max_dimension <= 0:
                    print("Please enter a positive number.")
                else:
                    break
            except ValueError:
                print("Please enter a valid number.")
    
    # Confirm before proceeding
    print(f"\nWARNING: This will resize all images in '{args.folder}' to a maximum dimension of {max_dimension}px")
    print("Original files will be DELETED after resizing.")
    confirm = input("Continue? (y/n): ")
    
    if confirm.lower() == 'y':
        resize_images(args.folder, max_dimension)
    else:
        print("Operation cancelled.")

if __name__ == "__main__":
    main()