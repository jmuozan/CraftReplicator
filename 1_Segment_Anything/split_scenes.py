import os
import shutil
import re
import argparse
from collections import defaultdict

def split_into_scenes(source_dir, output_dir):
    """
    Split images into different scene folders based on filename patterns.
    This is a simple approach - you may need to customize based on your naming convention.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get all image files from the source directory
    image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Group images by scene
    # This example uses a simple approach: looking for scene/sequence identifiers in filenames
    # Customize the pattern based on your actual file naming scheme
    scenes = defaultdict(list)
    
    # Example pattern: Extract scene identifier from filenames
    # This assumes filenames like "scene01_frame001.jpg" or similar
    # Adjust the regex pattern to match your actual naming scheme
    pattern = re.compile(r'(scene\d+|seq\d+|shot\d+|s\d+)')
    
    for image_file in image_files:
        match = pattern.search(image_file)
        if match:
            scene_id = match.group(1)
        else:
            # If no pattern match, try to guess based on numbering or other criteria
            # This is a very simple approach - sequence numbers
            # For documentary footage, frames might be numbered sequentially
            try:
                # Extract numbers from filename
                numbers = re.findall(r'\d+', image_file)
                if numbers:
                    # Use the first number group as a scene identifier
                    # Divide by 100 to group frames (assuming sequential numbering)
                    frame_num = int(numbers[0])
                    scene_id = f"scene_{frame_num // 100}"
                else:
                    scene_id = "unknown_scene"
            except:
                scene_id = "unknown_scene"
        
        scenes[scene_id].append(image_file)
    
    # Create scene directories and copy images
    for scene_id, files in scenes.items():
        scene_dir = os.path.join(output_dir, scene_id)
        if not os.path.exists(scene_dir):
            os.makedirs(scene_dir)
        
        for file in files:
            shutil.copy(
                os.path.join(source_dir, file),
                os.path.join(scene_dir, file)
            )
        
        print(f"Copied {len(files)} images to scene '{scene_id}'")
    
    return scenes.keys()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split images into scene folders based on filenames")
    parser.add_argument("--source", required=True, help="Source directory containing all images")
    parser.add_argument("--output", default="scenes", help="Output directory for scene folders")
    
    args = parser.parse_args()
    scene_ids = split_into_scenes(args.source, args.output)
    
    print(f"\nCreated {len(scene_ids)} scene folders in {args.output}")
    print("Run GLOMAP on each scene folder and then merge the results")