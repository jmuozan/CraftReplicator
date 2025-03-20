import os
import subprocess
import glob

# Define your scene folders - update this list with your actual scene folders
scene_folders = ['db/scene1', 'db/scene2', 'db/scene3', 'db/scene4']

# You can also automatically detect scene folders with a pattern
#scene_folders = glob.glob('db/scene*')

# Create main output directory
main_output_dir = 'glomap'
os.makedirs(main_output_dir, exist_ok=True)

def run_command(cmd):
    """Run a command and print its output"""
    print(f"Running: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    
    print("STDOUT:")
    print(stdout)
    
    if stderr:
        print("STDERR:")
        print(stderr)
    
    return process.returncode

def process_scene(scene_folder):
    """Process a single scene folder with COLMAP and GLOMAP"""
    scene_name = os.path.basename(scene_folder)
    print(f"\n\n{'='*50}")
    print(f"Processing scene: {scene_name}")
    print(f"{'='*50}")
    
    # Create output folders
    scene_dir = os.path.join(main_output_dir, scene_name)
    os.makedirs(scene_dir, exist_ok=True)
    
    recon_folder = os.path.join(scene_dir, 'reconstruction')
    os.makedirs(recon_folder, exist_ok=True)
    
    # Database path inside the scene directory
    db_path = os.path.join(scene_dir, f"{scene_name}_db.db")
    
    # Run feature extraction
    run_command(['colmap', 'feature_extractor', 
                 '--image_path', scene_folder, 
                 '--database_path', db_path])
    
    # Run feature matching
    run_command(['colmap', 'exhaustive_matcher', 
                 '--database_path', db_path])
    
    # Run GLOMAP mapper
    run_command(['glomap', 'mapper', 
                 '--database_path', db_path, 
                 '--image_path', scene_folder, 
                 '--output_path', recon_folder])
    
    # Return the reconstruction path for later analysis
    model_path = os.path.join(recon_folder, '0')
    if not os.path.exists(model_path):
        print(f"Warning: Reconstruction folder {model_path} does not exist!")
        return None
    
    return model_path

# Process each scene folder
model_paths = []
for scene_folder in scene_folders:
    if os.path.exists(scene_folder):
        model_path = process_scene(scene_folder)
        if model_path:
            model_paths.append(model_path)
    else:
        print(f"Warning: Scene folder {scene_folder} does not exist!")

# Analyze all models after all reconstructions are complete
print("\n\n" + "="*50)
print("All reconstructions complete! Analyzing models:")
print("="*50)

for model_path in model_paths:
    print(f"\nAnalyzing model: {model_path}")
    run_command(['colmap', 'model_analyzer', '--path', model_path])

print("\nAll operations completed successfully!")
print("Model paths:", model_paths)