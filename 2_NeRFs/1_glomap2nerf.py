import os
import subprocess
import glob
import shutil

# Detect all scene folders
scene_folders = glob.glob('db/scene*')
print(f"Found scene folders: {scene_folders}")

# Create main output directory
main_output_dir = 'glomap'
os.makedirs(main_output_dir, exist_ok=True)

def run_command(cmd, verbose=True):
    """Run a command and print its output"""
    cmd_str = ' '.join(cmd)
    print(f"Running: {cmd_str}")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    
    if verbose or process.returncode != 0:
        print("STDOUT:")
        print(stdout)
        
        if stderr:
            print("STDERR:")
            print(stderr)
    
    if process.returncode != 0:
        print(f"Command failed with return code: {process.returncode}")
    
    return process.returncode, stdout, stderr

def process_scene(scene_folder):
    """Process a single scene folder with both COLMAP and GLOMAP approaches"""
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
    
    # Check if the image folder is not empty
    image_files = glob.glob(os.path.join(scene_folder, '*.jpg')) + glob.glob(os.path.join(scene_folder, '*.png'))
    if not image_files:
        print(f"Warning: No images found in {scene_folder}!")
        return None
    
    print(f"Found {len(image_files)} images in {scene_folder}")
    
    # Remove existing database if it exists to start fresh
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"Removed existing database: {db_path}")
    
    # Run feature extraction
    ret_code, stdout, stderr = run_command([
        'colmap', 'feature_extractor', 
        '--image_path', scene_folder, 
        '--database_path', db_path,
        '--ImageReader.camera_model', 'SIMPLE_RADIAL'
    ])
    
    if ret_code != 0:
        print(f"Feature extraction failed for {scene_name}!")
        return None
    
    # Run feature matching
    ret_code, stdout, stderr = run_command([
        'colmap', 'exhaustive_matcher', 
        '--database_path', db_path
    ])
    
    if ret_code != 0:
        print(f"Feature matching failed for {scene_name}!")
        return None
    
    # Check database statistics
    print(f"Checking database statistics for {db_path}")
    run_command(['colmap', 'database_stats', '--database_path', db_path])
    
    # Try both GLOMAP and COLMAP approaches for each scene
    success = False
    
    # First try: GLOMAP mapper
    print(f"Attempting GLOMAP mapper for {scene_name}")
    ret_code, stdout, stderr = run_command([
        'glomap', 'mapper', 
        '--database_path', db_path, 
        '--image_path', scene_folder, 
        '--output_path', recon_folder
    ])
    
    # Check if GLOMAP produced results
    glomap_model_path = os.path.join(recon_folder, '0')
    if ret_code == 0 and os.path.exists(glomap_model_path) and os.path.isfile(os.path.join(glomap_model_path, 'points3D.bin')):
        print(f"GLOMAP mapper succeeded for {scene_name}")
        success = True
        model_path = glomap_model_path
    else:
        print(f"GLOMAP mapper did not produce complete results for {scene_name}")
        
        # Second try: COLMAP mapper
        print(f"Attempting COLMAP mapper for {scene_name}")
        colmap_sparse_dir = os.path.join(scene_dir, 'colmap_sparse')
        os.makedirs(colmap_sparse_dir, exist_ok=True)
        
        ret_code, stdout, stderr = run_command([
            'colmap', 'mapper',
            '--database_path', db_path,
            '--image_path', scene_folder,
            '--output_path', colmap_sparse_dir
        ])
        
        # Check if COLMAP produced results
        colmap_model_path = os.path.join(colmap_sparse_dir, '0')
        if ret_code == 0 and os.path.exists(colmap_model_path) and os.path.isfile(os.path.join(colmap_model_path, 'points3D.bin')):
            print(f"COLMAP mapper succeeded for {scene_name}")
            
            # Copy COLMAP results to reconstruction folder
            if os.path.exists(glomap_model_path):
                shutil.rmtree(glomap_model_path)
            
            shutil.copytree(colmap_model_path, glomap_model_path)
            print(f"Copied COLMAP results to {glomap_model_path}")
            
            success = True
            model_path = glomap_model_path
    
    if not success:
        print(f"WARNING: Both GLOMAP and COLMAP failed to produce results for {scene_name}")
        return None
    
    # Verify the model files exist
    expected_files = ['cameras.bin', 'images.bin', 'points3D.bin']
    missing_files = [f for f in expected_files if not os.path.exists(os.path.join(model_path, f))]
    
    if missing_files:
        print(f"WARNING: Model is missing these files: {missing_files}")
        return None
    
    print(f"Successfully created reconstruction for {scene_name}")
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

# Analyze all successful models
print("\n\n" + "="*50)
print("All reconstructions complete! Analyzing models:")
print("="*50)

for model_path in model_paths:
    print(f"\nAnalyzing model: {model_path}")
    run_command(['colmap', 'model_analyzer', '--path', model_path])

print("\nAll operations completed successfully!")
print("Model paths:", model_paths)