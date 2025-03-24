import os
import subprocess
import sys

def run_colmap_with_relaxed_params(image_dir, output_dir):
    """
    Run COLMAP with relaxed parameters to improve success with challenging datasets
    
    Args:
        image_dir: Input image directory
        output_dir: Output directory for COLMAP results
    """
    os.makedirs(output_dir, exist_ok=True)
    db_path = os.path.join(output_dir, "database.db")
    
    # Feature extraction with relaxed parameters
    feature_extractor_cmd = [
        "colmap", "feature_extractor",
        "--database_path", db_path,
        "--image_path", image_dir,
        "--ImageReader.single_camera", "1",
        "--SiftExtraction.use_gpu", "1",
        "--SiftExtraction.max_num_features", "8192",  # Extract more features
        "--SiftExtraction.first_octave", "-1",  # Start at a lower octave level
        "--SiftExtraction.peak_threshold", "0.004"  # Lower threshold for detection
    ]
    
    # Feature matching with relaxed parameters
    matcher_cmd = [
        "colmap", "exhaustive_matcher",
        "--database_path", db_path,
        "--SiftMatching.guided_matching", "1",
        "--SiftMatching.max_num_matches", "32768",  # Allow more matches
        "--SiftMatching.max_ratio", "0.9"  # Be more permissive with ratio test
    ]
    
    # Mapper
    mapper_output_path = os.path.join(output_dir, "sparse")
    os.makedirs(mapper_output_path, exist_ok=True)

    mapper_cmd = [
        "colmap", "mapper",
        "--database_path", db_path,
        "--image_path", image_dir,
        "--output_path", mapper_output_path,
        "--Mapper.ba_global_max_num_iterations", "30",
        "--Mapper.filter_max_reproj_error", "8.0",
        "--Mapper.init_min_num_inliers", "15"
    ]
    
    print("Running COLMAP feature extraction...")
    subprocess.run(feature_extractor_cmd)
    
    print("Running COLMAP exhaustive matching...")
    subprocess.run(matcher_cmd)
    
    print("Running COLMAP mapping...")
    subprocess.run(mapper_cmd)
    
    # Check if reconstruction succeeded
    if not os.path.exists(os.path.join(output_dir, "sparse/0")):
        print("COLMAP reconstruction failed. Will need to initialize poses.")
        return False
    return True

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python run_colmap.py <image_dir> <output_dir>")
        sys.exit(1)
    
    image_dir = sys.argv[1]
    output_dir = sys.argv[2]
    run_colmap_with_relaxed_params(image_dir, output_dir)