#!/usr/bin/env python3
"""
Improved Structure-from-Motion Pipeline for Camera Position Estimation
This script uses hloc to reconstruct 3D scene and camera positions from a set of images.
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt

from hloc import (
    extract_features,
    match_features,
    reconstruction,
    visualization,
    pairs_from_retrieval,
    pairs_from_exhaustive
)

def main():
    parser = argparse.ArgumentParser(description='Run SfM pipeline on image dataset')
    parser.add_argument('--images', type=str, default='db_split_highpass',
                        help='Path to the directory containing images')
    parser.add_argument('--outputs', type=str, default='outputs/sfm',
                        help='Path to the output directory')
    parser.add_argument('--exhaustive', action='store_true',
                        help='Use exhaustive matching instead of retrieval')
    parser.add_argument('--max_pairs', type=int, default=20,
                        help='Number of image pairs to match per image')
    parser.add_argument('--feature_type', type=str, default='superpoint',
                        choices=['superpoint', 'sift', 'r2d2', 'd2net'],
                        help='Type of feature extractor to use')
    parser.add_argument('--matcher_type', type=str, default='superglue',
                        choices=['superglue', 'lightglue', 'NN'],
                        help='Type of feature matcher to use')
    parser.add_argument('--clear_cache', action='store_true',
                        help='Clear existing feature and match files')
    args = parser.parse_args()

    # Set up paths
    images = Path(args.images)
    outputs = Path(args.outputs)
    outputs.mkdir(exist_ok=True, parents=True)
    
    print(f"Processing images from {images}")
    print(f"Storing results in {outputs}")
    
    # Set configurations based on command line arguments
    feature_types = {
        'superpoint': 'superpoint_aachen',
        'sift': 'sift',
        'r2d2': 'r2d2',
        'd2net': 'd2net-ss'
    }
    
    matcher_types = {
        'superglue': 'superglue',
        'lightglue': 'lightglue',
        'NN': 'NN-ratio'
    }
    
    # Configuration
    retrieval_conf = extract_features.confs["netvlad"]
    feature_conf = extract_features.confs[feature_types[args.feature_type]]
    matcher_conf = match_features.confs[matcher_types[args.matcher_type]]
    
    # Modify feature configuration to extract more keypoints
    if args.feature_type == 'superpoint':
        feature_conf['model']['max_keypoints'] = 8192  # Increase from default 4096
    
    # Create names for output files
    sfm_dir = outputs / f"sfm_{args.feature_type}+{args.matcher_type}"
    
    # Clear cache if requested
    if args.clear_cache:
        import os
        for file in outputs.glob(f"*{feature_conf['output']}*"):
            print(f"Removing {file}")
            os.remove(file)
        for file in outputs.glob(f"*{matcher_conf['output']}*"):
            print(f"Removing {file}")
            os.remove(file)
        for file in outputs.glob("*pairs*"):
            print(f"Removing {file}")
            os.remove(file)
    
    # Find image pairs
    if args.exhaustive:
        print("\n--- Finding image pairs via exhaustive matching ---")
        sfm_pairs = outputs / "pairs-exhaustive.txt"
        pairs_from_exhaustive.main(images, sfm_pairs, args.max_pairs)
    else:
        print("\n--- Finding image pairs via image retrieval ---")
        sfm_pairs = outputs / "pairs-netvlad.txt"
        retrieval_path = extract_features.main(retrieval_conf, images, outputs)
        pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=args.max_pairs)
    
    print(f"Image pairs written to {sfm_pairs}")
    
    # Extract local features
    print(f"\n--- Extracting {args.feature_type} features ---")
    feature_path = extract_features.main(feature_conf, images, outputs)
    print(f"Features written to {feature_path}")
    
    # Match local features
    print(f"\n--- Matching features with {args.matcher_type} ---")
    match_path = match_features.main(
        matcher_conf, 
        sfm_pairs, 
        feature_conf["output"],
        outputs
    )
    print(f"Matches written to {match_path}")
    
    # Run SfM reconstruction with modified parameters
    print("\n--- Running 3D reconstruction ---")
    mapper_options = {
        'ba_global_max_refinements': 5,  # More refinement iterations
        'ba_global_max_num_iterations': 100,  # More optimization iterations
        'min_num_matches': 8,  # Reduce from default (15/30)
        'init_min_num_inliers': 8,  # Reduce from default
        'abs_pose_min_num_inliers': 8,  # Reduce from default
        'abs_pose_min_inlier_ratio': 0.1,  # Reduce from default (0.25)
        'tri_min_angle': 1.0,  # Reduce minimum triangulation angle (degrees)
        'num_threads': -1  # Use all available threads
    }
    
    # Add COLMAP parameters optimized for difficult reconstructions
    model = reconstruction.main(
        sfm_dir, 
        images, 
        sfm_pairs, 
        feature_path, 
        match_path,
        mapper_options=mapper_options,
        verbose=True
    )
    print(f"Reconstruction completed and saved to {sfm_dir}")
    
    # Check if model is None (reconstruction failed)
    if model is None:
        print("\n--- Reconstruction failed! ---")
        print("Trying with relaxed settings...")
        
        # Even more relaxed settings for difficult cases
        mapper_options.update({
            'min_num_matches': 4,  
            'init_min_num_inliers': 4,
            'abs_pose_min_num_inliers': 4,
            'abs_pose_min_inlier_ratio': 0.05,
            'tri_min_angle': 0.5,
        })
        
        # Retry with exhaustive matching if not already used
        if not args.exhaustive:
            print("\n--- Retrying with exhaustive matching ---")
            sfm_pairs = outputs / "pairs-exhaustive.txt"
            pairs_from_exhaustive.main(images, sfm_pairs, args.max_pairs * 2)
            
            match_path = match_features.main(
                matcher_conf, 
                sfm_pairs, 
                feature_conf["output"],
                outputs
            )
            
            model = reconstruction.main(
                sfm_dir, 
                images, 
                sfm_pairs, 
                feature_path, 
                match_path,
                mapper_options=mapper_options,
                verbose=True
            )
    
    # Visualize results
    print("\n--- Creating visualizations ---")
    try:
        if model is not None:
            # Visualize by visibility
            fig = visualization.visualize_sfm_2d(model, images, color_by="visibility", n=5)
            vis_path = outputs / "visualization_visibility.png"
            fig.write_image(str(vis_path))
            print(f"Visibility visualization saved to {vis_path}")
            
            # Visualize by track length
            fig = visualization.visualize_sfm_2d(model, images, color_by="track_length", n=5)
            track_path = outputs / "visualization_track_length.png"
            fig.write_image(str(track_path))
            print(f"Track length visualization saved to {track_path}")
            
            # Visualize by depth
            fig = visualization.visualize_sfm_2d(model, images, color_by="depth", n=5)
            depth_path = outputs / "visualization_depth.png"
            fig.write_image(str(depth_path))
            print(f"Depth visualization saved to {depth_path}")
            
            # 3D visualization
            fig = visualization.visualize_sfm_3d(model)
            model3d_path = outputs / "visualization_3d_model.html"
            fig.write_html(str(model3d_path))
            print(f"3D model visualization saved to {model3d_path}")
        else:
            print("Cannot create visualizations because reconstruction failed.")
    except Exception as e:
        print(f"Visualization error: {e}")
        print("Visualization failed, but reconstruction should still be valid.")
    
    print("\n--- Pipeline complete ---")
    if model is not None:
        print(f"Camera positions and 3D model are available in {sfm_dir}")
        return model
    else:
        print("Reconstruction failed. Try with different settings or input images.")
        return None

if __name__ == "__main__":
    main()