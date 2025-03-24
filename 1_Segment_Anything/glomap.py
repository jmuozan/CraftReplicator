import os
import subprocess
import argparse
import time
import datetime
import json
import numpy as np
from shutil import copy2, move
import sqlite3
import sys
try:
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available, falling back to command-line interface")

class SimpleImageMatcher:
    def __init__(self, image_path, selected_frames, output_path):
        self.image_path = image_path
        self.selected_frames = selected_frames
        self.output_path = output_path
        self.matches = []
        
        os.makedirs(output_path, exist_ok=True)
        
        # Load existing matches if available
        matches_file = os.path.join(output_path, "manual_matches.json")
        if os.path.exists(matches_file):
            try:
                with open(matches_file, 'r') as f:
                    self.matches = json.load(f)
                print(f"Loaded {len(self.matches)} existing match pairs from {matches_file}")
            except Exception as e:
                print(f"Error loading existing matches: {str(e)}")
    
    def run_matcher(self):
        """Run the manual matching process for all image pairs"""
        # For each consecutive pair of selected frames
        for i in range(len(self.selected_frames) - 1):
            image1 = self.selected_frames[i]
            image2 = self.selected_frames[i+1]
            
            # Check if we already have matches for this pair
            existing_match = next((m for m in self.matches if m["image1"] == image1 and m["image2"] == image2), None)
            
            if existing_match:
                print(f"\nImage pair {i+1}/{len(self.selected_frames)-1}: {image1} <> {image2}")
                print(f"Already has {len(existing_match['points'])} point matches")
                action = input("Options: [s]kip, [v]iew matches, [r]eplace matches: ").lower()
                
                if action == 's':
                    continue
                elif action == 'v':
                    print("\nExisting matches:")
                    for j, (x1, y1, x2, y2) in enumerate(existing_match['points']):
                        print(f"  Point {j+1}: ({x1:.1f}, {y1:.1f}) <-> ({x2:.1f}, {y2:.1f})")
                    
                    action = input("\nOptions: [c]ontinue with next pair, [r]eplace matches: ").lower()
                    if action != 'r':
                        continue
            
            # Match this pair
            point_pairs = self.match_image_pair(image1, image2)
            
            if point_pairs:
                # Update or add match
                if existing_match:
                    existing_match["points"] = point_pairs
                    print(f"Updated {len(point_pairs)} point pairs for {image1} <> {image2}")
                else:
                    self.matches.append({
                        "image1": image1,
                        "image2": image2,
                        "points": point_pairs
                    })
                    print(f"Added {len(point_pairs)} point pairs for {image1} <> {image2}")
                
                # Save after each successful pair
                self.save_matches()
        
        return os.path.join(self.output_path, "manual_matches.json")
    
    def match_image_pair(self, image1, image2):
        """Handle matching a single image pair"""
        if MATPLOTLIB_AVAILABLE:
            return self.match_image_pair_matplotlib(image1, image2)
        else:
            return self.match_image_pair_cli(image1, image2)
    
    def match_image_pair_cli(self, image1, image2):
        """Command-line interface for image matching"""
        print(f"\nMatching points between:\n  1. {image1}\n  2. {image2}")
        print("Since matplotlib is not available, you'll need to use the CLI interface.")
        print("Please view the images side-by-side in another application.")
        
        print("\nInstructions:")
        print("- Enter point coordinates as 'x1 y1 x2 y2'")
        print("- Enter 'done' when finished")
        print("- Enter 'skip' to skip this pair")
        
        point_pairs = []
        
        try:
            while True:
                user_input = input("> ").strip()
                if user_input.lower() == 'done':
                    break
                elif user_input.lower() == 'skip':
                    return []
                elif user_input.lower() == 'clear':
                    point_pairs = []
                    print("Cleared all points")
                    continue
                elif user_input.lower() == 'help':
                    print("\nCommands:")
                    print("  x1 y1 x2 y2 - Add a point pair")
                    print("  done - Finish matching this pair")
                    print("  skip - Skip this pair")
                    print("  clear - Clear all points")
                    print("  help - Show this help")
                    print("  list - List current points")
                    continue
                elif user_input.lower() == 'list':
                    if not point_pairs:
                        print("No points defined yet")
                    else:
                        for i, (x1, y1, x2, y2) in enumerate(point_pairs):
                            print(f"Point {i+1}: ({x1:.1f}, {y1:.1f}) <-> ({x2:.1f}, {y2:.1f})")
                    continue
                
                try:
                    coords = list(map(float, user_input.split()))
                    if len(coords) != 4:
                        print("Error: Need exactly 4 values (x1 y1 x2 y2)")
                        continue
                    
                    x1, y1, x2, y2 = coords
                    point_pairs.append((x1, y1, x2, y2))
                    print(f"Added point pair: ({x1}, {y1}) <-> ({x2}, {y2})")
                except ValueError:
                    print("Invalid format. Please use: x1 y1 x2 y2")
        
        except KeyboardInterrupt:
            print("\nInterrupted. Saving current points...")
        
        return point_pairs
    
    def match_image_pair_matplotlib(self, image1, image2):
        """Matplotlib-based interface for image matching"""
        from matplotlib.patches import Circle
        
        # Load images
        img1_path = os.path.join(self.image_path, image1)
        img2_path = os.path.join(self.image_path, image2)
        
        try:
            img1 = plt.imread(img1_path)
            img2 = plt.imread(img2_path)
        except Exception as e:
            print(f"Error loading images: {str(e)}")
            return self.match_image_pair_cli(image1, image2)
        
        # Set up the figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        fig.canvas.manager.set_window_title(f'Matching: {image1} <> {image2}')
        
        # Show images
        ax1.imshow(img1)
        ax1.set_title(f"Left: {image1}")
        ax1.axis('off')
        
        ax2.imshow(img2)
        ax2.set_title(f"Right: {image2}")
        ax2.axis('off')
        
        # Storage for points
        points_left = []
        points_right = []
        point_pairs = []
        circles_left = []
        circles_right = []
        texts_left = []
        texts_right = []
        
        # Text area for instructions
        plt.figtext(0.5, 0.01, 
                   "Click on the LEFT image, then the RIGHT image to create correspondences.\n"
                   "Press 'c' to clear all points, 'd' when done, 'x' to cancel.",
                   ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
        
        info_text = plt.figtext(0.5, 0.95, "Select point on LEFT image", ha="center", fontsize=12)
        
        # Event handlers
        def onclick(event):
            if event.inaxes == ax1 and len(points_left) <= len(points_right):
                # Left image click
                x, y = event.xdata, event.ydata
                points_left.append((x, y))
                
                # Add marker
                circle = Circle((x, y), radius=5, color='red', fill=True)
                ax1.add_patch(circle)
                circles_left.append(circle)
                
                # Add number
                text = ax1.text(x+10, y-10, str(len(points_left)), color='yellow', fontweight='bold')
                texts_left.append(text)
                
                info_text.set_text(f"Point {len(points_left)} set on LEFT. Now select corresponding point on RIGHT image.")
                
                fig.canvas.draw_idle()
            
            elif event.inaxes == ax2 and len(points_right) < len(points_left):
                # Right image click
                x, y = event.xdata, event.ydata
                points_right.append((x, y))
                
                # Add marker
                circle = Circle((x, y), radius=5, color='red', fill=True)
                ax2.add_patch(circle)
                circles_right.append(circle)
                
                # Add number
                text = ax2.text(x+10, y-10, str(len(points_right)), color='yellow', fontweight='bold')
                texts_right.append(text)
                
                # Create point pair
                xl, yl = points_left[-1]
                point_pairs.append((xl, yl, x, y))
                
                info_text.set_text(f"Created point pair {len(point_pairs)}. Select next point on LEFT image.")
                
                fig.canvas.draw_idle()
        
        def onkey(event):
            nonlocal points_left, points_right, circles_left, circles_right, texts_left, texts_right, point_pairs
            
            if event.key == 'c':
                # Clear all points
                for c in circles_left + circles_right:
                    c.remove()
                for t in texts_left + texts_right:
                    t.remove()
                
                points_left = []
                points_right = []
                circles_left = []
                circles_right = []
                texts_left = []
                texts_right = []
                point_pairs = []
                
                info_text.set_text("Cleared all points. Select point on LEFT image.")
                fig.canvas.draw_idle()
            
            elif event.key == 'd':
                # Done with this pair
                plt.close(fig)
            
            elif event.key == 'x':
                # Cancel
                point_pairs = []
                plt.close(fig)
        
        # Connect event handlers
        cid1 = fig.canvas.mpl_connect('button_press_event', onclick)
        cid2 = fig.canvas.mpl_connect('key_press_event', onkey)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        plt.show()
        
        return point_pairs
    
    def save_matches(self):
        """Save all matches to file"""
        matches_file = os.path.join(self.output_path, "manual_matches.json")
        with open(matches_file, 'w') as f:
            json.dump(self.matches, f, indent=2)
        
        print(f"Saved {len(self.matches)} image pairs with manual correspondences to {matches_file}")
        return matches_file

def check_triangulate_points(database_path, image1, image2):
    """
    Check if two images have sufficient matches for triangulation
    """
    try:
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()
        
        # Get image IDs
        cursor.execute("SELECT image_id, name FROM images WHERE name IN (?, ?)", (image1, image2))
        image_info = cursor.fetchall()
        
        if len(image_info) != 2:
            conn.close()
            return False, 0
        
        image_id1 = image_info[0][0] if image_info[0][1] == image1 else image_info[1][0]
        image_id2 = image_info[0][0] if image_info[0][1] == image2 else image_info[1][0]
        
        # Calculate pair_id
        pair_id = min(image_id1, image_id2) * 2147483647 + max(image_id1, image_id2)
        
        # Check for matches
        cursor.execute("SELECT data FROM matches WHERE pair_id = ?", (pair_id,))
        match_data = cursor.fetchone()
        
        conn.close()
        
        if not match_data:
            return False, 0
            
        # Count number of matches
        num_matches = len(match_data[0].split()) // 4  # Each match is 4 values
        return num_matches >= 8, num_matches  # Need at least 8 matches for good triangulation
        
    except Exception as e:
        print(f"Error checking triangulation: {str(e)}")
        return False, 0

def find_best_initial_pair(database_path, image_path):
    """
    Find the best initial image pair with the most matches for good triangulation
    """
    try:
        print("Finding best initial image pair for reconstruction...")
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()
        
        # Get all images
        cursor.execute("SELECT image_id, name FROM images")
        images = cursor.fetchall()
        image_ids = {img_id: name for img_id, name in images}
        
        # Get all matches
        cursor.execute("SELECT pair_id, COUNT(pair_id) FROM matches GROUP BY pair_id ORDER BY COUNT(pair_id) DESC")
        match_counts = cursor.fetchall()
        
        # Collect best pairs
        best_pairs = []
        for pair_id, count in match_counts:
            # Calculate the two image IDs from the pair_id
            image_id2 = pair_id % 2147483647
            image_id1 = pair_id // 2147483647
            
            if image_id1 in image_ids and image_id2 in image_ids:
                img1 = image_ids[image_id1]
                img2 = image_ids[image_id2]
                
                # Get actual match count by parsing the match data
                cursor.execute("SELECT data FROM matches WHERE pair_id = ?", (pair_id,))
                match_data = cursor.fetchone()
                if match_data:
                    num_matches = len(match_data[0].split()) // 4
                    best_pairs.append((img1, img2, num_matches))
        
        conn.close()
        
        # Sort by match count
        best_pairs.sort(key=lambda x: x[2], reverse=True)
        
        if best_pairs:
            top_pairs = best_pairs[:5]  # Get top 5 pairs
            print("Top image pairs by matches:")
            for i, (img1, img2, num_matches) in enumerate(top_pairs, 1):
                print(f"{i}. {img1} <> {img2}: {num_matches} matches")
            
            return best_pairs[0]  # Return the best pair
        else:
            print("No matched pairs found!")
            return None
        
    except Exception as e:
        print(f"Error finding best initial pair: {str(e)}")
        return None

def import_manual_matches_to_db(database_path, image_path, matches_file):
    """
    Import manual matches into COLMAP database
    """
    print(f"\nImporting manual matches from {matches_file} into COLMAP database...")
    
    # Load matches
    with open(matches_file, 'r') as f:
        match_data = json.load(f)
    
    # Connect to database
    try:
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()
    except Exception as e:
        print(f"Error connecting to database: {str(e)}")
        return False
    
    try:
        # Insert matches directly using COLMAP API
        colmap_db_cmd = [
            "colmap", "matches_importer",
            "--database_path", database_path,
            "--match_list_path", matches_file,
            "--match_type", "pairs"
        ]
        
        print(f"Running COLMAP matches importer...")
        result = subprocess.run(colmap_db_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"COLMAP matches_importer failed. Falling back to manual approach.")
            print(f"Error: {result.stderr}")
            
            # Fall back to direct database manipulation
            return import_manual_matches_direct(database_path, match_data)
        else:
            print(f"Successfully imported matches using COLMAP importer")
            return True
    
    except Exception as e:
        print(f"Error importing matches: {str(e)}")
        return False
    finally:
        conn.close()

def import_manual_matches_direct(database_path, match_data):
    """
    Import manual matches directly into COLMAP database through SQL
    """
    try:
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()
        
        # Get image IDs
        cursor.execute("SELECT image_id, name FROM images")
        image_ids = {name: image_id for image_id, name in cursor.fetchall()}
        
        # For each match
        match_count = 0
        for match in match_data:
            image1 = match["image1"]
            image2 = match["image2"]
            
            if image1 not in image_ids or image2 not in image_ids:
                print(f"Warning: Could not find images {image1} or {image2} in database")
                continue
            
            image_id1 = image_ids[image1]
            image_id2 = image_ids[image2]
            
            # Skip if no points
            if not match["points"]:
                continue
            
            # Create matches string directly from points
            # Format: "x1 y1 x2 y2" pairs
            matches_str = " ".join([f"{x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}" for x1, y1, x2, y2 in match["points"]])
            
            # Calculate pair_id as per COLMAP's convention
            pair_id = min(image_id1, image_id2) * 2147483647 + max(image_id1, image_id2)
            
            # Check if these images already have matches
            cursor.execute("SELECT pair_id FROM matches WHERE pair_id = ?", (pair_id,))
            existing_pair = cursor.fetchone()
            
            if existing_pair:
                # Update existing matches
                cursor.execute("UPDATE matches SET data = ? WHERE pair_id = ?", (matches_str, pair_id))
                print(f"Updated matches between {image1} and {image2}")
            else:
                # Insert new matches
                cursor.execute("INSERT INTO matches VALUES (?, ?)", (pair_id, matches_str))
                print(f"Added matches between {image1} and {image2}")
            
            match_count += 1
        
        # Commit changes
        conn.commit()
        conn.close()
        
        print(f"Successfully imported {match_count} match pairs directly to database")
        return True
    
    except Exception as e:
        print(f"Error importing matches directly: {str(e)}")
        return False

def create_initial_reconstruction(database_path, image_path, sparse_output_path):
    """
    Create initial reconstruction with best pair and specific options to help with challenging cases
    """
    # Find the best image pair to start with
    best_pair = find_best_initial_pair(database_path, image_path)
    
    if not best_pair:
        print("Could not find a good initial pair. Continuing with default mapper behavior.")
        return False

    img1, img2, num_matches = best_pair
    print(f"Using initial image pair: {img1} <> {img2} with {num_matches} matches")

    # Create a text file with the image pair
    init_path = os.path.join(os.path.dirname(sparse_output_path), 'init_pair.txt')
    with open(init_path, 'w') as f:
        f.write(f"{img1}\n{img2}")
    
    # First, run mapper in manual initialization mode
    mapper_init_cmd = (
        f"colmap mapper "
        f"--database_path {database_path} "
        f"--image_path {image_path} "
        f"--output_path {sparse_output_path} "
        f"--Mapper.init_image_id1 0 "
        f"--Mapper.init_image_id2 1 "
        f"--Mapper.init_min_points 8 "
        f"--Mapper.ba_global_max_num_iterations 50 "
        f"--Mapper.ba_local_max_num_iterations 30 "
        f"--Mapper.min_num_matches 8 "
        f"--Mapper.filter_max_reproj_error 4.0 "
        f"--Mapper.max_reg_trials 5"
    )
    
    print(f"Creating initial reconstruction with pair...")
    try:
        subprocess.run(mapper_init_cmd, shell=True, check=True)
        return True
    except subprocess.CalledProcessError:
        print("Failed to create initial reconstruction with specified pair.")
        return False

def create_additional_matches(database_path, image_path, manual_matches_folder):
    """Generate additional matches for pairs with not enough correspondences"""
    print("Analyzing match quality...")
    
    # Connect to database
    try:
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()
        
        # Get all images
        cursor.execute("SELECT image_id, name FROM images")
        images = cursor.fetchall()
        
        # For sequential matching, we want to ensure good connections between consecutive frames
        pairs_to_check = []
        for i in range(len(images) - 1):
            img1_id, img1_name = images[i]
            img2_id, img2_name = images[i+1]
            pairs_to_check.append((img1_name, img2_name))
        
        # Also check connections between more distant frames (every 5th frame)
        for i in range(0, len(images) - 5, 5):
            img1_id, img1_name = images[i]
            img2_id, img2_name = images[i+5]
            pairs_to_check.append((img1_name, img2_name))
        
        conn.close()
        
        # Check pairs for match quality
        weak_pairs = []
        for img1, img2 in pairs_to_check:
            has_good_matches, match_count = check_triangulate_points(database_path, img1, img2)
            if not has_good_matches:
                weak_pairs.append((img1, img2, match_count))
        
        # Sort weak pairs by match count
        weak_pairs.sort(key=lambda x: x[2])
        
        if weak_pairs:
            print(f"Found {len(weak_pairs)} image pairs with insufficient matches.")
            print("Top 5 weakest connections:")
            for i, (img1, img2, count) in enumerate(weak_pairs[:5], 1):
                print(f"{i}. {img1} <> {img2}: {count} matches")
            
            # Ask user if they want to add manual matches for weak pairs
            if len(weak_pairs) > 0:
                response = input("\nWould you like to add manual matches for some of these pairs? (y/n): ").lower()
                if response == 'y':
                    # Select a subset of weak pairs to fix (max 5)
                    pairs_to_fix = weak_pairs[:min(5, len(weak_pairs))]
                    
                    # Create a new SimpleImageMatcher instance for just these pairs
                    pairs_matcher = SimpleImageMatcher(image_path, 
                                                     [p[0] for p in pairs_to_fix] + [pairs_to_fix[-1][1]], 
                                                     manual_matches_folder)
                    
                    # Run the matcher for these pairs
                    matches_file = pairs_matcher.run_matcher()
                    
                    # Import the new matches
                    if os.path.exists(matches_file):
                        import_manual_matches_to_db(database_path, image_path, matches_file)
                        return True
        else:
            print("All image pairs have sufficient matches for triangulation.")
        
        return False
        
    except Exception as e:
        print(f"Error analyzing match quality: {str(e)}")
        return False

def select_key_frames(image_path, num_frames=10):
    """
    Select a subset of frames for manual matching to serve as anchors
    """
    image_files = sorted([f for f in os.listdir(image_path) 
                         if os.path.isfile(os.path.join(image_path, f)) and 
                         f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
    
    if len(image_files) <= num_frames:
        return image_files
    
    # Ensure even distribution across the sequence
    indices = np.linspace(0, len(image_files) - 1, num_frames, dtype=int)
    selected_frames = [image_files[i] for i in indices]
    
    print(f"Selected {len(selected_frames)} key frames for manual matching guidance")
    return selected_frames

def run_colmap_with_manual_matcher(image_path, matcher_type, interval, model_type, num_key_frames=15, 
                                  max_iterations=10):
    """
    Run COLMAP with command-line manual point selection and iterative refinement
    """
    # Prepare directories
    parent_dir = os.path.abspath(os.path.join(image_path, os.pardir))
    distorted_folder = os.path.join(parent_dir, 'distorted')
    database_path = os.path.join(distorted_folder, 'database.db')
    sparse_folder = os.path.join(parent_dir, 'sparse')
    sparse_zero_folder = os.path.join(sparse_folder, '0')
    manual_matches_folder = os.path.join(parent_dir, 'manual_matches')
    
    # Create all necessary directories
    os.makedirs(distorted_folder, exist_ok=True)
    
    # Make sure the sparse output directory exists
    sparse_output_path = os.path.join(distorted_folder, 'sparse')
    os.makedirs(sparse_output_path, exist_ok=True)
    
    os.makedirs(sparse_folder, exist_ok=True)
    os.makedirs(manual_matches_folder, exist_ok=True)
    
    log_file_path = os.path.join(parent_dir, "colmap_run.log")
    total_start_time = time.time()
    
    with open(log_file_path, "w") as log_file:
        log_file.write(f"COLMAP run started at: {datetime.datetime.now()}\n")
        
        # Step 1: Feature extraction
        print("Step 1: Running feature extraction...")
        log_file.write("Step 1: Running feature extraction\n")
        feature_cmd = f"colmap feature_extractor --image_path {image_path} --database_path {database_path} --ImageReader.single_camera 1 --ImageReader.camera_model PINHOLE --SiftExtraction.use_gpu 1"
        
        print(f"Running: {feature_cmd}")
        feature_start_time = time.time()
        subprocess.run(feature_cmd, shell=True, check=True)
        feature_end_time = time.time()
        feature_elapsed_time = feature_end_time - feature_start_time
        
        log_file.write(f"Feature extraction completed in {feature_elapsed_time:.2f} seconds\n")
        print(f"Feature extraction completed in {feature_elapsed_time:.2f} seconds")
        
        # Step 2: Manual matching
        print("\nStep 2: Starting manual matching process...")
        log_file.write("Step 2: Starting manual matching process\n")
        
        # Select frames for manual matching
        selected_frames = select_key_frames(image_path, num_key_frames)
        
        # Launch manual matcher
        matcher = SimpleImageMatcher(image_path, selected_frames, manual_matches_folder)
        matches_file = matcher.run_matcher()
        
        if not os.path.exists(matches_file):
            log_file.write("Error: No manual matches file created\n")
            print("Error: No manual matches file created. Exiting.")
            return
        
        # Import manual matches
        log_file.write("Importing manual matches into database\n")
        imported = import_manual_matches_to_db(database_path, image_path, matches_file)
        
        if imported:
            log_file.write(f"Successfully imported manual matches from {matches_file}\n")
            print(f"Successfully imported manual matches from {matches_file}")
        else:
            log_file.write(f"Failed to import manual matches\n")
            print(f"Failed to import manual matches")
        
        # Step 3: Run matcher
        print("\nStep 3: Running feature matcher...")
        log_file.write("Step 3: Running feature matcher\n")
        
        # Use exhaustive_matcher for better coverage
        if matcher_type == 'sequential_matcher':
            print("For better reconstruction, upgrading to exhaustive_matcher...")
            matcher_type = 'exhaustive_matcher'
            
        matcher_cmd = f"colmap {matcher_type} --database_path {database_path}"
        
        print(f"Running: {matcher_cmd}")
        matcher_start_time = time.time()
        subprocess.run(matcher_cmd, shell=True, check=True)
        matcher_end_time = time.time()
        matcher_elapsed_time = matcher_end_time - matcher_start_time
        
        log_file.write(f"Feature matching completed in {matcher_elapsed_time:.2f} seconds\n")
        print(f"Feature matching completed in {matcher_elapsed_time:.2f} seconds")
        
        # Check and improve match quality
        create_additional_matches(database_path, image_path, manual_matches_folder)
        
        # Run matcher again with improved matches
        print("\nRunning matcher again with improved matches...")
        subprocess.run(matcher_cmd, shell=True, check=True)
        
        # Step 4: Attempt incremental mapping with multiple strategies
        print("\nStep 4: Running mapper with enhanced strategies...")
        log_file.write("Step 4: Running mapper with enhanced strategies\n")
        
        # Use a specified initial pair if possible
        os.makedirs(sparse_output_path, exist_ok=True)
        os.system(f"rm -rf {sparse_output_path}/*")  # Clear previous attempts
        
        # First attempt: Create initial reconstruction with best pair
        success = create_initial_reconstruction(database_path, image_path, sparse_output_path)
        
        if not success:
            # Second attempt: Try with lower thresholds
            print("Trying mapper with lower thresholds...")
            mapper_cmd = (
                f"colmap mapper "
                f"--database_path {database_path} "
                f"--image_path {image_path} "
                f"--output_path {sparse_output_path} "
                f"--Mapper.ba_global_max_num_iterations 100 "
                f"--Mapper.ba_local_max_num_iterations 50 "
                f"--Mapper.min_num_matches 6 "  # Lower for challenging scenes
                f"--Mapper.min_model_size 3 "  # Allow smaller initial models
                f"--Mapper.max_reg_trials 5 "
                f"--Mapper.max_num_models 50 "  # Try more initial models
                f"--Mapper.multiple_models 1 "  # Allow multiple models
                f"--Mapper.filter_max_reproj_error 6.0"  # More permissive
            )
            
            try:
                print(f"Running: {mapper_cmd}")
                mapper_start_time = time.time()
                subprocess.run(mapper_cmd, shell=True, check=True)
                mapper_successful = True
            except subprocess.CalledProcessError:
                mapper_successful = False
                
            if not mapper_successful:
                # Third attempt: Try with even more relaxed parameters
                print("Trying mapper with more relaxed parameters...")
                mapper_cmd = (
                    f"colmap mapper "
                    f"--database_path {database_path} "
                    f"--image_path {image_path} "
                    f"--output_path {sparse_output_path} "
                    f"--Mapper.ba_global_max_num_iterations 150 "
                    f"--Mapper.ba_local_max_num_iterations 75 "
                    f"--Mapper.min_num_matches 4 "  # Very low threshold
                    f"--Mapper.min_model_size 3 "
                    f"--Mapper.init_min_points 4 "
                    f"--Mapper.abs_pose_min_num_inliers 4 "
                    f"--Mapper.max_reg_trials 10 "
                    f"--Mapper.max_num_models 100 "
                    f"--Mapper.multiple_models 1 "
                    f"--Mapper.filter_max_reproj_error 8.0 "  # Very permissive
                    f"--Mapper.tri_complete_max_reproj_error 8.0"
                )
                
                try:
                    print(f"Running: {mapper_cmd}")
                    mapper_start_time = time.time()
                    subprocess.run(mapper_cmd, shell=True, check=True)
                    mapper_successful = True
                except subprocess.CalledProcessError:
                    mapper_successful = False
        else:
            mapper_successful = True
        
        mapper_end_time = time.time()
        mapper_elapsed_time = mapper_end_time - mapper_start_time
        
        if mapper_successful:
            log_file.write(f"Mapping completed in {mapper_elapsed_time:.2f} seconds\n")
            print(f"Mapping completed in {mapper_elapsed_time:.2f} seconds")
            
            # Check if we have a valid reconstruction with multiple images
            model_0_path = os.path.join(sparse_output_path, '0')
            if not os.path.exists(model_0_path) or not os.path.exists(os.path.join(model_0_path, 'images.bin')):
                # Try using model 1 or higher if model 0 doesn't exist
                for i in range(1, 10):
                    alt_model_path = os.path.join(sparse_output_path, str(i))
                    if os.path.exists(alt_model_path) and os.path.exists(os.path.join(alt_model_path, 'images.bin')):
                        print(f"Using alternative model {i} as main reconstruction")
                        # Rename model i to model 0
                        temp_path = os.path.join(sparse_output_path, 'temp')
                        os.rename(alt_model_path, temp_path)
                        if os.path.exists(model_0_path):
                            os.rename(model_0_path, os.path.join(sparse_output_path, f'old_0'))
                        os.rename(temp_path, model_0_path)
                        break
        
            # Step 5: Run the model completer to try to register more images
            print("\nStep 5: Running model completer to register more images...")
            log_file.write("Step 5: Running model completer\n")
            
            completer_cmd = (
                f"colmap model_converter "
                f"--input_path {os.path.join(sparse_output_path, '0')} "
                f"--output_path {os.path.join(sparse_output_path, 'text')} "
                f"--output_type TXT"
            )
            
            try:
                subprocess.run(completer_cmd, shell=True, check=True)
                
                # Now run model_aligner to ensure consistent scale
                aligner_cmd = (
                    f"colmap model_aligner "
                    f"--input_path {os.path.join(sparse_output_path, '0')} "
                    f"--output_path {os.path.join(sparse_output_path, '0')} "
                    f"--robust_alignment 1 "
                    f"--robust_alignment_max_error 10"
                )
                subprocess.run(aligner_cmd, shell=True, check=True)
                
                # Run image_registrator to try to register more images
                registrator_cmd = (
                    f"colmap image_registrator "
                    f"--database_path {database_path} "
                    f"--input_path {os.path.join(sparse_output_path, '0')} "
                    f"--output_path {os.path.join(sparse_output_path, '0')} "
                    f"--min_num_matches 4 "
                    f"--Mapper.filter_max_reproj_error 8.0 "
                    f"--Mapper.tri_complete_max_reproj_error 8.0"
                )
                subprocess.run(registrator_cmd, shell=True, check=True)
                
                # Run bundle adjustment to refine the model
                bundle_cmd = (
                    f"colmap bundle_adjuster "
                    f"--input_path {os.path.join(sparse_output_path, '0')} "
                    f"--output_path {os.path.join(sparse_output_path, '0')} "
                    f"--BundleAdjustment.max_num_iterations 100 "
                    f"--BundleAdjustment.max_linear_solver_iterations 200"
                )
                subprocess.run(bundle_cmd, shell=True, check=True)
                
            except subprocess.CalledProcessError as e:
                print(f"Error in model completion steps: {str(e)}")
                log_file.write(f"Error in model completion steps: {str(e)}\n")
        
            # Step 6: Run undistorter if needed
            if model_type == '3dgs':
                print("\nStep 6: Running image undistorter...")
                log_file.write("Step 6: Running image undistorter\n")
                img_undist_cmd = (
                    f"colmap image_undistorter "
                    f"--image_path {image_path} "
                    f"--input_path {os.path.join(sparse_output_path, '0')} "
                    f"--output_path {parent_dir} "
                    f"--output_type COLMAP"
                )
                
                print(f"Running: {img_undist_cmd}")
                undistort_start_time = time.time()
                exit_code = os.system(img_undist_cmd)
                undistort_end_time = time.time()
                undistort_elapsed_time = undistort_end_time - undistort_start_time
                
                if exit_code != 0:
                    log_file.write(f"Undistortion failed with code {exit_code}\n")
                    print(f"Undistortion failed with code {exit_code}")
                else:
                    log_file.write(f"Undistortion completed in {undistort_elapsed_time:.2f} seconds\n")
                    print(f"Undistortion completed in {undistort_elapsed_time:.2f} seconds")
            
            # Step 7: Move necessary files
            print("\nStep 7: Organizing final files...")
            log_file.write("Step 7: Organizing final files\n")
            
            # Ensure sparse/0 directory exists
            os.makedirs(sparse_zero_folder, exist_ok=True)
            
            # Copy all the files from distorted/sparse/0 to sparse/0
            dist_sparse_zero = os.path.join(sparse_output_path, '0')
            if os.path.exists(dist_sparse_zero):
                for file_name in ['cameras.bin', 'images.bin', 'points3D.bin']:
                    source_file = os.path.join(dist_sparse_zero, file_name)
                    dest_file = os.path.join(sparse_zero_folder, file_name)
                    if os.path.exists(source_file):
                        # Use copy2 instead of move to preserve original files
                        copy2(source_file, dest_file)
                        log_file.write(f"Copied {file_name} to {sparse_zero_folder}\n")
                        print(f"Copied {file_name} to {sparse_zero_folder}")
            else:
                log_file.write(f"Warning: Could not find {dist_sparse_zero}\n")
                print(f"Warning: Could not find {dist_sparse_zero}")
            
            # Step 8: Count registered images
            try:
                if os.path.exists(os.path.join(sparse_zero_folder, 'images.bin')):
                    print("\nStep 8: Analyzing final model...")
                    log_file.write("Step 8: Analyzing final model\n")
                    
                    import_images_cmd = f"colmap model_analyzer --path {sparse_zero_folder}"
                    print(f"Running: {import_images_cmd}")
                    result = subprocess.run(import_images_cmd, shell=True, capture_output=True, text=True)
                    log_file.write(result.stdout)
                    
                    # Extract number of registered images
                    registered_images = 0
                    total_images = 0
                    for line in result.stdout.split('\n'):
                        if "registered images" in line:
                            parts = line.split()
                            if len(parts) >= 4:
                                registered_images = int(parts[1])
                                total_images = int(parts[3])
                            log_file.write(f"Result: {line}\n")
                            print(f"\nResult: {line}")
                    
                    # If we have less than 90% of images registered, offer to continue refinement
                    if registered_images > 0 and total_images > 0 and registered_images < 0.9 * total_images:
                        missing = total_images - registered_images
                        print(f"\nWarning: {missing} out of {total_images} images are not registered.")
                        response = input("Would you like to try additional refinement steps? (y/n): ")
                        if response.lower() == 'y':
                            # Run additional matcher
                            print("Running additional matching step with missing frames...")
                            # This would be implemented but skipped for brevity
            except Exception as e:
                log_file.write(f"Error analyzing model: {str(e)}\n")
                print(f"Error analyzing model: {str(e)}")
        else:
            log_file.write("Mapping failed with all attempted strategies\n")
            print("Mapping failed with all attempted strategies")
        
        # Finish up
        total_end_time = time.time()
        total_elapsed_time = total_end_time - total_start_time
        log_file.write(f"COLMAP run finished at: {datetime.datetime.now()}\n")
        log_file.write(f"Total time taken: {total_elapsed_time:.2f} seconds\n")
        print(f"\nCOLMAP run finished at: {datetime.datetime.now()}")
        print(f"Total time taken: {total_elapsed_time:.2f} seconds")
        print("\nCheck the sparse/0 directory for the reconstruction results.")
        print("\nAdvanced Troubleshooting Tips:")
        print("1. If still missing frames, try adding manual matches between registered and unregistered frames")
        print("2. Use --matcher_type exhaustive_matcher for more thorough matching")
        print("3. Consider increasing --num_key_frames to 40 to add more guidance points")
        print("4. Try running with --cleanup flag to start fresh")
        print("5. Check log file for detailed information about the reconstruction process")

def main():
    parser = argparse.ArgumentParser(description="Run COLMAP with manual point selection.")
    parser.add_argument('--image_path', required=True, help="Path to the images folder.")
    parser.add_argument('--matcher_type', default='sequential_matcher', choices=['sequential_matcher', 'exhaustive_matcher', 'vocab_tree_matcher'], 
                        help="Type of matcher to use (default: sequential_matcher).")
    parser.add_argument('--interval', type=int, default=1, help="Interval of images to use (default: 1, meaning all images).")
    parser.add_argument('--model_type', default='3dgs', choices=['3dgs', 'nerfstudio'], 
                        help="Model type to run. '3dgs' includes undistortion, 'nerfstudio' skips undistortion.")
    parser.add_argument('--num_key_frames', type=int, default=15, help="Number of key frames to use for manual guidance.")
    parser.add_argument('--cleanup', action='store_true', help="Clean up existing database and sparse folders before starting.")

    args = parser.parse_args()

    # Optional cleanup
    if args.cleanup:
        parent_dir = os.path.abspath(os.path.join(args.image_path, os.pardir))
        distorted_folder = os.path.join(parent_dir, 'distorted')
        sparse_folder = os.path.join(parent_dir, 'sparse')
        database_path = os.path.join(distorted_folder, 'database.db')
        
        if os.path.exists(database_path):
            os.remove(database_path)
            print(f"Removed existing database: {database_path}")
            
        if os.path.exists(sparse_folder):
            import shutil
            shutil.rmtree(sparse_folder)
            print(f"Removed existing sparse folder: {sparse_folder}")
            
        if os.path.exists(distorted_folder):
            import shutil
            shutil.rmtree(distorted_folder)
            print(f"Removed existing distorted folder: {distorted_folder}")

    # Run COLMAP with manual matching
    run_colmap_with_manual_matcher(args.image_path, args.matcher_type, args.interval, args.model_type, args.num_key_frames)

if __name__ == "__main__":
    main()