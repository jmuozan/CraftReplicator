#!/bin/bash

# Create necessary directories
mkdir -p data/porcelain_processed

# Save the nerf_dataset.py file
cat > nerf_dataset.py << 'EOF'
import os
import json
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

def qvec2rotmat(qvec):
    """Convert quaternion to rotation matrix"""
    w, x, y, z = qvec
    R = np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
        [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
        [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]
    ])
    return R

class NeRFDataset(Dataset):
    """
    Dataset for NeRF training that loads data from transforms JSON files.
    """
    def __init__(self, root_dir, split='train', img_wh=(800, 800), device='cpu'):
        """
        Initialize the dataset.
        
        Args:
            root_dir: Root directory containing the dataset
            split: 'train', 'val', or 'test'
            img_wh: Image width and height after resizing
            device: Device to load tensors to
        """
        self.root_dir = root_dir
        self.split = split
        self.img_wh = img_wh
        self.device = device
        
        # Load transforms file
        transform_file = os.path.join(root_dir, f'transforms_{split}.json')
        with open(transform_file, 'r') as f:
            self.transforms = json.load(f)
        
        # Extract camera parameters
        self.camera_angle_x = self.transforms.get('camera_angle_x', 0.0)
        self.focal = 0.5 * img_wh[0] / np.tan(0.5 * self.camera_angle_x)
        
        # Prepare data
        self.all_rays = []
        self.all_rgbs = []
        self.image_indices = []  # Store image index for appearance embedding
        self.image_paths = []    # Store paths for debugging
        
        # Get all rays and pixel values from the dataset
        self._generate_rays()
    
    def _generate_rays(self):
        """Generate rays for all images in the dataset."""
        print(f"Generating rays for {len(self.transforms['frames'])} images...")
        
        for i, frame in enumerate(self.transforms['frames']):
            # Load image
            image_path = os.path.join(self.root_dir, frame['file_path'] + '.png')
            self.image_paths.append(image_path)
            
            # Handle missing images
            if not os.path.exists(image_path):
                print(f"Warning: Image not found: {image_path}")
                continue
            
            # Load and resize image
            img = Image.open(image_path)
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = np.array(img) / 255.0  # Normalize to [0, 1]
            
            # Extract poses
            c2w = np.array(frame['transform_matrix'], dtype=np.float32)
            
            # Generate rays
            rays_o, rays_d = self._get_rays(c2w)
            
            # Flatten image and rays
            img = img.reshape(-1, 3)
            rays_o = rays_o.reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)
            
            # Append to dataset
            for pixel_idx in range(len(img)):
                self.all_rays.append(np.concatenate([
                    rays_o[pixel_idx],
                    rays_d[pixel_idx],
                    [i],  # Image index for appearance embedding
                    [i]   # Also use image index for transient embedding
                ]))
                self.all_rgbs.append(img[pixel_idx])
            
            print(f"Processed image {i+1}/{len(self.transforms['frames'])}: {image_path}")
        
        # Convert to tensors
        if self.all_rays:
            self.all_rays = torch.tensor(np.array(self.all_rays), dtype=torch.float32)
            self.all_rgbs = torch.tensor(np.array(self.all_rgbs), dtype=torch.float32)
            print(f"Dataset loaded with {len(self.all_rays)} rays")
        else:
            print("Warning: No rays were generated. Check your dataset.")
            self.all_rays = torch.zeros((0, 8), dtype=torch.float32)
            self.all_rgbs = torch.zeros((0, 3), dtype=torch.float32)
    
    def _get_rays(self, c2w):
        """
        Generate rays for an image with the given camera-to-world transform.
        
        Args:
            c2w: 4x4 camera-to-world transform matrix
            
        Returns:
            Tuple of rays origin and direction tensors
        """
        w, h = self.img_wh
        
        # Create meshgrid for pixel coordinates
        i, j = np.meshgrid(
            np.arange(w, dtype=np.float32),
            np.arange(h, dtype=np.float32),
            indexing='xy'
        )
        
        # Convert pixel coordinates to camera coordinates
        dirs = np.stack(
            [(i - w * 0.5) / self.focal,
             -(j - h * 0.5) / self.focal,
             -np.ones_like(i)],
            axis=-1
        )
        
        # Apply camera-to-world rotation to get ray directions
        rays_d = dirs @ c2w[:3, :3].T
        rays_d = rays_d / np.linalg.norm(rays_d, axis=-1, keepdims=True)
        
        # Get ray origins (camera center)
        rays_o = np.broadcast_to(c2w[:3, 3], rays_d.shape)
        
        return rays_o, rays_d
    
    def __len__(self):
        return len(self.all_rays)
    
    def __getitem__(self, idx):
        # Return ray data and corresponding RGB value
        ray_data = self.all_rays[idx]
        rgb = self.all_rgbs[idx]
        
        return torch.cat([ray_data, rgb])
EOF

# Now run the pipeline
echo "Step 1: Create and extract COLMAP features"
mkdir -p data/porcelain/sparse
colmap feature_extractor \
    --database_path data/porcelain/database.db \
    --image_path data/porcelain

echo "Step 2: Match features"
colmap exhaustive_matcher \
    --database_path data/porcelain/database.db

echo "Step 3: Run COLMAP mapper"
colmap mapper \
    --database_path data/porcelain/database.db \
    --image_path data/porcelain \
    --output_path data/porcelain/sparse \
    --Mapper.filter_max_reproj_error 4 \
    --Mapper.tri_min_angle 4 \
    --Mapper.min_num_matches 15

echo "Step 4: Convert COLMAP data to NeRF format"
python colmap_to_nerf_fixed.py \
    --colmap_dir data/porcelain/sparse/0 \
    --output_dir data/porcelain_nerf \
    --image_dir data/porcelain \
    --square

echo "Step 5: Check generated files"
ls -la data/porcelain_nerf/
ls -la data/porcelain_nerf/images/ | wc -l

echo "Step 6: Train the NeRF-W model"
python instant-ngp-nerfw.py \
    --data_dir data/porcelain_nerf \
    --output_dir novel_views_porcelain \
    --img_wh 800 800 \
    --epochs 20 \
    --batch_size 1024 \
    --near 2.0 \
    --far 6.0 \
    --lambda_u 0.01

echo "Pipeline complete!"