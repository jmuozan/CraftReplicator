import os
import json
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import random

def qvec2rotmat(qvec):
    """Convert quaternion to rotation matrix"""
    w, x, y, z = qvec
    R = np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
        [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
        [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]
    ])
    return R

class MemoryEfficientNeRFDataset(Dataset):
    """
    Memory-efficient dataset for NeRF training that loads images on demand.
    """
    def __init__(self, root_dir, split='train', img_wh=(800, 800), ray_sample_count=4096):
        """
        Initialize the dataset.
        
        Args:
            root_dir: Root directory containing the dataset
            split: 'train', 'val', or 'test'
            img_wh: Image width and height after resizing
            ray_sample_count: Number of random rays to sample per image (for training)
        """
        self.root_dir = root_dir
        self.split = split
        self.img_wh = img_wh
        self.ray_sample_count = ray_sample_count
        
        # Load transforms file
        transform_file = os.path.join(root_dir, f'transforms_{split}.json')
        with open(transform_file, 'r') as f:
            self.transforms = json.load(f)
        
        # Extract camera parameters
        self.camera_angle_x = self.transforms.get('camera_angle_x', 0.0)
        self.focal = 0.5 * img_wh[0] / np.tan(0.5 * self.camera_angle_x)
        
        # Prepare image paths and corresponding transforms
        self.image_paths = []
        self.poses = []
        self.image_indices = []
        
        for i, frame in enumerate(self.transforms['frames']):
            image_path = os.path.join(self.root_dir, frame['file_path'] + '.png')
            if os.path.exists(image_path):
                self.image_paths.append(image_path)
                self.poses.append(np.array(frame['transform_matrix'], dtype=np.float32))
                self.image_indices.append(i)
                
        self.n_images = len(self.image_paths)
        self.rays_per_image = img_wh[0] * img_wh[1]
        
        # For validation/test we want all rays per image
        # For training we'll randomly sample rays
        if split == 'train':
            self.length = self.n_images * ray_sample_count
        else:
            self.length = self.n_images * self.rays_per_image
            
        print(f"Dataset loaded with {self.n_images} images")
    
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
    
    def _load_image(self, image_path):
        """Load and resize an image, handling RGBA images."""
        img = Image.open(image_path)
        img = img.resize(self.img_wh, Image.LANCZOS)
        img = np.array(img) / 255.0  # Normalize to [0, 1]
        
        # Handle RGBA images
        if img.shape[-1] == 4:
            img = img[:, :, :3]  # Take only RGB channels
            
        return img
    
    def _sample_pixels(self, img, rays_o, rays_d, img_idx, sample_all=False):
        """
        Sample pixels from an image and corresponding rays.
        
        Args:
            img: Image array
            rays_o: Ray origins
            rays_d: Ray directions
            img_idx: Image index for appearance embedding
            sample_all: Whether to return all pixels (True) or random samples (False)
            
        Returns:
            Dictionary containing sampled rays and RGB values
        """
        h, w, _ = img.shape
        
        if sample_all:
            # Return all pixels (for validation/testing)
            img = img.reshape(-1, 3)
            rays_o = rays_o.reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)
            img_idx_array = np.ones((rays_o.shape[0]), dtype=np.int64) * img_idx
        else:
            # Randomly sample pixels
            pixels_count = h * w
            pixel_indices = np.random.choice(pixels_count, size=self.ray_sample_count, replace=False)
            
            img = img.reshape(-1, 3)[pixel_indices]
            rays_o = rays_o.reshape(-1, 3)[pixel_indices]
            rays_d = rays_d.reshape(-1, 3)[pixel_indices]
            img_idx_array = np.ones((rays_o.shape[0]), dtype=np.int64) * img_idx
        
        # Combine ray origins, directions, and image indices
        rays = np.concatenate([
            rays_o, 
            rays_d, 
            img_idx_array[:, None],  # appearance embedding idx
            img_idx_array[:, None]   # transient embedding idx
        ], axis=1)
        
        return {
            'rays': rays,
            'rgbs': img
        }
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        """
        Get a batch of rays and corresponding RGB values.
        
        For training: Returns a batch of random rays from a random image
        For validation/testing: Returns rays in order from each image
        """
        if self.split == 'train':
            # For training, choose a random image and sample random rays
            img_idx = idx % self.n_images
            img_path = self.image_paths[img_idx]
            pose = self.poses[img_idx]
            appearance_idx = self.image_indices[img_idx]
            
            # Load image and generate rays
            img = self._load_image(img_path)
            rays_o, rays_d = self._get_rays(pose)
            
            # Sample rays randomly
            sample_data = self._sample_pixels(img, rays_o, rays_d, appearance_idx, sample_all=False)
            
            # Get a random batch of rays
            rays_sample_idx = np.random.randint(0, sample_data['rays'].shape[0])
            rays = torch.from_numpy(sample_data['rays'][rays_sample_idx])
            rgbs = torch.from_numpy(sample_data['rgbs'][rays_sample_idx])
            
        else:
            # For validation/testing, return all rays in order
            img_idx = idx // self.rays_per_image
            pixel_idx = idx % self.rays_per_image
            
            if img_idx >= self.n_images:
                # Fallback for edge cases
                img_idx = 0
                pixel_idx = 0
            
            img_path = self.image_paths[img_idx]
            pose = self.poses[img_idx]
            appearance_idx = self.image_indices[img_idx]
            
            # Load image and generate rays
            img = self._load_image(img_path)
            rays_o, rays_d = self._get_rays(pose)
            
            # Reshape to get all pixels
            sample_data = self._sample_pixels(img, rays_o, rays_d, appearance_idx, sample_all=True)
            
            # Get specific pixel
            rays = torch.from_numpy(sample_data['rays'][pixel_idx])
            rgbs = torch.from_numpy(sample_data['rgbs'][pixel_idx])
        
        # Combine rays and RGB values
        return torch.cat([rays, rgbs])


class BatchedMemoryEfficientNeRFDataset(Dataset):
    """
    Dataset that returns batches of rays from the same image for efficient training.
    """
    def __init__(self, root_dir, split='train', img_wh=(800, 800), batch_size=1024):
        """
        Initialize the dataset.
        
        Args:
            root_dir: Root directory containing the dataset
            split: 'train', 'val', or 'test'
            img_wh: Image width and height after resizing
            batch_size: Number of rays to return per batch
        """
        self.root_dir = root_dir
        self.split = split
        self.img_wh = img_wh
        self.batch_size = batch_size
        
        # Load transforms file
        transform_file = os.path.join(root_dir, f'transforms_{split}.json')
        with open(transform_file, 'r') as f:
            self.transforms = json.load(f)
        
        # Extract camera parameters
        self.camera_angle_x = self.transforms.get('camera_angle_x', 0.0)
        self.focal = 0.5 * img_wh[0] / np.tan(0.5 * self.camera_angle_x)
        
        # Prepare image paths and corresponding transforms
        self.image_paths = []
        self.poses = []
        self.image_indices = []
        
        for i, frame in enumerate(self.transforms['frames']):
            image_path = os.path.join(self.root_dir, frame['file_path'] + '.png')
            if os.path.exists(image_path):
                self.image_paths.append(image_path)
                self.poses.append(np.array(frame['transform_matrix'], dtype=np.float32))
                self.image_indices.append(i)
                
        self.n_images = len(self.image_paths)
        print(f"Dataset loaded with {self.n_images} images")

        # Precalculate dataset length based on number of batches
        if split == 'train':
            # For training, we use a fixed number of iterations per epoch
            # Calculate how many batches per epoch (using each image once)
            self.rays_per_image = img_wh[0] * img_wh[1]
            self.batches_per_image = max(1, self.rays_per_image // batch_size)
            self.length = self.n_images * self.batches_per_image
        else:
            # For validation/testing, we go through all pixels
            self.rays_per_image = img_wh[0] * img_wh[1]
            self.batches_per_image = (self.rays_per_image + batch_size - 1) // batch_size
            self.length = self.n_images * self.batches_per_image
    
    def _get_rays(self, c2w):
        """Generate rays for an image with the given camera-to-world transform."""
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
    
    def _load_image(self, image_path):
        """Load and resize an image, handling RGBA images."""
        img = Image.open(image_path)
        img = img.resize(self.img_wh, Image.LANCZOS)
        img = np.array(img) / 255.0  # Normalize to [0, 1]
        
        # Handle RGBA images
        if img.shape[-1] == 4:
            img = img[:, :, :3]  # Take only RGB channels
            
        return img
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        """Get a batch of rays from a single image."""
        if self.split == 'train':
            # For training, select a random image for each batch
            img_idx = idx % self.n_images
            batch_idx = idx // self.n_images
            
            # Load image and generate rays
            img_path = self.image_paths[img_idx]
            pose = self.poses[img_idx]
            appearance_idx = self.image_indices[img_idx]
            
            img = self._load_image(img_path)
            rays_o, rays_d = self._get_rays(pose)
            
            # Reshape to a list of pixels
            img = img.reshape(-1, 3)
            rays_o = rays_o.reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)
            
            # For random sampling during training
            total_pixels = img.shape[0]
            
            # Sample batch_size rays randomly
            if self.batch_size < total_pixels:
                indices = np.random.choice(total_pixels, size=self.batch_size, replace=False)
                img = img[indices]
                rays_o = rays_o[indices]
                rays_d = rays_d[indices]
            
            # Create indices for appearance and transient embeddings
            appearance_indices = np.ones(rays_o.shape[0], dtype=np.int64) * appearance_idx
            transient_indices = np.ones(rays_o.shape[0], dtype=np.int64) * appearance_idx
            
            # Combine ray origins, directions, indices, and RGB values
            all_rays = np.concatenate([
                rays_o,
                rays_d,
                appearance_indices[:, None],
                transient_indices[:, None],
                img
            ], axis=1)
            
            return torch.from_numpy(all_rays).float()
            
        else:
            # For validation/testing, go through images sequentially
            img_idx = idx // self.batches_per_image
            batch_idx = idx % self.batches_per_image
            
            if img_idx >= self.n_images:
                img_idx = 0
                batch_idx = 0
            
            # Load image and generate rays
            img_path = self.image_paths[img_idx]
            pose = self.poses[img_idx]
            appearance_idx = self.image_indices[img_idx]
            
            img = self._load_image(img_path)
            rays_o, rays_d = self._get_rays(pose)
            
            # Reshape to a list of pixels
            img = img.reshape(-1, 3)
            rays_o = rays_o.reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)
            
            # Get batch of pixels
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, self.rays_per_image)
            
            # Extract batch
            img_batch = img[start_idx:end_idx]
            rays_o_batch = rays_o[start_idx:end_idx]
            rays_d_batch = rays_d[start_idx:end_idx]
            
            # Create indices for appearance and transient embeddings
            appearance_indices = np.ones(rays_o_batch.shape[0], dtype=np.int64) * appearance_idx
            transient_indices = np.ones(rays_o_batch.shape[0], dtype=np.int64) * appearance_idx
            
            # Combine ray origins, directions, indices, and RGB values
            all_rays = np.concatenate([
                rays_o_batch,
                rays_d_batch,
                appearance_indices[:, None],
                transient_indices[:, None],
                img_batch
            ], axis=1)
            
            return torch.from_numpy(all_rays).float()