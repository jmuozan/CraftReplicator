import torch
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pickle

# Import the PhototourismDataset
from phototourism_dataset import PhototourismDataset

class PhototourismAdapterForNGPNeRFW:
    """
    Adapter to convert PhototourismDataset to the format needed for NGP-NeRF-W.
    """
    def __init__(self, root_dir, img_downscale=4, val_num=1, use_cache=True):
        """
        Initialize the adapter.
        
        Args:
            root_dir: Root directory of the Phototourism dataset
            img_downscale: Image downscale factor
            val_num: Number of validation images
            use_cache: Whether to use cached data
        """
        self.root_dir = root_dir
        self.img_downscale = img_downscale
        self.val_num = val_num
        self.use_cache = use_cache
        
        # Create output directory
        self.output_dir = os.path.join(root_dir, 'ngp_nerfw_data')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load PhototourismDataset
        self.dataset = PhototourismDataset(
            root_dir=root_dir,
            split='train',
            img_downscale=img_downscale,
            val_num=val_num,
            use_cache=use_cache
        )
        
        print(f"Loaded PhototourismDataset with {len(self.dataset)} rays")
        print(f"N_images_train: {self.dataset.N_images_train}")
        print(f"N_images_test: {self.dataset.N_images_test}")
    
    def prepare_training_data(self, batch_size=100000):
        """
        Prepare training data for NGP-NeRF-W.
        
        Args:
            batch_size: Batch size for processing rays
        
        Returns:
            Path to the saved training data
        """
        print("Preparing training data...")
        
        # Total number of rays in the dataset
        total_rays = len(self.dataset)
        
        # Initialize arrays for the processed data
        # Format: [ray_origin(3), ray_direction(3), appearance_idx(1), transient_idx(1), rgb(3)]
        all_processed_data = []
        
        for start_idx in tqdm(range(0, total_rays, batch_size)):
            end_idx = min(start_idx + batch_size, total_rays)
            
            # Get batch of rays
            batch_rays = []
            batch_ts = []
            batch_rgbs = []
            
            for idx in range(start_idx, end_idx):
                sample = self.dataset[idx]
                batch_rays.append(sample['rays'])
                batch_ts.append(sample['ts'])
                batch_rgbs.append(sample['rgbs'])
            
            # Convert to tensors
            batch_rays = torch.stack(batch_rays)
            batch_ts = torch.stack(batch_ts)
            batch_rgbs = torch.stack(batch_rgbs)
            
            # Extract ray origins and directions
            ray_origins = batch_rays[:, :3].numpy()
            ray_directions = batch_rays[:, 3:6].numpy()
            
            # Get appearance and transient indices
            # In NeRF-W, each image has its own appearance embedding
            # For transient objects, we can use the same index for simplicity
            appearance_indices = batch_ts.numpy().reshape(-1, 1)
            transient_indices = batch_ts.numpy().reshape(-1, 1)
            
            # Get RGB values
            rgbs = batch_rgbs.numpy()
            
            # Combine all data
            batch_processed_data = np.concatenate([
                ray_origins,
                ray_directions,
                appearance_indices,
                transient_indices,
                rgbs
            ], axis=1)
            
            all_processed_data.append(batch_processed_data)
        
        # Concatenate all batches
        all_processed_data = np.concatenate(all_processed_data, axis=0)
        
        # Shuffle the data
        np.random.shuffle(all_processed_data)
        
        # Save the processed data
        output_path = os.path.join(self.output_dir, 'training_data.npy')
        np.save(output_path, all_processed_data)
        
        print(f"Saved training data to {output_path}")
        print(f"Training data shape: {all_processed_data.shape}")
        
        return output_path
    
    def prepare_testing_data(self):
        """
        Prepare testing data for NGP-NeRF-W from validation set.
        
        Returns:
            Path to the saved testing data
        """
        print("Preparing testing data...")
        
        # Load validation dataset
        val_dataset = PhototourismDataset(
            root_dir=self.root_dir,
            split='test_train',  # Use test_train split to get all training images for testing
            img_downscale=self.img_downscale,
            val_num=self.val_num,
            use_cache=self.use_cache
        )
        
        # Initialize list for the processed data
        all_processed_data = []
        
        # Process each image
        for idx in tqdm(range(len(val_dataset))):
            sample = val_dataset[idx]
            
            # Extract rays
            rays = sample['rays']
            ray_origins = rays[:, :3].numpy()
            ray_directions = rays[:, 3:6].numpy()
            
            # Get appearance and transient indices
            # Use the same indices as in training
            ts = sample['ts'].numpy()
            appearance_indices = ts.reshape(-1, 1)
            transient_indices = ts.reshape(-1, 1)
            
            # Get RGB values
            rgbs = sample['rgbs'].numpy()
            
            # Combine all data
            processed_data = np.concatenate([
                ray_origins,
                ray_directions,
                appearance_indices,
                transient_indices,
                rgbs
            ], axis=1)
            
            all_processed_data.append(processed_data)
        
        # Concatenate all images
        all_processed_data = np.concatenate(all_processed_data, axis=0)
        
        # Save the processed data
        output_path = os.path.join(self.output_dir, 'testing_data.npy')
        np.save(output_path, all_processed_data)
        
        print(f"Saved testing data to {output_path}")
        print(f"Testing data shape: {all_processed_data.shape}")
        
        return output_path
    
    def save_metadata(self):
        """
        Save metadata for the dataset.
        
        Returns:
            Path to the saved metadata
        """
        print("Saving metadata...")
        
        metadata = {
            'img_downscale': self.img_downscale,
            'val_num': self.val_num,
            'use_cache': self.use_cache,
            'N_images_train': self.dataset.N_images_train,
            'N_images_test': self.dataset.N_images_test,
            'img_ids_train': self.dataset.img_ids_train,
            'img_ids_test': self.dataset.img_ids_test,
            'poses_dict': {k: v.tolist() for k, v in self.dataset.poses_dict.items()},
            'N_appearance': len(self.dataset.img_ids),  # Number of appearance embeddings
            'N_transient': len(self.dataset.img_ids),   # Number of transient embeddings
        }
        
        # Save metadata
        output_path = os.path.join(self.output_dir, 'metadata.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"Saved metadata to {output_path}")
        
        return output_path

    def convert_npy_to_pkl(self):
        """
        Convert .npy files to .pkl files for compatibility with the original model.
        
        Returns:
            Paths to the converted files
        """
        print("Converting .npy files to .pkl files...")
        
        # Convert training data
        training_data_npy = os.path.join(self.output_dir, 'training_data.npy')
        training_data_pkl = os.path.join(self.output_dir, 'training_data.pkl')
        
        training_data = np.load(training_data_npy)
        with open(training_data_pkl, 'wb') as f:
            pickle.dump(training_data, f)
        
        # Convert testing data
        testing_data_npy = os.path.join(self.output_dir, 'testing_data.npy')
        testing_data_pkl = os.path.join(self.output_dir, 'testing_data.pkl')
        
        testing_data = np.load(testing_data_npy)
        with open(testing_data_pkl, 'wb') as f:
            pickle.dump(testing_data, f)
        
        print(f"Converted files to {training_data_pkl} and {testing_data_pkl}")
        
        return training_data_pkl, testing_data_pkl

    def prepare_data(self):
        """
        Full data preparation pipeline.
        
        Returns:
            Paths to the saved data files
        """
        training_path = self.prepare_training_data()
        testing_path = self.prepare_testing_data()
        metadata_path = self.save_metadata()
        pkl_paths = self.convert_npy_to_pkl()
        
        return {
            'training_npy': training_path,
            'testing_npy': testing_path,
            'metadata': metadata_path,
            'training_pkl': pkl_paths[0],
            'testing_pkl': pkl_paths[1]
        }

def optimize_memory_batch_process(dataset, batch_size=10000):
    """
    Process the dataset in batches to avoid memory issues.
    
    Args:
        dataset: PhototourismDataset
        batch_size: Batch size for processing
    
    Returns:
        Processed data
    """
    # Total number of rays
    total_rays = len(dataset)
    all_processed_data = []
    
    # Process in batches
    for start_idx in tqdm(range(0, total_rays, batch_size)):
        end_idx = min(start_idx + batch_size, total_rays)
        batch_data = []
        
        for idx in range(start_idx, end_idx):
            sample = dataset[idx]
            rays = sample['rays']
            ts = sample['ts']
            rgbs = sample['rgbs']
            
            # Format: [ray_origin(3), ray_direction(3), appearance_idx(1), transient_idx(1), rgb(3)]
            data = np.concatenate([
                rays[:3].numpy(),          # ray_origin
                rays[3:6].numpy(),         # ray_direction
                np.array([ts.item()]),     # appearance_idx
                np.array([ts.item()]),     # transient_idx (same as appearance for simplicity)
                rgbs.numpy()               # rgb
            ])
            
            batch_data.append(data)
        
        batch_data = np.stack(batch_data)
        all_processed_data.append(batch_data)
        
        # Clear memory
        del batch_data
        
    # Concatenate all batches
    all_processed_data = np.concatenate(all_processed_data, axis=0)
    
    return all_processed_data

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare Phototourism data for NGP-NeRF-W')
    parser.add_argument('--root_dir', type=str, required=True, 
                        help='Root directory of the Phototourism dataset')
    parser.add_argument('--img_downscale', type=int, default=4,
                        help='Image downscale factor')
    parser.add_argument('--val_num', type=int, default=1,
                        help='Number of validation images')
    parser.add_argument('--use_cache', action='store_true',
                        help='Whether to use cached data')
    parser.add_argument('--batch_size', type=int, default=100000,
                        help='Batch size for processing rays')
    
    args = parser.parse_args()
    
    # Create adapter
    adapter = PhototourismAdapterForNGPNeRFW(
        root_dir=args.root_dir,
        img_downscale=args.img_downscale,
        val_num=args.val_num,
        use_cache=args.use_cache
    )
    
    # Prepare data
    output_paths = adapter.prepare_data()
    
    print("Data preparation complete!")
    print(f"Output files: {output_paths}")