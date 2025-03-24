from tqdm import tqdm
import torch
import numpy as np
from rendering import rendering


def training(model, optimizer, scheduler, tn, tf, nb_bins, nb_epochs, data_loader, device='cpu'):
    
    training_loss = []
    for epoch in (range(nb_epochs)):
        for batch in tqdm(data_loader):
            o = batch[:, :3].to(device)
            d = batch[:, 3:6].to(device)
            
            target = batch[:, 6:].to(device)
            
            prediction = rendering(model, o, d, tn, tf, nb_bins=nb_bins, device=device)
            
            loss = ((prediction - target)**2).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss.append(loss.item())
            
        scheduler.step()
        
        torch.save(model.cpu(), 'model_nerf')
        model.to(device)
        
    return training_loss


def training_with_pose_refinement(model, optimizer, scheduler, tn, tf, nb_bins, nb_epochs, camera_data, data_loader, device='cpu'):
    """Training loop with pose refinement"""
    
    training_loss = []
    
    # Get original poses
    original_poses = camera_data['poses']
    refined_poses = original_poses.copy()
    
    # Make sure we're using the correct model for rendering
    # If model is NeRFWithPoses, use model.nerf for rendering
    rendering_model = model.nerf if hasattr(model, 'nerf') else model
    
    for epoch in range(nb_epochs):
        print(f"Epoch {epoch+1}/{nb_epochs}")
        
        # Track batch losses for this epoch
        epoch_losses = []
        
        for batch in tqdm(data_loader):
            # Extract batch data
            o = batch[:, :3].to(device)
            d = batch[:, 3:6].to(device)
            target = batch[:, 6:9].to(device)
            img_idx = batch[:, 9].to(device).long()
            
            # Render with the current model
            prediction = rendering(rendering_model, o, d, tn, tf, nb_bins=nb_bins, device=device)
            
            # Calculate loss
            loss = ((prediction - target)**2).mean()
            
            # Backprop and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Record loss
            training_loss.append(loss.item())
            epoch_losses.append(loss.item())
        
        # Step the scheduler after each epoch
        scheduler.step()
        
        # Update refined poses after each epoch (only if we have a pose refinement model)
        if hasattr(model, 'get_refined_pose'):
            for i in range(len(original_poses)):
                refined_poses[i] = model.get_refined_pose(original_poses[i], i)
        
        # Print epoch stats
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
        print(f"Epoch {epoch+1} complete. Average loss: {avg_epoch_loss:.6f}")
    
    # Final pose update (only if we have a pose refinement model)
    if hasattr(model, 'get_refined_pose'):
        for i in range(len(original_poses)):
            refined_poses[i] = model.get_refined_pose(original_poses[i], i)
    
    return training_loss, refined_poses