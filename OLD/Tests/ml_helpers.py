# ml_helpers.py
from tqdm import tqdm
from rendering import rendering
import torch
import matplotlib.pyplot as plt

def train_model(model, optimizer, scheduler, dataloader, tn, tf, nb_bins, nb_epochs, device='mps'):
    training_loss = []
    
    for epoch in range(nb_epochs):
        epoch_loss = []
        for batch in tqdm(dataloader, desc=f'Epoch {epoch+1}/{nb_epochs}'):
            # Move batch to device
            o = batch[:, :3].to(device)
            d = batch[:, 3:6].to(device)
            target = batch[:, 6:].to(device)
            
            prediction = rendering(model, o, d, tn, tf, nb_bins=nb_bins, device=device)
            loss = ((prediction - target)**2).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss.append(loss.item())
        
        avg_loss = sum(epoch_loss) / len(epoch_loss)
        training_loss.append(avg_loss)
        print(f"Epoch {epoch+1}/{nb_epochs}, Loss: {avg_loss:.4f}")
        
        scheduler.step()
        
        # Save model checkpoints to CPU
        torch.save(model.cpu().state_dict(), f'model_nerf_epoch_{epoch+1}.pt')
        model.to(device)
    
    return training_loss

def plot_training_loss(loss_data, phase='full'):
    """
    Plot training loss with support for different training phases
    Args:
        loss_data: List of loss values or tuple of (warmup_loss, main_loss)
        phase: 'warmup', 'main', or 'full' (for both phases)
    """
    plt.figure(figsize=(12, 6))
    
    if isinstance(loss_data, tuple) and phase == 'full':
        warmup_loss, main_loss = loss_data
        # Plot warmup phase
        plt.plot(range(len(warmup_loss)), warmup_loss, 
                color='orange', label='Warm-up Phase')
        # Plot main training phase
        plt.plot(range(len(warmup_loss), len(warmup_loss) + len(main_loss)), 
                main_loss, color='blue', label='Main Training')
        plt.axvline(x=len(warmup_loss), color='r', linestyle='--', 
                   alpha=0.3, label='Phase Transition')
        plt.legend()
        plt.title('Complete Training Loss (Warm-up + Main Training)')
    else:
        # Single phase plotting
        loss_data = loss_data[0] if isinstance(loss_data, tuple) else loss_data
        plt.plot(loss_data, color='blue')
        plt.title(f'{phase.capitalize()} Phase Training Loss')
    
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.show()

# Enhanced training function with warmup support
def training_with_warmup(model, optimizer, scheduler, tn, tf, nb_bins, device='mps'):
    """
    Complete training pipeline with warmup and full training phases
    """
    print("Starting warm-up phase...")
    # Warmup training
    warmup_loss = train_model(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloader=dataloader_warmup,
        tn=tn,
        tf=tf,
        nb_bins=nb_bins,
        nb_epochs=1,
        device=device
    )
    plot_training_loss(warmup_loss)
    
    print("\nStarting main training phase...")
    # Main training
    main_loss = train_model(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloader=dataloader,
        tn=tn,
        tf=tf,
        nb_bins=nb_bins,
        nb_epochs=nb_epochs,
        device=device
    )
    plot_training_loss(main_loss)
    
    return warmup_loss, main_loss

# Keep the original training function for backwards compatibility
def training(model, optimizer, scheduler, tn, tf, nb_bins, nb_epochs, data_loader, device='mps'):
    print("Warning: This is the legacy training function. Consider using train_model instead.")
    
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
        
        torch.save(model.cpu().state_dict(), f'model_nerf_epoch_{epoch+1}.pt')
        model.to(device)
        
    return training_loss