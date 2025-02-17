# training_script.py
import torch
from torch.utils.data import DataLoader
from dataset import get_rays
from model import Nerf
from ml_helpers import train_model, plot_training_loss

def setup_training():
    # Check for MPS availability
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Hyperparameters
    batch_size = 1024
    tn = 8.
    tf = 12.
    nb_epochs = 10
    lr = 1e-3
    gamma = .5
    nb_bins = 100

    # Load data
    o, d, target_px_values = get_rays('fox', mode='train')
    
    # Create dataloaders
    dataloader = DataLoader(torch.cat((
        torch.from_numpy(o).reshape(-1, 3).type(torch.float),
        torch.from_numpy(d).reshape(-1, 3).type(torch.float),
        torch.from_numpy(target_px_values).reshape(-1, 3).type(torch.float)
    ), dim=1), batch_size=batch_size, shuffle=True)

    dataloader_warmup = DataLoader(torch.cat((
        torch.from_numpy(o).reshape(90, 400, 400, 3)[:, 100:300, 100:300, :].reshape(-1, 3).type(torch.float),
        torch.from_numpy(d).reshape(90, 400, 400, 3)[:, 100:300, 100:300, :].reshape(-1, 3).type(torch.float),
        torch.from_numpy(target_px_values).reshape(90, 400, 400, 3)[:, 100:300, 100:300, :].reshape(-1, 3).type(torch.float)
    ), dim=1), batch_size=batch_size, shuffle=True)

    # Initialize model and optimizer
    model = Nerf(hidden_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10], gamma=gamma)

    return {
        'model': model,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'dataloader': dataloader,
        'dataloader_warmup': dataloader_warmup,
        'device': device,
        'tn': tn,
        'tf': tf,
        'nb_bins': nb_bins,
        'nb_epochs': nb_epochs
    }

def run_training_with_warmup(training_setup):
    # Warmup phase
    print("Starting warm-up phase...")
    warmup_loss = train_model(
        model=training_setup['model'],
        optimizer=training_setup['optimizer'],
        scheduler=training_setup['scheduler'],
        dataloader=training_setup['dataloader_warmup'],
        tn=training_setup['tn'],
        tf=training_setup['tf'],
        nb_bins=training_setup['nb_bins'],
        nb_epochs=1,
        device=training_setup['device']
    )
    plot_training_loss(warmup_loss, phase='warmup')
    
    # Main training phase
    print("\nStarting main training phase...")
    main_loss = train_model(
        model=training_setup['model'],
        optimizer=training_setup['optimizer'],
        scheduler=training_setup['scheduler'],
        dataloader=training_setup['dataloader'],
        tn=training_setup['tn'],
        tf=training_setup['tf'],
        nb_bins=training_setup['nb_bins'],
        nb_epochs=training_setup['nb_epochs'],
        device=training_setup['device']
    )
    plot_training_loss(main_loss, phase='main')
    
    return warmup_loss, main_loss