"""
Main training script for SIAT model.

Usage:
    python train.py --data_dir ./data_npz --epochs 50 --batch_size 32
"""

import argparse
import glob
import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.models import SIAT
from src.data import TrajectoryDataset
from src.training import train_one_epoch, evaluate


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train SIAT model for trajectory prediction')
    parser.add_argument('--data_dir', type=str, default='./data_npz', 
                       help='Directory containing preprocessed npz files')
    parser.add_argument('--epochs', type=int, default=50, 
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, 
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, 
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='auto', 
                       help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--obs_len', type=int, default=8, 
                       help='Number of observed timesteps')
    parser.add_argument('--pred_len', type=int, default=12, 
                       help='Number of predicted timesteps')
    parser.add_argument('--model_save_path', type=str, default='./checkpoints/siat_model.pth',
                       help='Path to save the trained model')
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Initialize configuration
    config = Config()
    
    # Update config with command line arguments
    config.data.data_dir = args.data_dir
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.lr
    config.data.obs_len = args.obs_len
    config.data.pred_len = args.pred_len
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    config.training.device = str(device)
    
    print(f"Using device: {device}")
    print(f"Configuration: {config}")
    
    # Prepare dataset
    npz_files = glob.glob(f"{config.data.data_dir}/*.npz")
    if len(npz_files) == 0:
        raise RuntimeError(
            f'No .npz files found in {config.data.data_dir}. '
            'Please preprocess ETH/UCY into scene .npz files.'
        )
    
    print(f"Found {len(npz_files)} npz files for training")
    
    # Create dataset and dataloader
    dataset = TrajectoryDataset(
        npz_files, 
        obs_len=config.data.obs_len, 
        pred_len=config.data.pred_len
    )
    loader = DataLoader(
        dataset, 
        batch_size=config.training.batch_size, 
        shuffle=True, 
        drop_last=False
    )
    
    print(f"Dataset size: {len(dataset)} samples")
    
    # Initialize model
    model = SIAT(
        obs_len=config.model.obs_len,
        pred_len=config.model.pred_len,
        in_size=config.model.in_size,
        embed_size=config.model.embed_size,
        enc_layers=config.model.enc_layers,
        dec_layers=config.model.dec_layers,
        nhead=config.model.nhead,
        gcn_hidden=config.model.gcn_hidden,
        gcn_layers=config.model.gcn_layers,
        dropout=config.model.dropout
    ).to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    print("Starting training...")
    
    # Training loop
    best_ade = float('inf')
    for epoch in range(1, config.training.epochs + 1):
        # Train
        loss = train_one_epoch(model, optimizer, loader, device, config.training.grad_clip)
        
        # Evaluate
        ade, fde = evaluate(model, loader, device)
        
        print(f'Epoch {epoch:03d} | Loss {loss:.6f} | ADE {ade:.4f} | FDE {fde:.4f}')
        
        # Save best model
        if ade < best_ade:
            best_ade = ade
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ade': ade,
                'fde': fde,
                'config': config
            }, args.model_save_path)
            print(f'New best model saved with ADE: {ade:.4f}')
    
    print(f"Training completed! Best ADE: {best_ade:.4f}")
    print(f"Model saved to: {args.model_save_path}")


if __name__ == '__main__':
    main()
