"""
Main training entry point for SIAT (Social Interaction-Aware Transformer).

Usage:
    python train.py --data_dir ./data_npz --epochs 50 --batch_size 32
"""

import os
import sys
import argparse
import glob
import torch
from torch.utils.data import DataLoader, random_split

# Add project root and src to path
sys.path.insert(0, os.path.abspath('src'))

from models import SIAT
from data import TrajectoryDataset, collate_fn
from training import train_one_epoch, evaluate
from config import Config


def parse_args():
    parser = argparse.ArgumentParser(description='Train SIAT model for pedestrian trajectory prediction')
    
    # Dataset args
    parser.add_argument('--data_dir', type=str, default='./data_npz',
                        help='Directory containing preprocessed .npz files')
    parser.add_argument('--train_split', type=float, default=0.8,
                        help='Fraction of data used for training')
    
    # Model architecture args
    parser.add_argument('--obs_len', type=int, default=8,
                        help='Number of observed timesteps')
    parser.add_argument('--pred_len', type=int, default=12,
                        help='Number of predicted timesteps')
    parser.add_argument('--embed_size', type=int, default=64,
                        help='Transformer embedding dimension')
    parser.add_argument('--enc_layers', type=int, default=2,
                        help='Number of Transformer encoder layers')
    parser.add_argument('--dec_layers', type=int, default=1,
                        help='Number of Transformer decoder layers')
    parser.add_argument('--nhead', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--gcn_hidden', type=int, default=64,
                        help='GCN hidden dimension')
    parser.add_argument('--gcn_layers', type=int, default=2,
                        help='Number of GCN layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping max norm')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cuda, mps, cpu)')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory to save model checkpoints')
    
    return parser.parse_args()


def get_device(device_arg: str) -> torch.device:
    if device_arg == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    return torch.device(device_arg)


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Check for npz files
    npz_files = glob.glob(os.path.join(args.data_dir, '*.npz'))
    if not npz_files:
        print(f"Error: No .npz files found in {args.data_dir}.")
        print("Please download and preprocess ETH/UCY data first:")
        print("  bash scripts/step0_download_data.sh")
        print("  python scripts/step2_preprocess_data.py")
        sys.exit(1)

    print(f"Found {len(npz_files)} .npz files in {args.data_dir}")

    # Prepare datasets
    full_dataset = TrajectoryDataset(
        npz_files=npz_files,
        obs_len=args.obs_len,
        pred_len=args.pred_len
    )
    total_samples = len(full_dataset)
    train_size = int(args.train_split * total_samples)
    val_size = total_samples - train_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"Total samples: {total_samples} (Train: {train_size}, Val: {val_size})")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # Initialize SIAT Model
    model = SIAT(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embed_size=args.embed_size,
        enc_layers=args.enc_layers,
        dec_layers=args.dec_layers,
        nhead=args.nhead,
        gcn_hidden=args.gcn_hidden,
        gcn_layers=args.gcn_layers,
        dropout=args.dropout
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Initialized SIAT model with {total_params:,} trainable parameters.")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=5,
        factor=0.5
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_ade = float('inf')
    best_ckpt_path = os.path.join(args.checkpoint_dir, 'best_model.pth')

    print("\nStarting training loop...")
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, optimizer, train_loader, device, clip=args.grad_clip)
        val_ade, val_fde = evaluate(model, val_loader, device)
        scheduler.step(val_ade)

        print(f"Epoch {epoch:03d}/{args.epochs:03d} | Train Loss: {loss:.6f} | Val ADE: {val_ade:.4f} | Val FDE: {val_fde:.4f}")

        if val_ade < best_ade:
            best_ade = val_ade
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ade': val_ade,
                'fde': val_fde,
                'args': vars(args)
            }, best_ckpt_path)
            print(f"  --> Saved new best checkpoint with ADE {best_ade:.4f} to {best_ckpt_path}")

    print(f"\nTraining completed. Best Validation ADE: {best_ade:.4f}")


if __name__ == '__main__':
    main()
