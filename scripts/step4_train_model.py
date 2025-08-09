#!/usr/bin/env python3
"""
Step 4: Train SIAT Model

This script trains your SIAT model using the preprocessed dataset.
Run this after step3_test_compatibility.py passes successfully.

Features:
- Automatic train/validation split
- GPU acceleration if available
- Progress monitoring with metrics
- Model checkpointing
- Early stopping option
"""

import sys
import os
import time
import argparse
import glob
import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append('src')

from models import SIAT
from data import TrajectoryDataset, collate_fn
from config import Config
from training import train_one_epoch, evaluate


def setup_training_args():
    """Parse command line arguments for training."""
    parser = argparse.ArgumentParser(description='Train SIAT model for trajectory prediction')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='./data_npz',
                       help='Directory containing preprocessed npz files')
    parser.add_argument('--train_split', type=float, default=0.8,
                       help='Fraction of data for training')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Fraction of data for validation')
    
    # Model arguments
    parser.add_argument('--obs_len', type=int, default=8,
                       help='Number of observed timesteps')
    parser.add_argument('--pred_len', type=int, default=12,
                       help='Number of predicted timesteps')
    parser.add_argument('--embed_size', type=int, default=64,
                       help='Embedding dimension')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                       help='Weight decay for optimizer')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                       help='Gradient clipping threshold')
    
    # Device arguments
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--num_workers', type=int, default=2,
                       help='Number of dataloader workers')
    
    # Checkpoint arguments
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='Directory to save model checkpoints')
    parser.add_argument('--save_every', type=int, default=10,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    # Monitoring arguments
    parser.add_argument('--early_stop', type=int, default=15,
                       help='Early stopping patience (0 to disable)')
    parser.add_argument('--log_every', type=int, default=5,
                       help='Log progress every N epochs')
    
    return parser.parse_args()


def setup_device(device_arg):
    """Setup training device."""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    
    print(f"🖥️  Using device: {device}")
    
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    return device


def create_datasets(args):
    """Create training and validation datasets."""
    print("\n📊 Setting up datasets...")
    
    # Find npz files
    npz_files = glob.glob(f"{args.data_dir}/*.npz")
    
    if len(npz_files) == 0:
        raise RuntimeError(
            f'No .npz files found in {args.data_dir}. '
            'Please run: python scripts/step2_preprocess_data.py'
        )
    
    print(f"   Found {len(npz_files)} npz files")
    
    # Create full dataset
    full_dataset = TrajectoryDataset(
        npz_files,
        obs_len=args.obs_len,
        pred_len=args.pred_len
    )
    
    print(f"   Total samples: {len(full_dataset)}")
    
    # Split dataset
    total_size = len(full_dataset)
    train_size = int(args.train_split * total_size)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader


def create_model(args, device):
    """Create and initialize SIAT model."""
    print("\n🤖 Setting up model...")
    
    model = SIAT(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embed_size=args.embed_size
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   Model: SIAT")
    print(f"   Parameters: {total_params:,} total, {trainable_params:,} trainable")
    print(f"   Model size: ~{total_params * 4 / 1e6:.1f} MB")
    
    return model


def create_optimizer(model, args):
    """Create optimizer and scheduler."""
    print("\n⚙️  Setting up optimizer...")
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=5,
        factor=0.5,
        verbose=True
    )
    
    print(f"   Optimizer: Adam (lr={args.lr}, weight_decay={args.weight_decay})")
    print(f"   Scheduler: ReduceLROnPlateau")
    
    return optimizer, scheduler


def save_checkpoint(model, optimizer, epoch, best_ade, args, filename=None):
    """Save model checkpoint."""
    if filename is None:
        filename = f"siat_epoch_{epoch:03d}.pth"
    
    checkpoint_path = os.path.join(args.checkpoint_dir, filename)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_ade': best_ade,
        'args': args
    }, checkpoint_path)
    
    return checkpoint_path


def load_checkpoint(checkpoint_path, model, optimizer):
    """Load model checkpoint."""
    print(f"📂 Loading checkpoint from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    best_ade = checkpoint['best_ade']
    
    print(f"   Resumed from epoch {checkpoint['epoch']}")
    print(f"   Best ADE so far: {best_ade:.4f}")
    
    return start_epoch, best_ade


def format_time(seconds):
    """Format time in human readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def main():
    """Main training function."""
    args = setup_training_args()
    
    print("🚀 SIAT Model Training")
    print("=" * 50)
    print(f"Data directory: {args.data_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Observation length: {args.obs_len}")
    print(f"Prediction length: {args.pred_len}")
    
    # Setup
    device = setup_device(args.device)
    train_loader, val_loader = create_datasets(args)
    model = create_model(args, device)
    optimizer, scheduler = create_optimizer(model, args)
    
    # Resume from checkpoint if specified
    start_epoch = 1
    best_ade = float('inf')
    best_epoch = 0
    
    if args.resume:
        start_epoch, best_ade = load_checkpoint(args.resume, model, optimizer)
        best_epoch = start_epoch - 1
    
    # Training setup
    print(f"\n🏋️  Starting training from epoch {start_epoch}...")
    train_start_time = time.time()
    
    # Training history
    train_losses = []
    val_ades = []
    val_fdes = []
    
    # Early stopping
    patience_counter = 0
    
    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start_time = time.time()
        
        # Training
        train_loss = train_one_epoch(model, optimizer, train_loader, device, args.grad_clip)
        train_losses.append(train_loss)
        
        # Validation
        val_ade, val_fde = evaluate(model, val_loader, device)
        val_ades.append(val_ade)
        val_fdes.append(val_fde)
        
        # Learning rate scheduling
        scheduler.step(val_ade)
        
        # Timing
        epoch_time = time.time() - epoch_start_time
        elapsed_time = time.time() - train_start_time
        
        # Logging
        if epoch % args.log_every == 0 or epoch == 1:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:3d}/{args.epochs} | "
                  f"Loss {train_loss:.6f} | "
                  f"ADE {val_ade:.4f} | "
                  f"FDE {val_fde:.4f} | "
                  f"LR {current_lr:.2e} | "
                  f"Time {format_time(epoch_time)} | "
                  f"Elapsed {format_time(elapsed_time)}")
        
        # Save best model
        if val_ade < best_ade:
            best_ade = val_ade
            best_epoch = epoch
            patience_counter = 0
            
            best_path = save_checkpoint(model, optimizer, epoch, best_ade, args, "best_model.pth")
            print(f"💾 New best model saved! ADE: {best_ade:.4f}")
        else:
            patience_counter += 1
        
        # Regular checkpointing
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(model, optimizer, epoch, best_ade, args)
        
        # Early stopping
        if args.early_stop > 0 and patience_counter >= args.early_stop:
            print(f"\n⏰ Early stopping triggered after {patience_counter} epochs without improvement")
            break
    
    # Training completed
    total_time = time.time() - train_start_time
    print(f"\n🎉 Training completed!")
    print(f"Total time: {format_time(total_time)}")
    print(f"Best ADE: {best_ade:.4f} (epoch {best_epoch})")
    
    # Save final model
    final_path = save_checkpoint(model, optimizer, epoch, best_ade, args, "final_model.pth")
    
    # Summary
    print(f"\n📋 TRAINING SUMMARY")
    print("=" * 50)
    print(f"✅ Epochs completed: {epoch - start_epoch + 1}")
    print(f"✅ Best validation ADE: {best_ade:.4f}")
    print(f"✅ Best model saved: {os.path.join(args.checkpoint_dir, 'best_model.pth')}")
    print(f"✅ Final model saved: {final_path}")
    
    # Plot training curves if matplotlib available
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss curve
        ax1.plot(range(1, len(train_losses) + 1), train_losses)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Training Loss')
        ax1.set_title('Training Loss')
        ax1.grid(True)
        
        # Metrics curve
        ax2.plot(range(1, len(val_ades) + 1), val_ades, label='ADE', marker='o', markersize=3)
        ax2.plot(range(1, len(val_fdes) + 1), val_fdes, label='FDE', marker='s', markersize=3)
        ax2.axvline(x=best_epoch, color='r', linestyle='--', alpha=0.7, label='Best')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Error')
        ax2.set_title('Validation Metrics')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.checkpoint_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
        print(f"📊 Training curves saved: {os.path.join(args.checkpoint_dir, 'training_curves.png')}")
        
    except ImportError:
        print("📊 Install matplotlib to see training curves: pip install matplotlib")
    
    print(f"\n🎯 Next Steps:")
    print(f"   - Evaluate model: python evaluate.py --checkpoint {os.path.join(args.checkpoint_dir, 'best_model.pth')}")
    print(f"   - Test on new data: Use the saved model for inference")
    
    return True


if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print(f"\n⏹️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
