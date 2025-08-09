"""
Evaluation script for trained SIAT model.

Usage:
    python evaluate.py --model_path ./checkpoints/siat_model.pth --data_dir ./data_npz
"""

import argparse
import glob
import torch
from torch.utils.data import DataLoader

from src.models import SIAT
from src.data import TrajectoryDataset
from src.training import evaluate


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate trained SIAT model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model checkpoint')
    parser.add_argument('--data_dir', type=str, default='./data_npz',
                       help='Directory containing preprocessed npz files')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cuda, cpu)')
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load trained model
    print(f"Loading model from: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    config = checkpoint['config']
    
    # Initialize model with saved configuration
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
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Model loaded successfully (trained for {checkpoint['epoch']} epochs)")
    print(f"Training ADE: {checkpoint['ade']:.4f}, FDE: {checkpoint['fde']:.4f}")
    
    # Prepare dataset
    npz_files = glob.glob(f"{args.data_dir}/*.npz")
    if len(npz_files) == 0:
        raise RuntimeError(f'No .npz files found in {args.data_dir}')
    
    print(f"Found {len(npz_files)} npz files for evaluation")
    
    # Create dataset and dataloader
    dataset = TrajectoryDataset(
        npz_files,
        obs_len=config.data.obs_len,
        pred_len=config.data.pred_len
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False
    )
    
    print(f"Dataset size: {len(dataset)} samples")
    
    # Evaluate model
    print("Starting evaluation...")
    ade, fde = evaluate(model, loader, device)
    
    print(f"Evaluation Results:")
    print(f"ADE: {ade:.4f}")
    print(f"FDE: {fde:.4f}")


if __name__ == '__main__':
    main()
