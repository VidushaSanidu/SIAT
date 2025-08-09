#!/usr/bin/env python3
"""
Step 5: Evaluate Trained Model

This script evaluates your trained SIAT model on test data and generates
comprehensive metrics and visualizations.

Run this after step4_train_model.py completes successfully.
"""

import sys
import os
import argparse
import torch
import numpy as np
import glob
from pathlib import Path

# Add src to path
sys.path.append('src')

from models import SIAT
from data import TrajectoryDataset, collate_fn
from utils.metrics import ade_fde
from torch.utils.data import DataLoader


def setup_evaluation_args():
    """Parse command line arguments for evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate trained SIAT model')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', type=str, default='./data_npz',
                       help='Directory containing test data')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Directory to save evaluation results')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate trajectory visualizations')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of samples to visualize')
    
    return parser.parse_args()


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint."""
    print(f"📂 Loading model from {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model parameters from checkpoint
    if 'args' in checkpoint:
        args = checkpoint['args']
        obs_len = args.obs_len
        pred_len = args.pred_len
        embed_size = args.embed_size
    else:
        # Default parameters if not saved in checkpoint
        obs_len = 8
        pred_len = 12
        embed_size = 64
        print("⚠️  Using default model parameters (not found in checkpoint)")
    
    # Create model
    model = SIAT(
        obs_len=obs_len,
        pred_len=pred_len,
        embed_size=embed_size
    ).to(device)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"   ✅ Model loaded successfully")
    if 'epoch' in checkpoint:
        print(f"   - Trained for {checkpoint['epoch']} epochs")
    if 'best_ade' in checkpoint:
        print(f"   - Best training ADE: {checkpoint['best_ade']:.4f}")
    
    return model, obs_len, pred_len


def create_test_dataset(data_dir, obs_len, pred_len):
    """Create test dataset."""
    print(f"\n📊 Loading test data from {data_dir}")
    
    npz_files = glob.glob(f"{data_dir}/*.npz")
    
    if len(npz_files) == 0:
        raise RuntimeError(f'No .npz files found in {data_dir}')
    
    print(f"   Found {len(npz_files)} data files")
    
    dataset = TrajectoryDataset(
        npz_files,
        obs_len=obs_len,
        pred_len=pred_len
    )
    
    print(f"   Total test samples: {len(dataset)}")
    
    return dataset


def evaluate_model(model, dataset, device, batch_size=32):
    """Evaluate model on dataset."""
    print(f"\n🧪 Evaluating model...")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2
    )
    
    all_predictions = []
    all_ground_truth = []
    all_observations = []
    total_ade = 0.0
    total_fde = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            obs = batch['obs'].to(device)
            fut = batch['fut'].to(device)
            window = batch['window'].to(device)
            agent_mask = batch['agent_mask'].to(device)
            
            # Forward pass
            pred = model(obs, window, agent_mask)
            
            # Calculate metrics
            ade, fde = ade_fde(pred, fut)
            batch_size_actual = obs.size(0)
            
            total_ade += ade * batch_size_actual
            total_fde += fde * batch_size_actual
            num_samples += batch_size_actual
            
            # Store for analysis
            all_predictions.append(pred.cpu().numpy())
            all_ground_truth.append(fut.cpu().numpy())
            all_observations.append(obs.cpu().numpy())
            
            if (i + 1) % 10 == 0:
                print(f"   Processed {i + 1}/{len(dataloader)} batches")
    
    # Final metrics
    final_ade = total_ade / num_samples
    final_fde = total_fde / num_samples
    
    print(f"   ✅ Evaluation complete")
    print(f"   - Samples evaluated: {num_samples}")
    print(f"   - Final ADE: {final_ade:.4f}")
    print(f"   - Final FDE: {final_fde:.4f}")
    
    # Combine all predictions
    predictions = np.concatenate(all_predictions, axis=0)
    ground_truth = np.concatenate(all_ground_truth, axis=0)
    observations = np.concatenate(all_observations, axis=0)
    
    return {
        'ade': final_ade,
        'fde': final_fde,
        'predictions': predictions,
        'ground_truth': ground_truth,
        'observations': observations,
        'num_samples': num_samples
    }


def analyze_results(results):
    """Analyze evaluation results in detail."""
    print(f"\n📈 Detailed Analysis...")
    
    predictions = results['predictions']
    ground_truth = results['ground_truth']
    
    # Per-timestep errors
    timestep_errors = []
    for t in range(predictions.shape[1]):
        pred_t = predictions[:, t, :]
        gt_t = ground_truth[:, t, :]
        error_t = np.sqrt(np.sum((pred_t - gt_t) ** 2, axis=1)).mean()
        timestep_errors.append(error_t)
    
    print(f"   Per-timestep errors:")
    for t, error in enumerate(timestep_errors):
        print(f"     t+{t+1}: {error:.4f}")
    
    # Distance-based analysis
    distances = np.sqrt(np.sum((predictions - ground_truth) ** 2, axis=2))
    
    print(f"   Error statistics:")
    print(f"     Mean error: {distances.mean():.4f}")
    print(f"     Std error: {distances.std():.4f}")
    print(f"     Min error: {distances.min():.4f}")
    print(f"     Max error: {distances.max():.4f}")
    print(f"     Median error: {np.median(distances):.4f}")
    
    # Percentile analysis
    percentiles = [50, 75, 90, 95, 99]
    print(f"   Error percentiles:")
    for p in percentiles:
        value = np.percentile(distances, p)
        print(f"     {p}th percentile: {value:.4f}")
    
    return {
        'timestep_errors': timestep_errors,
        'error_stats': {
            'mean': distances.mean(),
            'std': distances.std(),
            'min': distances.min(),
            'max': distances.max(),
            'median': np.median(distances)
        }
    }


def visualize_trajectories(results, output_dir, num_samples=10):
    """Create trajectory visualizations."""
    print(f"\n🎨 Creating visualizations...")
    
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("   ⚠️  matplotlib not available, skipping visualizations")
        print("   Install with: pip install matplotlib")
        return
    
    predictions = results['predictions']
    ground_truth = results['ground_truth']
    observations = results['observations']
    
    # Select random samples to visualize
    num_total = len(predictions)
    indices = np.random.choice(num_total, min(num_samples, num_total), replace=False)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Individual trajectory plots
    for i, idx in enumerate(indices):
        plt.figure(figsize=(10, 8))
        
        obs = observations[idx]
        pred = predictions[idx]
        gt = ground_truth[idx]
        
        # Plot observed trajectory
        plt.plot(obs[:, 0], obs[:, 1], 'b-o', linewidth=2, markersize=4, 
                label='Observed', alpha=0.8)
        
        # Plot predicted trajectory
        plt.plot(pred[:, 0], pred[:, 1], 'r-s', linewidth=2, markersize=4,
                label='Predicted', alpha=0.8)
        
        # Plot ground truth trajectory
        plt.plot(gt[:, 0], gt[:, 1], 'g-^', linewidth=2, markersize=4,
                label='Ground Truth', alpha=0.8)
        
        # Connect observed to predicted/ground truth
        plt.plot([obs[-1, 0], pred[0, 0]], [obs[-1, 1], pred[0, 1]], 'r--', alpha=0.5)
        plt.plot([obs[-1, 0], gt[0, 0]], [obs[-1, 1], gt[0, 1]], 'g--', alpha=0.5)
        
        # Mark start and end points
        plt.plot(obs[0, 0], obs[0, 1], 'ko', markersize=8, label='Start')
        plt.plot(gt[-1, 0], gt[-1, 1], 'ks', markersize=8, label='True End')
        plt.plot(pred[-1, 0], pred[-1, 1], 'k^', markersize=8, label='Pred End')
        
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.title(f'Trajectory Prediction - Sample {idx}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        plt.savefig(f'{output_dir}/trajectory_{idx}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"   ✅ Saved {len(indices)} individual trajectory plots")
    
    # Error distribution plot
    plt.figure(figsize=(12, 8))
    
    distances = np.sqrt(np.sum((predictions - ground_truth) ** 2, axis=2))
    
    # Histogram of final displacement errors
    plt.subplot(2, 2, 1)
    final_errors = distances[:, -1]
    plt.hist(final_errors, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Final Displacement Error (FDE)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Final Displacement Errors')
    plt.grid(True, alpha=0.3)
    
    # Average displacement errors
    plt.subplot(2, 2, 2)
    avg_errors = distances.mean(axis=1)
    plt.hist(avg_errors, bins=50, alpha=0.7, edgecolor='black', color='orange')
    plt.xlabel('Average Displacement Error (ADE)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Average Displacement Errors')
    plt.grid(True, alpha=0.3)
    
    # Error over time
    plt.subplot(2, 2, 3)
    timestep_errors = distances.mean(axis=0)
    plt.plot(range(1, len(timestep_errors) + 1), timestep_errors, 'bo-', linewidth=2)
    plt.xlabel('Future Timestep')
    plt.ylabel('Average Error')
    plt.title('Error vs Prediction Horizon')
    plt.grid(True, alpha=0.3)
    
    # Error scatter plot
    plt.subplot(2, 2, 4)
    plt.scatter(avg_errors, final_errors, alpha=0.5)
    plt.xlabel('Average Displacement Error (ADE)')
    plt.ylabel('Final Displacement Error (FDE)')
    plt.title('ADE vs FDE')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/error_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved error analysis plots")


def save_results(results, analysis, output_dir):
    """Save evaluation results to files."""
    print(f"\n💾 Saving results to {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save numerical results
    results_summary = {
        'overall_metrics': {
            'ade': float(results['ade']),
            'fde': float(results['fde']),
            'num_samples': int(results['num_samples'])
        },
        'detailed_analysis': analysis
    }
    
    import json
    with open(f'{output_dir}/evaluation_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    # Save raw predictions as numpy arrays
    np.savez_compressed(f'{output_dir}/predictions.npz',
                       predictions=results['predictions'],
                       ground_truth=results['ground_truth'],
                       observations=results['observations'])
    
    print(f"   ✅ Results saved")
    print(f"   - Summary: {output_dir}/evaluation_results.json")
    print(f"   - Raw data: {output_dir}/predictions.npz")


def main():
    """Main evaluation function."""
    args = setup_evaluation_args()
    
    print("🧪 SIAT Model Evaluation")
    print("=" * 50)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Device: {device}")
    
    try:
        # Load model
        model, obs_len, pred_len = load_model(args.checkpoint, device)
        
        # Create test dataset
        dataset = create_test_dataset(args.data_dir, obs_len, pred_len)
        
        # Evaluate model
        results = evaluate_model(model, dataset, device, args.batch_size)
        
        # Detailed analysis
        analysis = analyze_results(results)
        
        # Visualizations
        if args.visualize:
            visualize_trajectories(results, args.output_dir, args.num_samples)
        
        # Save results
        save_results(results, analysis, args.output_dir)
        
        # Final summary
        print(f"\n🎉 Evaluation completed successfully!")
        print(f"📊 Final Results:")
        print(f"   - ADE: {results['ade']:.4f}")
        print(f"   - FDE: {results['fde']:.4f}")
        print(f"   - Samples: {results['num_samples']}")
        print(f"   - Results saved to: {args.output_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
