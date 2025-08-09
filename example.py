"""
Example usage of SIAT model for trajectory prediction.
"""

import torch
import numpy as np
from src.models import SIAT
from src.data import TrajectoryDataset
from src.config import Config


def create_dummy_data():
    """Create dummy trajectory data for demonstration."""
    # Create some dummy trajectory data
    n_agents = 5
    total_time = 20
    
    # Generate random trajectories
    trajectories = np.random.randn(n_agents, total_time, 2).astype(np.float32)
    
    # Add some smoothness to make it more realistic
    for i in range(1, total_time):
        trajectories[:, i] = 0.7 * trajectories[:, i-1] + 0.3 * trajectories[:, i]
    
    # Save as npz file
    np.savez('dummy_data.npz', trajectories=trajectories)
    print("Created dummy_data.npz with shape:", trajectories.shape)
    
    return ['dummy_data.npz']


def main():
    """Demonstrate SIAT model usage."""
    print("SIAT Model Example")
    print("=" * 50)
    
    # Configuration
    config = Config()
    print(f"Model config: obs_len={config.model.obs_len}, pred_len={config.model.pred_len}")
    
    # Create dummy data
    npz_files = create_dummy_data()
    
    # Create dataset
    dataset = TrajectoryDataset(
        npz_files, 
        obs_len=config.model.obs_len, 
        pred_len=config.model.pred_len
    )
    print(f"Dataset created with {len(dataset)} samples")
    
    # Initialize model
    model = SIAT(
        obs_len=config.model.obs_len,
        pred_len=config.model.pred_len,
        embed_size=config.model.embed_size
    )
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Get a sample
    sample = dataset[0]
    obs = sample['obs'].unsqueeze(0)  # Add batch dimension
    window = sample['window'].unsqueeze(0)  # Add batch dimension
    fut = sample['fut'].unsqueeze(0)  # Add batch dimension
    
    print(f"Sample shapes:")
    print(f"  Observed trajectory: {obs.shape}")
    print(f"  Full window: {window.shape}")
    print(f"  Future trajectory: {fut.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        pred = model(obs, window)
    
    print(f"Predicted trajectory shape: {pred.shape}")
    
    # Calculate error
    from src.utils import ade_fde
    ade, fde = ade_fde(pred, fut)
    print(f"ADE: {ade:.4f}, FDE: {fde:.4f}")
    
    print("\nExample completed successfully!")


if __name__ == '__main__':
    main()
