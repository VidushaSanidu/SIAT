"""
Test script to verify model-dataset compatibility.
"""

import torch
import numpy as np
import os
from src.models import SIAT
from src.data import TrajectoryDataset, collate_fn
from src.config import Config
from torch.utils.data import DataLoader


def test_compatibility():
    """Test if the model is compatible with dataset structure."""
    print("Testing SIAT Model-Dataset Compatibility")
    print("=" * 50)
    
    # Check if preprocessed data exists
    data_dir = './data_npz'
    if not os.path.exists(data_dir) or len(os.listdir(data_dir)) == 0:
        print("❌ No preprocessed data found. Please run:")
        print("   python preprocess_data.py --input_dir ./datasets --output_dir ./data_npz")
        return False
    
    config = Config()
    print(f"✓ Configuration loaded: obs_len={config.model.obs_len}, pred_len={config.model.pred_len}")
    
    # Find npz files
    npz_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npz')]
    if len(npz_files) == 0:
        print("❌ No .npz files found in data directory")
        return False
    
    print(f"✓ Found {len(npz_files)} preprocessed files")
    
    # Test dataset loading
    try:
        dataset = TrajectoryDataset(
            npz_files[:1],  # Use only first file for testing
            obs_len=config.model.obs_len,
            pred_len=config.model.pred_len
        )
        print(f"✓ Dataset loaded successfully with {len(dataset)} samples")
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return False
    
    # Test dataloader with collate function
    try:
        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=collate_fn
        )
        print("✓ DataLoader created successfully")
    except Exception as e:
        print(f"❌ DataLoader creation failed: {e}")
        return False
    
    # Test getting a batch
    try:
        batch = next(iter(loader))
        obs_shape = batch['obs'].shape
        fut_shape = batch['fut'].shape
        window_shape = batch['window'].shape
        mask_shape = batch['agent_mask'].shape
        
        print(f"✓ Batch loaded successfully:")
        print(f"  - obs: {obs_shape}")
        print(f"  - fut: {fut_shape}")
        print(f"  - window: {window_shape}")
        print(f"  - agent_mask: {mask_shape}")
    except Exception as e:
        print(f"❌ Batch loading failed: {e}")
        return False
    
    # Test model initialization
    try:
        model = SIAT(
            obs_len=config.model.obs_len,
            pred_len=config.model.pred_len,
            embed_size=config.model.embed_size
        )
        print(f"✓ Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        return False
    
    # Test forward pass
    try:
        model.eval()
        with torch.no_grad():
            pred = model(batch['obs'], batch['window'], batch['agent_mask'])
        
        expected_shape = (batch['obs'].size(0), config.model.pred_len, 2)
        if pred.shape == expected_shape:
            print(f"✓ Forward pass successful: {pred.shape}")
        else:
            print(f"❌ Wrong prediction shape: got {pred.shape}, expected {expected_shape}")
            return False
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False
    
    # Test loss computation
    try:
        loss = torch.nn.functional.mse_loss(pred, batch['fut'])
        print(f"✓ Loss computation successful: {loss.item():.6f}")
    except Exception as e:
        print(f"❌ Loss computation failed: {e}")
        return False
    
    print("\n🎉 All compatibility tests passed!")
    print("Your model is compatible with your dataset structure.")
    
    # Additional recommendations
    print("\n📋 Next Steps:")
    print("1. Run preprocessing: python preprocess_data.py --input_dir ./datasets")
    print("2. Install dependencies: pip install -r requirements.txt")
    print("3. Start training: python train.py --data_dir ./data_npz")
    
    return True


if __name__ == '__main__':
    test_compatibility()
