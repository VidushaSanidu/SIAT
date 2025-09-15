#!/usr/bin/env python3
"""
Step 3: Test Model-Dataset Compatibility

This script tests if your SIAT model is compatible with the preprocessed dataset.
Run this after step2_preprocess_data.py completes successfully.

This will verify:
- Dataset loading works correctly
- Model can process the data format
- Forward pass produces correct output shapes
- Training loop components work
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path

# Add src and project root to sys.path so we can import modules
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from models import SIAT
from data import TrajectoryDataset, collate_fn
from config import Config
from utils.metrics import ade_fde
from torch.utils.data import DataLoader


def test_environment():
    """Test basic environment setup."""
    print("🔍 Testing Environment...")
    
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")
    
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    return True


def test_config():
    """Test configuration loading."""
    print("\n⚙️  Testing Configuration...")
    
    try:
        config = Config()
        print(f"   ✅ Config loaded successfully")
        print(f"   - obs_len: {config.model.obs_len}")
        print(f"   - pred_len: {config.model.pred_len}")
        print(f"   - embed_size: {config.model.embed_size}")
        print(f"   - batch_size: {config.training.batch_size}")
        return config
    except Exception as e:
        print(f"   ❌ Config loading failed: {e}")
        return None


def test_dataset_loading(data_dir='./data_npz'):
    """Test dataset loading with preprocessed data."""
    print("\n📊 Testing Dataset Loading...")
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"   ❌ Data directory {data_dir} not found!")
        print(f"   🔧 Run: python scripts/step2_preprocess_data.py")
        return None, None
    
    # Find npz files
    npz_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npz')]
    
    if len(npz_files) == 0:
        print(f"   ❌ No .npz files found in {data_dir}")
        print(f"   🔧 Run: python scripts/step2_preprocess_data.py")
        return None, None
    
    print(f"   ✅ Found {len(npz_files)} .npz files")
    
    # Test loading one file
    try:
        config = Config()
        dataset = TrajectoryDataset(
            npz_files[:1],  # Use only first file for testing
            obs_len=config.model.obs_len,
            pred_len=config.model.pred_len
        )
        print(f"   ✅ Dataset created with {len(dataset)} samples")
        
        # Test getting a single sample
        sample = dataset[0]
        print(f"   ✅ Sample structure:")
        print(f"     - obs: {sample['obs'].shape}")
        print(f"     - fut: {sample['fut'].shape}")
        print(f"     - window: {sample['window'].shape}")
        
        return dataset, config
        
    except Exception as e:
        print(f"   ❌ Dataset loading failed: {e}")
        return None, None


def test_dataloader(dataset, config):
    """Test DataLoader with collate function."""
    print("\n🔄 Testing DataLoader...")
    
    try:
        loader = DataLoader(
            dataset,
            batch_size=min(4, len(dataset)),  # Small batch for testing
            shuffle=False,
            collate_fn=collate_fn
        )
        print(f"   ✅ DataLoader created successfully")
        
        # Test getting a batch
        batch = next(iter(loader))
        print(f"   ✅ Batch loaded successfully:")
        print(f"     - obs: {batch['obs'].shape}")
        print(f"     - fut: {batch['fut'].shape}")
        print(f"     - window: {batch['window'].shape}")
        print(f"     - agent_mask: {batch['agent_mask'].shape}")
        
        # Check batch consistency
        batch_size = batch['obs'].size(0)
        assert batch['fut'].size(0) == batch_size
        assert batch['window'].size(0) == batch_size
        assert batch['agent_mask'].size(0) == batch_size
        
        print(f"   ✅ Batch shapes are consistent")
        
        return loader, batch
        
    except Exception as e:
        print(f"   ❌ DataLoader failed: {e}")
        return None, None


def test_model_initialization(config):
    """Test SIAT model initialization."""
    print("\n🤖 Testing Model Initialization...")
    
    try:
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
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"   ✅ Model initialized successfully")
        print(f"   - Total parameters: {total_params:,}")
        print(f"   - Trainable parameters: {trainable_params:,}")
        print(f"   - Model size: ~{total_params * 4 / 1e6:.1f} MB")
        
        return model
        
    except Exception as e:
        print(f"   ❌ Model initialization failed: {e}")
        return None


def test_forward_pass(model, batch, config):
    """Test model forward pass."""
    print("\n⚡ Testing Forward Pass...")
    
    try:
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model and batch to device
        model = model.to(device)
        obs = batch['obs'].to(device)
        window = batch['window'].to(device)
        agent_mask = batch['agent_mask'].to(device)
        fut = batch['fut'].to(device)
        
        print(f"   Using device: {device}")
        
        # Forward pass
        with torch.no_grad():
            pred = model(obs, window, agent_mask)
        
        # Check output shape
        expected_shape = (obs.size(0), config.model.pred_len, 2)
        if pred.shape == expected_shape:
            print(f"   ✅ Forward pass successful")
            print(f"   - Input obs: {obs.shape}")
            print(f"   - Input window: {window.shape}")
            print(f"   - Output pred: {pred.shape}")
            print(f"   - Expected: {expected_shape}")
        else:
            print(f"   ❌ Wrong output shape: got {pred.shape}, expected {expected_shape}")
            return None
        
        # Test loss computation
        loss = torch.nn.functional.mse_loss(pred, fut)
        print(f"   ✅ Loss computation: {loss.item():.6f}")
        
        # Test metrics
        ade, fde = ade_fde(pred, fut)
        print(f"   ✅ Metrics computation: ADE={ade:.4f}, FDE={fde:.4f}")
        
        return pred
        
    except Exception as e:
        print(f"   ❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_training_step(model, batch, config):
    """Test a single training step."""
    print("\n🏋️  Testing Training Step...")
    
    try:
        model.train()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Move batch to device
        obs = batch['obs'].to(device)
        window = batch['window'].to(device)
        agent_mask = batch['agent_mask'].to(device)
        fut = batch['fut'].to(device)
        
        # Training step
        optimizer.zero_grad()
        pred = model(obs, window, agent_mask)
        loss = torch.nn.functional.mse_loss(pred, fut)
        loss.backward()
        optimizer.step()
        
        print(f"   ✅ Training step successful")
        print(f"   - Loss: {loss.item():.6f}")
        print(f"   - Gradients computed and applied")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Training step failed: {e}")
        return False


def test_multiple_batches(loader, model, config):
    """Test processing multiple batches."""
    print("\n📦 Testing Multiple Batches...")
    
    try:
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        batch_count = 0
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in loader:
                if batch_count >= 3:  # Test first 3 batches
                    break
                
                obs = batch['obs'].to(device)
                window = batch['window'].to(device)
                agent_mask = batch['agent_mask'].to(device)
                fut = batch['fut'].to(device)
                
                pred = model(obs, window, agent_mask)
                loss = torch.nn.functional.mse_loss(pred, fut)
                total_loss += loss.item()
                batch_count += 1
        
        avg_loss = total_loss / batch_count
        print(f"   ✅ Processed {batch_count} batches successfully")
        print(f"   - Average loss: {avg_loss:.6f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Multiple batch processing failed: {e}")
        return False


def main():
    """Main compatibility testing function."""
    print("🧪 SIAT Model-Dataset Compatibility Test")
    print("=" * 50)
    
    test_results = {}
    
    # Run all tests
    tests = [
        ("Environment", test_environment),
        ("Configuration", test_config),
    ]
    
    # Test environment and config first
    test_results["Environment"] = test_environment()
    config = test_config()
    test_results["Configuration"] = config is not None
    
    if not config:
        print("\n❌ Cannot proceed without valid configuration")
        return False
    
    # Test dataset
    dataset, _ = test_dataset_loading()
    test_results["Dataset Loading"] = dataset is not None
    
    if not dataset:
        print("\n❌ Cannot proceed without valid dataset")
        return False
    
    # Test dataloader
    loader, batch = test_dataloader(dataset, config)
    test_results["DataLoader"] = loader is not None and batch is not None
    
    if not loader or not batch:
        print("\n❌ Cannot proceed without valid dataloader")
        return False
    
    # Test model
    model = test_model_initialization(config)
    test_results["Model Initialization"] = model is not None
    
    if not model:
        print("\n❌ Cannot proceed without valid model")
        return False
    
    # Test forward pass
    pred = test_forward_pass(model, batch, config)
    test_results["Forward Pass"] = pred is not None
    
    # Test training step
    test_results["Training Step"] = test_training_step(model, batch, config)
    
    # Test multiple batches
    test_results["Multiple Batches"] = test_multiple_batches(loader, model, config)
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 COMPATIBILITY TEST SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for test_name, passed in test_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status:8} {test_name}")
        all_passed = all_passed and passed
    
    if all_passed:
        print(f"\n🎉 All compatibility tests passed!")
        print(f"Your SIAT model is fully compatible with your dataset.")
        print(f"\n📋 Next Step:")
        print(f"   Run: python scripts/step4_train_model.py")
        
        # Additional info
        print(f"\n💡 Quick Training Tips:")
        print(f"   - Start with a small learning rate (0.001)")
        print(f"   - Use batch size 16-32 for better GPU utilization")
        print(f"   - Monitor ADE/FDE metrics during training")
        print(f"   - Save checkpoints regularly")
        
    else:
        print(f"\n❌ Some compatibility tests failed!")
        print(f"Please fix the issues above before proceeding to training.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
