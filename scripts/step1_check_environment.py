#!/usr/bin/env python3
"""
Step 1: Check Environment Setup

This script verifies that your environment is properly set up before proceeding.
Run this first to ensure all dependencies are installed and paths are correct.
"""

import sys
import os
import subprocess
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} - Need Python 3.8+")
        return False


def check_dependencies():
    """Check if required packages are installed."""
    print("\n📦 Checking required packages...")
    required_packages = [
        'numpy',
        'pandas', 
        'torch',
        'torchvision',
        'torchaudio'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("   Run: pip install -r requirements.txt")
        return False
    
    return True


def check_project_structure():
    """Check if project structure is correct."""
    print("\n📁 Checking project structure...")
    
    required_dirs = [
        'src',
        'src/models',
        'src/data',
        'src/training',
        'src/utils',
        'datasets'
    ]
    
    required_files = [
        'src/models/siat.py',
        'src/data/dataset.py',
        'src/training/trainer.py',
        'src/config.py',
        'requirements.txt'
    ]
    
    missing_items = []
    
    # Check directories
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"   ✅ {dir_path}/")
        else:
            print(f"   ❌ {dir_path}/ - Missing")
            missing_items.append(dir_path)
    
    # Check files
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} - Missing")
            missing_items.append(file_path)
    
    return len(missing_items) == 0


def check_dataset_structure():
    """Check if dataset files exist."""
    print("\n📊 Checking dataset files...")
    
    dataset_dirs = [
        'datasets/eth',
        'datasets/hotel', 
        'datasets/univ',
        'datasets/zara1',
        'datasets/zara2'
    ]
    
    found_datasets = []
    for dataset_dir in dataset_dirs:
        if os.path.exists(dataset_dir):
            txt_files = list(Path(dataset_dir).rglob("*.txt"))
            if txt_files:
                print(f"   ✅ {dataset_dir} - {len(txt_files)} .txt files")
                found_datasets.append(dataset_dir)
            else:
                print(f"   ⚠️  {dataset_dir} - No .txt files found")
        else:
            print(f"   ❌ {dataset_dir} - Directory not found")
    
    if found_datasets:
        print(f"\n   Found {len(found_datasets)} dataset directories with data")
        return True
    else:
        print("\n   ❌ No dataset files found!")
        print("      Please ensure your dataset .txt files are in the datasets/ directory")
        return False


def check_gpu_availability():
    """Check if GPU is available for training."""
    print("\n🖥️  Checking GPU availability...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"   ✅ CUDA available - {gpu_count} GPU(s)")
            print(f"   🎮 GPU: {gpu_name}")
            return True
        else:
            print("   ⚠️  CUDA not available - will use CPU")
            print("   💡 Training will be slower but still functional")
            return True
    except Exception as e:
        print(f"   ❌ Error checking GPU: {e}")
        return False


def main():
    """Main environment check function."""
    print("🔍 SIAT Environment Check")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies), 
        ("Project Structure", check_project_structure),
        ("Dataset Files", check_dataset_structure),
        ("GPU Support", check_gpu_availability)
    ]
    
    all_passed = True
    results = {}
    
    for check_name, check_func in checks:
        try:
            results[check_name] = check_func()
            all_passed = all_passed and results[check_name]
        except Exception as e:
            print(f"   ❌ Error in {check_name}: {e}")
            results[check_name] = False
            all_passed = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 SUMMARY")
    print("=" * 50)
    
    for check_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status:8} {check_name}")
    
    if all_passed:
        print("\n🎉 All checks passed! You're ready to proceed.")
        print("\n📋 Next Steps:")
        print("   1. Run: python scripts/step2_preprocess_data.py")
        print("   2. Run: python scripts/step3_test_compatibility.py") 
        print("   3. Run: python scripts/step4_train_model.py")
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above before proceeding.")
        if not results.get("Dependencies", True):
            print("\n🔧 To fix dependencies:")
            print("   pip install -r requirements.txt")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
