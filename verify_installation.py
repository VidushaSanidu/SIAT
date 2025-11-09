#!/usr/bin/env python3
"""
Installation verification script for video processing pipeline.

This script verifies that all dependencies are properly installed
and all components are working correctly.
"""

import sys
from pathlib import Path

def check_python_version():
    """Check Python version."""
    print("🔍 Checking Python version...")
    if sys.version_info < (3, 8):
        print(f"❌ Python 3.8+ required, got {sys.version_info.major}.{sys.version_info.minor}")
        return False
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}")
    return True


def check_imports():
    """Check all required imports."""
    print("\n🔍 Checking required packages...")
    
    required = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'cv2': 'OpenCV',
        'ultralytics': 'YOLOv8',
        'scipy': 'SciPy',
        'sklearn': 'Scikit-learn',
        'tqdm': 'tqdm',
        'pandas': 'Pandas',
        'matplotlib': 'Matplotlib',
    }
    
    failed = []
    for module, name in required.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"❌ {name} - NOT INSTALLED")
            failed.append(name)
    
    return len(failed) == 0, failed


def check_video_module():
    """Check video processing module."""
    print("\n🔍 Checking video processing module...")
    
    try:
        from src.video import (
            VideoProcessor,
            TrajectoryExtractor,
            Trajectory,
            extract_trajectories_from_detections,
            visualize_detections
        )
        print("✓ VideoProcessor")
        print("✓ TrajectoryExtractor")
        print("✓ Trajectory")
        print("✓ extract_trajectories_from_detections()")
        print("✓ visualize_detections()")
        return True
    except ImportError as e:
        print(f"❌ Error importing video module: {e}")
        return False


def check_model_module():
    """Check SIAT model module."""
    print("\n🔍 Checking SIAT model module...")
    
    try:
        from src.models import SIAT
        from src.config import Config
        print("✓ SIAT model")
        print("✓ Config")
        return True
    except ImportError as e:
        print(f"❌ Error importing model module: {e}")
        return False


def check_data_module():
    """Check data loading module."""
    print("\n🔍 Checking data module...")
    
    try:
        from src.data import TrajectoryDataset
        print("✓ TrajectoryDataset")
        return True
    except ImportError as e:
        print(f"❌ Error importing data module: {e}")
        return False


def check_scripts():
    """Check that main scripts exist."""
    print("\n🔍 Checking pipeline scripts...")
    
    scripts = {
        'scripts/step0_process_videos.py': 'Video Processing',
        'scripts/step1_check_environment.py': 'Environment Check',
        'scripts/step2_preprocess_data.py': 'Data Preprocessing',
        'scripts/step3_test_compatibility.py': 'Compatibility Test',
        'scripts/step4_train_model.py': 'Model Training',
        'scripts/step5_evaluate_model.py': 'Model Evaluation',
    }
    
    all_exist = True
    for script_path, name in scripts.items():
        if Path(script_path).exists():
            print(f"✓ {name}")
        else:
            print(f"❌ {name} - {script_path} not found")
            all_exist = False
    
    return all_exist


def check_cuda():
    """Check CUDA availability."""
    print("\n🔍 Checking CUDA...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available - {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
            return True
        else:
            print("⚠ CUDA not available (will use CPU - slower)")
            return False
    except Exception as e:
        print(f"⚠ Could not check CUDA: {e}")
        return False


def check_yolo_models():
    """Check YOLO model availability."""
    print("\n🔍 Checking YOLO models...")
    
    try:
        from ultralytics import YOLO
        
        models = ['yolov8n.pt', 'yolov8s.pt']
        downloaded = 0
        
        for model in models:
            try:
                print(f"  Checking {model}...")
                model_obj = YOLO(model)
                downloaded += 1
                print(f"  ✓ {model}")
            except Exception as e:
                print(f"  ⚠ {model} not cached (will download on first use)")
        
        if downloaded > 0:
            print(f"✓ Found {downloaded} YOLO models cached")
        return True
    except Exception as e:
        print(f"❌ Error checking YOLO models: {e}")
        return False


def check_documentation():
    """Check that documentation files exist."""
    print("\n🔍 Checking documentation...")
    
    docs = {
        'QUICK_START.md': 'Quick Start Guide',
        'VIDEO_PROCESSING.md': 'Detailed Guide',
        'PIPELINE_VISUALS.md': 'Visual Diagrams',
        'IMPLEMENTATION_SUMMARY.md': 'Implementation Summary',
        'PROJECT_STRUCTURE.md': 'Project Structure',
    }
    
    all_exist = True
    for doc_path, name in docs.items():
        if Path(doc_path).exists():
            print(f"✓ {name}")
        else:
            print(f"⚠ {name} - {doc_path} not found")
            all_exist = False
    
    return all_exist


def print_summary(all_checks):
    """Print verification summary."""
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    total = len(all_checks)
    passed = sum(1 for check in all_checks.values() if check)
    failed = total - passed
    
    print(f"\n✓ Passed: {passed}/{total}")
    if failed > 0:
        print(f"❌ Failed: {failed}/{total}")
    
    if failed == 0:
        print("\n🎉 All checks passed! Ready to use video processing pipeline.")
        return True
    else:
        print("\n⚠ Some checks failed. See above for details.")
        print("\nTo fix issues:")
        print("1. Install missing packages: pip install -r requirements.txt")
        print("2. Verify Python 3.8+: python --version")
        print("3. Check CUDA (if using GPU): python -c 'import torch; print(torch.cuda.is_available())'")
        return False


def print_next_steps():
    """Print next steps."""
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    
    print("""
1. Read the quick start guide:
   cat QUICK_START.md

2. Prepare your videos:
   mkdir -p videos
   # Copy your .avi or .mp4 files here

3. Process videos:
   python scripts/step0_process_videos.py \\
       --video_dir ./videos \\
       --output_dir ./data_npz

4. Train the model:
   python scripts/step4_train_model.py --data_dir ./data_npz

5. Evaluate:
   python scripts/step5_evaluate_model.py \\
       --checkpoint ./checkpoints/best_model.pth

For more details:
   cat VIDEO_PROCESSING.md     # Comprehensive guide
   cat PIPELINE_VISUALS.md     # Visual explanations
   python example_video_processing.py  # Code examples
    """)


def main():
    """Run all verification checks."""
    print("="*60)
    print("VIDEO PROCESSING PIPELINE VERIFICATION")
    print("="*60)
    
    all_checks = {}
    
    # Run all checks
    all_checks['Python version'] = check_python_version()
    
    success, failed = check_imports()
    all_checks['Packages'] = success
    if not success:
        print(f"\nMissing packages: {', '.join(failed)}")
        print("Install with: pip install -r requirements.txt")
        return 1
    
    all_checks['Video module'] = check_video_module()
    all_checks['Model module'] = check_model_module()
    all_checks['Data module'] = check_data_module()
    all_checks['Pipeline scripts'] = check_scripts()
    all_checks['CUDA'] = check_cuda()
    all_checks['YOLO models'] = check_yolo_models()
    all_checks['Documentation'] = check_documentation()
    
    # Print summary
    success = print_summary(all_checks)
    
    # Print next steps
    print_next_steps()
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
