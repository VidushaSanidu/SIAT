#!/usr/bin/env python3
"""
SIAT Training Pipeline - Master Script

This script runs the complete SIAT training pipeline from start to finish.
It executes all steps in sequence and provides a guided experience.

Steps:
1. Environment check
2. Data preprocessing  
3. Compatibility testing
4. Model training
5. Model evaluation

Usage:
    python run_pipeline.py [--skip-step STEP] [--only-step STEP] [--help]
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path


def print_header(title, char="="):
    """Print a formatted header."""
    print(f"\n{char * 60}")
    print(f"{title:^60}")
    print(f"{char * 60}")


def print_step(step_num, title, description):
    """Print step information."""
    print(f"\n🚀 STEP {step_num}: {title}")
    print(f"   {description}")
    print("-" * 50)


def run_script(script_path, args=None, capture_output=False):
    """Run a Python script and return success status."""
    cmd = [sys.executable, script_path]
    if args:
        cmd.extend(args)
    
    print(f"🔧 Running: {' '.join(cmd)}")
    
    try:
        if capture_output:
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0, result.stdout, result.stderr
        else:
            result = subprocess.run(cmd)
            return result.returncode == 0, "", ""
    except Exception as e:
        print(f"❌ Error running script: {e}")
        return False, "", str(e)


def check_prerequisites():
    """Check if basic prerequisites are met."""
    print("🔍 Checking Prerequisites...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print(f"❌ Python 3.8+ required, found {sys.version}")
        return False
    
    # Check if we're in the right directory
    required_files = ['src/models/siat.py', 'src/data/dataset.py', 'requirements.txt']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        print("   Make sure you're running this from the SIAT project root directory")
        return False
    
    print("✅ Prerequisites check passed")
    return True


def step1_environment_check():
    """Step 1: Check environment setup."""
    print_step(1, "ENVIRONMENT CHECK", "Verify dependencies and project structure")
    
    script_path = "scripts/step1_check_environment.py"
    if not os.path.exists(script_path):
        print(f"❌ Script not found: {script_path}")
        return False
    
    success, _, _ = run_script(script_path)
    
    if success:
        print("✅ Environment check passed!")
        return True
    else:
        print("❌ Environment check failed!")
        print("   Please fix the issues reported above before continuing.")
        return False


def step2_preprocess_data():
    """Step 2: Preprocess dataset."""
    print_step(2, "DATA PREPROCESSING", "Convert ETH/UCY files to SIAT format")
    
    script_path = "scripts/step2_preprocess_data.py"
    if not os.path.exists(script_path):
        print(f"❌ Script not found: {script_path}")
        return False
    
    # Default arguments for preprocessing
    args = ["--input_dir", "./datasets", "--output_dir", "./data_npz"]
    
    success, _, _ = run_script(script_path, args)
    
    if success:
        print("✅ Data preprocessing completed!")
        return True
    else:
        print("❌ Data preprocessing failed!")
        return False


def step3_test_compatibility():
    """Step 3: Test model-dataset compatibility."""
    print_step(3, "COMPATIBILITY TEST", "Verify model works with preprocessed data")
    
    script_path = "scripts/step3_test_compatibility.py"
    if not os.path.exists(script_path):
        print(f"❌ Script not found: {script_path}")
        return False
    
    success, _, _ = run_script(script_path)
    
    if success:
        print("✅ Compatibility test passed!")
        return True
    else:
        print("❌ Compatibility test failed!")
        return False


def step4_train_model():
    """Step 4: Train the model."""
    print_step(4, "MODEL TRAINING", "Train SIAT model on preprocessed data")
    
    script_path = "scripts/step4_train_model.py"
    if not os.path.exists(script_path):
        print(f"❌ Script not found: {script_path}")
        return False
    
    # Default training arguments
    args = [
        "--data_dir", "./data_npz",
        "--epochs", "50",
        "--batch_size", "32",
        "--lr", "0.001",
        "--checkpoint_dir", "./checkpoints"
    ]
    
    success, _, _ = run_script(script_path, args)
    
    if success:
        print("✅ Model training completed!")
        return True
    else:
        print("❌ Model training failed!")
        return False


def step5_evaluate_model():
    """Step 5: Evaluate the trained model."""
    print_step(5, "MODEL EVALUATION", "Evaluate trained model and generate results")
    
    script_path = "scripts/step5_evaluate_model.py"
    if not os.path.exists(script_path):
        print(f"❌ Script not found: {script_path}")
        return False
    
    # Check if model checkpoint exists
    checkpoint_path = "./checkpoints/best_model.pth"
    if not os.path.exists(checkpoint_path):
        print(f"❌ Model checkpoint not found: {checkpoint_path}")
        print("   Please complete training first (step 4)")
        return False
    
    # Default evaluation arguments
    args = [
        "--checkpoint", checkpoint_path,
        "--data_dir", "./data_npz",
        "--output_dir", "./results",
        "--visualize",
        "--num_samples", "10"
    ]
    
    success, _, _ = run_script(script_path, args)
    
    if success:
        print("✅ Model evaluation completed!")
        return True
    else:
        print("❌ Model evaluation failed!")
        return False


def main():
    """Main pipeline function."""
    parser = argparse.ArgumentParser(description='SIAT Training Pipeline')
    parser.add_argument('--skip-step', type=int, choices=[1, 2, 3, 4, 5],
                       help='Skip a specific step (useful if already completed)')
    parser.add_argument('--only-step', type=int, choices=[1, 2, 3, 4, 5],
                       help='Run only a specific step')
    parser.add_argument('--quick', action='store_true',
                       help='Run with reduced epochs for quick testing')
    
    args = parser.parse_args()
    
    print_header("🚀 SIAT TRAINING PIPELINE 🚀")
    print("This script will guide you through the complete SIAT training process.")
    print("Each step will be executed automatically with progress feedback.")
    
    # Check prerequisites
    if not check_prerequisites():
        return False
    
    # Define all steps
    steps = [
        (1, "Environment Check", step1_environment_check),
        (2, "Data Preprocessing", step2_preprocess_data),
        (3, "Compatibility Test", step3_test_compatibility),
        (4, "Model Training", step4_train_model),
        (5, "Model Evaluation", step5_evaluate_model)
    ]
    
    # Filter steps based on arguments
    if args.only_step:
        steps = [(num, name, func) for num, name, func in steps if num == args.only_step]
    elif args.skip_step:
        steps = [(num, name, func) for num, name, func in steps if num != args.skip_step]
    
    # Execute steps
    start_time = time.time()
    failed_steps = []
    
    for step_num, step_name, step_func in steps:
        try:
            if not step_func():
                failed_steps.append((step_num, step_name))
                print(f"\n⚠️  Step {step_num} failed. You can:")
                print(f"   1. Fix the issue and run: python run_pipeline.py --only-step {step_num}")
                print(f"   2. Skip this step and continue: python run_pipeline.py --skip-step {step_num}")
                print(f"   3. Check individual step scripts in the scripts/ directory")
                
                # Ask user if they want to continue
                if step_num < 5:  # Don't ask for last step
                    response = input(f"\nContinue to next step? (y/n): ").lower().strip()
                    if response != 'y':
                        break
        except KeyboardInterrupt:
            print(f"\n⏹️  Pipeline interrupted by user at step {step_num}")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error in step {step_num}: {e}")
            failed_steps.append((step_num, step_name))
            break
    
    # Final summary
    total_time = time.time() - start_time
    print_header("📋 PIPELINE SUMMARY")
    
    if not failed_steps:
        print("🎉 All steps completed successfully!")
        print(f"⏱️  Total time: {total_time/60:.1f} minutes")
        
        print(f"\n📊 Results:")
        if os.path.exists("./checkpoints/best_model.pth"):
            print(f"   ✅ Trained model: ./checkpoints/best_model.pth")
        if os.path.exists("./results/evaluation_results.json"):
            print(f"   ✅ Evaluation results: ./results/evaluation_results.json")
        if os.path.exists("./data_npz"):
            npz_files = len([f for f in os.listdir("./data_npz") if f.endswith('.npz')])
            print(f"   ✅ Preprocessed data: {npz_files} files in ./data_npz/")
        
        print(f"\n🎯 What's Next:")
        print(f"   - Use your trained model for trajectory prediction")
        print(f"   - Experiment with different hyperparameters")
        print(f"   - Try the model on new datasets")
        print(f"   - Check results visualizations in ./results/")
        
    else:
        print(f"❌ Pipeline completed with {len(failed_steps)} failed steps:")
        for step_num, step_name in failed_steps:
            print(f"   - Step {step_num}: {step_name}")
        
        print(f"\n🔧 To fix issues:")
        print(f"   - Check the error messages above")
        print(f"   - Run individual steps: python scripts/stepN_*.py")
        print(f"   - Re-run failed steps: python run_pipeline.py --only-step N")
    
    return len(failed_steps) == 0


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print(f"\n⏹️  Pipeline interrupted by user")
        sys.exit(1)
