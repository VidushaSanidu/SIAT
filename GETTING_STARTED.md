# 🚀 SIAT Training Pipeline - Getting Started

This guide will help you get started with training your SIAT model step by step.

## 📋 Quick Start

**Option 1: Run Complete Pipeline (Recommended)**
```bash
python run_pipeline.py
```

**Option 2: Run Individual Steps**
```bash
# Step 1: Check environment
python scripts/step1_check_environment.py

# Step 2: Preprocess data
python scripts/step2_preprocess_data.py --input_dir ./datasets --output_dir ./data_npz

# Step 3: Test compatibility
python scripts/step3_test_compatibility.py

# Step 4: Train model
python scripts/step4_train_model.py --data_dir ./data_npz --epochs 50

# Step 5: Evaluate model
python scripts/step5_evaluate_model.py --checkpoint ./checkpoints/best_model.pth
```

## 📁 File Structure

```
SIAT/
├── 🔧 run_pipeline.py              # Master script - START HERE
├── scripts/
│   ├── step1_check_environment.py  # Verify setup
│   ├── step2_preprocess_data.py    # Convert data format
│   ├── step3_test_compatibility.py # Test model+data
│   ├── step4_train_model.py        # Train the model
│   └── step5_evaluate_model.py     # Evaluate results
├── src/                            # Source code
├── datasets/                       # Your ETH/UCY .txt files
├── data_npz/                       # Preprocessed data (auto-created)
├── checkpoints/                    # Saved models (auto-created)
└── results/                        # Evaluation results (auto-created)
```

## 🔍 Step-by-Step Guide

### **Step 1: Environment Check** 🔧
**What it does:** Verifies Python version, dependencies, and project structure
**When to run:** Always run this first
**Script:** `scripts/step1_check_environment.py`

```bash
python scripts/step1_check_environment.py
```

**Expected output:**
```
✅ PASS Python Version
✅ PASS Dependencies  
✅ PASS Project Structure
✅ PASS Dataset Files
✅ PASS GPU Support
```

**If it fails:**
- Install dependencies: `pip install -r requirements.txt`
- Check that dataset .txt files are in `datasets/` directory
- Verify you're in the correct project directory

---

### **Step 2: Data Preprocessing** 📊
**What it does:** Converts ETH/UCY .txt files to .npz format required by SIAT
**Input:** `.txt` files in `datasets/` (format: frame_id, pedestrian_id, x, y)
**Output:** `.npz` files in `data_npz/`
**Script:** `scripts/step2_preprocess_data.py`

```bash
python scripts/step2_preprocess_data.py --input_dir ./datasets --output_dir ./data_npz
```

**Options:**
- `--obs_len 8`: Number of observed timesteps
- `--pred_len 12`: Number of prediction timesteps

**Expected output:**
```
✅ Successfully processed: 15 files
📁 Output directory: ./data_npz
📦 Created 15 .npz files
```

---

### **Step 3: Compatibility Test** 🧪
**What it does:** Tests if your model can process the preprocessed data
**Script:** `scripts/step3_test_compatibility.py`

```bash
python scripts/step3_test_compatibility.py
```

**Expected output:**
```
✅ PASS Dataset Loading
✅ PASS DataLoader
✅ PASS Model Initialization
✅ PASS Forward Pass
✅ PASS Training Step
```

**If it fails:** Check that step 2 completed successfully

---

### **Step 4: Model Training** 🏋️
**What it does:** Trains the SIAT model on your preprocessed data
**Script:** `scripts/step4_train_model.py`

```bash
python scripts/step4_train_model.py --data_dir ./data_npz --epochs 50
```

**Key options:**
- `--epochs 50`: Number of training epochs
- `--batch_size 32`: Batch size
- `--lr 0.001`: Learning rate
- `--checkpoint_dir ./checkpoints`: Where to save models
- `--early_stop 15`: Early stopping patience

**Expected output:**
```
Epoch  10/50 | Loss 0.045123 | ADE 0.8234 | FDE 1.5432 | Time 2.3m
💾 New best model saved! ADE: 0.8234
...
🎉 Training completed! Best ADE: 0.7845
```

**Monitor training:**
- Loss should decrease over time
- ADE/FDE should decrease (lower is better)
- Models are saved to `checkpoints/`

---

### **Step 5: Model Evaluation** 📈
**What it does:** Evaluates trained model and creates visualizations
**Script:** `scripts/step5_evaluate_model.py`

```bash
python scripts/step5_evaluate_model.py --checkpoint ./checkpoints/best_model.pth --visualize
```

**Options:**
- `--visualize`: Create trajectory visualizations
- `--num_samples 10`: Number of samples to visualize
- `--output_dir ./results`: Where to save results

**Expected output:**
```
📊 Final Results:
   - ADE: 0.7845
   - FDE: 1.4532
   - Samples: 5000
   - Results saved to: ./results
```

**Generated files:**
- `results/evaluation_results.json`: Numerical results
- `results/trajectory_*.png`: Individual trajectory plots
- `results/error_analysis.png`: Error analysis plots

## 🚨 Troubleshooting

### **Common Issues:**

**"No .txt files found"**
- Solution: Ensure your dataset files are in the correct format and location
- Check: `datasets/` directory contains .txt files with format: `frame_id	pedestrian_id	x	y`

**"Import errors"**
- Solution: Install dependencies: `pip install -r requirements.txt`
- Add pandas if missing: `pip install pandas`

**"CUDA out of memory"**
- Solution: Reduce batch size: `--batch_size 16` or `--batch_size 8`
- Use CPU: `--device cpu`

**"No .npz files found"**
- Solution: Run preprocessing first: `python scripts/step2_preprocess_data.py`

**"Checkpoint not found"**
- Solution: Complete training first: `python scripts/step4_train_model.py`

### **Performance Tips:**

**For faster training:**
- Use GPU if available (automatically detected)
- Increase batch size if you have enough memory
- Use multiple workers: `--num_workers 4`

**For better results:**
- Train for more epochs: `--epochs 100`
- Experiment with learning rate: `--lr 0.0005`
- Try different model sizes: `--embed_size 128`

**For debugging:**
- Run with fewer epochs: `--epochs 10`
- Use smaller datasets
- Check individual steps: `python run_pipeline.py --only-step 3`

## 📞 Getting Help

**Check logs:** Each script provides detailed progress information

**Run specific steps:** Use `--only-step N` to run individual steps

**Skip problematic steps:** Use `--skip-step N` to skip steps

**Individual scripts:** Run scripts directly from `scripts/` directory

## 🎯 Next Steps After Training

1. **Analyze Results:** Check `results/evaluation_results.json`
2. **Visualize Predictions:** Look at `results/trajectory_*.png`
3. **Experiment:** Try different hyperparameters
4. **Deploy:** Use saved model for inference
5. **Improve:** Collect more data or try ensemble methods

## 📊 Expected Performance

Typical results on ETH/UCY datasets:
- **ADE:** 0.7-1.2 (lower is better)
- **FDE:** 1.2-2.5 (lower is better)
- **Training time:** 30-60 minutes (depends on dataset size and hardware)

Good luck with your SIAT model training! 🚀
