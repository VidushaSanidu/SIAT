# SIAT: Social Interaction-Aware Transformer

A PyTorch implementation of the **Social Interaction-Aware Transformer (SIAT)** for multi-agent pedestrian trajectory prediction. This model combines Transformer encoders/decoders with Graph Convolutional Networks (GCN) to capture both temporal motion dependencies and spatial/social interactions in crowded scenarios.

This project implements the model proposed in:
> **Research Paper:**  
> **Title:** *SIAT: Pedestrian trajectory prediction via social interaction-aware transformer*  
> **Authors:** Chengdong Wang, Jianming Wang, Wenbo Gao, Lei Guo  
> **Published in:** *Complex & Intelligent Systems*  
> **Link:** *https://link.springer.com/article/10.1007/s40747-025-01944-3*

*The implementation aims to reproduce the methodology described in the original paper as closely as possible. Since the original source code was not publicly available, certain implementation details not explicitly specified in the paper were necessarily inferred or selected based on the authors' descriptions.*

---



## 🏗️ Architecture

SIAT integrates two primary processing streams:
1. **Transformer Encoder-Decoder**: Captures temporal trajectory dynamics and sequential dependencies.
2. **Pedestrian Social Processing Module (GCN)**: Models social interactions dynamically by building an adjacency graph based on pedestrian spatial proximity (Gaussian kernel) at the observation horizon.
3. **Feature Fusion**: Combines temporal transformer representations and spatial GCN representations using learned weighting parameters ($\lambda_1, \lambda_2$).
4. **Regression Head**: Outputs future coordinate predictions across the prediction horizon.

```
                  ┌──────────────────────┐
                  │ Observed & Neighbor  │
                  │     Trajectories     │
                  └──────────┬───────────┘
                             │
                      Embedding Layer
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
   ┌──────────────────────┐      ┌──────────────────────┐
   │ Transformer Encoder  │      │ Spatial Graph & GCN  │
   │  (Temporal Dynamics) │      │ (Social Interaction) │
   └──────────┬───────────┘      └──────────┬───────────┘
              │                             │
              └──────────────┬──────────────┘
                             ▼
                    Feature Fusion (λ)
                             │
                             ▼
                    Transformer Decoder
                             │
                             ▼
                    Regression Head
                             │
                             ▼
                  Future Trajectories (x, y)
```

- **Observed horizon**: 8 timesteps (default, 3.2s @ 2.5Hz)
- **Prediction horizon**: 12 timesteps (default, 4.8s @ 2.5Hz)
- **Target benchmarks**: ETH / UCY datasets (ETH, HOTEL, UNIV, ZARA1, ZARA2)

---

## 📁 Repository Structure

```
SIAT/
├── src/
│   ├── models/
│   │   ├── siat.py          # SIAT model architecture
│   │   └── gcn.py           # Graph Convolutional Network layer (used in SIAT)
│   ├── data/
│   │   └── dataset.py       # TrajectoryDataset and collate_fn with masking
│   ├── training/
│   │   └── trainer.py       # Training loop and evaluation routines
│   ├── utils/
│   │   └── metrics.py       # ADE and FDE metrics calculation
│   └── config.py            # Dataclass configuration settings
├── scripts/
│   ├── step0_download_data.sh       # Script to download ETH/UCY datasets
│   ├── step1_check_environment.py   # Verify Python & PyTorch dependencies
│   ├── step2_preprocess_data.py     # Preprocess raw ETH/UCY txt files to .npz
│   ├── step3_test_compatibility.py  # Test model forward & backward passes
│   ├── step4_train_model.py         # Advanced training script with logging
│   └── step5_evaluate_model.py      # Evaluate checkpoints on ADE/FDE
├── train.py                 # Main root training entry point
├── SIAT_colab.ipynb         # Google Colab notebook for GPU training
├── requirements.txt         # Core Python dependencies
└── README.md
```

---

## 🚀 Getting Started

### 1. Installation

Create a virtual environment and install the required dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Dataset Preparation

Download the standard ETH/UCY benchmark datasets and convert them into model-ready `.npz` sliding windows:

```bash
# Step 0: Download raw ETH/UCY datasets
bash scripts/step0_download_data.sh

# Step 2: Preprocess trajectories into .npz files
python scripts/step2_preprocess_data.py --input_dir ./datasets --output_dir ./data_npz
```

### 3. Model Compatibility Test

Verify that the model architecture and data loaders run smoothly:

```bash
python scripts/step3_test_compatibility.py
```

---

## 🏋️ Training

### Quick Training
Run standard training with default hyperparameters:

```bash
python train.py --data_dir ./data_npz --epochs 50 --batch_size 32
```

### Advanced Training Options
You can configure model capacity and training settings via CLI flags:

```bash
python train.py \
    --data_dir ./data_npz \
    --epochs 60 \
    --batch_size 32 \
    --lr 0.001 \
    --embed_size 64 \
    --enc_layers 2 \
    --dec_layers 1 \
    --nhead 4 \
    --gcn_hidden 64 \
    --gcn_layers 2 \
    --device auto \
    --checkpoint_dir ./checkpoints
```

Alternatively, use the modular training script:

```bash
python scripts/step4_train_model.py --data_dir ./data_npz --epochs 50 --batch_size 32
```

---

## 📊 Evaluation

Evaluate a saved checkpoint against the test dataset to compute Average Displacement Error (**ADE**) and Final Displacement Error (**FDE**):

```bash
python scripts/step5_evaluate_model.py \
    --checkpoint ./checkpoints/best_model.pth \
    --data_dir ./data_npz
```

---

## 💡 Python Usage Example

```python
import torch
from src.models import SIAT

# Dimensions: batch_size=4, num_agents=5, obs_len=8, pred_len=12
obs_len, pred_len = 8, 12
B, N = 4, 5

# Initialize model
model = SIAT(obs_len=obs_len, pred_len=pred_len, embed_size=64)
model.eval()

# Inputs
target_obs = torch.randn(B, obs_len, 2)              # (B, 8, 2)
scene_window = torch.randn(B, N, obs_len+pred_len, 2) # (B, N, 20, 2)
agent_mask = torch.ones(B, N, dtype=torch.bool)       # (B, N)

# Forward pass
with torch.no_grad():
    predicted_fut = model(target_obs, scene_window, agent_mask)

print("Predicted future shape:", predicted_fut.shape)  # torch.Size([4, 12, 2])
```

---

## ☁️ Google Colab

To train on Google Colab with GPU acceleration, open `SIAT_colab.ipynb`. The notebook includes end-to-end data setup, model training, and evaluation cells.

---

## 📜 Metrics

- **ADE (Average Displacement Error)**: Mean Euclidean distance over all predicted timesteps between predicted trajectory and ground truth:
  $$\text{ADE} = \frac{1}{T_{pred}} \sum_{t=1}^{T_{pred}} \| \hat{Y}_t - Y_t \|_2$$
- **FDE (Final Displacement Error)**: Euclidean distance at the destination / final predicted timestep:
  $$\text{FDE} = \| \hat{Y}_{T_{pred}} - Y_{T_{pred}} \|_2$$
