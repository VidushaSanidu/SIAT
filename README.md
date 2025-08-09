# SIAT: Social Interaction-Aware Transformer

A PyTorch implementation of the Social Interaction-Aware Transformer for pedestrian trajectory prediction.

## Overview

SIAT combines Transformer encoders/decoders with Graph Convolutional Networks to capture both temporal dependencies and social interactions for accurate pedestrian trajectory prediction.

## Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules for models, data, training, and utilities
- **Configurable**: Easy configuration management through dataclasses
- **Extensible**: Well-documented code structure for easy extension and modification
- **Reproducible**: Consistent training and evaluation protocols

## Project Structure

```
SIAT/
├── src/
│   ├── __init__.py
│   ├── config.py                 # Configuration management
│   ├── models/
│   │   ├── __init__.py
│   │   ├── siat.py              # Main SIAT model
│   │   └── gcn.py               # Graph Convolutional Network layers
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py           # Dataset classes
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py           # Training and evaluation functions
│   └── utils/
│       ├── __init__.py
│       └── metrics.py           # Evaluation metrics (ADE, FDE)
├── train.py                     # Main training script
├── evaluate.py                  # Model evaluation script
├── requirements.txt             # Python dependencies
├── setup.py                     # Package installation
└── README.md                    # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd SIAT
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Data Preparation

The model expects preprocessed data in NumPy `.npz` format. Each file should contain:

- `trajectories`: Array of shape `(N_agents, T_total, 2)` where:
  - `N_agents`: Number of agents in the scene
  - `T_total`: Total time steps
  - `2`: x, y coordinates

### ETH/UCY Dataset

For ETH/UCY datasets, convert track files into `.npz` format with the expected structure. Ensure coordinates are in world coordinates (e.g., meters) and consider per-scene normalization if coordinates vary significantly between scenes.

## Usage

### Training

```bash
python train.py --data_dir ./data_npz --epochs 50 --batch_size 32 --lr 0.001
```

### Evaluation

```bash
python evaluate.py --model_path ./checkpoints/siat_model.pth --data_dir ./data_npz
```

### Programmatic Usage

```python
import torch
from src.models import SIAT
from src.data import TrajectoryDataset
from src.training import train_one_epoch, evaluate

# Initialize model
model = SIAT(obs_len=8, pred_len=12, embed_size=64)

# Load data
dataset = TrajectoryDataset(npz_files, obs_len=8, pred_len=12)
loader = torch.utils.data.DataLoader(dataset, batch_size=32)

# Train
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss = train_one_epoch(model, optimizer, loader, device='cuda')

# Evaluate
ade, fde = evaluate(model, loader, device='cuda')
```

## Model Architecture

The SIAT model consists of:

1. **Embedding Layer**: Projects flattened trajectory sequences to embedding space
2. **Transformer Encoder**: Processes agent embeddings to capture temporal dependencies
3. **Graph Convolutional Network**: Models social interactions through dynamic adjacency matrices
4. **Feature Fusion**: Combines transformer and GCN features with learnable weights
5. **Transformer Decoder**: Generates future trajectory predictions
6. **Regression Head**: Outputs final coordinate predictions

## Configuration

Model and training parameters can be configured through the `Config` class in `src/config.py`:

```python
from src.config import Config

config = Config()
config.model.embed_size = 128
config.training.learning_rate = 0.0005
config.data.obs_len = 8
```

## Evaluation Metrics

- **ADE (Average Displacement Error)**: Mean Euclidean distance between predicted and ground truth across all timesteps
- **FDE (Final Displacement Error)**: Euclidean distance between predicted and ground truth final positions

## Citation

If you use this code in your research, please cite the original SIAT paper:

```bibtex
@article{siat2023,
  title={SIAT: Pedestrian trajectory prediction via social interaction-aware transformer},
  author={[Original Authors]},
  journal={[Journal Name]},
  year={2023}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Troubleshooting

### Common Issues

1. **No .npz files found**: Ensure your data is preprocessed and placed in the correct directory
2. **CUDA out of memory**: Reduce batch size or model dimensions
3. **Import errors**: Make sure the package is installed with `pip install -e .`

For more issues, please check the GitHub issues page or create a new issue.
