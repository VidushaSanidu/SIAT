# SIAT: Social Interaction-Aware Transformer

A PyTorch implementation of the **Social Interaction-Aware Transformer (SIAT)** for pedestrian trajectory prediction. This model combines Transformer encoders/decoders with Graph Convolutional Networks to capture both temporal dependencies and social interactions in multi-agent scenarios.

## 🏗️ Architecture

SIAT integrates two key components:
- **Transformer Networks**: Capture temporal dependencies in pedestrian trajectories
- **Graph Convolutional Networks (GCN)**: Model social interactions between pedestrians based on spatial proximity

The model processes observed trajectories (8 timesteps) to predict future trajectories (12 timesteps) while considering the influence of nearby pedestrians.

## 📁 Project Structure

```
SIAT/
├── src/                          # Source code
│   ├── models/                   # Model implementations
│   │   ├── siat.py              # Main SIAT model
│   │   └── gcn.py               # GCN layer implementation
│   ├── data/                     # Data loading and preprocessing
│   ├── training/                 # Training utilities
│   ├── utils/                    # Evaluation metrics and utilities
│   └── config.py                # Configuration settings
├── scripts/                      # Training pipeline scripts
├── data_npz/                     # Preprocessed trajectory data
├── datasets/                     # Raw dataset files
├── checkpoints/                  # Saved model checkpoints
├── requirements.txt              # Python dependencies
├── run_pipeline.py              # Master training script
└── example.py                   # Usage example
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PyTorch
- Additional dependencies listed in `requirements.txt`

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd SIAT
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

#### Option 1: Automated Pipeline (Recommended)

Run the complete training pipeline:
```bash
python run_pipeline.py
```

This script will automatically:
1. Check environment setup
2. Preprocess data
3. Test model compatibility
4. Train the model
5. Evaluate results

#### Option 2: Manual Steps

1. **Preprocess data:**
```bash
python scripts/step2_preprocess_data.py --input_dir ./datasets --output_dir ./data_npz
```

2. **Train the model:**
```bash
python scripts/step4_train_model.py --data_dir ./data_npz --epochs 50 --batch_size 32
```

3. **Evaluate the model:**
```bash
python scripts/step5_evaluate_model.py --checkpoint ./checkpoints/best_model.pth --data_dir ./data_npz
```

#### Option 3: Basic Example

For a quick demonstration:
```bash
python example.py
```

## 📊 Model Configuration

The model can be configured through `src/config.py`:

```python
@dataclass
class ModelConfig:
    obs_len: int = 8          # Observation length (timesteps)
    pred_len: int = 12        # Prediction length (timesteps)
    embed_size: int = 64      # Embedding dimension
    enc_layers: int = 2       # Transformer encoder layers
    dec_layers: int = 1       # Transformer decoder layers
    nhead: int = 4            # Attention heads
    gcn_hidden: int = 64      # GCN hidden dimension
    gcn_layers: int = 2       # Number of GCN layers
    dropout: float = 0.1      # Dropout rate
```

## 📈 Datasets

The model supports trajectory datasets in NPZ format. The pipeline includes preprocessing scripts for common pedestrian datasets:

- **ETH/UCY datasets**: Standard benchmarks for pedestrian trajectory prediction
- **Custom datasets**: Any trajectory data can be preprocessed using the provided scripts

### Data Format

Input trajectories should be in NPZ format with:
- `trajectories`: Array of shape `(n_agents, timesteps, 2)` containing x,y coordinates
- Preprocessed data includes both observed and future timesteps

## 🧠 Model Details

### Input
- **obs**: Target pedestrian observations `(batch_size, obs_len, 2)`
- **full_window**: All agents in scene `(batch_size, n_agents, obs_len+pred_len, 2)`
- **agent_mask**: Valid agent indicators `(batch_size, n_agents)`

### Output
- **pred**: Predicted future trajectory `(batch_size, pred_len, 2)`

### Key Features
- **Transformer Encoder**: Processes agent embeddings to capture temporal patterns
- **GCN**: Models social interactions via distance-based adjacency matrices
- **Feature Fusion**: Combines transformer and GCN outputs with learnable weights
- **Transformer Decoder**: Generates future trajectory predictions

## 📊 Evaluation Metrics

The model is evaluated using standard trajectory prediction metrics:
- **ADE** (Average Displacement Error): Average L2 distance across all predicted points
- **FDE** (Final Displacement Error): L2 distance at the final predicted point

## 🔧 Pipeline Options

The `run_pipeline.py` script supports various options:

```bash
# Run specific step only
python run_pipeline.py --only-step 4

# Skip a step (useful if already completed)
python run_pipeline.py --skip-step 1

# Quick test with reduced epochs
python run_pipeline.py --quick
```

## 📝 Example Usage

```python
import torch
from src.models import SIAT
from src.config import Config

# Initialize model
config = Config()
model = SIAT(
    obs_len=config.model.obs_len,
    pred_len=config.model.pred_len,
    embed_size=config.model.embed_size
)

# Forward pass
obs = torch.randn(1, 8, 2)          # Observed trajectory
window = torch.randn(1, 5, 20, 2)   # Full scene window
pred = model(obs, window)            # Predicted trajectory
```

## 🏆 Results

After training, you'll find:
- **Model checkpoint**: `./checkpoints/best_model.pth`
- **Evaluation results**: `./results/evaluation_results.json`
- **Visualizations**: `./results/` (if enabled)

## 🤝 Contributing

This implementation is designed for research purposes. Key areas for contribution:
- Additional dataset support
- Alternative attention mechanisms
- Multi-modal prediction capabilities
- Performance optimizations

## 📚 Citation

If you use this implementation in your research, please cite the original SIAT paper:

```bibtex
@article{siat2023,
  title={Social Interaction-Aware Transformer for Pedestrian Trajectory Prediction},
  author={[Authors]},
  journal={[Journal]},
  year={2023}
}
```

## 📄 License

This project is released under the MIT License. See `LICENSE` file for details.

## 🔍 Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size in config
2. **Data not found**: Ensure datasets are in the correct directory structure
3. **Import errors**: Verify all dependencies are installed

### Getting Help

- Check the pipeline logs for detailed error messages
- Run individual steps to isolate issues
- Verify data preprocessing completed successfully

---

**Note**: This implementation is for research and educational purposes. Performance may vary depending on dataset and hyperparameter settings.
