"""
Configuration settings for SIAT model training and evaluation.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for SIAT model architecture."""
    obs_len: int = 8
    pred_len: int = 12
    in_size: int = 2
    embed_size: int = 64
    enc_layers: int = 2
    dec_layers: int = 1
    nhead: int = 4
    gcn_hidden: int = 64
    gcn_layers: int = 2
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    grad_clip: Optional[float] = 1.0
    device: str = 'cuda'
    
    
@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    data_dir: str = './data_npz'
    obs_len: int = 8
    pred_len: int = 12
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1


@dataclass
class Config:
    """Main configuration class combining all config components."""
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    data: DataConfig = DataConfig()
    
    def __post_init__(self):
        """Ensure consistency between configs."""
        # Sync observation and prediction lengths
        self.model.obs_len = self.data.obs_len
        self.model.pred_len = self.data.pred_len
