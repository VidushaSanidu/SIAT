"""
SIAT: Social Interaction-Aware Transformer

A PyTorch implementation of the SIAT architecture for pedestrian trajectory prediction.
"""

from .models import SIAT, GCNLayer
from .data import TrajectoryDataset
from .utils import ade_fde
from .training import train_one_epoch, evaluate

__version__ = "1.0.0"
__author__ = "Generated for SIAT Research"

__all__ = [
    'SIAT',
    'GCNLayer', 
    'TrajectoryDataset',
    'ade_fde',
    'train_one_epoch',
    'evaluate'
]
