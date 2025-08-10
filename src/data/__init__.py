"""
Data module initialization.
"""

from .dataset import TrajectoryDataset, collate_fn

__all__ = ['TrajectoryDataset', 'collate_fn']
