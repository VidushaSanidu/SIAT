"""
Evaluation metrics for trajectory prediction.
"""

from typing import Tuple
import torch


def ade_fde(pred: torch.Tensor, gt: torch.Tensor) -> Tuple[float, float]:
    """
    Calculate Average Displacement Error (ADE) and Final Displacement Error (FDE).
    
    Args:
        pred: Predicted trajectories of shape (B, Tpred, 2)
        gt: Ground truth trajectories of shape (B, Tpred, 2)
    
    Returns:
        Tuple of (ADE, FDE) averaged over batch
    """
    err = torch.norm(pred - gt, dim=-1)  # (B, Tpred)
    ade = err.mean().item()
    fde = err[:, -1].mean().item()
    return ade, fde
