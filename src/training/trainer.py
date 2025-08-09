"""
Training utilities for SIAT model.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..utils.metrics import ade_fde


def train_one_epoch(model: nn.Module, 
                   optimizer, 
                   loader: DataLoader, 
                   device: torch.device, 
                   clip: Optional[float] = 1.0) -> float:
    """
    Train the model for one epoch.
    
    Args:
        model: The SIAT model to train
        optimizer: Optimizer for training
        loader: Training data loader
        device: Device to run training on
        clip: Gradient clipping value (None to disable)
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    
    for batch in loader:
        obs = batch['obs'].to(device)               # (B, obs_len, 2)
        fut = batch['fut'].to(device)               # (B, pred_len, 2)
        window = batch['window'].to(device)         # (B, N, obs+pred, 2)

        optimizer.zero_grad()
        pred = model(obs, window)
        loss = F.mse_loss(pred, fut)
        loss.backward()
        
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            
        optimizer.step()
        total_loss += loss.item() * obs.size(0)
        
    return total_loss / len(loader.dataset)


def evaluate(model: nn.Module, 
            loader: DataLoader, 
            device: torch.device) -> tuple[float, float]:
    """
    Evaluate the model on validation/test data.
    
    Args:
        model: The SIAT model to evaluate
        loader: Evaluation data loader
        device: Device to run evaluation on
        
    Returns:
        Tuple of (ADE, FDE) metrics
    """
    model.eval()
    total_ade = 0.0
    total_fde = 0.0
    
    with torch.no_grad():
        for batch in loader:
            obs = batch['obs'].to(device)
            fut = batch['fut'].to(device)
            window = batch['window'].to(device)
            pred = model(obs, window)
            ade, fde = ade_fde(pred, fut)
            total_ade += ade * obs.size(0)
            total_fde += fde * obs.size(0)
            
    n = len(loader.dataset)
    return total_ade / n, total_fde / n
