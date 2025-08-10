"""
Training utilities for SIAT model.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.metrics import ade_fde


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
    
    # Create progress bar for training batches
    pbar = tqdm(loader, desc="Training", leave=False, ncols=80, dynamic_ncols=True)
    
    for batch in pbar:
        obs = batch['obs'].to(device)               # (B, obs_len, 2)
        fut = batch['fut'].to(device)               # (B, pred_len, 2)1
        
        window = batch['window'].to(device)         # (B, N, obs+pred, 2)
        agent_mask = batch['agent_mask'].to(device) # (B, N)

        optimizer.zero_grad()
        pred = model(obs, window, agent_mask)
        loss = F.mse_loss(pred, fut)
        loss.backward()
        
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            
        optimizer.step()
        batch_loss = loss.item()
        total_loss += batch_loss * obs.size(0)
        
        # Update progress bar with current loss
        pbar.set_postfix({'loss': f'{batch_loss:.6f}'})
        
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
    
    # Create progress bar for evaluation batches
    pbar = tqdm(loader, desc="Evaluating", leave=False, ncols=80, dynamic_ncols=True)
    
    with torch.no_grad():
        for batch in pbar:
            obs = batch['obs'].to(device)
            fut = batch['fut'].to(device)
            window = batch['window'].to(device)
            agent_mask = batch['agent_mask'].to(device)
            
            pred = model(obs, window, agent_mask)
            ade, fde = ade_fde(pred, fut)
            total_ade += ade * obs.size(0)
            total_fde += fde * obs.size(0)
            
            # Update progress bar with current metrics
            pbar.set_postfix({'ADE': f'{ade:.4f}', 'FDE': f'{fde:.4f}'})
            
    n = len(loader.dataset)
    return total_ade / n, total_fde / n
