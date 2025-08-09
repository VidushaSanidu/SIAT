"""
Dataset classes for trajectory prediction.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    """
    Dataset class for ETH-UCY style trajectory prediction.
    
    This class loads preprocessed NumPy files containing trajectory samples.
    Each .npz file should contain:
    - 'observations': Target agent observed trajectories (N_samples, obs_len, 2)
    - 'futures': Target agent future trajectories (N_samples, pred_len, 2)  
    - 'windows': Full scene windows (N_samples,) with varying N_agents per sample
    
    The preprocessing ensures target agent is always at index 0 in each window.
    """

    def __init__(self, npz_files: list, obs_len: int = 8, pred_len: int = 12, transform=None):
        """
        Initialize the trajectory dataset.
        
        Args:
            npz_files: List of paths to .npz files containing preprocessed trajectory data
            obs_len: Number of observed timesteps
            pred_len: Number of predicted timesteps
            transform: Optional transform to apply to the data
        """
        self.samples = []
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.transform = transform

        for f in npz_files:
            data = np.load(f, allow_pickle=True)
            
            # Check if this is the new preprocessed format
            if 'observations' in data and 'futures' in data and 'windows' in data:
                observations = data['observations']
                futures = data['futures']
                windows = data['windows']
                
                for i in range(len(observations)):
                    obs = observations[i].astype(np.float32)
                    fut = futures[i].astype(np.float32)
                    window = windows[i].astype(np.float32)
                    self.samples.append((obs, fut, window))
            
            # Legacy format support
            elif 'trajectories' in data:
                trajs = data['trajectories']  # (N, T, 2)
                N, T, _ = trajs.shape
                for i in range(N):
                    for start in range(0, T - (obs_len + pred_len) + 1):
                        obs = trajs[i, start:start + obs_len]
                        fut = trajs[i, start + obs_len:start + obs_len + pred_len]
                        # Create window with target at index 0
                        window = trajs[:, start:start + obs_len + pred_len]
                        # Reorder so target is first
                        reordered_window = np.zeros_like(window)
                        reordered_window[0] = window[i]  # Target first
                        other_idx = 1
                        for j in range(N):
                            if j != i:
                                reordered_window[other_idx] = window[j]
                                other_idx += 1
                        self.samples.append((obs.astype(np.float32), fut.astype(np.float32), reordered_window.astype(np.float32)))
            
            else:
                raise ValueError(f"Unsupported .npz format in {f}. Expected 'observations'/'futures'/'windows' or 'trajectories' keys.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        obs, fut, window = self.samples[idx]
        if self.transform:
            obs, fut, window = self.transform(obs, fut, window)
        return {
            'obs': torch.from_numpy(obs),        # (obs_len, 2)
            'fut': torch.from_numpy(fut),        # (pred_len, 2)
            'window': torch.from_numpy(window)   # (N_agents, obs_len+pred_len, 2)
        }


def collate_fn(batch):
    """
    Custom collate function to handle variable number of agents per scene.
    
    Args:
        batch: List of samples from __getitem__
        
    Returns:
        Batched data with padded windows
    """
    obs_batch = torch.stack([item['obs'] for item in batch])
    fut_batch = torch.stack([item['fut'] for item in batch])
    
    # Find max number of agents in this batch
    max_agents = max(item['window'].size(0) for item in batch)
    
    # Pad windows to same size
    batch_size = len(batch)
    seq_len = batch[0]['window'].size(1)
    
    window_batch = torch.zeros(batch_size, max_agents, seq_len, 2)
    agent_masks = torch.zeros(batch_size, max_agents, dtype=torch.bool)
    
    for i, item in enumerate(batch):
        n_agents = item['window'].size(0)
        window_batch[i, :n_agents] = item['window']
        agent_masks[i, :n_agents] = True
    
    return {
        'obs': obs_batch,
        'fut': fut_batch,
        'window': window_batch,
        'agent_mask': agent_masks
    }
