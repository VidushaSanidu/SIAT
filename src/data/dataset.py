"""
Dataset classes for trajectory prediction.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    """
    Minimal dataset class for ETH-UCY / SDD style sample files.
    Expected input format per scene: a list/array of agent trajectories where each
    trajectory is (T_total, 2) in absolute coordinates or normalized coordinates.

    This class expects preprocessed NumPy files (one .npz per scene) with keys:
    - 'trajectories': shape (N_agents, T_total, 2)

    You must preprocess ETH/UCY sequences to split into sliding windows of
    obs_len + pred_len. See DATA_PREPARE section in documentation.
    """

    def __init__(self, npz_files: list, obs_len: int = 8, pred_len: int = 12, transform=None):
        """
        Initialize the trajectory dataset.
        
        Args:
            npz_files: List of paths to .npz files containing trajectory data
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
            trajs = data['trajectories']  # (N, T, 2)
            # sliding windows per agent
            N, T, _ = trajs.shape
            for i in range(N):
                for start in range(0, T - (obs_len + pred_len) + 1):
                    obs = trajs[i, start:start + obs_len]
                    fut = trajs[i, start + obs_len:start + obs_len + pred_len]
                    # surrounding context: other agents in the same time window
                    # For simplicity we include all agents (including target) as nodes
                    window = trajs[:, start:start + obs_len + pred_len]
                    # store a tuple: (obs_target, fut_target, full_window_of_all_agents)
                    self.samples.append((obs.astype(np.float32), fut.astype(np.float32), window.astype(np.float32)))

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
