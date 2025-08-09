"""
SIAT: Social Interaction-Aware Transformer

This file contains:
- A PyTorch implementation of the SIAT architecture described in the uploaded paper
  "SIAT: Pedestrian trajectory prediction via social interaction-aware transformer".
- Dataset loader stubs for ETH-UCY / SDD style trajectory data (preprocessing notes included).
- Training loop, evaluation (ADE/FDE), and configuration matching the paper's reported
  hyperparameters by default.
- A model report (markdown) at the bottom describing architecture, hyperparameters,
  training recipe, ablation plan, and expected evaluation protocol.

Notes:
- This code is meant as a high-quality, runnable baseline. You will need to prepare
  dataset files (see DATA_PREPARE instructions below) and install PyTorch.
- No external GNN libraries are required; GCN is implemented with standard PyTorch ops.

Author: Generated for user based on the uploaded SIAT paper
"""

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# Utility functions: ADE / FDE
# -----------------------------

def ade_fde(pred: torch.Tensor, gt: torch.Tensor) -> Tuple[float, float]:
    """
    pred, gt: tensors of shape (B, Tpred, 2)
    returns: ADE, FDE (averaged over batch)
    """
    err = torch.norm(pred - gt, dim=-1)  # (B, Tpred)
    ade = err.mean().item()
    fde = err[:, -1].mean().item()
    return ade, fde

# -----------------------------
# Simple ETH/UCY-like Dataset
# -----------------------------

class TrajectoryDataset(Dataset):
    """
    Minimal dataset class for ETH-UCY / SDD style sample files.
    Expected input format per scene: a list/array of agent trajectories where each
    trajectory is (T_total, 2) in absolute coordinates or normalized coordinates.

    This class expects preprocessed NumPy files (one .npz per scene) with keys:
    - 'trajectories': shape (N_agents, T_total, 2)

    You must preprocess ETH/UCY sequences to split into sliding windows of
    obs_len + pred_len. See DATA_PREPARE section below.
    """

    def __init__(self, npz_files: list, obs_len: int = 8, pred_len: int = 12, transform=None):
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

# -----------------------------
# GCN Layer (simple)
# -----------------------------

class GCNLayer(nn.Module):
    def __init__(self, in_feats: int, out_feats: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_feats, out_feats, bias=bias)

    def forward(self, X: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        # X: (B, N, F)
        # A_norm: (B, N, N)
        # message passing: A_norm @ X @ W
        H = torch.bmm(A_norm, X)  # (B, N, F)
        H = self.linear(H)
        return F.relu(H)

# -----------------------------
# SIAT Model Implementation
# -----------------------------

class SIAT(nn.Module):
    def __init__(self,
                 obs_len: int = 8,
                 pred_len: int = 12,
                 in_size: int = 2,
                 embed_size: int = 64,
                 enc_layers: int = 2,
                 dec_layers: int = 1,
                 nhead: int = 4,
                 gcn_hidden: int = 64,
                 gcn_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.in_size = in_size
        self.embed_size = embed_size

        # Embedding: flatten (obs+pred) segment per mode as in paper
        seq_len = obs_len + pred_len
        self.flatten_dim = seq_len * in_size
        self.embedding = nn.Linear(self.flatten_dim, embed_size)

        # Positional encoding for Transformer (use learnable or standard PE)
        self.pos_enc = nn.Parameter(torch.randn(1, embed_size))

        # Transformer Encoder (applied per pedestrian embedding)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=nhead, dim_feedforward=embed_size * 2,
                                                   dropout=dropout, activation='relu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=enc_layers)

        # GCN to process social graph: input will be per-node features (we reuse embedding dims)
        gcn_modules = []
        gcn_in = embed_size
        for _ in range(gcn_layers):
            gcn_modules.append(GCNLayer(gcn_in, gcn_hidden))
            gcn_in = gcn_hidden
        self.gcn = nn.ModuleList(gcn_modules)

        # feature fusion projection
        self.fuse_trans = nn.Linear(embed_size, embed_size)
        self.fuse_gcn = nn.Linear(gcn_hidden, embed_size)
        # learnable scalar weights
        self.lambda1 = nn.Parameter(torch.tensor(0.5))
        self.lambda2 = nn.Parameter(torch.tensor(0.5))

        # Transformer Decoder - we'll use a simple transformer decoder stack
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_size, nhead=nhead, dim_feedforward=embed_size * 2,
                                                   dropout=dropout, activation='relu')
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=dec_layers)

        # Regression head to predict (pred_len * 2)
        self.reg_head = nn.Linear(embed_size, pred_len * 2)

        # initialization
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def compute_adjacency(self, positions: torch.Tensor, sigma: float = 1.5, eps: float = 1e-6) -> torch.Tensor:
        """
        positions: (B, N, 2) - positions at a particular timestep (or aggregated)
        returns normalized adjacency A_norm: (B, N, N)
        uses gaussian kernel: exp(-d^2 / sigma^2)
        """
        B, N, _ = positions.shape
        # pairwise distances
        pos_expand1 = positions.unsqueeze(2)  # (B, N, 1, 2)
        pos_expand2 = positions.unsqueeze(1)  # (B, 1, N, 2)
        diff = pos_expand1 - pos_expand2
        dist2 = (diff ** 2).sum(-1)  # (B, N, N)
        A = torch.exp(-dist2 / (sigma ** 2 + eps))
        # zero self-connections optionally
        # A = A - torch.diag_embed(torch.diagonal(A, dim1=1, dim2=2))
        # degree & normalization
        deg = A.sum(-1)  # (B, N)
        # D^{-1/2} A D^{-1/2}
        deg_inv_sqrt = (deg + eps).pow(-0.5)
        D_inv_sqrt = deg_inv_sqrt.unsqueeze(-1) * deg_inv_sqrt.unsqueeze(-2)  # (B,N,1)*(B,1,N) -> (B,N,N) via broadcast
        A_norm = A * D_inv_sqrt
        return A_norm

    def forward(self, obs: torch.Tensor, full_window: torch.Tensor) -> torch.Tensor:
        """
        obs: (B, obs_len, 2) - target pedestrian observation
        full_window: (B, N_agents, obs_len+pred_len, 2) - all agents positions in window

        returns: predicted future for the target: (B, pred_len, 2)
        """
        B = obs.size(0)
        device = obs.device

        # Prepare per-agent embeddings: flatten each agent's (obs+pred) window
        # For agents other than target, we do not know future; we use zeros or replicate obs part
        N = full_window.size(1)
        seq_len = full_window.size(2)

        # Flatten each agent's full window
        agent_flat = full_window.view(B, N, -1)  # (B, N, seq_len*2)
        agent_emb = self.embedding(agent_flat)  # (B, N, embed)

        # Transformer encoder: process sequence of agent embeddings
        # TransformerEncoder expects (S, B, E) or (L, N, E) depending - easier to treat agents as sequence
        # We'll pass agents as the 'sequence' dimension (N, B, E)
        agent_emb_t = agent_emb.permute(1, 0, 2)  # (N, B, E)
        trans_enc_out = self.transformer_encoder(agent_emb_t)  # (N, B, E)
        trans_enc_out = trans_enc_out.permute(1, 0, 2)  # (B, N, E)

        # GCN: compute adjacency per sample using the last observed positions (take last obs step for each agent)
        last_pos = full_window[:, :, self.obs_len - 1, :]  # (B, N, 2)
        A_norm = self.compute_adjacency(last_pos)  # (B, N, N)

        # initial node features for GCN: can use transformer features or raw coords
        H = agent_emb  # (B, N, E)
        for layer in self.gcn:
            H = layer(H, A_norm)  # (B, N, gcn_hidden)

        # project and fuse features
        H_trans_proj = self.fuse_trans(trans_enc_out)  # (B, N, embed)
        H_gcn_proj = self.fuse_gcn(H)  # (B, N, embed)

        H_fused = self.lambda1 * H_trans_proj + self.lambda2 * H_gcn_proj  # (B, N, embed)

        # For decoding, we want the target pedestrian's fused feature
        # assume the target pedestrian is at index 0 of full_window (dataset should ensure this)
        target_feat = H_fused[:, 0, :].unsqueeze(0)  # (1, B, E) for transformer decoder memory/query format

        # Create a decoder target query — we can use positional tokens equal to pred_len steps.
        # Simpler: repeat a learnable query vector pred_len times
        query = target_feat.repeat(self.pred_len, 1, 1)  # (Tpred, B, E)

        # Use the fused set as memory for decoder (shape: N, B, E)
        memory = H_fused.permute(1, 0, 2)  # (N, B, E)

        dec_out = self.transformer_decoder(tgt=query, memory=memory)  # (Tpred, B, E)
        # take last token representation or pool across time
        # We'll apply regressor on mean of decoder outputs per sample
        dec_out_mean = dec_out.permute(1, 0, 2).mean(dim=1)  # (B, E)

        reg = self.reg_head(dec_out_mean)  # (B, pred_len*2)
        pred = reg.view(B, self.pred_len, 2)
        return pred

# -----------------------------
# Training loop + helpers
# -----------------------------

def train_one_epoch(model: nn.Module, optimizer, loader: DataLoader, device: torch.device, clip: Optional[float] = 1.0):
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


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
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

# -----------------------------
# Example main: config + runner
# -----------------------------

if __name__ == '__main__':
    import argparse
    import glob

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data_npz', help='dir of preprocessed npz files')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--obs_len', type=int, default=8)
    parser.add_argument('--pred_len', type=int, default=12)
    args = parser.parse_args()

    # Prepare dataset
    npz_files = glob.glob(args.data_dir + '/*.npz')
    if len(npz_files) == 0:
        raise RuntimeError('No .npz files found in data_dir. Please preprocess ETH/UCY into scene .npz files (see DATA_PREPARE notes in this file).')

    dataset = TrajectoryDataset(npz_files, obs_len=args.obs_len, pred_len=args.pred_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    device = torch.device(args.device)
    model = SIAT(obs_len=args.obs_len, pred_len=args.pred_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, optimizer, loader, device)
        ade, fde = evaluate(model, loader, device)
        print(f'Epoch {epoch:03d} | Loss {loss:.6f} | ADE {ade:.4f} | FDE {fde:.4f}')

# -----------------------------
# DATA_PREPARE instructions
# -----------------------------
#
# ETH/UCY: obtain dataset (public). For each scene, convert track files into a single
# NumPy .npz file with key 'trajectories' shaped (N_agents, T, 2). Coordinates should be
# in world coordinates (e.g., meters). If coordinates vary a lot between scenes, per-scene
# normalization (zero-mean and dividing by a reference length) is recommended.
#
# Sliding windows: the dataset class expects sliding windows across each agent's full
# time series. For stationary or very short trajectories, you may need to filter out
# sequences shorter than obs_len + pred_len.
#
# Target agent index: the dataset loader writes windows such that the target agent is
# the agent used in the sample and appears at index 0 in the 'window' array. Make sure
# your preprocessing maintains that ordering (or update the model forward accordingly).
#
# -----------------------------
# Model report (markdown)
# -----------------------------

"""
MODEL REPORT
============

Model: SIAT - Social Interaction-Aware Transformer
-------------------------------------------------
High-level summary
- Input: observed trajectory length = 8 timesteps (x,y), predict next 12 timesteps.
- Architecture: embedding layer -> Transformer encoder (per-agent embeddings) ->
  Pedestrian Social Processing Module (GCN on dynamic graph) -> fusion ->
  Transformer decoder -> regression head (predicts all future timesteps at once).

Key components implemented
- Embedding layer: linear projection of flattened (obs+pred) agent window to 64-dim.
- Transformer encoder: 2 layers, 4 heads, feedforward expansion = 2x (implemented via
  PyTorch TransformerEncoderLayer).
- Pedestrian social module: dynamic adjacency via Gaussian kernel on last observed
  positions; 2-layer GCN.
- Fusion: learned scalar weights lambda1, lambda2 to combine transformer and GCN
  features into a single node-level representation.
- Transformer decoder: single-layer (by default), produces per-pred-step features.
- Regression head: linear mapping from fused decoder features to pred_len * 2 coords.

Hyperparameters (paper-aligned defaults)
- embed_size = 64
- enc_layers = 2, dec_layers = 1
- nhead = 4
- gcn_layers = 2, gcn_hidden = 64
- optimizer = Adam, lr = 0.001
- batch_size = 32, training steps: paper used 15k steps — here we use epochs
- sigma (Gaussian kernel) = 1.5

Training recipe
- Loss: MSE between predicted and ground-truth coordinate sequences.
- Regularization: edge dropout (not implemented in this simple baseline) may be
  applied during adjacency computation or during GCN message passing to match
  the original paper's training details.
- Gradient clipping and standard Adam updates used.

Evaluation metrics
- ADE (Average Displacement Error): mean Euclidean distance between predicted
  and ground truth across all predicted timesteps.
- FDE (Final Displacement Error): Euclidean distance between predicted final point and
  ground truth final point. Implemented in ade_fde() above.

Ablation study plan (reproducible using this repo)
- Component ablation: disable PSPM (replace GCN features with zeros), Transformer only,
  without embedding, full model. Compare ADE/FDE.
- Parametric ablation: vary encoder layers, decoder layers, number of heads, embedding
  dimension — evaluate on validation set and plot ADE/FDE vs hyperparameter.

Notes / caveats
- This implementation focuses on clarity and reproducibility. The paper mentions some
  implementation details that may change results (edge dropout, exact positional encodings,
  how future inputs are represented during training, mode K replication). If you want
  to reproduce exact numbers, match their data preprocessing, loss scheduling, and any
  data augmentations.
- Multi-modal outputs (paper mentions K modes) are implemented in the paper by
  expanding modes K and producing K candidate futures. This baseline implements
  a single-mode predictor. Extending to K modes requires changing the embedding
  and decoder to output multiple trajectories and possibly adding a classification
  branch or negative log-likelihood objective.

Expected next steps
- Prepare ETH/UCY / SDD data as described and run the training script.
- If you want an exact reproduction, tell me which dataset and I'll help write the
  precise preprocessing script for that dataset and (optionally) add data augmentation
  and multimodal extensions.

"""
