"""
SIAT: Social Interaction-Aware Transformer model implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .gcn import GCNLayer


class SIAT(nn.Module):
    """
    Social Interaction-Aware Transformer for pedestrian trajectory prediction.
    
    This model combines Transformer encoders/decoders with Graph Convolutional Networks
    to capture both temporal dependencies and social interactions.
    """

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
        """
        Initialize SIAT model.
        
        Args:
            obs_len: Number of observed timesteps
            pred_len: Number of predicted timesteps
            in_size: Input feature dimension (typically 2 for x,y coordinates)
            embed_size: Embedding dimension
            enc_layers: Number of transformer encoder layers
            dec_layers: Number of transformer decoder layers
            nhead: Number of attention heads
            gcn_hidden: Hidden dimension for GCN layers
            gcn_layers: Number of GCN layers
            dropout: Dropout rate
        """
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
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size, 
            nhead=nhead, 
            dim_feedforward=embed_size * 2,
            dropout=dropout, 
            activation='relu'
        )
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
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size, 
            nhead=nhead, 
            dim_feedforward=embed_size * 2,
            dropout=dropout, 
            activation='relu'
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=dec_layers)

        # Regression head to predict (pred_len * 2)
        self.reg_head = nn.Linear(embed_size, pred_len * 2)

        # initialization
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize model parameters using Xavier uniform initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def compute_adjacency(self, positions: torch.Tensor, sigma: float = 1.5, eps: float = 1e-6) -> torch.Tensor:
        """
        Compute normalized adjacency matrix using Gaussian kernel.
        
        Args:
            positions: Agent positions of shape (B, N, 2)
            sigma: Standard deviation for Gaussian kernel
            eps: Small epsilon for numerical stability
            
        Returns:
            Normalized adjacency matrix of shape (B, N, N)
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

    def forward(self, obs: torch.Tensor, full_window: torch.Tensor, agent_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of SIAT model.
        
        Args:
            obs: Target pedestrian observation of shape (B, obs_len, 2)
            full_window: All agents positions in window of shape (B, N_agents, obs_len+pred_len, 2)
            agent_mask: Boolean mask indicating valid agents of shape (B, N_agents)

        Returns:
            Predicted future trajectory for the target of shape (B, pred_len, 2)
        """
        B = obs.size(0)
        device = obs.device

        # Prepare per-agent embeddings: flatten each agent's (obs+pred) window
        N = full_window.size(1)
        seq_len = full_window.size(2)

        # Flatten each agent's full window
        agent_flat = full_window.view(B, N, -1)  # (B, N, seq_len*2)
        agent_emb = self.embedding(agent_flat)  # (B, N, embed)

        # Apply agent mask if provided
        if agent_mask is not None:
            # Zero out embeddings for invalid agents
            agent_emb = agent_emb * agent_mask.unsqueeze(-1).float()

        # Transformer encoder: process sequence of agent embeddings
        # TransformerEncoder expects (S, B, E) format
        agent_emb_t = agent_emb.permute(1, 0, 2)  # (N, B, E)
        
        # Create attention mask for transformer if we have agent_mask
        if agent_mask is not None:
            # Transformer mask: True means ignore, False means attend
            src_key_padding_mask = ~agent_mask  # (B, N)
        else:
            src_key_padding_mask = None
            
        trans_enc_out = self.transformer_encoder(
            agent_emb_t, 
            src_key_padding_mask=src_key_padding_mask
        )  # (N, B, E)
        trans_enc_out = trans_enc_out.permute(1, 0, 2)  # (B, N, E)

        # GCN: compute adjacency per sample using the last observed positions
        last_pos = full_window[:, :, self.obs_len - 1, :]  # (B, N, 2)
        A_norm = self.compute_adjacency(last_pos)  # (B, N, N)
        
        # Apply agent mask to adjacency matrix if provided
        if agent_mask is not None:
            mask_matrix = agent_mask.unsqueeze(-1) & agent_mask.unsqueeze(-2)  # (B, N, N)
            A_norm = A_norm * mask_matrix.float()

        # Initial node features for GCN
        H = agent_emb  # (B, N, E)
        for layer in self.gcn:
            H = layer(H, A_norm)  # (B, N, gcn_hidden)

        # Project and fuse features
        H_trans_proj = self.fuse_trans(trans_enc_out)  # (B, N, embed)
        H_gcn_proj = self.fuse_gcn(H)  # (B, N, embed)

        H_fused = self.lambda1 * H_trans_proj + self.lambda2 * H_gcn_proj  # (B, N, embed)

        # For decoding, we want the target pedestrian's fused feature
        # Target pedestrian is guaranteed to be at index 0 after preprocessing
        target_feat = H_fused[:, 0, :].unsqueeze(0)  # (1, B, E)

        # Create decoder query tokens for prediction steps
        query = target_feat.repeat(self.pred_len, 1, 1)  # (pred_len, B, E)

        # Use the fused features as memory for decoder
        memory = H_fused.permute(1, 0, 2)  # (N, B, E)
        
        # Create memory key padding mask if needed
        if agent_mask is not None:
            memory_key_padding_mask = ~agent_mask  # (B, N)
        else:
            memory_key_padding_mask = None

        dec_out = self.transformer_decoder(
            tgt=query, 
            memory=memory,
            memory_key_padding_mask=memory_key_padding_mask
        )  # (pred_len, B, E)
        
        # Apply regressor on mean of decoder outputs per sample
        dec_out_mean = dec_out.permute(1, 0, 2).mean(dim=1)  # (B, E)

        reg = self.reg_head(dec_out_mean)  # (B, pred_len*2)
        pred = reg.view(B, self.pred_len, 2)
        return pred
