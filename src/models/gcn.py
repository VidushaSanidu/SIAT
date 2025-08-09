"""
Graph Convolutional Network layers for social interaction modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    """
    Simple Graph Convolutional Network layer.
    """
    
    def __init__(self, in_feats: int, out_feats: int, bias: bool = True):
        """
        Initialize GCN layer.
        
        Args:
            in_feats: Input feature dimension
            out_feats: Output feature dimension
            bias: Whether to use bias in linear transformation
        """
        super().__init__()
        self.linear = nn.Linear(in_feats, out_feats, bias=bias)

    def forward(self, X: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of GCN layer.
        
        Args:
            X: Node features of shape (B, N, F)
            A_norm: Normalized adjacency matrix of shape (B, N, N)
            
        Returns:
            Updated node features of shape (B, N, out_feats)
        """
        # X: (B, N, F)
        # A_norm: (B, N, N)
        # message passing: A_norm @ X @ W
        H = torch.bmm(A_norm, X)  # (B, N, F)
        H = self.linear(H)
        return F.relu(H)
