import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class FeatureGatedGNN(nn.Module):
    def __init__(self, in_channels: int = 165, hidden_channels: int = 128, out_channels: int = 2, dropout: float = 0.3):
        super(FeatureGatedGNN, self).__init__()
        self.dropout = dropout

        # 1. Input Normalization
        self.input_bn = nn.BatchNorm1d(in_channels)

        # 2. Branch 1: Dynamic Graph Attention (GATv2)
        self.gat1 = GATv2Conv(in_channels, hidden_channels // 2, heads=2, concat=True)
        self.gat_bn1 = nn.BatchNorm1d(hidden_channels)
        
        self.gat2 = GATv2Conv(hidden_channels, hidden_channels // 2, heads=2, concat=True)
        self.gat_bn2 = nn.BatchNorm1d(hidden_channels)

        # 3. Branch 2: Tabular Residual Pathway
        self.tabular_proj1 = nn.Linear(in_channels, hidden_channels)
        self.tabular_bn = nn.BatchNorm1d(hidden_channels)
        self.tabular_proj2 = nn.Linear(hidden_channels, hidden_channels)

        # 4. Adaptive Gating Module (Gamma: decides Graph vs. Tabular trust)
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid()
        )

        # 5. Output Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        # Normalize raw node attributes
        x_norm = self.input_bn(x)

        # Graph Attention Branch (Z)
        z = self.gat1(x_norm, edge_index)
        z = self.gat_bn1(z)
        z = F.elu(z)
        z = F.dropout(z, p=self.dropout, training=self.training)
        
        z = self.gat2(z, edge_index)
        z = self.gat_bn2(z)
        z = F.elu(z)

        # Tabular Residual Branch (S)
        s = self.tabular_proj1(x_norm)
        s = self.tabular_bn(s)
        s = F.elu(s)
        s = F.dropout(s, p=self.dropout, training=self.training)
        s = self.tabular_proj2(s)
        s = F.elu(s)

        # Node-Wise Adaptive Gating
        gate_input = torch.cat([z, s], dim=-1)
        gamma = self.gate_mlp(gate_input)

        # Fuse Graph Structure with Tabular Features
        h = gamma * z + (1.0 - gamma) * s

        # Final Classification
        out = self.classifier(h)
        return out, gamma, h