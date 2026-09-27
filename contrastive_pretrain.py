import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.utils import dropout_edge
from torch_geometric.datasets import EllipticBitcoinDataset

from feature_enrichment import enrich_elliptic_features
from fg_gnn_model import FeatureGatedGNN

# 1. GRAPH PERTURBATION AUGMENTATIONS
def augment_graph(x, edge_index, feat_drop_prob=0.15, edge_drop_prob=0.10):
    """
    Generates stochastic views of the transaction graph:
    - Feature Masking: Zeroes out random node attributes
    - Edge Dropping: Randomly drops transaction links
    """
    # 1. Feature Masking
    mask = torch.rand_like(x) > feat_drop_prob
    x_aug = x * mask

    # 2. Edge Dropping
    edge_index_aug, _ = dropout_edge(edge_index, p=edge_drop_prob, training=True)

    return x_aug, edge_index_aug

# 2. INFONCE CONTRASTIVE LOSS

class InfoNCELoss(nn.Module):
    """
    Multi-node InfoNCE Contrastive Loss to maximize mutual information
    between positive node pairs across augmented graph views.
    """
    def __init__(self, temperature=0.2):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        # Normalize representations to unit sphere
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        # Batch-wise cosine similarity computation
        # Positive pairs: diagonal elements
        sim_matrix = torch.mm(z1, z2.t()) / self.temperature
        
        # Cross-entropy alignment
        labels = torch.arange(z1.size(0), device=z1.device)
        loss_1 = F.cross_entropy(sim_matrix, labels)
        loss_2 = F.cross_entropy(sim_matrix.t(), labels)

        return (loss_1 + loss_2) / 2.0

# 3. PRE-TRAINING ENGINE

def pretrain_gcpal(data, epochs=30, batch_size=4096, hidden_channels=128):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f" Starting GCPAL Contrastive Pre-Training on {device}...")
    print(f" Total Unlabeled + Labeled Transactions: {data.num_nodes:,}")

    # Initialize GNN with 168 input features
    model = FeatureGatedGNN(in_channels=data.x.size(1), hidden_channels=hidden_channels, out_channels=2).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    criterion = InfoNCELoss(temperature=0.2)

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)

    model.train()
    num_nodes = x.size(0)

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        # Step A: Create two augmented views of the entire transaction graph
        x1, edge1 = augment_graph(x, edge_index, feat_drop_prob=0.15, edge_drop_prob=0.10)
        x2, edge2 = augment_graph(x, edge_index, feat_drop_prob=0.15, edge_drop_prob=0.10)

        # Step B: Forward pass on both views through GNN encoder
        out1, gamma1, h1 = model(x1, edge1)
        out2, gamma2, h2 = model(x2, edge2)

        # Step C: Subsample batch of nodes to compute contrastive loss within memory bounds
        sampled_indices = torch.randperm(num_nodes)[:batch_size]
        z1_batch = out1[sampled_indices]
        z2_batch = out2[sampled_indices]

        loss = criterion(z1_batch, z2_batch)
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0 or epoch == 1:
            print(f"    Pre-train Epoch {epoch:02d}/{epochs:02d} | Contrastive InfoNCE Loss: {loss.item():.4f}")
    
    torch.save(model.state_dict(), "pretrained_gcpal_encoder.pth")

if __name__ == "__main__":
    # Enrich dataset with degree & flow attributes
    data = enrich_elliptic_features()
    # Run self-supervised contrastive pre-training
    pretrain_gcpal(data, epochs=25, batch_size=4096)