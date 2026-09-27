import torch
import numpy as np
from torch_geometric.datasets import EllipticBitcoinDataset
from torch_geometric.utils import degree

def enrich_elliptic_features(data_dir='./data/Elliptic'):
    
    dataset = EllipticBitcoinDataset(root=data_dir)
    data = dataset[0]

    row, col = data.edge_index

    # 1. Compute In-Degree and Out-Degree per node
    in_deg = degree(col, num_nodes=data.num_nodes).unsqueeze(1)
    out_deg = degree(row, num_nodes=data.num_nodes).unsqueeze(1)

    # 2. Compute Total Degree and Flow Imbalance Ratio: (Out - In) / (Out + In + eps)
    total_deg = in_deg + out_deg
    flow_imbalance = (out_deg - in_deg) / (total_deg + 1e-6)

    # 3. Log-transform degrees to normalize scale
    log_in_deg = torch.log1p(in_deg)
    log_out_deg = torch.log1p(out_deg)

    # 4. Concatenate new engineered features with original 165 features
    engineered_features = torch.cat([log_in_deg, log_out_deg, flow_imbalance], dim=-1)
    enriched_x = torch.cat([data.x, engineered_features], dim=-1)

    print("\n" + "="*50)
    print("      DATASET FEATURE ENRICHMENT SUMMARY")
    print("="*50)
    print(f"Original Feature Dimension:   {data.x.size(1)}")
    print(f"New Injected Features:        {engineered_features.size(1)} (In-Deg, Out-Deg, Flow Imbalance)")
    print(f"Enriched Feature Dimension:   {enriched_x.size(1)}")
    print("="*50)

    # Update data object
    data.x = enriched_x
    return data

if __name__ == "__main__":
    enriched_data = enrich_elliptic_features()