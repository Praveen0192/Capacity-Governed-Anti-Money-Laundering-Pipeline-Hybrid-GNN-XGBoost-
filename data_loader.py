import torch
from torch_geometric.datasets import EllipticBitcoinDataset

def load_and_split_elliptic(data_dir='./data/Elliptic'):
    print("[*] Loading Elliptic Bitcoin Dataset...")
    dataset = EllipticBitcoinDataset(root=data_dir)
    data = dataset[0]

    # y mapping: 0 = Licit (Legal), 1 = Illicit (Laundering), 2 = Unknown
    
    train_mask = data.train_mask
    test_mask = data.test_mask
    labeled_mask = (data.y != 2)


    total_nodes = data.num_nodes
    total_edges = data.edge_index.size(1)
    num_features = data.num_features

    train_labeled = train_mask.sum().item()
    test_labeled = test_mask.sum().item()
    train_illicit = ((data.y == 1) & train_mask).sum().item()
    test_illicit = ((data.y == 1) & test_mask).sum().item()

    print("\n" + "="*50)
    print("      ELLIPTIC DATASET SUMMARY")
    print("="*50)
    print(f"Total Transactions (Nodes): {total_nodes:,}")
    print(f"Total Payment Flows (Edges):{total_edges:,}")
    print(f"Feature Dimensions per Node:{num_features}")
    print(f"Training Labeled Nodes:     {train_labeled:,} (Illicit: {train_illicit})")
    print(f"Testing Labeled Nodes:      {test_labeled:,} (Illicit: {test_illicit})")
    print("="*50)

    return data, train_mask, test_mask

if __name__ == "__main__":
    data, train_mask, test_mask = load_and_split_elliptic()