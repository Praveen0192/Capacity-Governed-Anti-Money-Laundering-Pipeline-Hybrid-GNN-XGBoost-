import torch
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph

def extract_forensic_evidence(node_idx: int, data, model, k_hops: int = 2):
    node_idx = int(node_idx)
    
    model.eval()
    with torch.no_grad():
        logits, gamma, _ = model(data.x, data.edge_index)
        probs = F.softmax(logits, dim=-1)[:, 1]
        
        target_prob = float(probs[node_idx].item())
        gate_val = float(gamma[node_idx].item())

    # Step 1: Extract the k-hop subgraph around the target node
    subset_nodes, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx=node_idx,
        num_hops=k_hops,
        edge_index=data.edge_index,
        relabel_nodes=True
    )

    # Step 2: Grab key raw feature proxies (Amount proxy: col 1, Fee proxy: col 2, Timestep: col 0)
    raw_timestep = int(data.x[node_idx, 0].item())
    amount_proxy = round(float(data.x[node_idx, 1].item()), 4)
    fee_proxy = round(float(data.x[node_idx, 2].item()), 4)

    # Step 3: Construct the structured forensic dictionary
    forensic_payload = {
        "alert_metadata": {
            "target_transaction_id": int(node_idx),
            "time_step": raw_timestep,
            "illicit_probability": round(target_prob, 4),
            "alert_severity": "CRITICAL" if target_prob > 0.85 else "HIGH"
        },
        "model_reasoning_weights": {
            "structural_network_influence": round(gate_val, 4),
            "raw_transaction_feature_influence": round(1.0 - gate_val, 4),
            "primary_decision_driver": "Graph Topology (Layering/Funneling)" if gate_val > 0.5 else "Transaction Features (Amount/Velocity)"
        },
        "subgraph_topology": {
            "k_hop_depth": k_hops,
            "connected_counterparty_nodes": len(subset_nodes) - 1,
            "inter_entity_transactions": sub_edge_index.size(1),
            "network_density_flag": "High Connectivity Subgraph" if sub_edge_index.size(1) > (len(subset_nodes) * 1.5) else "Linear Chain"
        },
        "transaction_signals": {
            "normalized_amount_level": amount_proxy,
            "fee_anomaly_level": fee_proxy,
            "regulatory_typology": "Rapid Multi-Hop Layering and Smurfing Pattern"
        }
    }

    return forensic_payload

if __name__ == "__main__":
    print("[*] Subgraph Extractor is ready to serve.")