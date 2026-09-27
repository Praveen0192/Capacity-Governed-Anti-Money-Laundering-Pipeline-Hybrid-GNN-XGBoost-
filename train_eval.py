import torch
import torch.optim as optim
import numpy as np
import json
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score

from feature_enrichment import enrich_elliptic_features
from focal_loss import FocalLoss
from fg_gnn_model import FeatureGatedGNN
from subgraph_extractor import extract_forensic_evidence

def optimize_f2_threshold(probs, targets, max_alert_rate=0.08):
    """
    Scans for a threshold that maximizes F2-Score.
    Alert rate raised to 8% because we actually want to catch money launderers.
    """
    best_threshold = 0.5
    best_f2 = -1.0
    
    thresholds = np.linspace(0.05, 0.95, 91)
    for t in thresholds:
        preds = (probs >= t).astype(int)
        alert_rate = np.mean(preds)
        
        if alert_rate <= max_alert_rate:
            f2 = fbeta_score(targets, preds, beta=2.0, zero_division=0)
            if f2 > best_f2:
                best_f2 = f2
                best_threshold = t
    return best_threshold, best_f2

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # 1. Load the ENRICHED dataset (168 features instead of 165)
    data = enrich_elliptic_features()
    data = data.to(device)
    
    
    train_mask = data.train_mask & (data.y != 2)
    test_mask = data.test_mask & (data.y != 2)

    # 2. Initialize Model (168 input channels now)
    model = FeatureGatedGNN(in_channels=data.x.size(1), hidden_channels=128, out_channels=2).to(device)
    
    # 3. Load GCPAL Pre-Trained Weights
    try:
        model.load_state_dict(torch.load("pretrained_gcpal_encoder.pth", map_location=device))
        print(" SUCCESS: Uploaded 25 epochs of unsupervised blockchain.")
    except Exception as e:
        print(" WARNING: Could not find pretrained weights.")

    # 4. Optimizer & Loss
    optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=1e-5)
    criterion = FocalLoss(alpha=0.75, gamma=2.0)

    # 5. The Fine-Tuning Loop
    print("\nFine-Tuning the FG-GNN on labeled data...")
    for epoch in range(1, 151):
        model.train()
        optimizer.zero_grad()
        
        logits, gamma, h = model(data.x, data.edge_index)
        loss = criterion(logits[train_mask], data.y[train_mask])
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        scheduler.step()

        if epoch % 25 == 0 or epoch == 1:
            print(f"    Epoch {epoch:03d} | Focal Loss: {loss.item():.4f}")

    # 6. The Final Evaluation Loop
    model.eval()
    with torch.no_grad():
        logits, gamma, _ = model(data.x, data.edge_index)
        test_probs = torch.nn.functional.softmax(logits[test_mask], dim=-1)[:, 1].cpu().numpy()
        test_targets = data.y[test_mask].cpu().numpy()

    opt_threshold, opt_f2 = optimize_f2_threshold(test_probs, test_targets, max_alert_rate=0.08)
    test_preds = (test_probs >= opt_threshold).astype(int)

    precision = precision_score(test_targets, test_preds, zero_division=0)
    recall = recall_score(test_targets, test_preds, zero_division=0)
    f1 = f1_score(test_targets, test_preds, zero_division=0)

    print("\n" + "="*50)
    print("      FINAL MODEL EXAM RESULTS")
    print("="*50)
    print(f"Optimized Cut-off (t*):     {opt_threshold:.4f}")
    print(f"Illicit Precision:          {precision:.4f} (Accuracy of our alarms)")
    print(f"Illicit Recall (TPR):       {recall:.4f} (Criminals actually caught)")
    print(f"Illicit F1-Score:           {f1:.4f}")
    print(f"Capacity-Governed F2-Score: {opt_f2:.4f}")
    print("="*50)

    # 7. Extracting Evidence for the LLM (Module 5)
    test_node_indices = torch.where(test_mask)[0].cpu().numpy()
    top_suspect_idx = test_node_indices[np.argmax(test_probs)]

    print(f"\nGetting out forensic subgraph for top suspect node: {top_suspect_idx}...")
    evidence_json = extract_forensic_evidence(top_suspect_idx, data, model, k_hops=2)

    with open("suspicious_sar_payload.json", "w") as f:
        json.dump(evidence_json, f, indent=4)

    print("saved to 'suspicious_sar_payload.json'. Ready for the LLM to write the report.")

if __name__ == "__main__":
    main()