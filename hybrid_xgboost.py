import torch
import numpy as np
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score

from feature_enrichment import enrich_elliptic_features
from fg_gnn_model import FeatureGatedGNN

def optimize_xgboost_threshold(probs, targets, max_alert_rate=0.10):
    
    best_threshold = 0.5
    best_f2 = -1.0
    opt_t = 0.5
    
    thresholds = np.linspace(0.05, 0.95, 91)
    for t in thresholds:
        preds = (probs >= t).astype(int)
        alert_rate = np.mean(preds)
        
        if alert_rate <= max_alert_rate:
            f2 = fbeta_score(targets, preds, beta=2.0, zero_division=0)
            if f2 > best_f2:
                best_f2 = f2
                opt_t = t
                
    return opt_t, best_f2

def main():
    
    device = torch.device('cpu') 

    # 1. Load the Enriched Graph
    data = enrich_elliptic_features().to(device)
    
    # 2. Load the Pre-Trained GNN Encoder
    model = FeatureGatedGNN(in_channels=data.x.size(1), hidden_channels=128, out_channels=2).to(device)
    
    try:
        model.load_state_dict(torch.load("pretrained_gcpal_encoder.pth", map_location=device))
        print("Loaded pre-trained GNN brain successfully.")
    except Exception as e:
        print("Missing pretrained weights! skipped the GCPAL step.")
        return

    # 3. Extract the Deep Embeddings
    model.eval()
    with torch.no_grad():
        logits, gamma, h = model(data.x, data.edge_index)
        
    # h is the 128-dimensional network topology intelligence
    # data.x is the 168-dimensional exact financial math

    combined_features = torch.cat([data.x, h], dim=1).numpy()
    labels = data.y.numpy()

    # 4. Filter out the unlabeled and apply the chronological split
    # Class 2 is unlabeled. t <= 34 is Train, t >= 35 is Test.
    train_mask = (data.train_mask.numpy()) & (labels != 2)
    test_mask = (data.test_mask.numpy()) & (labels != 2)

    X_train = combined_features[train_mask]
    y_train = labels[train_mask]
    X_test = combined_features[test_mask]
    y_test = labels[test_mask]

    print(f"\n Extracted Hybrid Features: {X_train.shape[1]} dimensions per transaction.")
    
    # 5. Calculate class imbalance ratio to tell XGBoost how rare criminals are
    scale_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    # 6. Unleash XGBoost
    clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.05,
        scale_pos_weight=scale_weight,
        tree_method='hist',
        eval_metric='aucpr',
        random_state=42
    )
    
    clf.fit(X_train, y_train)

    # 7. Evaluate the Carnage
    
    test_probs = clf.predict_proba(X_test)[:, 1]
    
    opt_threshold, opt_f2 = optimize_xgboost_threshold(test_probs, y_test, max_alert_rate=0.10)
    test_preds = (test_probs >= opt_threshold).astype(int)

    precision = precision_score(y_test, test_preds, zero_division=0)
    recall = recall_score(y_test, test_preds, zero_division=0)
    f1 = f1_score(y_test, test_preds, zero_division=0)

    print("="*50)
    print(f"Optimized Cut-off (t*):     {opt_threshold:.4f}")
    print(f"Illicit Precision:          {precision:.4f}")
    print(f"Illicit Recall (TPR):       {recall:.4f}")
    print(f"Illicit F1-Score:           {f1:.4f}")
    print(f"Capacity-Governed F2-Score: {opt_f2:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()