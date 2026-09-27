# Capacity-Governed Anti-Money Laundering Pipeline (Hybrid GNN + XGBoost)

## Project Overview
This repository contains the code for an advanced Machine Learning pipeline designed to detect illicit money laundering transactions within cryptocurrency networks. This project was developed as part of an M.Tech dissertation in Artificial Intelligence and Data Science.

Standard fraud detection models often struggle with complex financial data: pure tabular models (like standard Random Forests) cannot see the flow of money between accounts, while standard Graph Neural Networks (GNNs) tend to overlook exact financial thresholds (like specific transaction fees). 

This project solves this by combining the topological awareness of a Feature-Gated GNN with the precise decision boundaries of an XGBoost classifier, capped off with an automated Large Language Model (LLM) agent that generates regulatory paperwork.

## Dataset & Evaluation
* **Dataset:** The Elliptic Bitcoin Dataset (203k transactions, 234k edges, 165 raw features).
* **Data Split:** A strict chronological split is used to simulate real-world conditions (Train on Time Steps $\le 34$, Test on Time Steps $\ge 35$). This prevents the model from "time-traveling" and inflating accuracy metrics.

## Pipeline Architecture
The project is divided into five distinct modules:

1. **Feature Engineering (`feature_enrichment.py`):** 
   Extracts network metrics directly from the graph edges (In-Degree, Out-Degree, and Flow Imbalance) to help the model identify structural laundering typologies like "smurfing" (fan-out) or "pooling" (fan-in). Extends the feature space to 168 dimensions.

2. **Unsupervised Graph Pre-Training (`contrastive_pretrain.py`):** 
   Because 77% of the dataset is unlabeled, this module uses Graph Contrastive Learning (GCPAL) with InfoNCE loss to train the network on the entire dataset. By masking features and dropping edges, the model learns the global flow of the blockchain before looking at any crime labels.

3. **Feature-Gated GNN (`train_eval.py` & `fg_gnn_model.py`):** 
   A custom GNN that uses an adaptive gate ($\gamma$) to dynamically decide whether to trust a node's graph neighborhood or its raw transaction features. It is fine-tuned using Focal Loss to handle severe class imbalance.

4. **Hybrid XGBoost Classifier (`hybrid_xgboost.py`):** 
   The pipeline extracts the 128-dimensional topological embeddings from the GNN and concatenates them with the 168 raw/engineered features. This 296-dimensional vector is passed into an XGBoost ensemble, which achieves exceptional precision on the tabular data while maintaining graph awareness.

5. **Automated LLM Compliance Agent (`llm_sar_agent.py` & `subgraph_extractor.py`):** 
   When a suspicious transaction is flagged, a subgraph extractor isolates the local transaction network and saves it as a forensic JSON file. An automated agent powered by the Google GenAI API (Gemini 3.6-flash) then reads this evidence and writes a formal, FinCEN-compliant Suspicious Activity Report (SAR).

## Final Model Performance
Tested on unseen future transactions under a capacity-governed threshold optimizer:
* **Precision:** 92.24%
* **Recall (TPR):** 72.48%
* **F1-Score:** 81.18%

## Installation & Setup

1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/your-repo-name.git](https://github.com/yourusername/your-repo-name.git)
   cd your-repo-name
