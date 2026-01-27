import sys
import os
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

# --- SYSTEM PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import spatial_training.gnn as gnn
from spatial_training.main import get_random_subgraph

# --- CONFIGURATION ---
CACHE_DIR = os.path.join(parent_dir, "cache_spatial")
OUTPUT_FOLDER = os.path.join(parent_dir, "saved_models_spatial")

PATH_X = os.path.join(CACHE_DIR, "x_features.pt")
PATH_TEST_EDGES = os.path.join(CACHE_DIR, "graph_test_edges.pt")
MODEL_PATH = os.path.join(OUTPUT_FOLDER, "best_model.pth")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_hard_negatives():
    print("\n--- HARD NEGATIVE STRESS TEST ---")
    
    # 1. Load Data
    if not os.path.exists(PATH_X):
        print("Error: Feature file not found.")
        return

    x_global = torch.load(PATH_X, weights_only=False)
    test_edges = torch.load(PATH_TEST_EDGES, weights_only=False).cpu()

    # --- CRITICAL: MATCH THE TRAINED 6-FEATURE MODEL ---
    # Dropping Soma Coordinates (Indices 3,4,5)
    features_to_keep = [0, 1, 2, 6, 7, 8]
    x_global = x_global[:, features_to_keep]
    print(f"Data Loaded. Nodes: {x_global.size(0)}. Features sliced to: {x_global.shape[1]}")
    # ---------------------------------------------------

    # 2. Load Model
    print(f"Loading Model from {MODEL_PATH}...")
    model = gnn.SynapsePredictor(in_channels=x_global.shape[1], hidden_channels=128).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=False))
    model.eval()

    # 3. Stress Test Loop
    # We will aggregate scores to calculate one massive metric at the end
    all_pos_scores = []
    all_rand_neg_scores = []
    all_hard_neg_scores = []
    
    # Mining Settings
    num_subgraphs = 10
    mining_pool_factor = 20 # We check 20x random negatives to find the 1x hardest ones
    
    print(f"Mining Hard Negatives (Sampling {mining_pool_factor}x candidates per positive)...")
    
    for i in range(num_subgraphs):
        # Sample a subgraph
        local_edge_index, node_indices = get_random_subgraph(test_edges, x_global.size(0), sample_size=8000)
        
        if local_edge_index.size(1) == 0: continue
            
        local_edge_index = local_edge_index.to(device)
        batch_x = x_global[node_indices].to(device)
        
        with torch.no_grad():
            z = model.encode(batch_x, local_edge_index)
            
            # A. Get Positive Scores (Real Synapses)
            pos_src, pos_dst = local_edge_index[0], local_edge_index[1]
            pos_scores = (z[pos_src] * z[pos_dst]).sum(dim=1).sigmoid()
            all_pos_scores.append(pos_scores.cpu())
            
            # B. Generate Random Pool (for Mining)
            num_pos = pos_src.size(0)
            pool_size = num_pos * mining_pool_factor

            num_nodes_in_batch = batch_x.size(0)

            cand_src = torch.randint(0, num_nodes_in_batch, (pool_size,), device=device)
            cand_dst = torch.randint(0, num_nodes_in_batch, (pool_size,), device=device)
            
            cand_scores = (z[cand_src] * z[cand_dst]).sum(dim=1).sigmoid()
            
            # C. Select "Hardest" Negatives (Highest Scores)
            # We take the top 'num_pos' scores from the pool
            hard_values, _ = torch.topk(cand_scores, k=num_pos)
            all_hard_neg_scores.append(hard_values.cpu())
            
            # D. Select "Random" Negatives (for Baseline)
            # Just take the first 'num_pos' from the pool (since the pool is random)
            rand_values = cand_scores[:num_pos]
            all_rand_neg_scores.append(rand_values.cpu())
            
        print(f"  Batch {i+1}/{num_subgraphs}: Processed {num_pos} positives vs {pool_size} candidate negatives.")

    # 4. Final Calculation
    pos_scores_all = torch.cat(all_pos_scores).numpy()
    hard_scores_all = torch.cat(all_hard_neg_scores).numpy()
    rand_scores_all = torch.cat(all_rand_neg_scores).numpy()
    
    print("\n====== STRESS TEST RESULTS ======")
    print(f"Total Positives Evaluated: {len(pos_scores_all):,}")
    
    # --- SCENARIO 1: STANDARD EVALUATION (Random Negatives) ---
    y_true_rand = np.concatenate([np.ones(len(pos_scores_all)), np.zeros(len(rand_scores_all))])
    y_score_rand = np.concatenate([pos_scores_all, rand_scores_all])
    auc_rand = roc_auc_score(y_true_rand, y_score_rand)
    
    # --- SCENARIO 2: HARD EVALUATION (Hardest Negatives) ---
    y_true_hard = np.concatenate([np.ones(len(pos_scores_all)), np.zeros(len(hard_scores_all))])
    y_score_hard = np.concatenate([pos_scores_all, hard_scores_all])
    auc_hard = roc_auc_score(y_true_hard, y_score_hard)

    print(f"\n1. Baseline (vs Random Negatives)")
    print(f"   ROC AUC: {auc_rand:.5f}  (Should be ~0.999)")
    
    print(f"\n2. Stress Test (vs Hardest Mined Negatives)")
    print(f"   ROC AUC: {auc_hard:.5f}")
    
    print("\n-------------------------------------------")
    print(f"Average 'Hard Negative' Score: {np.mean(hard_scores_all):.4f}")
    print(f"Average 'True Synapse' Score:  {np.mean(pos_scores_all):.4f}")
    print("-------------------------------------------")
    
    if auc_hard > 0.95:
        print("[SUCCESS] Model distinguishes synapses even from the most confusing impostors!")
    elif auc_hard > 0.90:
        print("[PASS] Model is robust, though hard cases reduce certainty.")
    else:
        print("[WARNING] Model struggles with hard negatives (Potential False Positives in dense regions).")

if __name__ == "__main__":
    evaluate_hard_negatives()