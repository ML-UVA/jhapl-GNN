import sys
import os
from main import log_to_file

# --- SYSTEM PATH SETUP (Fixes ModuleNotFoundErrors) ---
# This adds the parent directory (edge_classification_gnn) to Python's path
# so that 'import spatial_training.gnn' works correctly.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import torch
import numpy as np
# Now these imports will work because Python knows where 'spatial_training' is
import spatial_training.gnn as gnn  
from spatial_training.main import get_random_subgraph, analyze_model_performance

# CONFIGURATION
CACHE_DIR = "cache_spatial"
OUTPUT_FOLDER = "saved_models_spatial"
PATH_X = os.path.join(CACHE_DIR, "x_features.pt")
PATH_TEST_EDGES = os.path.join(CACHE_DIR, "graph_test_edges.pt")
MODEL_PATH = os.path.join(OUTPUT_FOLDER, "best_model.pth")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def final_eval():
    log_to_file("--- Loading Data for Final Evaluation ---")
    
    # 1. Load Data
    if not os.path.exists(PATH_X) or not os.path.exists(PATH_TEST_EDGES):
        log_to_file(f"Error: Data not found in {CACHE_DIR}")
        return

    x_global = torch.load(PATH_X, weights_only=False)
    test_edges = torch.load(PATH_TEST_EDGES, weights_only=False).cpu()
    
    (f"Nodes: {x_global.size(0)}")
    log_to_file(f"Test Edges: {test_edges.size(1):,}")

    # 2. Load Model
    log_to_file(f"Loading Model from {MODEL_PATH}...")
    model = gnn.SynapsePredictor(in_channels=x_global.shape[1], hidden_channels=128).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=False))
    model.eval()

    # 3. Large Scale Evaluation
    log_to_file("Running Final Inference on Test Set...")
    
    all_y_true = []
    all_y_scores = []
    
    # Aggregate results from multiple subgraphs for robustness
    for i in range(10):
        log_to_file(f"Sampling Test Subgraph {i+1}/10...")
        local_edge_index, node_indices = get_random_subgraph(test_edges, x_global.size(0), sample_size=8000)
        
        if local_edge_index.size(1) == 0: continue
            
        local_edge_index = local_edge_index.to(device)
        batch_x = x_global[node_indices].to(device)
        
        with torch.no_grad():
            z = model.encode(batch_x, local_edge_index)
            
            # Positive Edges
            pos_src, pos_dst = local_edge_index[0], local_edge_index[1]
            pos_scores = (z[pos_src] * z[pos_dst]).sum(dim=1)
            
            # Negative Edges (Random)
            num_pos = pos_src.size(0)
            neg_src = torch.randint(0, 8000, (num_pos,), device=device)
            neg_dst = torch.randint(0, 8000, (num_pos,), device=device)
            neg_scores = (z[neg_src] * z[neg_dst]).sum(dim=1)
            
            # Append
            scores = torch.cat([torch.sigmoid(pos_scores), torch.sigmoid(neg_scores)]).cpu().numpy()
            labels = np.concatenate([np.ones(num_pos), np.zeros(num_pos)])
            
            all_y_scores.append(scores)
            all_y_true.append(labels)

    # 4. Final Stats
    if not all_y_true:
        log_to_file("Error: No edges evaluated.")
        return

    y_true_final = np.concatenate(all_y_true)
    y_scores_final = np.concatenate(all_y_scores)
    
    log_to_file(f"\nEvaluated on {len(y_true_final):,} edge pairs.")
    analyze_model_performance(y_true_final, y_scores_final)

if __name__ == "__main__":
    final_eval()