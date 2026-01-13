import torch
import numpy as np
from sklearn.metrics import roc_auc_score
import os
import sys

# 1. Get the path to the folder this script is in (visualization_scripts)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Get the parent folder (spatial_training)
parent_dir = os.path.dirname(current_dir)

# 3. Add the parent folder to the system path so Python can find 'gnn.py'
sys.path.append(parent_dir)

# ---------------------------------------------------------
# NOW you can import modules from the parent folder
# ---------------------------------------------------------
import gnn # Your model file

# --- CONFIG ---
CACHE_DIR = "cache_spatial"
OUTPUT_FOLDER = "saved_models_spatial"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_random_subgraph(edge_index_cpu, num_nodes, sample_size=10000):
    perm = torch.randperm(num_nodes)[:sample_size]
    node_mask = torch.zeros(num_nodes, dtype=torch.bool)
    node_mask[perm] = True
    
    row, col = edge_index_cpu
    edge_mask = node_mask[row] & node_mask[col]
    subset_edge_index = edge_index_cpu[:, edge_mask]
    
    dense_map = torch.full((num_nodes,), -1, dtype=torch.long)
    dense_map[perm] = torch.arange(sample_size)
    
    new_src = dense_map[subset_edge_index[0]]
    new_dst = dense_map[subset_edge_index[1]]
    return torch.stack([new_src, new_dst], dim=0), perm

def calculate_auc(model, x, edge_index):
    model.eval()
    with torch.no_grad():
        local_edge_index, node_idx = get_random_subgraph(edge_index, x.size(0))
        batch_x = x[node_idx].to(device)
        local_edge_index = local_edge_index.to(device)
        
        z = model.encode(batch_x, local_edge_index)
        
        src, dst = local_edge_index[0], local_edge_index[1]
        pos_scores = (z[src] * z[dst]).sum(dim=1).cpu()
        
        neg_src = torch.randint(0, len(batch_x), (len(src),), device=device)
        neg_dst = torch.randint(0, len(batch_x), (len(src),), device=device)
        neg_scores = (z[neg_src] * z[neg_dst]).sum(dim=1).cpu()
        
        y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
        y_scores = torch.cat([pos_scores, neg_scores]).numpy()
        
        return roc_auc_score(y_true, y_scores)

def run_importance_test():
    print("Loading Data...")
    x_global = torch.load(f"{CACHE_DIR}/x_features.pt", weights_only=False)
    test_edges = torch.load(f"{CACHE_DIR}/graph_test_edges.pt", weights_only=False).cpu()
        
    model = gnn.SynapsePredictor(x_global.shape[1], 128).to(device)
    model.load_state_dict(torch.load(f"{OUTPUT_FOLDER}/best_model.pth"))
    
    # Correct names for the reduced 6 features
    feature_names = [
        "Soma_Vol",   # 0
        "Total_Vol",  # 1
        "Total_Len",  # 2
        "Centroid_X", # 3 (was 6)
        "Centroid_Y", # 4 (was 7)
        "Centroid_Z"  # 5 (was 8)
    ]
    
    print("\n--- Permutation Feature Importance ---")
    baseline_auc = calculate_auc(model, x_global, test_edges)
    print(f"Baseline Model AUC: {baseline_auc:.5f}")
    print("-" * 50)
    print(f"{'Feature':<15} | {'Drop in AUC':<12} | {'Interpretation'}")
    print("-" * 50)
    
    for i, name in enumerate(feature_names):
        x_shuffled = x_global.clone()
        perm = torch.randperm(x_global.size(0))
        x_shuffled[:, i] = x_shuffled[perm, i] # Scramble ONE column
        
        shuffled_auc = calculate_auc(model, x_shuffled, test_edges)
        drop = baseline_auc - shuffled_auc
        
        if drop < 0.0001:
            interp = "Useless (Candidate for removal)"
        elif drop < 0.01:
            interp = "Minor Impact"
        else:
            interp = "CRITICAL FEATURE"
            
        print(f"{name:<15} | {drop:.5f}      | {interp}")

if __name__ == "__main__":
    run_importance_test()