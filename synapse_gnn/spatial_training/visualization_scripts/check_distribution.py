import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from spatial_training import gnn

# --- 1. CONFIGURATION LOADER ---
def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

# --- 2. UPDATED SUBGRAPH SAMPLER (FIXED TYPEERROR) ---
def get_random_subgraph(edge_index_cpu, candidates_cpu, num_nodes_total, sample_size=6000):
    node_mask = torch.zeros(num_nodes_total, dtype=torch.bool)
    perm = torch.randperm(num_nodes_total)[:int(sample_size)]
    node_mask[perm] = True
    
    dense_map = torch.full((num_nodes_total,), -1, dtype=torch.long)
    dense_map[perm] = torch.arange(len(perm))
    
    # Filter Positives
    row, col = edge_index_cpu
    edge_mask = node_mask[row] & node_mask[col]
    local_edge_index = torch.stack([dense_map[edge_index_cpu[0, edge_mask]], 
                                    dense_map[edge_index_cpu[1, edge_mask]]], dim=0)
    
    # Filter Candidates (Negatives)
    c_row, c_col = candidates_cpu
    cand_mask = node_mask[c_row] & node_mask[c_col]
    local_candidates = torch.stack([dense_map[candidates_cpu[0, cand_mask]], 
                                    dense_map[candidates_cpu[1, cand_mask]]], dim=0)
    
    return local_edge_index, local_candidates, perm

# --- 3. VISUALIZATION LOGIC ---
def visualize_scores_zoomed():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="config.json")
    args = parser.parse_args()
    config = load_config(args.config)

    # Paths and Params from Config
    CACHE_DIR = config["paths"]["data_dir"]
    MODEL_OUT = config["paths"]["model_out"]
    VIS_DIR = config["paths"]["visualization_output"]
    THRESH_NM = config["graph_generation"]["spatial_threshold_nm"]
    
    # DYNAMIC NAMING: Matches the new train_and_eval.py structure
    graph_filename = os.path.splitext(config["paths"]["input_nx_graph"])[0]
    
    # FIX: Corrected variable casing (THRESH_NM) and un-indented the print statement
    if 'adp' in graph_filename.lower():
        MODEL_PATH = os.path.join(MODEL_OUT, f"best_model_{graph_filename}_added_adp_weights_{THRESH_NM}nm.pth")
    else:
        MODEL_PATH = os.path.join(MODEL_OUT, f"best_model_{graph_filename}_{THRESH_NM}nm.pth")
        
    print(f"--- Generating Distribution Plot for {graph_filename} at {THRESH_NM}nm ---")
    
    # Load Data
    x = torch.load(os.path.join(CACHE_DIR, "x_features.pt"), weights_only=False)
    test_edges = torch.load(os.path.join(CACHE_DIR, "graph_test_edges.pt"), weights_only=False)
    test_cands = torch.load(os.path.join(CACHE_DIR, "graph_test_spatial_candidates.pt"), weights_only=False)
    
    num_nodes_total = x.size(0)
    num_features = config["architecture"]["in_channels"]
    
    # Load Model
    model = gnn.SynapsePredictor(in_channels=num_features, hidden_channels=config["architecture"]["hidden_dim"])
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Could not find model at {MODEL_PATH}")
        return
        
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu', weights_only=False))
    model.eval()

    # Get Subgraph
    local_pos, local_neg, node_idx = get_random_subgraph(test_edges, test_cands, num_nodes_total, sample_size=8000)
    
    batch_x = x[node_idx, :num_features]
    z = model.encode(batch_x, local_pos) 

    # Calculate Scores
    with torch.no_grad():
        pos_scores = torch.sigmoid((z[local_pos[0]] * z[local_pos[1]]).sum(dim=1)).numpy()
        neg_scores = torch.sigmoid((z[local_neg[0]] * z[local_neg[1]]).sum(dim=1)).numpy()

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.hist(pos_scores, bins=50, alpha=0.5, label='True Synapses (Pos)', color='green', density=True)
    plt.hist(neg_scores, bins=50, alpha=0.5, label='Candidates (Neg)', color='red', density=True)
    plt.axvline(x=0.5, color='black', linestyle='--', label='Default Threshold')
    plt.title(f"Score Distribution: {graph_filename} ({THRESH_NM}nm)")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Density")
    plt.legend()

    # Save inside the specialized folder
    os.makedirs(VIS_DIR, exist_ok=True)
    out_path = os.path.join(VIS_DIR, f"distribution_{graph_filename}_{THRESH_NM}nm.png")
    plt.savefig(out_path)
    print(f"Success: Saved plot to {out_path}")

if __name__ == "__main__":
    visualize_scores_zoomed()