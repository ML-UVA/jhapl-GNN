import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score

# --- SYSTEM PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(current_dir) == "visualization_scripts":
    parent_dir = os.path.dirname(current_dir)
    grandparent_dir = os.path.dirname(parent_dir)
    sys.path.append(grandparent_dir)
    sys.path.append(parent_dir)
else:
    sys.path.append(current_dir)

import spatial_training.gnn as gnn
from spatial_training.main import get_random_subgraph

# --- CONFIGURATION ---
CACHE_DIR = "cache_spatial" if os.path.exists("cache_spatial") else "../cache_spatial"
OUTPUT_FOLDER = "saved_models_spatial" if os.path.exists("saved_models_spatial") else "../saved_models_spatial"

PATH_X = os.path.join(CACHE_DIR, "x_features.pt")
PATH_TEST_EDGES = os.path.join(CACHE_DIR, "graph_test_edges.pt")
MODEL_PATH = os.path.join(OUTPUT_FOLDER, "best_model.pth")

# --- LOAD DATA ---
print("Loading data...")
x_global = torch.load(PATH_X, weights_only=False) # RAW 9-feature data (for metadata lookup)
test_edges = torch.load(PATH_TEST_EDGES, weights_only=False)

# 3 Features: [CentrX, CentrY, CentrZ]
# Original Indices: 6, 7, 8
features_to_keep = [6, 7, 8]
x_global_sliced = x_global[:, features_to_keep] # Model Input
feature_names = ["Centroid X", "Centroid Y", "Centroid Z"]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# FIX: Input channels is now 3
model = gnn.SynapsePredictor(in_channels=3, hidden_channels=128).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=False))
model.eval()

def analyze_feature_importance():
    print("Calculating Permutation Importance (X vs Y vs Z)...")
    
    # 1. Sample a batch for analysis
    local_edge_index, node_indices = get_random_subgraph(test_edges, x_global.size(0), sample_size=10000)
    batch_x = x_global_sliced[node_indices].to(device)
    local_edge_index = local_edge_index.to(device)
    
    # 3. Establish Baseline
    original_x = batch_x.clone()
    
    with torch.no_grad():
        z = model.encode(original_x, local_edge_index)
        base_scores = (z[local_edge_index[0]] * z[local_edge_index[1]]).sum(dim=1).sigmoid()
        baseline_avg = base_scores.mean().item()
    
    importances = []
    
    # FIX: Loop only over the 3 existing features
    for i in range(3): 
        print(f"  Permuting {feature_names[i]}...")
        permuted_x = original_x.clone()
        perm_idx = torch.randperm(permuted_x.size(0))
        permuted_x[:, i] = permuted_x[perm_idx, i]
        
        with torch.no_grad():
            z = model.encode(permuted_x, local_edge_index)
            new_scores = (z[local_edge_index[0]] * z[local_edge_index[1]]).sum(dim=1).sigmoid()
            new_avg = new_scores.mean().item()
            
        drop = baseline_avg - new_avg
        importances.append(drop)

    # --- PLOT 1: FEATURE IMPORTANCE ---
    plt.figure(figsize=(8, 6))
    sns.barplot(x=importances, y=feature_names, palette="viridis")
    plt.title("Geometric Feature Importance (X vs Y vs Z)", fontsize=14)
    plt.xlabel("Drop in Confidence when Coordinate is Randomized", fontsize=12)
    plt.tight_layout()
    plt.savefig("demo_analysis_importance.png", dpi=300)
    print("Saved 'demo_analysis_importance.png'")

def analyze_distance_vs_prob():
    print("Generating Distance vs. Probability Plot...")
    
    # Sample data
    local_edge_index, node_indices = get_random_subgraph(test_edges, x_global.size(0), sample_size=5000)
    batch_x = x_global_sliced[node_indices].to(device)
    
    # Calculate Distances for REAL synapses
    src_idx, dst_idx = local_edge_index[0], local_edge_index[1]
    
    # FIX: Coordinates are now at indices 0, 1, 2 in the sliced tensor
    coords = batch_x[:, 0:3] 
    src_coords = coords[src_idx]
    dst_coords = coords[dst_idx]
    
    dists = torch.norm(src_coords - dst_coords, dim=1).cpu().numpy()
    
    # Model Probabilities
    with torch.no_grad():
        z = model.encode(batch_x, local_edge_index.to(device))
        probs = (z[src_idx] * z[dst_idx]).sum(dim=1).sigmoid().cpu().numpy()
        
    # Generate HARD NEGATIVES
    neg_src = torch.randint(0, 5000, (5000,), device=device)
    neg_dst = torch.randint(0, 5000, (5000,), device=device)
    
    neg_src_coords = coords[neg_src]
    neg_dst_coords = coords[neg_dst]
    neg_dists = torch.norm(neg_src_coords - neg_dst_coords, dim=1).cpu().numpy()
    
    mask = neg_dists < 5.0
    neg_dists = neg_dists[mask]
    neg_src = neg_src[mask]
    neg_dst = neg_dst[mask]
    
    with torch.no_grad():
        neg_probs = (z[neg_src] * z[neg_dst]).sum(dim=1).sigmoid().cpu().numpy()

    # --- PLOT 2: DISTANCE CLOUD ---
    plt.figure(figsize=(10, 8))
    plt.scatter(neg_dists, neg_probs, alpha=0.1, color='red', s=10, label="Non-Synapses (Close Proximity)")
    plt.scatter(dists, probs, alpha=0.1, color='green', s=10, label="True Synapses")
    
    plt.axhline(0.73, color='black', linestyle='--', label="Decision Threshold (0.73)")
    plt.title("Is it just learning distance? (Distance vs. Confidence)", fontsize=14)
    plt.xlabel("Euclidean Distance between Centroids (microns)", fontsize=12)
    plt.ylabel("Model Predicted Probability", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 5) 
    plt.tight_layout()
    plt.savefig("demo_analysis_distance_check.png", dpi=300)
    print("Saved 'demo_analysis_distance_check.png'")

def analyze_feature_distributions():
    # NOTE: This function now compares Volume (Metadata) vs Model Confidence
    # It proves that even though the model doesn't see volume, it's not biased.
    print("Generating Volume vs. Probability Analysis...")
    
    local_edge_index, node_indices = get_random_subgraph(test_edges, x_global.size(0), sample_size=15000)
    
    # Input for Model (Centroids)
    batch_x_input = x_global_sliced[node_indices].to(device)
    # Metadata for Plotting (Volume is Index 0 in RAW global)
    batch_x_raw = x_global[node_indices]
    
    local_edge_index = local_edge_index.to(device)
    
    pos_src, pos_dst = local_edge_index[0], local_edge_index[1]
    
    # Generate Negatives
    num_pos = pos_src.size(0)
    neg_src = torch.randint(0, 15000, (num_pos*5,), device=device)
    neg_dst = torch.randint(0, 15000, (num_pos*5,), device=device)
    
    # Distance Check using Input Centroids (Index 0,1,2 of input)
    coords = batch_x_input[:, 0:3]
    dists = torch.norm(coords[neg_src] - coords[neg_dst], dim=1)
    
    mask_close = dists < 2.0
    neg_src_close = neg_src[mask_close]
    neg_dst_close = neg_dst[mask_close]
    
    print(f"Comparing {len(pos_src)} True Synapses vs {len(neg_src_close)} Hard Negatives...")

    # Helper to get Volume from RAW data (Index 0 = Soma Vol)
    def get_avg_volume(src_idx, dst_idx):
        val_src = batch_x_raw[src_idx, 0].numpy()
        val_dst = batch_x_raw[dst_idx, 0].numpy()
        return (val_src + val_dst) / 2.0

    pos_vols = get_avg_volume(pos_src.cpu(), pos_dst.cpu())
    neg_vols = get_avg_volume(neg_src_close.cpu(), neg_dst_close.cpu())
    
    # Get Model Probs (Using Centroids Only)
    with torch.no_grad():
        z = model.encode(batch_x_input, local_edge_index)
        pos_probs = (z[pos_src] * z[pos_dst]).sum(dim=1).sigmoid().cpu().numpy()
        neg_probs = (z[neg_src_close] * z[neg_dst_close]).sum(dim=1).sigmoid().cpu().numpy()

    # --- PLOT 3: Scatter Volume vs Probability ---
    plt.figure(figsize=(10, 8))
    # Downsample for cleaner plot
    indices_pos = np.random.choice(len(pos_vols), min(2000, len(pos_vols)))
    indices_neg = np.random.choice(len(neg_vols), min(2000, len(neg_vols)))
    
    plt.scatter(neg_vols[indices_neg], neg_probs[indices_neg], alpha=0.1, color='red', label='Hard Impostors')
    plt.scatter(pos_vols[indices_pos], pos_probs[indices_pos], alpha=0.1, color='green', label='True Synapses')
    
    plt.axhline(0.73, color='black', linestyle='--')
    plt.title("Correlation Check: Does Hidden Volume correlate with Confidence?", fontsize=14)
    plt.xlabel("Soma Volume (Hidden Metadata)", fontsize=12)
    plt.ylabel("Model Predicted Probability", fontsize=12)
    plt.legend()
    plt.savefig("demo_scatter_vol_prob.png", dpi=300)
    print("Saved 'demo_scatter_vol_prob.png'")

if __name__ == "__main__":
    analyze_feature_importance()
    analyze_distance_vs_prob()
    analyze_feature_distributions()