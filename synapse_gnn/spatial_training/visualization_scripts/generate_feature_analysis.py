import os
import json
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score

from spatial_training import gnn
from spatial_training.train_and_eval import get_random_subgraph 

# --- 2. CONFIGURATION LOADER ---
# Set the default config path relative to this script's location
current_file = os.path.abspath(__file__)
visualization_scripts_dir = os.path.dirname(current_file)
spatial_training_dir = os.path.dirname(visualization_scripts_dir)
synapse_gnn_dir = os.path.dirname(spatial_training_dir)
default_config_path = os.path.join(synapse_gnn_dir, "config.json")

parser = argparse.ArgumentParser(description="Visualize GNN Model Metrics")
parser.add_argument('--config', type=str, default=default_config_path, help="Path to config file")
args, _ = parser.parse_known_args()

config_path = os.path.abspath(args.config)
print(f"Loading configuration from: {config_path}")
with open(config_path, 'r') as f:
    config = json.load(f)

# --- 3. ROBUST PATH SETUP ---
# Map paths from config (Resolving relative to the config file's location)
config_dir = os.path.dirname(config_path)
CACHE_DIR = os.path.normpath(os.path.join(config_dir, config["paths"]["data_dir"]))
MODEL_OUTPUT_FOLDER = os.path.normpath(os.path.join(config_dir, config["paths"]["model_out"]))
OUTPUT_FOLDER = os.path.join(spatial_training_dir, "best_model_spatial_evals")

# File Paths
PATH_X = os.path.join(CACHE_DIR, "x_features.pt")
PATH_TEST_EDGES = os.path.join(CACHE_DIR, "graph_test_edges.pt")
PATH_TRAIN_EDGES = os.path.join(CACHE_DIR, "graph_train_edges.pt")
PATH_TEST_CANDS = os.path.join(CACHE_DIR, "graph_test_spatial_candidates.pt")

# Graph name inference for dynamic model loading
graph_name = os.path.splitext(config["paths"]["input_nx_graph"])[0]
MODEL_PATH = os.path.join(MODEL_OUTPUT_FOLDER, f"best_model_{graph_name}.pth")

# Verify paths exist
print(f"Cache Directory: {CACHE_DIR}")
print(f"Model Directory: {MODEL_OUTPUT_FOLDER}")
if not os.path.exists(CACHE_DIR):
    print(f"WARNING: Cache directory not found at {CACHE_DIR}")
if not os.path.exists(MODEL_PATH):
    print(f"WARNING: Model not found at {MODEL_PATH}")

# Create output dir
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
print(f"Output Directory: {OUTPUT_FOLDER}\n")

# --- 4. LOAD DATA ---
print("Loading data...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x_global_raw = torch.load(PATH_X, weights_only=False)

morph_indices = [0, 1, 2, 3, 4, 5, 6, 7]
morph_names = [
    "Soma Vol", "Axon Len", "Basal Len", "Apical Len", 
    "Max Axon Reach", "Max Dendrite Reach", "Total Spines", "Total Spine Vol"
]
spatial_indices = [11, 12, 13] 

test_edges = torch.load(PATH_TEST_EDGES, weights_only=False).cpu()
train_edges = torch.load(PATH_TRAIN_EDGES, weights_only=False).cpu()
test_cands = torch.load(PATH_TEST_CANDS, weights_only=False).cpu()

print(f"Loaded {test_cands.size(1):,} Spatial Candidates for evaluation.")

# --- 5. LOAD MODEL ---
print(f"Loading Morphology Model from {MODEL_PATH}...")
model = gnn.SynapsePredictor(in_channels=8, hidden_channels=128).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=False))
model.eval()

# --- HELPER: ADVANCED SUBGRAPH SAMPLER (Now handles candidates!) ---
def get_eval_subgraph(edge_index_cpu, candidates_cpu, num_nodes_total, sample_size=10000):
    node_mask = torch.zeros(num_nodes_total, dtype=torch.bool)
    perm = torch.randperm(num_nodes_total)[:sample_size]
    node_mask[perm] = True
    
    dense_map = torch.full((num_nodes_total,), -1, dtype=torch.long)
    dense_map[perm] = torch.arange(sample_size)
    
    # 1. Extract True Test Edges
    row, col = edge_index_cpu
    edge_mask = node_mask[row] & node_mask[col]
    subset_edge_index = edge_index_cpu[:, edge_mask]
    local_edge_index = torch.stack([dense_map[subset_edge_index[0]], dense_map[subset_edge_index[1]]], dim=0)
    
    # 2. Extract Hard Negative Candidates
    c_row, c_col = candidates_cpu
    cand_mask = node_mask[c_row] & node_mask[c_col]
    subset_candidates = candidates_cpu[:, cand_mask]
    local_candidates = torch.stack([dense_map[subset_candidates[0]], dense_map[subset_candidates[1]]], dim=0)
    
    return local_edge_index, local_candidates, perm

# --- HELPER: INDUCTIVE ENCODING ---
def get_inductive_embeddings(subset_x, subset_node_indices):
    num_nodes_total = x_global_raw.size(0)
    subset_mask = torch.zeros(num_nodes_total, dtype=torch.bool)
    subset_mask[subset_node_indices] = True
    
    train_row, train_col = train_edges
    edge_mask = subset_mask[train_row] & subset_mask[train_col]
    batch_train_edges = train_edges[:, edge_mask]
    
    node_idx_map = torch.full((num_nodes_total,), -1, dtype=torch.long)
    node_idx_map[subset_node_indices] = torch.arange(len(subset_node_indices))
    
    local_context_edges = torch.stack([
        node_idx_map[batch_train_edges[0]], node_idx_map[batch_train_edges[1]]
    ], dim=0)
    
    with torch.no_grad():
        z = model.encode(subset_x.to(device), local_context_edges.to(device))
    return z

# --- HELPER: SCORE AUC ---
def calculate_auc(z, pos_edges, neg_edges):
    if pos_edges.size(1) == 0 or neg_edges.size(1) == 0: return 0.5
    
    # Downsample negatives to match positives for balanced AUC
    num_pos = pos_edges.size(1)
    if neg_edges.size(1) > num_pos:
        perm = torch.randperm(neg_edges.size(1))[:num_pos]
        neg_edges = neg_edges[:, perm]
        
    pos_scores = (z[pos_edges[0].to(device)] * z[pos_edges[1].to(device)]).sum(dim=1).sigmoid().cpu().numpy()
    neg_scores = (z[neg_edges[0].to(device)] * z[neg_edges[1].to(device)]).sum(dim=1).sigmoid().cpu().numpy()
    
    y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    y_scores = np.concatenate([pos_scores, neg_scores])
    return roc_auc_score(y_true, y_scores)

# --- ANALYSIS 1: FEATURE IMPORTANCE ---
def analyze_feature_importance():
    print("\nRunning Analysis 1: Feature Importance (via Hard Negatives)...")
    
    # Use config sample sizes for evaluation
    eval_sample_size = config["evaluation"]["test_node_sample_size"]
    local_pos, local_neg, node_indices = get_eval_subgraph(test_edges, test_cands, x_global_raw.size(0), sample_size=eval_sample_size)
    batch_x = x_global_raw[node_indices][:, morph_indices]
    
    z_base = get_inductive_embeddings(batch_x, node_indices)
    baseline_auc = calculate_auc(z_base, local_pos, local_neg)
    print(f"  Baseline AUC on Subgraph: {baseline_auc:.4f}")
    
    importances = []
    
    for i in range(len(morph_names)):
        print(f"  Permuting {morph_names[i]}...")
        permuted_x = batch_x.clone()
        perm_idx = torch.randperm(permuted_x.size(0))
        permuted_x[:, i] = permuted_x[perm_idx, i] 
        
        z_perm = get_inductive_embeddings(permuted_x, node_indices)
        shuffled_auc = calculate_auc(z_perm, local_pos, local_neg)
        
        drop = baseline_auc - shuffled_auc
        importances.append(drop)
        
    plt.figure(figsize=(12, 7))
    sns.barplot(x=importances, y=morph_names, palette="magma")
    plt.title(f"Feature Importance ({graph_name} Graph Candidates)", fontsize=14)
    plt.xlabel("Drop in ROC-AUC (Higher = More Important)", fontsize=12)
    plt.axvline(0, color='k', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, "1_feature_importance.png"), dpi=300)
    print("Saved '1_feature_importance.png'")

# --- ANALYSIS 2: DISTANCE CHECK ---
def analyze_distance_vs_prob():
    print("\nRunning Analysis 2: Distance vs Probability Check...")
    
    eval_sample_size = config["evaluation"]["test_node_sample_size"]
    local_pos, local_neg, node_indices = get_eval_subgraph(test_edges, test_cands, x_global_raw.size(0), sample_size=eval_sample_size)
    
    batch_x_morph = x_global_raw[node_indices][:, morph_indices] 
    batch_x_spatial = x_global_raw[node_indices][:, spatial_indices] 
    
    z = get_inductive_embeddings(batch_x_morph, node_indices)
    
    # 1. Real Synapses (Green)
    pos_src, pos_dst = local_pos[0], local_pos[1]
    dists_real = torch.norm(batch_x_spatial[pos_src] - batch_x_spatial[pos_dst], dim=1).numpy()
    probs_real = (z[pos_src.to(device)] * z[pos_dst.to(device)]).sum(dim=1).sigmoid().cpu().numpy()
    
    # 2. Hard Negatives (Red)
    num_plot = min(len(pos_src) * 3, local_neg.size(1))
    neg_src, neg_dst = local_neg[0, :num_plot], local_neg[1, :num_plot]
    
    dists_neg = torch.norm(batch_x_spatial[neg_src] - batch_x_spatial[neg_dst], dim=1).numpy()
    probs_neg = (z[neg_src.to(device)] * z[neg_dst.to(device)]).sum(dim=1).sigmoid().cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    plt.scatter(dists_neg, probs_neg, alpha=0.3, color='red', s=15, label="Structural Candidates (False Positives)")
    plt.scatter(dists_real, probs_real, alpha=0.5, color='green', s=20, label="True Synapses")
    
    plt.title("Model Confidence vs. True Euclidean Distance", fontsize=14)
    plt.xlabel("Euclidean Distance in Microns", fontsize=12)
    plt.ylabel("Predicted Probability", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_FOLDER, "2_distance_check.png"), dpi=300)
    print("Saved '2_distance_check.png'")

# --- ANALYSIS 3: VOLUME IMPACT ---
def analyze_volume_impact():
    print("\nRunning Analysis 3: Volume Impact...")
    
    eval_sample_size = config["evaluation"]["test_node_sample_size"]
    local_pos, _, node_indices = get_eval_subgraph(test_edges, test_cands, x_global_raw.size(0), sample_size=eval_sample_size)
    batch_x_morph = x_global_raw[node_indices][:, morph_indices]
    
    z = get_inductive_embeddings(batch_x_morph, node_indices)
    
    src, dst = local_pos[0], local_pos[1]
    
    # Feature 0 is Soma Volume
    vol_src = batch_x_morph[src, 0]
    vol_dst = batch_x_morph[dst, 0]
    avg_vol = (vol_src + vol_dst) / 2.0
    
    probs = (z[src.to(device)] * z[dst.to(device)]).sum(dim=1).sigmoid().cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(avg_vol.numpy(), probs, alpha=0.4, c=probs, cmap='viridis')
    plt.colorbar(sc, label="Model Confidence")
    plt.title("Impact of Soma Volume on Prediction Confidence (True Synapses)", fontsize=14)
    plt.xlabel("Average Soma Volume (microns^3)", fontsize=12)
    plt.ylabel("Predicted Probability", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_FOLDER, "3_volume_impact.png"), dpi=300)
    print("Saved '3_volume_impact.png'")

if __name__ == "__main__":
    analyze_feature_importance()
    analyze_distance_vs_prob()
    analyze_volume_impact()
    print("\nAll visualizations completed successfully.")