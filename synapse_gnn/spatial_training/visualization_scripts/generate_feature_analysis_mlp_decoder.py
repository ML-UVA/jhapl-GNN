import os
import json
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score

from synapse_gnn.models import gnn

# --- 1. CONFIGURATION LOADER ---
current_file = os.path.abspath(__file__)
visualization_scripts_dir = os.path.dirname(current_file)
spatial_training_dir = os.path.dirname(visualization_scripts_dir)
synapse_gnn_dir = os.path.dirname(spatial_training_dir)
default_config_path = os.path.join(synapse_gnn_dir, "config.json")

parser = argparse.ArgumentParser(description="Visualize GNN MLP Decoder Metrics")
parser.add_argument('--config', type=str, default=default_config_path, help="Path to config file")
args, _ = parser.parse_known_args()

config_path = os.path.abspath(args.config)
print(f"Loading configuration from: {config_path}")
with open(config_path, 'r') as f:
    config = json.load(f)

# --- 2. ROBUST PATH SETUP ---
config_dir = os.path.dirname(config_path)
CACHE_DIR = os.path.normpath(os.path.join(config_dir, config["paths"]["data_dir"]))
MODEL_OUTPUT_FOLDER = os.path.normpath(os.path.join(config_dir, config["paths"]["model_out"]))
OUTPUT_FOLDER = os.path.normpath(os.path.join(config_dir, config["paths"]["visualization_output"]))

# File Paths
PATH_X = os.path.join(CACHE_DIR, "x_features.pt")
PATH_TEST_EDGES = os.path.join(CACHE_DIR, "graph_test_edges.pt")
PATH_TRAIN_EDGES = os.path.join(CACHE_DIR, "graph_train_edges.pt")
PATH_TEST_CANDS = os.path.join(CACHE_DIR, "graph_test_spatial_candidates.pt")
PATH_TRAIN_CANDS = os.path.join(CACHE_DIR, "graph_train_spatial_candidates.pt")


graph_name = os.path.splitext(config["paths"]["input_nx_graph"])[0]
thresh_nm = config["graph_generation"]["spatial_threshold_nm"]

is_adp = 'adp' in graph_name.lower()

# GLOBALLY LOAD AND NORMALIZE WEIGHTS
if is_adp:
    PATH_TRAIN_WEIGHTS = os.path.join(CACHE_DIR, "graph_train_spatial_weights.pt")
    PATH_TEST_WEIGHTS = os.path.join(CACHE_DIR, "graph_test_spatial_weights.pt")
    
    if os.path.exists(PATH_TEST_WEIGHTS) and os.path.exists(PATH_TRAIN_WEIGHTS):
        train_weights_raw = torch.load(PATH_TRAIN_WEIGHTS, weights_only=False).cpu()
        test_weights_raw = torch.load(PATH_TEST_WEIGHTS, weights_only=False).cpu()
        
        max_weight = train_weights_raw.max()
        train_weights = train_weights_raw / max_weight  
        test_weights = test_weights_raw / max_weight    
        print("Loaded and Normalized Continuous ADP Weights for Evaluation.")
    else:
        print("WARNING: ADP graph detected, but weight tensors were missing!")
        train_weights = None
        test_weights = None
else:
    train_weights = None
    test_weights = None

if is_adp:
    MODEL_PATH = os.path.join(MODEL_OUTPUT_FOLDER, f"best_model_{graph_name}_added_adp_weights_{thresh_nm}nm.pth")
else:
    MODEL_PATH = os.path.join(MODEL_OUTPUT_FOLDER, f"best_model_{graph_name}_{thresh_nm}nm.pth")
    
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
print(f"Output Directory: {OUTPUT_FOLDER}\n")

# --- 3. LOAD DATA ---
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
train_cands = torch.load(PATH_TRAIN_CANDS, weights_only=False).cpu()

# --- 4. LOAD MODEL ---
print(f"Loading MLP Decoder Model from {MODEL_PATH}...")
model = gnn.SynapsePredictor(in_channels=8, hidden_channels=128).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=False))
model.eval()

# --- HELPER: ROBUST SUBGRAPH SAMPLER ---
def get_eval_subgraph(edge_index_cpu, candidates_cpu, num_nodes_total, weights_cpu=None, sample_size=10000):
    node_mask = torch.zeros(num_nodes_total, dtype=torch.bool)
    perm = torch.randperm(num_nodes_total)[:int(sample_size)]
    node_mask[perm] = True
    dense_map = torch.full((num_nodes_total,), -1, dtype=torch.long)
    dense_map[perm] = torch.arange(len(perm))
    
    row, col = edge_index_cpu
    edge_mask = node_mask[row] & node_mask[col]
    local_edge_index = torch.stack([dense_map[edge_index_cpu[0, edge_mask]], dense_map[edge_index_cpu[1, edge_mask]]], dim=0)
    
    c_row, c_col = candidates_cpu
    cand_mask = node_mask[c_row] & node_mask[c_col]
    local_candidates = torch.stack([dense_map[candidates_cpu[0, cand_mask]], dense_map[candidates_cpu[1, cand_mask]]], dim=0)
    
    if weights_cpu is not None:
        if local_candidates.size(1) > 0 and local_edge_index.size(1) > 0:
            local_cand_weights = weights_cpu[cand_mask]
            
            max_node = sample_size
            hash_cands = local_candidates[0] * max_node + local_candidates[1]
            hash_pos = local_edge_index[0] * max_node + local_edge_index[1]
            
            sort_idx = torch.argsort(hash_cands)
            sorted_hash_cands = hash_cands[sort_idx]
            sorted_weights = local_cand_weights[sort_idx]
            
            idx = torch.searchsorted(sorted_hash_cands, hash_pos)
            idx = idx.clamp(0, len(sorted_hash_cands) - 1)
            
            valid_match = sorted_hash_cands[idx] == hash_pos
            local_weights = torch.ones(local_edge_index.size(1), dtype=torch.float32)
            local_weights[valid_match] = sorted_weights[idx[valid_match]]
        else:
            local_weights = torch.ones(local_edge_index.size(1), dtype=torch.float32)
            local_cand_weights = torch.ones(local_candidates.size(1), dtype=torch.float32)
    else:
        local_weights = None
        local_cand_weights = None
        
    return local_edge_index, local_candidates, perm, local_weights, local_cand_weights

# --- HELPER: INDUCTIVE ENCODING ---
# --- HELPER: INDUCTIVE ENCODING ---
def get_inductive_embeddings(subset_x, subset_node_indices):
    num_nodes_total = x_global_raw.size(0)
    subset_mask = torch.zeros(num_nodes_total, dtype=torch.bool)
    subset_mask[subset_node_indices] = True
    
    train_row, train_col = train_edges
    edge_mask = subset_mask[train_row] & subset_mask[train_col]
    batch_train_edges = train_edges[:, edge_mask]
    
    if train_weights is not None:
        # Match train_edges to train_cands to extract the correct weights
        cand_row, cand_col = train_cands
        cand_mask = subset_mask[cand_row] & subset_mask[cand_col]
        local_cands = train_cands[:, cand_mask]
        local_cand_weights = train_weights[cand_mask]
        
        if local_cands.size(1) > 0 and batch_train_edges.size(1) > 0:
            max_node = num_nodes_total
            hash_cands = local_cands[0] * max_node + local_cands[1]
            hash_pos = batch_train_edges[0] * max_node + batch_train_edges[1]
            
            sort_idx = torch.argsort(hash_cands)
            sorted_hash_cands = hash_cands[sort_idx]
            sorted_weights = local_cand_weights[sort_idx]
            
            idx = torch.searchsorted(sorted_hash_cands, hash_pos)
            idx = idx.clamp(0, len(sorted_hash_cands) - 1)
            
            valid_match = sorted_hash_cands[idx] == hash_pos
            local_context_weights = torch.ones(batch_train_edges.size(1), dtype=torch.float32)
            local_context_weights[valid_match] = sorted_weights[idx[valid_match]]
            local_context_weights = local_context_weights.to(device)
        else:
            local_context_weights = torch.ones(batch_train_edges.size(1), dtype=torch.float32).to(device)
    else:
        local_context_weights = None
    
    node_idx_map = torch.full((num_nodes_total,), -1, dtype=torch.long)
    node_idx_map[subset_node_indices] = torch.arange(len(subset_node_indices))
    
    local_context_edges = torch.stack([
        node_idx_map[batch_train_edges[0]], node_idx_map[batch_train_edges[1]]
    ], dim=0)
    
    with torch.no_grad():
        z = model.encode(subset_x.to(device), local_context_edges.to(device), edge_weight=local_context_weights)
    return z
# --- HELPER: SCORE AUC ---
def calculate_auc(z, pos_edges, neg_edges, explicit_weights):
    if pos_edges.size(1) == 0 or neg_edges.size(1) == 0: return 0.5
    
    num_pos = pos_edges.size(1)
    target_edges = torch.cat([pos_edges.to(device), neg_edges.to(device)], dim=1)
    
    with torch.no_grad():
        preds = model.decode(z, target_edges, explicit_weight=explicit_weights)
        all_scores = torch.sigmoid(preds).cpu().numpy()
        
    pos_scores = all_scores[:num_pos]
    neg_scores = all_scores[num_pos:]
    
    y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    y_scores = np.concatenate([pos_scores, neg_scores])
    return roc_auc_score(y_true, y_scores)

# --- ANALYSIS 1: FEATURE IMPORTANCE ---
def analyze_feature_importance():
    print("\nRunning Analysis 1: Feature Importance (via MLP Decoder)...")
    
    eval_sample_size = config["evaluation"]["test_node_sample_size"]
    local_pos, local_neg, node_indices, local_weights, local_cand_weights = get_eval_subgraph(
        test_edges, test_cands, x_global_raw.size(0), weights_cpu=test_weights, sample_size=eval_sample_size)
    
    num_pos = local_pos.size(1)
    num_neg = local_neg.size(1)
    if num_neg > num_pos:
        perm = torch.randperm(num_neg)[:num_pos]
        local_neg = local_neg[:, perm]
        if local_cand_weights is not None:
            local_cand_weights = local_cand_weights[perm]
            
    if local_weights is not None and local_cand_weights is not None:
        explicit_weights = torch.cat([local_weights, local_cand_weights]).to(device)
    else:
        explicit_weights = None

    batch_x = x_global_raw[node_indices][:, morph_indices]
    
    z_base = get_inductive_embeddings(batch_x, node_indices)
    baseline_auc = calculate_auc(z_base, local_pos, local_neg, explicit_weights)
    print(f"  Baseline AUC on Subgraph: {baseline_auc:.4f}")
    
    importances = []
    
    for i in range(len(morph_names)):
        permuted_x = batch_x.clone()
        perm_idx = torch.randperm(permuted_x.size(0))
        permuted_x[:, i] = permuted_x[perm_idx, i] 
        
        z_perm = get_inductive_embeddings(permuted_x, node_indices)
        shuffled_auc = calculate_auc(z_perm, local_pos, local_neg, explicit_weights)
        
        drop = baseline_auc - shuffled_auc
        importances.append(drop)
        
    plt.figure(figsize=(12, 7))
    sns.barplot(x=importances, y=morph_names, palette="magma", hue=morph_names, legend=False)
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
    local_pos, local_neg, node_indices, local_weights, local_cand_weights = get_eval_subgraph(
        test_edges, test_cands, x_global_raw.size(0), weights_cpu=test_weights, sample_size=eval_sample_size)
    
    batch_x_morph = x_global_raw[node_indices][:, morph_indices] 
    batch_x_spatial = x_global_raw[node_indices][:, spatial_indices] 
    
    z = get_inductive_embeddings(batch_x_morph, node_indices)
    
    num_plot = min(local_pos.size(1) * 3, local_neg.size(1))
    local_neg_plot = local_neg[:, :num_plot]
    
    if local_cand_weights is not None:
        cand_weights_plot = local_cand_weights[:num_plot]
    else:
        cand_weights_plot = None

    with torch.no_grad():
        probs_real = torch.sigmoid(model.decode(z, local_pos.to(device), explicit_weight=local_weights.to(device) if local_weights is not None else None)).cpu().numpy()
        probs_neg = torch.sigmoid(model.decode(z, local_neg_plot.to(device), explicit_weight=cand_weights_plot.to(device) if cand_weights_plot is not None else None)).cpu().numpy()
    
    pos_src, pos_dst = local_pos[0], local_pos[1]
    dists_real = torch.norm(batch_x_spatial[pos_src] - batch_x_spatial[pos_dst], dim=1).numpy()
    
    neg_src, neg_dst = local_neg_plot[0], local_neg_plot[1]
    dists_neg = torch.norm(batch_x_spatial[neg_src] - batch_x_spatial[neg_dst], dim=1).numpy()
    
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
    local_pos, _, node_indices, local_weights, _ = get_eval_subgraph(
        test_edges, test_cands, x_global_raw.size(0), weights_cpu=test_weights, sample_size=eval_sample_size)
    batch_x_morph = x_global_raw[node_indices][:, morph_indices]
    
    z = get_inductive_embeddings(batch_x_morph, node_indices)
    
    src, dst = local_pos[0], local_pos[1]
    vol_src = batch_x_morph[src, 0]
    vol_dst = batch_x_morph[dst, 0]
    avg_vol = (vol_src + vol_dst) / 2.0
    
    with torch.no_grad():
        probs = torch.sigmoid(model.decode(z, local_pos.to(device), explicit_weight=local_weights.to(device) if local_weights is not None else None)).cpu().numpy()
    
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