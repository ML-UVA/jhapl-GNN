import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score

# Import the subgraph sampler from your engine
from synapse_gnn.training.train_engine import get_random_subgraph

# Map the feature indices to their human-readable biological names
ALL_MORPH_NAMES = [
    "Soma Vol", "Axon Len", "Basal Len", "Apical Len", 
    "Max Axon Reach", "Max Dendrite Reach", "Total Spines", "Total Spine Vol"
]

def calculate_auc(model, z, pos_edges, neg_edges, explicit_weights, device):
    """Helper to quickly decode embeddings and score them for the feature permutation test."""
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


def plot_feature_importance(model, data_dict, config, device):
    print("\nGenerating Feature Importance Plot (via MLP Decoder)...")
    
    selected_features = config["architecture"].get("selected_features", [0, 1, 2, 3, 4, 5, 6, 7])
    active_morph_names = [ALL_MORPH_NAMES[i] for i in selected_features]
    
    eval_sample_size = config["evaluation"]["test_node_sample_size"]
    num_nodes_total = data_dict["x_raw"].size(0)
    
    local_pos, local_neg, node_indices, local_weights, local_cand_weights = get_random_subgraph(
        data_dict["test_edges"], data_dict["test_cands"], num_nodes_total, 
        weights_cpu=data_dict["test_weights"], sample_size=eval_sample_size
    )
    
    # Downsample negatives to match positives for balanced metric
    num_pos = local_pos.size(1)
    num_neg = local_neg.size(1)
    if num_neg > num_pos:
        perm = torch.randperm(num_neg)[:num_pos]
        local_neg = local_neg[:, perm]
        if local_cand_weights is not None:
            local_cand_weights = local_cand_weights[perm]
            
    explicit_weights = torch.cat([local_weights, local_cand_weights]).to(device) if (local_weights is not None and local_cand_weights is not None) else None

    # Base Embeddings
    batch_x = data_dict["x_raw"][node_indices][:, selected_features].to(device)
    
    msg_edges = local_pos.to(device) 
    msg_weights = local_weights.to(device) if local_weights is not None else None
    
    with torch.no_grad():
        z_base = model.encode(batch_x, msg_edges, edge_weight=msg_weights)
        
    baseline_auc = calculate_auc(model, z_base, local_pos, local_neg, explicit_weights, device)
    
    importances = []
    
    # Permutation Test
    for i in range(len(selected_features)):
        permuted_x = batch_x.clone()
        perm_idx = torch.randperm(permuted_x.size(0))
        permuted_x[:, i] = permuted_x[perm_idx, i] 
        
        with torch.no_grad():
            z_perm = model.encode(permuted_x, msg_edges, edge_weight=msg_weights)
        shuffled_auc = calculate_auc(model, z_perm, local_pos, local_neg, explicit_weights, device)
        
        importances.append(baseline_auc - shuffled_auc)
        
    # Plotting
    graph_type = os.path.splitext(config["paths"]["input_nx_graph"])[0]
    thresh_nm = config["graph_generation"]["spatial_threshold_nm"]
    
    plt.figure(figsize=(12, 7))
    sns.barplot(x=importances, y=active_morph_names, palette="magma", hue=active_morph_names, legend=False)
    plt.title(f"Feature Importance ({graph_type} graph - {thresh_nm}nm)", fontsize=14)
    plt.xlabel("Drop in ROC-AUC (Higher = More Important)", fontsize=12)
    plt.axvline(0, color='k', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    out_path = os.path.join(config["paths"]["visualization_output"], f"feature_importance_{graph_type}_{thresh_nm}nm.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved Feature Importance to: {out_path}")


def plot_score_distribution(model, data_dict, config, device):
    print("\nGenerating Score Distribution Plot...")
    
    eval_sample_size = config["evaluation"]["test_node_sample_size"]
    num_nodes_total = data_dict["x_raw"].size(0)
    
    local_pos, local_neg, node_indices, local_weights, local_cand_weights = get_random_subgraph(
        data_dict["test_edges"], data_dict["test_cands"], num_nodes_total, 
        weights_cpu=data_dict["test_weights"], sample_size=eval_sample_size
    )
    
    selected_features = config["architecture"].get("selected_features", [0, 1, 2, 3, 4, 5, 6, 7])
    batch_x = data_dict["x_raw"][node_indices][:, selected_features].to(device)
    
    msg_edges = local_pos.to(device)
    msg_weights = local_weights.to(device) if local_weights is not None else None
    
    with torch.no_grad():
        z = model.encode(batch_x, msg_edges, edge_weight=msg_weights)
        
        # Score True Synapses
        pos_preds = model.decode(z, local_pos.to(device), explicit_weight=local_weights.to(device) if local_weights is not None else None)
        pos_scores = torch.sigmoid(pos_preds).cpu().numpy()
        
        # Score Candidates (Negatives)
        neg_preds = model.decode(z, local_neg.to(device), explicit_weight=local_cand_weights.to(device) if local_cand_weights is not None else None)
        neg_scores = torch.sigmoid(neg_preds).cpu().numpy()

    # Plotting (Mirroring the stacked style from your uploaded graphs)
    graph_type = os.path.splitext(config["paths"]["input_nx_graph"])[0]
    thresh_nm = config["graph_generation"]["spatial_threshold_nm"]
    
    plt.figure(figsize=(10, 6))
    
    # Using 'stepfilled' histtype with transparency to show overlapping distributions cleanly
    plt.hist([neg_scores, pos_scores], bins=50, stacked=True, 
             color=['#ff7f7f', '#7fbf7f'], label=['Candidates (Neg)', 'True Synapses (Pos)'])
    
    plt.axvline(x=0.5, color='k', linestyle='--', label='Default Threshold')
    
    plt.title(f"Score Distribution: {graph_type} ({thresh_nm}nm)", fontsize=14)
    plt.xlabel("Predicted Probability", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.legend(loc='upper left')
    plt.tight_layout()
    
    out_path = os.path.join(config["paths"]["visualization_output"], f"score_distribution_{graph_type}_{thresh_nm}nm.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved Score Distribution to: {out_path}")

# Main wrapper you can call from main.py
def generate_all_visualizations(model, data_dict, config, device):
    model.eval()
    plot_feature_importance(model, data_dict, config, device)
    plot_score_distribution(model, data_dict, config, device)
    print("\nAll visualizations completed successfully.")