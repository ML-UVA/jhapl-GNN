import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from synapse_gnn.training.train_engine import get_random_subgraph

ALL_MORPH_NAMES = [
    "Soma Vol", "Axon Len", "Basal Len", "Apical Len", 
    "Max Axon Reach", "Max Dendrite Reach", "Total Spines", "Total Spine Vol"
]

def calculate_auc(model, z, pos_edges, neg_edges, explicit_weights, device):
    if pos_edges.size(1) == 0 or neg_edges.size(1) == 0: return 0.5
    num_pos = pos_edges.size(1)
    target_edges = torch.cat([pos_edges.to(device), neg_edges.to(device)], dim=1)
    with torch.no_grad():
        preds = model.decode(z, target_edges, explicit_weight=explicit_weights)
        all_scores = torch.sigmoid(preds).cpu().numpy()
        
    pos_scores, neg_scores = all_scores[:num_pos], all_scores[num_pos:]
    y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    y_scores = np.concatenate([pos_scores, neg_scores])
    return roc_auc_score(y_true, y_scores)


def plot_feature_importance(model, train_data, test_data, config, device):
    print("\nGenerating Feature Importance Plot...")
    
    # 1. Unpack PyG Object
    x_raw = test_data.x
    test_edges = test_data.edge_label_index
    test_cands = test_data.edge_index
    test_weights = test_data.edge_attr.squeeze(-1) if test_data.edge_attr is not None else None
    
    selected_features = config["architecture"].get("selected_features", [0, 1, 2, 3, 4, 5, 6, 7])
    active_morph_names = [ALL_MORPH_NAMES[i] for i in selected_features]
    
    local_pos, local_neg, node_indices, local_weights, local_cand_weights = get_random_subgraph(
        test_edges, test_cands, x_raw.size(0), 
        weights_cpu=test_weights, sample_size=config["evaluation"]["test_node_sample_size"]
    )
    
    num_pos, num_neg = local_pos.size(1), local_neg.size(1)
    if num_neg > num_pos:
        perm = torch.randperm(num_neg)[:num_pos]
        local_neg = local_neg[:, perm]
        if local_cand_weights is not None: local_cand_weights = local_cand_weights[perm]
            
    explicit_weights = torch.cat([local_weights, local_cand_weights]).to(device) if (local_weights is not None and local_cand_weights is not None) else None

    batch_x = x_raw[node_indices][:, selected_features].to(device)
    msg_edges = local_pos.to(device) 
    msg_weights = local_weights.to(device) if local_weights is not None else None
    
    with torch.no_grad():
        z_base = model.encode(batch_x, msg_edges, edge_weight=msg_weights)
    baseline_auc = calculate_auc(model, z_base, local_pos, local_neg, explicit_weights, device)
    
    importances = []
    for i in range(len(selected_features)):
        permuted_x = batch_x.clone()
        perm_idx = torch.randperm(permuted_x.size(0))
        permuted_x[:, i] = permuted_x[perm_idx, i] 
        with torch.no_grad():
            z_perm = model.encode(permuted_x, msg_edges, edge_weight=msg_weights)
        shuffled_auc = calculate_auc(model, z_perm, local_pos, local_neg, explicit_weights, device)
        importances.append(baseline_auc - shuffled_auc)
        
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


def plot_score_distribution(model, train_data, test_data, config, device):
    print("\nGenerating Score Distribution Plot...")
    
    x_raw = test_data.x
    test_edges = test_data.edge_label_index
    test_cands = test_data.edge_index
    test_weights = test_data.edge_attr.squeeze(-1) if test_data.edge_attr is not None else None
    
    local_pos, local_neg, node_indices, local_weights, local_cand_weights = get_random_subgraph(
        test_edges, test_cands, x_raw.size(0), 
        weights_cpu=test_weights, sample_size=config["evaluation"]["test_node_sample_size"]
    )
    
    selected_features = config["architecture"].get("selected_features", [0, 1, 2, 3, 4, 5, 6, 7])
    batch_x = x_raw[node_indices][:, selected_features].to(device)
    
    msg_edges = local_pos.to(device)
    msg_weights = local_weights.to(device) if local_weights is not None else None
    
    with torch.no_grad():
        z = model.encode(batch_x, msg_edges, edge_weight=msg_weights)
        pos_preds = model.decode(z, local_pos.to(device), explicit_weight=local_weights.to(device) if local_weights is not None else None)
        pos_scores = torch.sigmoid(pos_preds).cpu().numpy()
        
        neg_preds = model.decode(z, local_neg.to(device), explicit_weight=local_cand_weights.to(device) if local_cand_weights is not None else None)
        neg_scores = torch.sigmoid(neg_preds).cpu().numpy()

    graph_type = os.path.splitext(config["paths"]["input_nx_graph"])[0]
    thresh_nm = config["graph_generation"]["spatial_threshold_nm"]
    
    plt.figure(figsize=(10, 6))
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


def plot_distance_vs_probability(model, train_data, test_data, config, device):
    print("\nGenerating Distance vs. Probability Check...")
    
    x_raw = test_data.x
    test_edges = test_data.edge_label_index
    test_cands = test_data.edge_index
    test_weights = test_data.edge_attr.squeeze(-1) if test_data.edge_attr is not None else None
    
    local_pos, local_neg, node_indices, local_weights, local_cand_weights = get_random_subgraph(
        test_edges, test_cands, x_raw.size(0), 
        weights_cpu=test_weights, sample_size=config["evaluation"]["test_node_sample_size"]
    )
    
    morph_indices = config["architecture"].get("selected_features", [0, 1, 2, 3, 4, 5, 6, 7])
    spatial_indices = config["architecture"].get("spatial_features", [8, 9, 10]) 
    
    batch_x_morph = x_raw[node_indices][:, morph_indices].to(device)
    batch_x_spatial = x_raw[node_indices][:, spatial_indices]
    
    with torch.no_grad():
        z = model.encode(batch_x_morph, local_pos.to(device), 
                         edge_weight=local_weights.to(device) if local_weights is not None else None)
        
        num_neg_plot = min(local_pos.size(1) * 3, local_neg.size(1))
        local_neg_plot = local_neg[:, :num_neg_plot]
        cand_weights_plot = local_cand_weights[:num_neg_plot] if local_cand_weights is not None else None

        probs_real = torch.sigmoid(model.decode(z, local_pos.to(device), 
                                                explicit_weight=local_weights.to(device) if local_weights is not None else None)).cpu().numpy()
        probs_neg = torch.sigmoid(model.decode(z, local_neg_plot.to(device), 
                                               explicit_weight=cand_weights_plot.to(device) if cand_weights_plot is not None else None)).cpu().numpy()
    
    pos_src, pos_dst = local_pos[0], local_pos[1]
    dists_real = torch.norm(batch_x_spatial[pos_src] - batch_x_spatial[pos_dst], dim=1).numpy()
    
    neg_src, neg_dst = local_neg_plot[0], local_neg_plot[1]
    dists_neg = torch.norm(batch_x_spatial[neg_src] - batch_x_spatial[neg_dst], dim=1).numpy()
    
    plt.figure(figsize=(10, 8))
    plt.scatter(dists_neg, probs_neg, alpha=0.3, color='red', s=15, label="Structural Candidates")
    plt.scatter(dists_real, probs_real, alpha=0.5, color='green', s=20, label="True Synapses")
    plt.title("Model Confidence vs. True Euclidean Distance", fontsize=14)
    plt.xlabel("Euclidean Distance in Microns", fontsize=12)
    plt.ylabel("Predicted Probability", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_path = os.path.join(config["paths"]["visualization_output"], "distance_vs_prob.png")
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_volume_impact(model, train_data, test_data, config, device):
    print("\nGenerating Volume Impact Analysis...")
    
    x_raw = test_data.x
    test_edges = test_data.edge_label_index
    test_cands = test_data.edge_index
    test_weights = test_data.edge_attr.squeeze(-1) if test_data.edge_attr is not None else None
    
    local_pos, _, node_indices, local_weights, _ = get_random_subgraph(
        test_edges, test_cands, x_raw.size(0), 
        weights_cpu=test_weights, sample_size=config["evaluation"]["test_node_sample_size"]
    )
    
    morph_indices = config["architecture"].get("selected_features", [0, 1, 2, 3, 4, 5, 6, 7])
    batch_x_morph = x_raw[node_indices][:, morph_indices].to(device)
    
    with torch.no_grad():
        z = model.encode(batch_x_morph, local_pos.to(device), 
                         edge_weight=local_weights.to(device) if local_weights is not None else None)
        probs = torch.sigmoid(model.decode(z, local_pos.to(device), 
                                           explicit_weight=local_weights.to(device) if local_weights is not None else None)).cpu().numpy()
    
    src, dst = local_pos[0], local_pos[1]
    vol_src = batch_x_morph[src, 0].cpu()
    vol_dst = batch_x_morph[dst, 0].cpu()
    avg_vol = (vol_src + vol_dst) / 2.0
    
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(avg_vol.numpy(), probs, alpha=0.4, c=probs, cmap='viridis')
    plt.colorbar(sc, label="Model Confidence")
    plt.title("Impact of Soma Volume on Prediction Confidence (True Synapses)", fontsize=14)
    plt.xlabel("Average Soma Volume (microns^3)", fontsize=12)
    plt.ylabel("Predicted Probability", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    out_path = os.path.join(config["paths"]["visualization_output"], "volume_impact.png")
    plt.savefig(out_path, dpi=300)
    plt.close()


def generate_all_visualizations(model, train_data, test_data, config, device):
    model.eval()
    plot_feature_importance(model, train_data, test_data, config, device)
    plot_score_distribution(model, train_data, test_data, config, device)
    plot_distance_vs_probability(model, train_data, test_data, config, device)
    plot_volume_impact(model, train_data, test_data, config, device)
    print("\nAll visualizations completed successfully.")