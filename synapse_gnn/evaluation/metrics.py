import os
import json
import torch
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, average_precision_score, brier_score_loss, precision_recall_curve

# Correctly import the helper from train_engine!
from synapse_gnn.training.train_engine import get_random_subgraph

def export_metrics(y_true, y_scores, config, data_dict):
    try:
        auc = roc_auc_score(y_true, y_scores)
        pr_auc = average_precision_score(y_true, y_scores)
        brier = brier_score_loss(y_true, y_scores)
    except:
        auc = pr_auc = brier = 0.0
        
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    numerator = 2 * precisions * recalls
    denominator = precisions + recalls
    f1_scores = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
    
    if len(f1_scores) > 0:
        best_idx = np.argmax(f1_scores)
        best_f1 = f1_scores[best_idx]
        best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    else:
        best_f1, best_thresh = 0.0, 0.5
            
    final_preds = (y_scores >= best_thresh).astype(int)
    cm = confusion_matrix(y_true, final_preds)
    
    graph_type = os.path.splitext(config["paths"]["input_nx_graph"])[0]
    thresh_nm = config["graph_generation"]["spatial_threshold_nm"]
    
    results = {
        "graph_type": graph_type,
        "spatial_threshold_nm": thresh_nm,
        "roc_auc": float(auc),
        "pr_auc": float(pr_auc),
        "brier_score": float(brier),
        "optimal_threshold": float(best_thresh),
        "best_f1": float(best_f1),
        "confusion_matrix": {"tn": int(cm[0,0]), "fp": int(cm[0,1]), "fn": int(cm[1,0]), "tp": int(cm[1,1])}
    }
    
    weight_tag = "_with_continuous_weights" if data_dict["train_weights"] is not None else ""
    json_out = os.path.join(config["paths"]["visualization_output"], f"metrics_{graph_type}{weight_tag}_{thresh_nm}nm.json")
    
    with open(json_out, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nFinal Metrics exported to: {json_out}")

@torch.no_grad()
def run_inductive_evaluation(model, model_path, data_dict, config, device):
    print("\nRunning Inductive Analysis on Test Set...")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()
    
    selected_features = config["architecture"].get("selected_features", [0, 1, 2, 3, 4, 5, 6, 7])
    num_nodes_total = data_dict["x_raw"].size(0)
    all_y_true, all_y_scores = [], []

    for _ in range(config["evaluation"]["test_aggregation_runs"]):
        local_test_edges, local_test_cands, node_indices, local_weights, local_cand_weights = get_random_subgraph(
            data_dict["test_edges"], data_dict["test_cands"], num_nodes_total, weights_cpu=data_dict["test_weights"], 
            sample_size=config["evaluation"]["test_node_sample_size"])
            
        if local_test_edges.size(1) == 0: continue
        
        # --- NEW: Move everything to the GPU before indexing! ---
        local_test_edges = local_test_edges.to(device)
        local_test_cands = local_test_cands.to(device)
        if local_weights is not None: local_weights = local_weights.to(device)
        if local_cand_weights is not None: local_cand_weights = local_cand_weights.to(device)
        
        num_edges = local_test_edges.size(1)
        perm = torch.randperm(num_edges, device=device)
        split_idx = int(num_edges * 0.5)
        
        # Slicing now happens safely on the GPU
        msg_edges = local_test_edges[:, perm[split_idx:]]
        target_edges = local_test_edges[:, perm[:split_idx]]
        msg_weights = local_weights[perm[split_idx:]] if local_weights is not None else None
        
        batch_x = data_dict["x_raw"][node_indices][:, selected_features].to(device)
        z = model.encode(batch_x, msg_edges, edge_weight=msg_weights) 
        
        pos_src, pos_dst = target_edges[0], target_edges[1]            
        num_pos = pos_src.size(0)
        num_cands = local_test_cands.size(1)
        
        pos_weights_target, neg_weights_target = None, None
        if local_weights is not None: pos_weights_target = local_weights[perm[:split_idx]][:num_pos]
        
        if num_cands > num_pos:
            cand_perm = torch.randperm(num_cands, device=device)[:num_pos]
            neg_src, neg_dst = local_test_cands[0, cand_perm], local_test_cands[1, cand_perm]
            if local_cand_weights is not None: neg_weights_target = local_cand_weights[cand_perm]
        elif num_cands > 0:
            neg_src, neg_dst, num_pos = local_test_cands[0], local_test_cands[1], num_cands
            pos_src, pos_dst = pos_src[:num_pos], pos_dst[:num_pos]
            if local_cand_weights is not None:
                neg_weights_target = local_cand_weights
                pos_weights_target = pos_weights_target[:num_pos] if pos_weights_target is not None else None
        else: continue 

        target_weights = torch.cat([pos_weights_target, neg_weights_target]) if pos_weights_target is not None and neg_weights_target is not None else None
        target_edges_eval = torch.cat([torch.stack([pos_src, pos_dst], dim=0), torch.stack([neg_src, neg_dst], dim=0)], dim=1)
        
        preds = model.decode(z, target_edges_eval, explicit_weight=target_weights)
        all_y_scores.append(torch.sigmoid(preds).cpu().numpy())
        all_y_true.append(np.concatenate([np.ones(num_pos), np.zeros(neg_src.size(0))]))

    if all_y_true:
        export_metrics(np.concatenate(all_y_true), np.concatenate(all_y_scores), config, data_dict)