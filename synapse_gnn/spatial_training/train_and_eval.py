import os
import argparse
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, f1_score
import numpy as np
import json

# Import your model architecture using the new module structure
from spatial_training import gnn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 1. CONFIGURATION LOADER ---
def parse_args():
    parser = argparse.ArgumentParser(description="Config-Driven GraphSAGE Synapse Predictor")
    parser.add_argument('--config', type=str, default="config.json", help="Path to the JSON configuration file")
    return parser.parse_args()

def load_config(config_path):
    print(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

# --- 2. DATA LOADER ---
def load_graph_data(data_dir, config):
    print(f"Loading standard graph tensors from: {data_dir}")
    paths = {
        "x": os.path.join(data_dir, "x_features.pt"),
        "train_pos": os.path.join(data_dir, "graph_train_edges.pt"),
        "test_pos": os.path.join(data_dir, "graph_test_edges.pt"),
        "train_cands": os.path.join(data_dir, "graph_train_spatial_candidates.pt"),
        "test_cands": os.path.join(data_dir, "graph_test_spatial_candidates.pt")
    }
    
    data = {k: torch.load(v, weights_only=False) for k, v in paths.items()}
    
    for name, tensor in data.items():
        print(f" -> {name}: shape {tensor.shape}")
        if tensor.numel() == 0 or len(tensor.shape) < 2:
            print(f"CRITICAL WARNING: {name} is empty or malformed!")

    # --- VERSION 2: LOAD WEIGHTS IF ADP ---
    graph_type = os.path.splitext(config["paths"]["input_nx_graph"])[0]
    has_weights = "adp" in graph_type.lower()
    
    if has_weights:
        print(" -> Detected ADP Graph: Loading continuous edge weights...")
        train_weights = torch.load(os.path.join(data_dir, "graph_train_spatial_weights.pt"), weights_only=False)
        test_weights = torch.load(os.path.join(data_dir, "graph_test_spatial_weights.pt"), weights_only=False)

        print("Normalizing continuous edge weights to prevent exploding loss")
        # Find the massive max value in the training set
        max_weight = train_weights.max() 
        # Scale both sets linearly between 0 and 1
        train_weights = train_weights / max_weight
        test_weights = test_weights / max_weight
        print(f" -> train_weights: shape {train_weights.shape}")
        print(f" -> test_weights: shape {test_weights.shape}")
    else:
        train_weights = None
        test_weights = None

    return data["x"], data["train_pos"], data["test_pos"], data["train_cands"], data["test_cands"], train_weights, test_weights

def log_to_file(message, log_file):
    print(message)
    with open(log_file, "a") as f:
        f.write(message + "\n")

# --- 3. PERFORMANCE ANALYSIS & JSON EXPORT ---
def analyze_model_performance(y_true, y_scores, graph_type, thresh_nm, model_out, log_file, vis_dir):
    try:
        auc = roc_auc_score(y_true, y_scores)
    except:
        auc = 0.0
        
    log_to_file(f"\n====== FINAL MODEL EVALUATION ======", log_file)
    log_to_file(f"ROC AUC: {auc:.4f}", log_file)

    thresholds = np.arange(0.01, 1.00, 0.01)
    best_f1 = 0.0
    best_thresh = 0.5
    
    for thresh in thresholds:
        preds = (y_scores >= thresh).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            
    final_preds = (y_scores >= best_thresh).astype(int)
    acc = accuracy_score(y_true, final_preds)
    prec = precision_score(y_true, final_preds, zero_division=0)
    rec = recall_score(y_true, final_preds, zero_division=0)
    cm = confusion_matrix(y_true, final_preds)
    
    log_to_file(f"Optimal Threshold: {best_thresh:.2f} | Max F1: {best_f1:.4f}", log_file)
    log_to_file(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}", log_file)
    log_to_file(f"TN: {cm[0,0]:,} | FP: {cm[0,1]:,} | FN: {cm[1,0]:,} | TP: {cm[1,1]:,}", log_file)

    results_summary = {
            "graph_type": graph_type,
            "spatial_threshold_nm": thresh_nm,
            "roc_auc": float(auc),
            "optimal_threshold": float(best_thresh),
            "best_f1": float(best_f1),
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "confusion_matrix": {"tn": int(cm[0,0]), "fp": int(cm[0,1]), "fn": int(cm[1,0]), "tp": int(cm[1,1])}
        }

    # NEW: Conditional JSON naming
    if 'adp' in graph_type.lower():
        json_filename = f"metrics_{graph_type}_added_adp_weights_{thresh_nm}nm.json"
    else:
        json_filename = f"metrics_{graph_type}_{thresh_nm}nm.json"
        
    # FIX: Route the JSON directly to the unique evals visualization directory
    json_out = os.path.join(vis_dir, json_filename)
    
    with open(json_out, 'w') as f:
        json.dump(results_summary, f, indent=4)
    log_to_file(f"Results exported to: {json_out}", log_file)
    log_to_file(f"Visualization Directory set to: {vis_dir}\n====================================", log_file)

# --- 4. GRAPH SAMPLERS ---
def get_random_subgraph(edge_index_cpu, candidates_cpu, num_nodes_total, weights_cpu=None, sample_size=6000):
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
            local_cand_weights = torch.ones(local_candidates.size(1), dtype=torch.float32) # FIX: Initialize!
    else:
        local_weights = None
        local_cand_weights = None
        
    return local_edge_index, local_candidates, perm, local_weights, local_cand_weights


def train_step(model, x_global, edge_index_train, cands_train, optimizer, weights_train=None, node_sample_size=6000):
    model.train()
    optimizer.zero_grad()
    num_nodes = x_global.size(0)
    
    local_edge_index, local_candidates, node_indices, local_weights, local_cand_weights = get_random_subgraph(
        edge_index_train, cands_train, num_nodes, weights_train, node_sample_size)
    if local_edge_index.size(1) < 2: return 0.0
    
    local_edge_index, local_candidates = local_edge_index.to(device), local_candidates.to(device)
    batch_x = x_global[node_indices].to(device)
    if local_weights is not None: local_weights = local_weights.to(device)
    
    num_edges = local_edge_index.size(1)
    perm = torch.randperm(num_edges, device=device)
    split_idx = int(num_edges * 0.5)
    
    msg_edges, target_edges = local_edge_index[:, perm[split_idx:]], local_edge_index[:, perm[:split_idx]]
    msg_weights = local_weights[perm[split_idx:]] if local_weights is not None else None

    # Pass dynamic weights into encoder
    z = model.encode(batch_x, msg_edges, edge_weight=msg_weights)
    
    pos_src, pos_dst = target_edges[0], target_edges[1]
    num_pos = pos_src.size(0)
    num_cands = local_candidates.size(1)
    
    # Initialize variables to hold the true weights for the target edges
    pos_weights_target = None
    neg_weights_target = None
    
    if local_weights is not None:
        pos_weights_target = local_weights[perm[:split_idx]][:num_pos]
    
    if num_cands > num_pos:
        cand_perm = torch.randperm(num_cands, device=device)[:num_pos]
        neg_src, neg_dst = local_candidates[0, cand_perm], local_candidates[1, cand_perm]
        if local_cand_weights is not None:
            neg_weights_target = local_cand_weights[cand_perm].to(device)
            
    elif num_cands > 0:
        neg_src, neg_dst, num_pos = local_candidates[0], local_candidates[1], num_cands
        pos_src, pos_dst = pos_src[:num_pos], pos_dst[:num_pos]
        if local_cand_weights is not None:
            neg_weights_target = local_cand_weights.to(device)
            pos_weights_target = pos_weights_target[:num_pos] if pos_weights_target is not None else None
    else:
        neg_src = neg_dst = torch.randint(0, node_indices.size(0), (num_pos,), device=device)
        if local_cand_weights is not None:
            neg_weights_target = torch.zeros(num_pos, device=device) # Fallback if no structural cands exist
    
    # Combine the true positive and true negative weights for the decoder
    if pos_weights_target is not None and neg_weights_target is not None:
        target_weights = torch.cat([pos_weights_target, neg_weights_target])
    else:
        target_weights = None

    target_edges = torch.cat([
        torch.stack([pos_src, pos_dst], dim=0),
        torch.stack([neg_src, neg_dst], dim=0)
    ], dim=1)
    
    preds = model.decode(z, target_edges, explicit_weight=target_weights)
    targets = torch.cat([torch.ones(num_pos, device=device), torch.zeros(num_pos, device=device)])    
    loss = F.binary_cross_entropy_with_logits(preds, targets)
    loss.backward(); optimizer.step()
    return loss.item()


@torch.no_grad()
def validate_step(model, x_global, edge_index_test, cands_test, weights_test=None, node_sample_size=6000):
    model.eval()
    num_nodes = x_global.size(0)
    
    # FIX: Unpack local_cand_weights
    local_edge_index, local_candidates, node_indices, local_weights, local_cand_weights = get_random_subgraph(
        edge_index_test, cands_test, num_nodes, weights_test, node_sample_size)
    if local_edge_index.size(1) < 2: return 0.0, 0.0
    
    local_edge_index, local_candidates = local_edge_index.to(device), local_candidates.to(device)
    batch_x = x_global[node_indices].to(device)
    if local_weights is not None: local_weights = local_weights.to(device)
    
    num_edges = local_edge_index.size(1)
    perm = torch.randperm(num_edges, device=device)
    split_idx = int(num_edges * 0.5)
    
    msg_edges, target_edges = local_edge_index[:, perm[split_idx:]], local_edge_index[:, perm[:split_idx]]
    msg_weights = local_weights[perm[split_idx:]] if local_weights is not None else None

    z = model.encode(batch_x, msg_edges, edge_weight=msg_weights)
    
    pos_src, pos_dst = target_edges[0], target_edges[1]
    num_pos = pos_src.size(0)
    num_cands = local_candidates.size(1)
    
    # Initialize variables to hold the true weights for the target edges
    pos_weights_target = None
    neg_weights_target = None
    
    if local_weights is not None:
        pos_weights_target = local_weights[perm[:split_idx]][:num_pos]
    
    if num_cands > num_pos:
        cand_perm = torch.randperm(num_cands, device=device)[:num_pos]
        neg_src, neg_dst = local_candidates[0, cand_perm], local_candidates[1, cand_perm]
        if local_cand_weights is not None:
            neg_weights_target = local_cand_weights[cand_perm].to(device)
            
    elif num_cands > 0:
        neg_src, neg_dst, num_pos = local_candidates[0], local_candidates[1], num_cands
        pos_src, pos_dst = pos_src[:num_pos], pos_dst[:num_pos]
        if local_cand_weights is not None:
            neg_weights_target = local_cand_weights.to(device)
            pos_weights_target = pos_weights_target[:num_pos] if pos_weights_target is not None else None
    else:
        neg_src = neg_dst = torch.randint(0, node_indices.size(0), (num_pos,), device=device)
        if local_cand_weights is not None:
            neg_weights_target = torch.zeros(num_pos, device=device)
    
    if pos_weights_target is not None and neg_weights_target is not None:
        target_weights = torch.cat([pos_weights_target, neg_weights_target])
    else:
        target_weights = None

    target_edges = torch.cat([
        torch.stack([pos_src, pos_dst], dim=0),
        torch.stack([neg_src, neg_dst], dim=0)
    ], dim=1)
    
    # FIX: Use the new MLP decoder instead of hardcoded dot-product
    preds = model.decode(z, target_edges, explicit_weight=target_weights)
    
    y_scores = torch.sigmoid(preds).cpu().numpy()
    num_neg = neg_src.size(0)
    y_true = np.concatenate([np.ones(num_pos), np.zeros(num_neg)])
    
    try:
        auc = roc_auc_score(y_true, y_scores)
        f1 = f1_score(y_true, (y_scores >= 0.5).astype(int), zero_division=0)
    except:
        auc = f1 = 0.0
    return auc, f1

# --- 5. MAIN PIPELINE ---
# --- 5. MAIN PIPELINE ---
def main():
    args = parse_args()
    config = load_config(args.config)
    
    data_dir = config["paths"]["data_dir"]
    model_out = config["paths"]["model_out"]
    
    # DYNAMIC PATHING
    graph_type = os.path.splitext(config["paths"]["input_nx_graph"])[0]
    thresh_nm = config["graph_generation"]["spatial_threshold_nm"]
    log_file = config["training"]["log_file_name"]
    
    vis_dir = config["paths"]["visualization_output"]
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(model_out, exist_ok=True)
    
    x_global_raw, train_edges, test_edges, train_cands, test_cands, train_weights, test_weights = load_graph_data(data_dir, config)
    num_features = config["architecture"]["in_channels"]
    x_morph = x_global_raw[:, :num_features].to(device)
    
    model = gnn.SynapsePredictor(
        in_channels=num_features, 
        hidden_channels=config["architecture"]["hidden_dim"]
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    print(f"\n--- Starting Training for {graph_type} at {thresh_nm}nm ---")
    
    best_val_auc = 0.0
    patience_counter = 0
    patience_limit = 10 
    
    # NEW: Conditional Model Weights naming
    if 'adp' in graph_type.lower():
        MODEL_SAVE_PATH = os.path.join(model_out, f"best_model_{graph_type}_added_adp_weights_{thresh_nm}nm.pth")
    else:
        MODEL_SAVE_PATH = os.path.join(model_out, f"best_model_{graph_type}_{thresh_nm}nm.pth")

    for epoch in range(1, config["training"]["epochs"] + 1):
        total_loss = 0
        for _ in range(config["training"]["steps_per_epoch"]):
            total_loss += train_step(model, x_morph, train_edges, train_cands, optimizer,
                                     weights_train=train_weights, 
                                     node_sample_size=config["training"]["train_node_sample_size"])
        
        val_aucs, val_f1s = zip(*[validate_step(model, x_morph, test_edges, test_cands,
                                                weights_test=test_weights, 
                                                node_sample_size=config["training"]["validation_node_sample_size"]) 
                                  for _ in range(config["training"]["validation_averaging_runs"])])
        avg_auc, avg_f1 = np.mean(val_aucs), np.mean(val_f1s)
        
        log_to_file(f"Epoch {epoch:03d} | Loss: {total_loss/config['training']['steps_per_epoch']:.4f} | Val AUC: {avg_auc:.4f} | Val F1: {avg_f1:.4f}", log_file)
        
        if avg_auc > best_val_auc:
            best_val_auc = avg_auc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience_limit:
            log_to_file(f"\n--- Early Stopping Triggered! ---", log_file)
            break

    # --- 6. FINAL AGGREGATED EVALUATION ---
    log_to_file("\nTraining Complete. Running Inductive Analysis on Test Set...", log_file)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=False))
    model.eval()
    
    all_y_true, all_y_scores = [], []
    num_nodes_total = x_global_raw.size(0)

    with torch.no_grad():
        for i in range(config["evaluation"]["test_aggregation_runs"]):
            # FIX: Unpack local_cand_weights
            local_test_edges, local_test_cands, node_indices, local_weights, local_cand_weights = get_random_subgraph(
                test_edges, test_cands, num_nodes_total, weights_cpu=test_weights, sample_size=config["evaluation"]["test_node_sample_size"])
            if local_test_edges.size(1) == 0: continue
            
            num_edges = local_test_edges.size(1)
            perm = torch.randperm(num_edges, device=device)
            split_idx = int(num_edges * 0.5)
            
            msg_edges = local_test_edges[:, perm[split_idx:]].to(device)
            target_edges = local_test_edges[:, perm[:split_idx]].to(device)
            
            if local_weights is not None:
                local_weights = local_weights.to(device)
                msg_weights = local_weights[perm[split_idx:]]
            else:
                msg_weights = None
            
            batch_x = x_global_raw[node_indices, :num_features].to(device)
            
            z = model.encode(batch_x, msg_edges, edge_weight=msg_weights) 
            
            pos_src, pos_dst = target_edges[0], target_edges[1]            
            num_pos = pos_src.size(0)
            num_cands = local_test_cands.size(1)
            
            pos_weights_target = None
            neg_weights_target = None
            
            if local_weights is not None:
                pos_weights_target = local_weights[perm[:split_idx]][:num_pos]
            
            if num_cands > num_pos:
                cand_perm = torch.randperm(num_cands, device=device)[:num_pos]
                neg_src, neg_dst = local_test_cands[0, cand_perm], local_test_cands[1, cand_perm]
                if local_cand_weights is not None:
                    neg_weights_target = local_cand_weights[cand_perm].to(device)
            elif num_cands > 0:
                neg_src, neg_dst, num_pos = local_test_cands[0], local_test_cands[1], num_cands
                pos_src, pos_dst = pos_src[:num_pos], pos_dst[:num_pos]
                if local_cand_weights is not None:
                    neg_weights_target = local_cand_weights.to(device)
                    pos_weights_target = pos_weights_target[:num_pos] if pos_weights_target is not None else None
            else: 
                continue 

            if pos_weights_target is not None and neg_weights_target is not None:
                target_weights = torch.cat([pos_weights_target, neg_weights_target])
            else:
                target_weights = None

            target_edges_eval = torch.cat([
                torch.stack([pos_src, pos_dst], dim=0),
                torch.stack([neg_src, neg_dst], dim=0)
            ], dim=1)
            
            # FIX: Use the new MLP decoder instead of hardcoded dot-product
            preds = model.decode(z, target_edges_eval, explicit_weight=target_weights)
            
            all_y_scores.append(torch.sigmoid(preds).cpu().numpy())
            num_neg = neg_src.size(0)
            all_y_true.append(np.concatenate([np.ones(num_pos), np.zeros(num_neg)]))

    if all_y_true:
        analyze_model_performance(np.concatenate(all_y_true), np.concatenate(all_y_scores), graph_type, thresh_nm, model_out, log_file, vis_dir)
    log_to_file("Pipeline Complete.", log_file)
    
if __name__ == "__main__":
    main()