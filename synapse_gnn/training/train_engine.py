import os
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

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
            local_cand_weights = torch.ones(local_candidates.size(1), dtype=torch.float32) 
    else:
        local_weights = None
        local_cand_weights = None
        
    return local_edge_index, local_candidates, perm, local_weights, local_cand_weights

def train_step(model, x_global, edge_index_train, cands_train, optimizer, device, weights_train=None, node_sample_size=6000):
    model.train()
    optimizer.zero_grad()
    num_nodes = x_global.size(0)
    
    local_edge_index, local_candidates, node_indices, local_weights, local_cand_weights = get_random_subgraph(
        edge_index_train, cands_train, num_nodes, weights_train, node_sample_size)
    if local_edge_index.size(1) < 2: return 0.0
    
    local_edge_index, local_candidates = local_edge_index.to(device), local_candidates.to(device)
    batch_x = x_global[node_indices].to(device)
    if local_weights is not None: local_weights = local_weights.to(device)
    if local_cand_weights is not None: local_cand_weights = local_cand_weights.to(device)
    
    num_edges = local_edge_index.size(1)
    perm = torch.randperm(num_edges, device=device)
    split_idx = int(num_edges * 0.5)
    
    msg_edges, target_edges = local_edge_index[:, perm[split_idx:]], local_edge_index[:, perm[:split_idx]]
    msg_weights = local_weights[perm[split_idx:]] if local_weights is not None else None

    z = model.encode(batch_x, msg_edges, edge_weight=msg_weights)
    
    pos_src, pos_dst = target_edges[0], target_edges[1]
    num_pos = pos_src.size(0)
    num_cands = local_candidates.size(1)
    
    pos_weights_target, neg_weights_target = None, None
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
        if local_cand_weights is not None: neg_weights_target = torch.zeros(num_pos, device=device) 
    
    target_weights = torch.cat([pos_weights_target, neg_weights_target]) if pos_weights_target is not None and neg_weights_target is not None else None
    target_edges = torch.cat([torch.stack([pos_src, pos_dst], dim=0), torch.stack([neg_src, neg_dst], dim=0)], dim=1)
    
    preds = model.decode(z, target_edges, explicit_weight=target_weights)
    targets = torch.cat([torch.ones(num_pos, device=device), torch.zeros(num_pos, device=device)])    
    loss = F.binary_cross_entropy_with_logits(preds, targets)
    loss.backward(); optimizer.step()
    return loss.item()

@torch.no_grad()
def validate_step(model, x_global, edge_index_test, cands_test, device, weights_test=None, node_sample_size=6000):
    model.eval()
    num_nodes = x_global.size(0)
    
    local_edge_index, local_candidates, node_indices, local_weights, local_cand_weights = get_random_subgraph(
        edge_index_test, cands_test, num_nodes, weights_test, node_sample_size)
    if local_edge_index.size(1) < 2: return 0.0, 0.0
    
    local_edge_index, local_candidates = local_edge_index.to(device), local_candidates.to(device)
    batch_x = x_global[node_indices].to(device)
    if local_weights is not None: local_weights = local_weights.to(device)
    if local_cand_weights is not None: local_cand_weights = local_cand_weights.to(device) 
    
    num_edges = local_edge_index.size(1)
    perm = torch.randperm(num_edges, device=device)
    split_idx = int(num_edges * 0.5)
    
    msg_edges, target_edges = local_edge_index[:, perm[split_idx:]], local_edge_index[:, perm[:split_idx]]
    msg_weights = local_weights[perm[split_idx:]] if local_weights is not None else None

    z = model.encode(batch_x, msg_edges, edge_weight=msg_weights)
    
    pos_src, pos_dst = target_edges[0], target_edges[1]
    num_pos = pos_src.size(0)
    num_cands = local_candidates.size(1)
    
    pos_weights_target, neg_weights_target = None, None
    if local_weights is not None: pos_weights_target = local_weights[perm[:split_idx]][:num_pos]
    
    if num_cands > num_pos:
        cand_perm = torch.randperm(num_cands, device=device)[:num_pos]
        neg_src, neg_dst = local_candidates[0, cand_perm], local_candidates[1, cand_perm]
        if local_cand_weights is not None: neg_weights_target = local_cand_weights[cand_perm].to(device)
    elif num_cands > 0:
        neg_src, neg_dst, num_pos = local_candidates[0], local_candidates[1], num_cands
        pos_src, pos_dst = pos_src[:num_pos], pos_dst[:num_pos]
        if local_cand_weights is not None:
            neg_weights_target = local_cand_weights.to(device)
            pos_weights_target = pos_weights_target[:num_pos] if pos_weights_target is not None else None
    else:
        neg_src = neg_dst = torch.randint(0, node_indices.size(0), (num_pos,), device=device)
        if local_cand_weights is not None: neg_weights_target = torch.zeros(num_pos, device=device)
    
    target_weights = torch.cat([pos_weights_target, neg_weights_target]) if pos_weights_target is not None and neg_weights_target is not None else None
    target_edges = torch.cat([torch.stack([pos_src, pos_dst], dim=0), torch.stack([neg_src, neg_dst], dim=0)], dim=1)
    
    preds = model.decode(z, target_edges, explicit_weight=target_weights)
    y_scores = torch.sigmoid(preds).cpu().numpy()
    num_neg = neg_src.size(0)
    y_true = np.concatenate([np.ones(num_pos), np.zeros(num_neg)])
    
    try: return roc_auc_score(y_true, y_scores), f1_score(y_true, (y_scores >= 0.5).astype(int), zero_division=0)
    except: return 0.0, 0.0

def run_training(model, train_data, test_data, config, device):
    """
    Updated to ingest PyTorch Geometric Data objects (train_data, test_data)
    instead of the legacy data_dict.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    
    selected_features = config["architecture"].get("selected_features", [0, 1, 2, 3, 4, 5, 6, 7])
    
    # 1. Extract node features
    x_morph = train_data.x[:, selected_features].to(device)
    
    # 2. Extract ADP Weights (Crucial: PyG edge_attr is 2D [E, 1], we MUST squeeze it to 1D)
    train_weights = train_data.edge_attr.squeeze(-1) if train_data.edge_attr is not None else None
    test_weights = test_data.edge_attr.squeeze(-1) if test_data.edge_attr is not None else None

    # 3. Map PyG Attributes to Legacy Variable Names
    train_cands = train_data.edge_index             # Structural ADP Graph
    train_edges = train_data.edge_label_index       # Ground Truth Synapses
    
    test_cands = test_data.edge_index               
    test_edges = test_data.edge_label_index         

    model_out = config["paths"]["model_out"]
    graph_type = "adp_graph" # Or dynamic extraction from config
    thresh_nm = config["graph_generation"]["spatial_threshold_nm"]
    
    weight_tag = "_with_continuous_weights" if train_weights is not None else ""
    MODEL_SAVE_PATH = os.path.join(model_out, f"best_model_{graph_type}{weight_tag}_{thresh_nm}nm.pth")

    best_val_auc = 0.0
    patience_counter = 0
    patience_limit = 10 

    for epoch in range(1, config["training"]["epochs"] + 1):
        total_loss = 0
        for _ in range(config["training"]["steps_per_epoch"]):
            # Pass mapped variables to the original train_step
            total_loss += train_step(model, x_morph, train_edges, train_cands, 
                                     optimizer, device, weights_train=train_weights, 
                                     node_sample_size=config["training"]["train_node_sample_size"])
        
        # Pass mapped variables to the original validate_step
        val_aucs, val_f1s = zip(*[validate_step(model, x_morph, test_edges, test_cands, 
                                                device, weights_test=test_weights, 
                                                node_sample_size=config["training"]["validation_node_sample_size"]) 
                                  for _ in range(config["training"]["validation_averaging_runs"])])
        avg_auc, avg_f1 = np.mean(val_aucs), np.mean(val_f1s)
        print(f"Epoch {epoch:03d} | Loss: {total_loss/config['training']['steps_per_epoch']:.4f} | Val AUC: {avg_auc:.4f} | Val F1: {avg_f1:.4f}")
        
        if avg_auc > best_val_auc:
            best_val_auc = avg_auc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print("Early Stopping Triggered!")
                break
                
    return MODEL_SAVE_PATH