import os
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
import preprocessing  # Your updated script
import gnn  # Your GNN model

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
# FOLDERS
CACHE_DIR = "cache_spatial"       # Separate folder for spatial split data
OUTPUT_FOLDER = "saved_models_spatial"
LOG_FILE = os.path.join(CACHE_DIR, "training_log_spatial_v2.txt")

# RAW DATA LOCATION (Where your .pbz2 files are)
RAW_GRAPH_DIR = "graph_exports/"  

# FILE PATHS
PATH_X = os.path.join(CACHE_DIR, "x_features.pt")
PATH_INDICES = os.path.join(CACHE_DIR, "valid_indices.pt")
PATH_TRAIN_EDGES = os.path.join(CACHE_DIR, "graph_train_edges.pt")
PATH_TEST_EDGES  = os.path.join(CACHE_DIR, "graph_test_edges.pt")
PATH_SPLIT_MASK  = os.path.join(CACHE_DIR, "spatial_split_masks.pt")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def log_to_file(message):
    print(message)
    with open(LOG_FILE, "a") as f:
        f.write(message + "\n")

# ---------------------------------------------------------
# INITIALIZATION & CHECKING
# ---------------------------------------------------------
def initialize_dataset():
    """
    Checks if features exist in cache_spatial.
    If not, generates them from raw files.
    """
    if os.path.exists(PATH_X) and os.path.exists(PATH_INDICES):
        log_to_file(f"Found cached features at {PATH_X}")
        return

    log_to_file(f"Features missing in {CACHE_DIR}. Generating now...")
    
    # 1. Get IDs
    neuron_ids = preprocessing.get_neuron_ids_from_folder(RAW_GRAPH_DIR)
    if not neuron_ids:
        log_to_file(f"CRITICAL ERROR: No .pbz2 files found in {RAW_GRAPH_DIR}")
        exit()
        
    log_to_file(f"Found {len(neuron_ids)} raw neuron files.")

    # 2. Run Preprocessing
    x, indices = preprocessing.build_node_features(neuron_ids, RAW_GRAPH_DIR)
    
    if x is None:
        log_to_file("Feature generation failed.")
        exit()
        
    valid_neuron_ids = [neuron_ids[i] for i in indices]

    # 3. Save to Cache
    log_to_file(f"Saving {x.shape[0]} feature vectors to {PATH_X}...")
    torch.save(x, PATH_X)
    torch.save(valid_neuron_ids, PATH_INDICES) # Save the IDs/Indices too!
    log_to_file("Feature generation complete.\n")

# ---------------------------------------------------------
# SAMPLER (GraphSAINT)
# ---------------------------------------------------------
def get_random_subgraph(edge_index_cpu, num_nodes_total, sample_size=8000):
    node_mask = torch.zeros(num_nodes_total, dtype=torch.bool)
    perm = torch.randperm(num_nodes_total)[:sample_size]
    node_mask[perm] = True
    
    row, col = edge_index_cpu
    edge_mask = node_mask[row] & node_mask[col]
    subset_edge_index = edge_index_cpu[:, edge_mask]
    
    dense_map = torch.full((num_nodes_total,), -1, dtype=torch.long)
    dense_map[perm] = torch.arange(sample_size)
    
    new_src = dense_map[subset_edge_index[0]]
    new_dst = dense_map[subset_edge_index[1]]
    
    local_edge_index = torch.stack([new_src, new_dst], dim=0)
    return local_edge_index, perm

def train_step(model, x_global, edge_index_train, optimizer, node_sample_size):
    model.train()
    optimizer.zero_grad()
    num_nodes = x_global.size(0)
    
    # 1. Sample Subgraph
    local_edge_index, node_indices = get_random_subgraph(edge_index_train, num_nodes, node_sample_size)
    if local_edge_index.size(1) == 0: return 0.0

    local_edge_index = local_edge_index.to(device)
    batch_x = x_global[node_indices].to(device)
    
    # 2. Forward Pass
    z = model.encode(batch_x, local_edge_index)
    
    pos_src, pos_dst = local_edge_index[0], local_edge_index[1]
    num_pos = pos_src.size(0)
    
    # ---------------------------------------------------------
    # HARD NEGATIVE MINING BLOCK
    # ---------------------------------------------------------
    # Strategy: Sample 4x candidates, keep the hardest 1x
    # ---------------------------------------------------------
    
    # A. Generate Candidates (4x more than needed)
    candidate_factor = 4
    cand_src = torch.randint(0, node_sample_size, (num_pos * candidate_factor,), device=device)
    cand_dst = torch.randint(0, node_sample_size, (num_pos * candidate_factor,), device=device)
    
    # B. Score Candidates (No grad needed for selection to save memory, 
    #    but we need grad for the final loss. We do a quick check here.)
    with torch.no_grad():
        cand_scores = (z[cand_src] * z[cand_dst]).sum(dim=1)
        
    # C. Select "Hardest" Negatives 
    # We want candidates with HIGHEST scores (model thinks they are positive)
    # torch.topk returns values and indices. We just need indices.
    _, hard_indices = torch.topk(cand_scores, k=num_pos)
    
    neg_src = cand_src[hard_indices]
    neg_dst = cand_dst[hard_indices]
    
    # ---------------------------------------------------------
    # END MINING BLOCK
    # ---------------------------------------------------------
    
    # 3. Calculate Final Scores (With Gradients)
    pos_scores = (z[pos_src] * z[pos_dst]).sum(dim=1)
    neg_scores = (z[neg_src] * z[neg_dst]).sum(dim=1) # Recalculate with grad
    
    preds = torch.cat([pos_scores, neg_scores])
    targets = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
    
    loss = F.binary_cross_entropy_with_logits(preds, targets)
    loss.backward()
    optimizer.step()
    return loss.item()
    
@torch.no_grad()
def validate_step(model, x_global, edge_index_test, node_sample_size=8000):
    """
    Returns AUC and F1. 
    Note: F1 here uses a hard 0.5 threshold just for monitoring during training.
    The final evaluation will use the optimal threshold.
    """
    model.eval()
    num_nodes = x_global.size(0)
    
    local_edge_index, node_indices = get_random_subgraph(edge_index_test, num_nodes, node_sample_size)
    if local_edge_index.size(1) == 0: return 0.0, 0.0

    local_edge_index = local_edge_index.to(device)
    batch_x = x_global[node_indices].to(device)
    
    z = model.encode(batch_x, local_edge_index)
    
    pos_src, pos_dst = local_edge_index[0], local_edge_index[1]
    num_pos = pos_src.size(0)
    neg_src = torch.randint(0, node_sample_size, (num_pos,), device=device)
    neg_dst = torch.randint(0, node_sample_size, (num_pos,), device=device)
    
    pos_scores = (z[pos_src] * z[pos_dst]).sum(dim=1)
    neg_scores = (z[neg_src] * z[neg_dst]).sum(dim=1)
    
    y_scores = torch.cat([torch.sigmoid(pos_scores), torch.sigmoid(neg_scores)]).cpu().numpy()
    y_true = np.concatenate([np.ones(num_pos), np.zeros(num_pos)])
    
    try:
        auc = roc_auc_score(y_true, y_scores)
        preds = (y_scores >= 0.5).astype(int)
        f1 = f1_score(y_true, preds)
    except:
        auc, f1 = 0.0, 0.0
    return auc, f1

def analyze_model_performance(y_true, y_scores):
    """
    Calculates comprehensive metrics and finds the optimal decision threshold.
    """
    # 1. AUC
    try:
        auc = roc_auc_score(y_true, y_scores)
    except:
        auc = 0.0
        
    log_to_file(f"\n====== FINAL MODEL EVALUATION ======")
    log_to_file(f"ROC AUC: {auc:.4f}")

    # 2. Find Optimal Threshold (Maximize F1)
    thresholds = np.arange(0.01, 1.00, 0.01)
    best_f1 = 0.0
    best_thresh = 0.5
    
    for thresh in thresholds:
        preds = (y_scores >= thresh).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            
    # --- EXPLICITLY PRINT OPTIMAL THRESHOLD ---
    log_to_file(f"\n--- Threshold Optimization ---")
    log_to_file(f"Optimal Decision Threshold: {best_thresh:.2f}")
    log_to_file(f"Max Achieved F1 Score:      {best_f1:.4f}")
    
    # 3. Metrics at Best Threshold
    final_preds = (y_scores >= best_thresh).astype(int)
    
    acc = accuracy_score(y_true, final_preds)
    prec = precision_score(y_true, final_preds, zero_division=0)
    rec = recall_score(y_true, final_preds, zero_division=0)
    
    log_to_file(f"\n--- Metrics at Optimal Threshold ({best_thresh:.2f}) ---")
    log_to_file(f"Accuracy:  {acc:.4f}")
    log_to_file(f"Precision: {prec:.4f}")
    log_to_file(f"Recall:    {rec:.4f}")
    log_to_file(f"F1 Score:  {best_f1:.4f}")
    
    # 4. Confusion Matrix
    cm = confusion_matrix(y_true, final_preds)
    log_to_file(f"\n--- Confusion Matrix (Thresh={best_thresh:.2f}) ---")
    log_to_file(f"TN: {cm[0,0]:,} | FP: {cm[0,1]:,}")
    log_to_file(f"FN: {cm[1,0]:,} | TP: {cm[1,1]:,}")
    log_to_file(f"====================================")
# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------
if __name__ == "__main__":
    with open(LOG_FILE, "w") as f: f.write("Training Log (Spatial Pipeline)\n")
    
    # 1. GENERATE FEATURES (If Needed)
    initialize_dataset()

    # 2. CHECK FOR EDGE FILES
    if not (os.path.exists(PATH_TRAIN_EDGES) and os.path.exists(PATH_TEST_EDGES)):
        log_to_file("CRITICAL: Edge files not found in cache_spatial.")
        log_to_file("Please run your Spatial Split and Stitching scripts now.")
        exit()


    log_to_file("Loading Data...")
    x_global = torch.load(PATH_X, weights_only=False)
    
    # --- FEATURE SELECTION: KEEP ONLY 6 ---
    # Index Mapping:
    # 0: Soma_Vol, 1: Total_Vol, 2: Total_Len
    # 3: Soma_X,   4: Soma_Y,    5: Soma_Z    <-- DROPPING THESE because it is redundant data(centroids cover the feature)
    # 6: Centr_X,  7: Centr_Y,   8: Centr_Z
    
    features_to_keep = [6, 7, 8]
    x_global = x_global[:, features_to_keep]
    
    log_to_file(f"TRAINING ON REDUCED SUBSET: {x_global.shape[1]} Features")
    # --------------------------------------    # Load edges to CPU to allow GraphSAINT sampling
    train_edges = torch.load(PATH_TRAIN_EDGES, weights_only=False).cpu()
    test_edges = torch.load(PATH_TEST_EDGES, weights_only=False).cpu()

    # --- SANITY CHECK: Ensure edge indices don't exceed feature count ---
    max_idx = max(train_edges.max(), test_edges.max())
    if max_idx >= x_global.size(0):
        log_to_file(f"CRITICAL ERROR: Max edge index ({max_idx}) >= Node count ({x_global.size(0)})")
        log_to_file("This means your 'stitching' process generated IDs that don't map to the feature matrix row indices.")
        exit()
    # ------------------------------------------------------------------
    
    log_to_file(f"Nodes: {x_global.size(0)}")
    log_to_file(f"Train Edges: {train_edges.size(1):,}")

    model = gnn.SynapsePredictor(in_channels=x_global.shape[1], hidden_channels=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    log_to_file("\nStarting Training...")
    best_val_auc = 0.0

    # Training Loop
    for epoch in range(1, 101):
        total_loss = 0
        steps = 100 
        
        for _ in range(steps):
            loss = train_step(model, x_global, train_edges, optimizer, node_sample_size=6000)
            total_loss += loss
            
        avg_loss = total_loss / steps
        
        val_aucs, val_f1s = [], []
        for _ in range(5):
            auc, f1 = validate_step(model, x_global, test_edges, node_sample_size=6000)
            val_aucs.append(auc)
            val_f1s.append(f1)
            
        avg_auc = np.mean(val_aucs)
        avg_f1 = np.mean(val_f1s)
        
        log_to_file(f"{epoch:03d}   | Loss: {avg_loss:.4f}   | Val AUC: {avg_auc:.4f}  | Val F1 (0.5): {avg_f1:.4f}")
        
        if avg_auc > best_val_auc:
            best_val_auc = avg_auc
            torch.save(model.state_dict(), os.path.join(OUTPUT_FOLDER, "best_model.pth"))

    # 3. FINAL EVALUATION (The crucial part that was missing!)
    log_to_file("\nTraining Complete. Loading best model for full evaluation...")
    
    # Load Best Model
    model.load_state_dict(torch.load(os.path.join(OUTPUT_FOLDER, "best_model.pth")))
    model.eval()
    
    log_to_file("Running comprehensive analysis on Test set (Large Sample)...")
    
    # We take a larger sample for the final report to get stable metrics
    # 15,000 nodes gives us decent coverage for the confusion matrix
    local_edge_index, node_indices = get_random_subgraph(test_edges, x_global.size(0), sample_size=15000)
    
    local_edge_index = local_edge_index.to(device)
    batch_x = x_global[node_indices].to(device)
    
    with torch.no_grad():
        z = model.encode(batch_x, local_edge_index)
        pos_src, pos_dst = local_edge_index[0], local_edge_index[1]
        num_pos = pos_src.size(0)
        
        # Negative sampling (Same size as positive)
        neg_src = torch.randint(0, 15000, (num_pos,), device=device)
        neg_dst = torch.randint(0, 15000, (num_pos,), device=device)
        
        pos_scores = (z[pos_src] * z[pos_dst]).sum(dim=1)
        neg_scores = (z[neg_src] * z[neg_dst]).sum(dim=1)
        
        # Concatenate and move to CPU
        y_scores = torch.cat([torch.sigmoid(pos_scores), torch.sigmoid(neg_scores)]).cpu().numpy()
        y_true = np.concatenate([np.ones(num_pos), np.zeros(num_pos)])

    # CALL THE ANALYSIS FUNCTION
    analyze_model_performance(y_true, y_scores)
    
    log_to_file("Done.")