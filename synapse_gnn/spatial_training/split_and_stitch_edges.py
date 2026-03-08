import numpy as np
import torch
import os
import json
import argparse

# --- CONFIGURATION LOADER ---
def parse_args():
    parser = argparse.ArgumentParser(description="Config-Driven Split and Stitch")
    parser.add_argument('--config', type=str, default="config.json", help="Path to the JSON configuration file")
    return parser.parse_args()

def load_config(config_path):
    print(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def split_and_stitch():
    args = parse_args()
    config = load_config(args.config)
    
    print(f"\n--- Starting Spatial Split & Stitch (Universal Pipeline) ---")
    
    CACHE_DIR = config["paths"]["data_dir"]
    
    # INPUTS
    SYNAPSES_FILE = os.path.join(CACHE_DIR, "synapses.json")
    PATH_INDICES   = os.path.join(CACHE_DIR, "node_mapping.json") 
    PATH_MASKS     = os.path.join(CACHE_DIR, "spatial_split_masks.pt")
    PATH_EDGES     = os.path.join(CACHE_DIR, "adp_base_edges.pt")

    # OUTPUTS
    TRAIN_POS_PATH = os.path.join(CACHE_DIR, "graph_train_edges.pt")
    TEST_POS_PATH  = os.path.join(CACHE_DIR, "graph_test_edges.pt")
    TRAIN_CANDIDATES_PATH = os.path.join(CACHE_DIR, "graph_train_spatial_candidates.pt")
    TEST_CANDIDATES_PATH  = os.path.join(CACHE_DIR, "graph_test_spatial_candidates.pt")
    
    # 1. Load ID Map
    if not os.path.exists(PATH_INDICES):
        print(f"Error: {PATH_INDICES} not found.")
        return
    
    with open(PATH_INDICES, 'r') as f:
        valid_ids_list = json.load(f)
    # Ensure IDs are strings for dictionary matching
    id_to_idx = {str(k): v for v, k in enumerate(valid_ids_list)}
    
    # 2. Load Spatial Masks
    if not os.path.exists(PATH_MASKS):
        print(f"Error: {PATH_MASKS} not found.")
        return
        
    masks = torch.load(PATH_MASKS, weights_only=False)
    train_mask = masks['train_mask']
    test_mask  = masks['test_mask']
    
    print(f"Train Neurons in Mask: {train_mask.sum():,}")
    print(f"Test Neurons in Mask:  {test_mask.sum():,}")

    # --- PART A: GROUND TRUTH (POSITIVES) ---
    print("\n--- Processing Ground Truth Synapses ---")
    with open(SYNAPSES_FILE, 'r') as f:
        syn_data = json.load(f)
        
    train_pos_edges, test_pos_edges = [], []
    
    for synid, val in syn_data.items():
        pair = val[0]
        src_str, dst_str = str(pair[0]), str(pair[1])
        
        if src_str in id_to_idx and dst_str in id_to_idx:
            s, d = id_to_idx[src_str], id_to_idx[dst_str]
            
            # Use two SEPARATE if statements, not an elif
            if train_mask[s] and train_mask[d]:
                train_pos_edges.append([s, d])
            
            if test_mask[s] and test_mask[d]:
                test_pos_edges.append([s, d])
    # Convert to Tensors and Save
    if len(train_pos_edges) > 0:
        train_pos_tensor = torch.tensor(train_pos_edges, dtype=torch.long).t().contiguous()
        torch.save(train_pos_tensor, TRAIN_POS_PATH)
        print(f"Saved Train POSITIVES: {train_pos_tensor.size(1):,} edges")
    else:
        print("CRITICAL WARNING: No training synapses found! Check cluster alignment.")

    if len(test_pos_edges) > 0:
        test_pos_tensor = torch.tensor(test_pos_edges, dtype=torch.long).t().contiguous()
        torch.save(test_pos_tensor, TEST_POS_PATH)
        print(f"Saved Test POSITIVES:  {test_pos_tensor.size(1):,} edges")

    # --- PART B: ADP CANDIDATES (HARD NEGATIVES) ---
    print("\n--- Processing Structural Candidates ---")
    if not os.path.exists(PATH_EDGES):
        print(f"Error: {PATH_EDGES} missing. Re-run adapter.")
        return

    try:
        base_edges = torch.load(PATH_EDGES, weights_only=False)
    except Exception as e:
        print(f"CRITICAL ERROR: {PATH_EDGES} is corrupted. {e}")
        return

    print(f"Filtering {base_edges.size(1):,} ADP edges...")
    
    # Train Filter
    train_edge_mask = train_mask[base_edges[0]] & train_mask[base_edges[1]]
    train_cands = base_edges[:, train_edge_mask]
    if train_cands.size(1) > 0:
        torch.save(train_cands.contiguous(), TRAIN_CANDIDATES_PATH)
        print(f"Saved Train CANDIDATES: {train_cands.size(1):,}")
        
    # Test Filter
    test_edge_mask = test_mask[base_edges[0]] & test_mask[base_edges[1]]
    test_cands = base_edges[:, test_edge_mask]
    if test_cands.size(1) > 0:
        torch.save(test_cands.contiguous(), TEST_CANDIDATES_PATH)
        print(f"Saved Test CANDIDATES:  {test_cands.size(1):,}")

    print("\n--- Split & Stitch Complete! ---")

if __name__ == "__main__":
    split_and_stitch()