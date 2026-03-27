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
    
    # DYNAMIC GRAPH PATHING
    graph_type = os.path.splitext(config["paths"]["input_nx_graph"])[0]
    if "adp" in graph_type.lower():
        PATH_EDGES = os.path.join(CACHE_DIR, "adp_base_edges.pt")
        PATH_WEIGHTS = os.path.join(CACHE_DIR, "adp_base_weights.pt") # NEW
    else:
        PATH_EDGES = os.path.join(CACHE_DIR, "base_edges.pt")
        PATH_WEIGHTS = None # Euclidean doesn't use weights
        
    # OUTPUTS
    TRAIN_POS_PATH = os.path.join(CACHE_DIR, "graph_train_edges.pt")
    TEST_POS_PATH  = os.path.join(CACHE_DIR, "graph_test_edges.pt")
    TRAIN_CANDIDATES_PATH = os.path.join(CACHE_DIR, "graph_train_spatial_candidates.pt")
    TEST_CANDIDATES_PATH  = os.path.join(CACHE_DIR, "graph_test_spatial_candidates.pt")
    
    # NEW WEIGHT OUTPUTS
    TRAIN_WEIGHTS_PATH = os.path.join(CACHE_DIR, "graph_train_spatial_weights.pt")
    TEST_WEIGHTS_PATH  = os.path.join(CACHE_DIR, "graph_test_spatial_weights.pt")
    
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
            
            if train_mask[s] and train_mask[d]:
                train_pos_edges.append([s, d])
            
            if test_mask[s] and test_mask[d]:
                test_pos_edges.append([s, d])
                
    # Convert to Tensors and Save
    if len(train_pos_edges) > 0:
        train_pos_tensor = torch.tensor(train_pos_edges, dtype=torch.long).t().contiguous()
        torch.save(train_pos_tensor, TRAIN_POS_PATH)
        print(f"Saved Train POSITIVES: {train_pos_tensor.size(1):,} edges")

    if len(test_pos_edges) > 0:
        test_pos_tensor = torch.tensor(test_pos_edges, dtype=torch.long).t().contiguous()
        torch.save(test_pos_tensor, TEST_POS_PATH)
        print(f"Saved Test POSITIVES:  {test_pos_tensor.size(1):,} edges")

    # --- PART B: STRUCTURAL CANDIDATES & WEIGHTS ---
    print("\n--- Processing Structural Candidates ---")
    if not os.path.exists(PATH_EDGES):
        print(f"Error: {PATH_EDGES} missing. Re-run adapter.")
        return

    base_edges = torch.load(PATH_EDGES, weights_only=False)
    print(f"Loaded {base_edges.size(1):,} structural edges...")
    
    # Load weights if they exist (For ADP Version 2)
    has_weights = PATH_WEIGHTS and os.path.exists(PATH_WEIGHTS)
    if has_weights:
        base_weights = torch.load(PATH_WEIGHTS, weights_only=False)
        print(f"Loaded {base_weights.size(0):,} continuous edge weights...")

    # Train Filter
    train_edge_mask = train_mask[base_edges[0]] & train_mask[base_edges[1]]
    train_cands = base_edges[:, train_edge_mask]
    if train_cands.size(1) > 0:
        torch.save(train_cands.contiguous(), TRAIN_CANDIDATES_PATH)
        print(f"Saved Train CANDIDATES: {train_cands.size(1):,}")
        
        if has_weights:
            train_weights = base_weights[train_edge_mask]
            torch.save(train_weights.contiguous(), TRAIN_WEIGHTS_PATH)
            print(f"Saved Train WEIGHTS:    {train_weights.size(0):,}")
        
    # Test Filter
    test_edge_mask = test_mask[base_edges[0]] & test_mask[base_edges[1]]
    test_cands = base_edges[:, test_edge_mask]
    if test_cands.size(1) > 0:
        torch.save(test_cands.contiguous(), TEST_CANDIDATES_PATH)
        print(f"Saved Test CANDIDATES:  {test_cands.size(1):,}")
        
        if has_weights:
            test_weights = base_weights[test_edge_mask]
            torch.save(test_weights.contiguous(), TEST_WEIGHTS_PATH)
            print(f"Saved Test WEIGHTS:     {test_weights.size(0):,}")

    print("\n--- Split & Stitch Complete! ---")

if __name__ == "__main__":
    split_and_stitch()