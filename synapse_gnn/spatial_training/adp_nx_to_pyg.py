import os
import json
import argparse
import pickle
import torch
import networkx as nx

# --- 1. CONFIGURATION LOADER ---
def parse_args():
    parser = argparse.ArgumentParser(description="Convert Teammate's ADP NetworkX Graph to PyTorch Tensor")
    parser.add_argument('--config', type=str, default="config.json", help="Path to the JSON configuration file")
    parser.add_argument('--nx_path', type=str, required=True, help="Path to the teammate's adp_graph.pkl")
    return parser.parse_args()

def load_config(config_path):
    print(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

# --- 2. MAIN CONVERSION LOGIC ---
def convert_adp_nx_to_pytorch():
    args = parse_args()
    config = load_config(args.config)
    
    # Extract paths from config
    CACHE_DIR = config["paths"]["data_dir"]
    PATH_INDICES = os.path.join(CACHE_DIR, "node_mapping.json")
    
    # Outputs
    OUTPUT_PT_PATH = os.path.join(CACHE_DIR, "adp_base_edges.pt")
    OUTPUT_WEIGHTS_PATH = os.path.join(CACHE_DIR, "adp_base_weights.pt")

    print(f"\n--- Converting ADP NetworkX Graph to PyTorch Tensor ---")
    
    # 1. Load the ID Map
    if not os.path.exists(PATH_INDICES):
        print(f"Error: Could not find {PATH_INDICES}.")
        return
        
    with open(PATH_INDICES, 'r') as f:
        valid_ids_list = json.load(f)
        
    # Create a mapping from Biological String ID to PyTorch Index
    id_to_idx = {str(bio_id): int(pt_idx) for pt_idx, bio_id in enumerate(valid_ids_list)}
    print(f"Loaded mapping for {len(valid_ids_list):,} neurons.")

    # 2. Load the NetworkX Graph
    print(f"Loading NetworkX Graph from {args.nx_path}...")
    if not os.path.exists(args.nx_path):
        print(f"Error: {args.nx_path} does not exist.")
        return

    with open(args.nx_path, 'rb') as f:
        nx_graph = pickle.load(f)

    print(f"Raw NetworkX Graph loaded with {nx_graph.number_of_nodes():,} nodes and {nx_graph.number_of_edges():,} edges.")

    # 3. Parse Edges and Weights
    src_list = []
    dst_list = []
    weight_list = []
    skipped_edges = 0

    print("Mapping biological IDs to PyTorch indices and extracting ADP weights...")
    
    # Extract edges with their attributes (the 'adp' weight)
    for u, v, data in nx_graph.edges(data=True):
        u_str, v_str = str(u), str(v)
        
        # Check if BOTH neurons exist in our valid volume
        if u_str in id_to_idx and v_str in id_to_idx:
            src_list.append(id_to_idx[u_str])
            dst_list.append(id_to_idx[v_str])
            
            # Extract the continuous weight (default to 1.0 if missing)
            weight = data.get('adp', 1.0)
            weight_list.append(weight)
        else:
            skipped_edges += 1

    print(f"\nData parsed successfully:")
    print(f"  -> Total Aligned Edges: {len(src_list):,}")
    print(f"  -> Ignored (Out of Volume): {skipped_edges:,}")

    # 4. Convert to PyTorch Tensors
    print("\nConverting to PyTorch Tensors...")
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_weights = torch.tensor(weight_list, dtype=torch.float32)

    # 5. Handle Directed vs Undirected (Crucial for GNN message passing)
    make_undirected = config["graph_conversion"]["make_undirected"]
    if make_undirected:
        print("Config set to 'make_undirected: true'. Duplicating edges in reverse...")
        # Concatenate reverse edges
        edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
        # Duplicate weights for the reverse edges
        edge_weights = torch.cat([edge_weights, edge_weights], dim=0)

    # 6. Save Tensors
    print(f"Saving highly compressed tensors...")
    torch.save(edge_index, OUTPUT_PT_PATH)
    torch.save(edge_weights, OUTPUT_WEIGHTS_PATH)
    
    print(f"Saved Edge Index to: {OUTPUT_PT_PATH}")
    print(f"Saved Edge Weights to: {OUTPUT_WEIGHTS_PATH}")
    print("\nDone! You can now proceed to split_and_stitch.")

if __name__ == "__main__":
    convert_adp_nx_to_pytorch()