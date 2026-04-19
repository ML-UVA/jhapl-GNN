import os
import argparse
import json
import torch
import numpy as np
from scipy.spatial import cKDTree

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Demo Euclidean Graph")
    parser.add_argument('--config', type=str, default="synapse_gnn/config.json")
    return parser.parse_args()

def main(config_path=None):
    # If called from the command line, grab it from argparse
    if config_path is None:
        args = parse_args()
        config_path = args.config
        
    # Load the config using the path
    print(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)     
   
    CACHE_DIR = config["paths"]["data_dir"]
    SPATIAL_THRESHOLD = 100000 # 100 microns
    
    PATH_X = os.path.join(CACHE_DIR, "x_features.pt")
    PATH_MAPPING = os.path.join(CACHE_DIR, "node_mapping.json")
    
    # Force output to .pt
    OUTPUT_PT_PATH = os.path.join(CACHE_DIR, "euc_graph.pt")

    print(f"\n--- Building Demo Euclidean PyTorch Graph (< {SPATIAL_THRESHOLD}nm) ---")
    
    # 1. Load Node Features and Coordinates
    x = torch.load(PATH_X, weights_only=False)
    num_nodes = x.size(0)
    print(f"Loaded {num_nodes} demo neurons.")

    # Spatial Coordinates (X, Y, Z are at 8, 9, 10 in your unnormalized tensor)
    coords = x[:, 8:11].numpy()
    
    # 2. Load the Node ID Map (to pack into the .pt file)
    with open(PATH_MAPPING, 'r') as f:
        node_ids = json.load(f)
        
    if len(node_ids) != num_nodes:
        raise ValueError("CRITICAL: The number of features does not match the node ID map!")

    # 3. Spatial Queries
    print("Building KD-Tree for fast distance calculation...")
    tree = cKDTree(coords)
    
    print(f"Finding all neuron pairs within {SPATIAL_THRESHOLD}nm...")
    close_pairs = tree.query_pairs(r=SPATIAL_THRESHOLD)

    # 4. Construct PyTorch Tensors natively (No NetworkX needed!)
    print("Constructing PyTorch edge_index tensor...")
    sources = []
    targets = []
    
    for u, v in close_pairs:
        # Euclidean distances are undirected, so we add both directions
        sources.extend([u, v])
        targets.extend([v, u])
        
    # Create the [2, num_edges] PyTorch Tensor
    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    
    print(f"Graph created successfully:")
    print(f"  -> Nodes: {num_nodes:,}")
    print(f"  -> Edges: {edge_index.shape[1]:,} (Directed)")

    # 5. Save the Standardized PyG Dictionary format
    graph_dict = {
        'edge_index': edge_index,
        'node_ids': node_ids  # Passes the string mapping to the loader
    }

    os.makedirs(CACHE_DIR, exist_ok=True)
    torch.save(graph_dict, OUTPUT_PT_PATH)
        
    print(f"\nSaved PyTorch demo graph to: {OUTPUT_PT_PATH}")

if __name__ == "__main__":
    main()