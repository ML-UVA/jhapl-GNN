import os
import argparse
import json
import torch
import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
import pickle

# --- CONFIGURATION LOADER ---
def parse_args():
    parser = argparse.ArgumentParser(description="Config-Driven Euclidean Graph Generator")
    parser.add_argument('--config', type=str, default="config.json", help="Path to the JSON configuration file")
    return parser.parse_args()

def load_config(config_path):
    print(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def build_euclidean_nx_graph():
    args = parse_args()
    config = load_config(args.config)
    
    CACHE_DIR = config["paths"]["data_dir"]
    SPATIAL_THRESHOLD = 100000
    
    PATH_X = os.path.join(CACHE_DIR, "x_features.pt")
    # Dynamically name the output based on the config to ensure it matches
    OUTPUT_NX_PATH = os.path.join(CACHE_DIR, config["paths"]["input_nx_graph"])

    print(f"\n--- Building Euclidean NetworkX Graph (< {SPATIAL_THRESHOLD}nm) ---")
    
    # 1. Load the biological features and coordinates
    if not os.path.exists(PATH_X):
        print(f"Error: Could not find {PATH_X}. Run step_0_extract_features.py first.")
        return
        
    x = torch.load(PATH_X, weights_only=False)
    num_nodes = x.size(0)
    print(f"Loaded {num_nodes} neurons.")

    # 2. Extract Spatial Coordinates
    # Soma X, Y, Z are at indices 8, 9, 10
    coords = x[:, 8:11].numpy()

    # 3. Fast Spatial Search using a KD-Tree
    print("Building KD-Tree for fast distance calculation...")
    tree = cKDTree(coords)
    
    print(f"Finding all neuron pairs within {SPATIAL_THRESHOLD}nm...")
    close_pairs = tree.query_pairs(r=SPATIAL_THRESHOLD)

    # 4. Construct the NetworkX Object
    print("Constructing NetworkX graph...")
    G = nx.Graph()
    
    # CRITICAL: Add all nodes first to lock in the PyTorch ID mapping!
    G.add_nodes_from(range(num_nodes))
    
    # Add the edges we found
    G.add_edges_from(close_pairs)
    
    print(f"Graph created successfully:")
    print(f"  -> Nodes: {G.number_of_nodes():,}")
    print(f"  -> Edges: {G.number_of_edges():,}")

    # 5. Save the NetworkX object
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(OUTPUT_NX_PATH, 'wb') as f:
        pickle.dump(G, f)
        
    print(f"\nSaved NetworkX graph to: {OUTPUT_NX_PATH}")

if __name__ == "__main__":
    build_euclidean_nx_graph()