import os
import argparse
import json
import torch
import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Demo Euclidean Graph")
    parser.add_argument('--config', type=str, default="config.json")
    return parser.parse_args()

def main(config_path = None):
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
    OUTPUT_NX_PATH = os.path.join(CACHE_DIR, config["paths"]["input_nx_graph"])

    print(f"\n--- Building Demo Euclidean NetworkX Graph (< {SPATIAL_THRESHOLD}nm) ---")
    
    x = torch.load(PATH_X, weights_only=False)
    num_nodes = x.size(0)
    print(f"Loaded {num_nodes} demo neurons.")

    # Spatial Coordinates (X, Y, Z are at 8, 9, 10 in your unnormalized tensor)
    coords = x[:, 8:11].numpy()

    print("Building KD-Tree for fast distance calculation...")
    tree = cKDTree(coords)
    
    print(f"Finding all neuron pairs within {SPATIAL_THRESHOLD}nm...")
    close_pairs = tree.query_pairs(r=SPATIAL_THRESHOLD)

    print("Constructing NetworkX graph...")
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(close_pairs)
    
    print(f"Graph created successfully:")
    print(f"  -> Nodes: {G.number_of_nodes():,}")
    print(f"  -> Edges: {G.number_of_edges():,}")

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(OUTPUT_NX_PATH, 'wb') as f:
        pickle.dump(G, f)
        
    print(f"\nSaved demo NetworkX graph to: {OUTPUT_NX_PATH}")

if __name__ == "__main__":
    main()