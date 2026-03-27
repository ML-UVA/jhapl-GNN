import os
import argparse
import json
import torch
import networkx as nx
import pickle

# --- CONFIGURATION LOADER ---
def parse_args():
    parser = argparse.ArgumentParser(description="Config-Driven NetworkX to PyG Adapter")
    parser.add_argument('--config', type=str, default="config.json", help="Path to the JSON configuration file")
    return parser.parse_args()

def load_config(config_path):
    print(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def convert_nx_to_pyg():
    args = parse_args()
    config = load_config(args.config)
    
    print(f"\n--- Converting NetworkX Graph to PyTorch Tensor ---")
    
    # Construct paths dynamically from the config
    data_dir = config["paths"]["data_dir"]
    nx_filename = config["paths"]["input_nx_graph"]
    
    input_nx_path = os.path.join(data_dir, nx_filename)
    output_edges_path = os.path.join(data_dir, "base_edges.pt")
    
    make_undirected = config["graph_conversion"]["make_undirected"]

    if not os.path.exists(input_nx_path):
        print(f"Error: Could not find {input_nx_path}")
        return
        
    # 1. Load the NetworkX object
    print(f"Loading {input_nx_path}...")
    with open(input_nx_path, 'rb') as f:
        nx_graph = pickle.load(f)
        
    print(f"Loaded graph with {nx_graph.number_of_nodes():,} nodes and {nx_graph.number_of_edges():,} edges.")

    # 2. Extract Edges
    edges = list(nx_graph.edges())
    if len(edges) == 0:
        print("Error: The NetworkX graph has no edges!")
        return

    # 3. Convert to PyTorch Tensor format [2, num_edges]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    # 4. Handle Directed vs Undirected mapping based on JSON
    if make_undirected:
        print("Config set to 'make_undirected: true'. Duplicating edges in reverse (A->B and B->A)...")
        edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
    else:
        print("Config set to 'make_undirected: false'. Keeping edges strictly directed.")
        
    print(f"Successfully converted to PyTorch tensor. Final Edge Shape: {edge_index.shape}")

    # 5. Save the base edge index
    os.makedirs(data_dir, exist_ok=True)
    torch.save(edge_index, output_edges_path)
    
    print(f"Saved base PyTorch edges to: {output_edges_path}")
    print("\nNext Step: Run split_and_stitch.py!")

if __name__ == "__main__":
    convert_nx_to_pyg()