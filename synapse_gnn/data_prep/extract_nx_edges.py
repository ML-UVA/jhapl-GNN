import os
import json
import torch
import pickle

import os
import json
import torch
import pickle

def extract_base_tensors(config):
    graph_filename = config["paths"]["input_nx_graph"]
    CACHE_DIR = config["paths"]["data_dir"]
    load_weights = config.get("graph_generation", {}).get("load_edge_weights", False)
    print(f"\n--- Extracting Base Edges from {graph_filename} ---")
    
    # 1. Load Node Mapping
    with open(os.path.join(CACHE_DIR, "node_mapping.json"), 'r') as f:
        id_to_idx = {str(k): v for v, k in enumerate(json.load(f))}
        
    # 2. Load NetworkX Graph
    graph_path = os.path.join(CACHE_DIR, graph_filename)
    with open(graph_path, 'rb') as f:
        G = pickle.load(f)
        
    # --- NEW: AUTO-DETECT GRAPH FORMAT ---
    first_node = list(G.nodes())[0]
    is_integer_indexed = isinstance(first_node, int)
    
    # Allows pipeline to handle both ADP graph(string indexed) and integer based euclidean graphs
    # without breaking
    if is_integer_indexed:
        print(" -> Detected Integer-Indexed Graph (Direct PyTorch Mapping)")
    else:
        print(" -> Detected String-Indexed Graph (Biological ID Mapping)")

    # 3. Extract to Lists
    base_edges, base_weights = [], []
    for u, v, data in G.edges(data=True):
        if is_integer_indexed:
            # If nodes are already integers, they are the PyTorch indices!
            u_idx, v_idx = u, v
        else:
            # If they are strings, we map them
            u_str, v_str = str(u), str(v)
            if u_str not in id_to_idx or v_str not in id_to_idx:
                continue
            u_idx, v_idx = id_to_idx[u_str], id_to_idx[v_str]
            
        base_edges.extend([[u_idx, v_idx], [v_idx, u_idx]]) # Bidirectional
        if load_weights:
            w = data.get('adp', 1.0)
            base_weights.extend([w, w])
                
    # 4. Convert and Save
    if len(base_edges) == 0:
        print("\n[CRITICAL ERROR] 0 edges were extracted!")
        return

    base_edges_tensor = torch.tensor(base_edges, dtype=torch.long).t().contiguous()
    torch.save(base_edges_tensor, os.path.join(CACHE_DIR, "base_edges.pt"))
    print(f"Saved {base_edges_tensor.size(1):,} structural edges.")

    if load_weights: 
        base_weights_tensor = torch.tensor(base_weights, dtype=torch.float)
        torch.save(base_weights_tensor, os.path.join(CACHE_DIR, "base_weights.pt"))
        print(f"Saved {base_weights_tensor.size(0):,} continuous weights.")