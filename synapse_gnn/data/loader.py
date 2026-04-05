import os
import torch

from synapse_gnn.data_prep.extract_nx_edges import extract_base_tensors
from synapse_gnn.data.spatial_split import generate_spatial_masks_and_stitch
def load_graph_data(config):
    data_dir = config["paths"]["data_dir"]
    data_dir = config["paths"]["data_dir"]
    
    # --- AUTO-TRIGGER LOGIC ---
    # 1. Check if the final PyG tensors exist
    check_file = os.path.join(data_dir, "graph_train_spatial_candidates.pt")
    
    if not os.path.exists(check_file):
        print(f"Detected missing dataset files for {config['paths']['input_nx_graph']}. Rebuilding...")
        
        # 2. Check if we need to extract the base edges from the gpickle
        base_edge_file = os.path.join(data_dir, "base_edges.pt")
        if not os.path.exists(base_edge_file):
            extract_base_tensors(config)
            
        # 3. Generate the spatial split and stitch the final graph
        generate_spatial_masks_and_stitch(config)
    # ------------------------------
    print(f"Loading standard graph tensors from: {data_dir}")
    
    paths = {
        "x": os.path.join(data_dir, "x_features.pt"),
        "train_pos": os.path.join(data_dir, "graph_train_edges.pt"),
        "test_pos": os.path.join(data_dir, "graph_test_edges.pt"),
        "train_cands": os.path.join(data_dir, "graph_train_spatial_candidates.pt"),
        "test_cands": os.path.join(data_dir, "graph_test_spatial_candidates.pt")
    }
    
    data = {k: torch.load(v, weights_only=False).cpu() for k, v in paths.items()}
    
    # Dynamically check for continuous weights (Replaces hardcoded ADP check)
    path_train_weights = os.path.join(data_dir, "graph_train_spatial_weights.pt")
    path_test_weights = os.path.join(data_dir, "graph_test_spatial_weights.pt")
    
    if os.path.exists(path_train_weights) and os.path.exists(path_test_weights):
        print(" -> Detected continuous edge weights. Normalizing...")
        train_weights_raw = torch.load(path_train_weights, weights_only=False).cpu()
        test_weights_raw = torch.load(path_test_weights, weights_only=False).cpu()
        
        max_weight = train_weights_raw.max() 
        train_weights = train_weights_raw / max_weight
        test_weights = test_weights_raw / max_weight
    else:
        print(" -> No continuous weights detected. Defaulting to unweighted message passing.")
        train_weights = None
        test_weights = None

    return {
        "x_raw": data["x"],
        "train_edges": data["train_pos"],
        "test_edges": data["test_pos"],
        "train_cands": data["train_cands"],
        "test_cands": data["test_cands"],
        "train_weights": train_weights,
        "test_weights": test_weights
    }