import os
import json
import torch
from torch_geometric.data import Data

def load_pyg_data(cache_dir, graph_filename="euc_graph.pt", labels_filename="synapses.pt"):
    """
    Loads node features, structural edges, and ground truth labels from .pt files
    and securely maps them into a single PyTorch Geometric Data object.
    """
    features_path = os.path.join(cache_dir, "x_features.pt")
    mapping_path = os.path.join(cache_dir, "node_mapping.json")
    graph_path = os.path.join(cache_dir, graph_filename)
    labels_path = os.path.join(cache_dir, labels_filename)

    # ---------------------------------------------------------
    # 1. Load Node Features and ID Mapping
    # ---------------------------------------------------------
    try:
        x_features = torch.load(features_path)
        with open(mapping_path, 'r') as f:
            feature_nodes = json.load(f)
            
        # Create a fast lookup: string ID -> PyTorch feature row index
        feat_id_to_idx = {node_id: idx for idx, node_id in enumerate(feature_nodes)}
        print(f"Loaded {len(feature_nodes)} features from {features_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load node features or mapping: {e}")

    # ---------------------------------------------------------
    # 2. Define Tensor Translation Function
    # ---------------------------------------------------------
    def translate_edges(raw_dict, dict_name):
        """Translates local graph indices to global feature row indices."""
        if 'edge_index' not in raw_dict or 'node_ids' not in raw_dict:
            raise ValueError(f"Invalid format in {dict_name}: missing 'edge_index' or 'node_ids'.")
            
        raw_edge_index = raw_dict['edge_index']
        graph_nodes = raw_dict['node_ids']
        
        # Build translation tensor: graph_idx -> feature_idx
        translation_map = []
        for g_idx, node_id in enumerate(graph_nodes):
            translation_map.append(feat_id_to_idx.get(node_id, -1)) # -1 if node missing in features
            
        translation_tensor = torch.tensor(translation_map, dtype=torch.long)
        
        # Apply mapping
        mapped_edge_index = translation_tensor[raw_edge_index]
        
        # Filter out edges where a node is missing from the feature matrix
        valid_mask = (mapped_edge_index[0] != -1) & (mapped_edge_index[1] != -1)
        final_edge_index = mapped_edge_index[:, valid_mask]
        
        final_edge_attr = None
        if 'edge_attr' in raw_dict:
            # Filter attributes to match the valid edges
            final_edge_attr = raw_dict['edge_attr'][valid_mask]
            
        dropped = raw_edge_index.shape[1] - final_edge_index.shape[1]
        if dropped > 0:
            print(f"  [Warning] Dropped {dropped} edges in {dict_name} due to missing node features.")
            
        return final_edge_index, final_edge_attr

    # ---------------------------------------------------------
    # 3. Load Structural Input Graph (Euclidean or ADP)
    # ---------------------------------------------------------
    try:
        # Backward compatibility check
        if graph_path.endswith('.gpickle') or graph_path.endswith('.pkl'):
            raise ValueError(f"Legacy file format detected ({graph_path}). Pipeline now requires .pt files.")
            
        graph_dict = torch.load(graph_path)
        print(f"Processing input graph: {graph_path}")
        edge_index, edge_attr = translate_edges(graph_dict, graph_filename)
    except FileNotFoundError:
        raise FileNotFoundError(f"Missing input graph file at {graph_path}")
    except Exception as e:
        raise RuntimeError(f"Error parsing graph tensors: {e}")

    # ---------------------------------------------------------
    # 4. Load Ground Truth Labels (Synapses)
    # ---------------------------------------------------------
    try:
        labels_dict = torch.load(labels_path)
        print(f"Processing ground truth labels: {labels_path}")
        edge_label_index, _ = translate_edges(labels_dict, labels_filename)
    except FileNotFoundError:
        raise FileNotFoundError(f"Missing ground truth file at {labels_path}")

    # ---------------------------------------------------------
    # 5. Construct Final PyG Data Object
    # ---------------------------------------------------------
    data = Data(x=x_features, edge_index=edge_index)
    
    # Store labels under a different key to prevent model data leakage
    data.edge_label_index = edge_label_index
    
    # Add weights if dealing with ADP graph
    if edge_attr is not None:
        data.edge_attr = edge_attr
        
    print(f"\nFinal PyG Data Object Built:")
    print(f"  -> Nodes (Features): {data.x.shape}")
    print(f"  -> Edges (Structure): {data.edge_index.shape}")
    print(f"  -> Labels (Ground Truth): {data.edge_label_index.shape}")
    
    return data