import os
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data

# Import the new loader we built to fetch the master Data object
from synapse_gnn.data.loader import load_pyg_data 

def generate_spatial_masks_and_stitch(config):
    CACHE_DIR = config["paths"]["data_dir"]
    SPATIAL_OFFSET = config["graph_generation"]["spatial_threshold_nm"]
    TRAIN_SIZE = config["spatial_split"]["train_cluster_size"]
    TEST_SIZE = config["spatial_split"]["test_cluster_size"]
    
    # Determine which graph to load based on the weights toggle
    load_weights = config.get("graph_generation", {}).get("load_edge_weights", False)
    graph_filename = "adp_graph.pt" if load_weights else "euc_graph.pt"
    
    print(f"\n--- Generating {SPATIAL_OFFSET}nm Spatial Split & Stitching ---")
    
    # 1. Load the unified Master Data Object
    try:
        data = load_pyg_data(CACHE_DIR, graph_filename=graph_filename, labels_filename="synapses.pt")
    except Exception as e:
        raise RuntimeError(f"Failed to load master Data object: {e}")
        
    # Extract coordinates (Features 11, 12, 13)
    coords = data.x[:, [11, 12, 13]].numpy()
    
    # 2. Generate Spatial Masks
    # Vectorized calculation of synapse midpoints
    src_coords = coords[data.edge_label_index[0].numpy()]
    dst_coords = coords[data.edge_label_index[1].numpy()]
    syn_locations = (src_coords + dst_coords) / 2.0
    
    center_of_synapses = np.median(syn_locations, axis=0)
    train_seed = center_of_synapses.copy()
    train_seed[0] -= SPATIAL_OFFSET
    test_seed = center_of_synapses.copy()
    test_seed[0] += SPATIAL_OFFSET

    nbrs = NearestNeighbors(n_neighbors=max(TRAIN_SIZE, TEST_SIZE)).fit(coords)
    _, train_indices = nbrs.kneighbors(train_seed.reshape(1, -1))
    _, test_indices = nbrs.kneighbors(test_seed.reshape(1, -1))

    train_mask = torch.zeros(coords.shape[0], dtype=torch.bool)
    test_mask = torch.zeros(coords.shape[0], dtype=torch.bool)
    train_mask[train_indices[0][:TRAIN_SIZE]] = True
    test_mask[test_indices[0][:TEST_SIZE]] = True

    # 3. Stitch Positives (True Synapses) using Boolean Masking
    # Keep edges where BOTH the source and target nodes are in the spatial mask
    train_pos_mask = train_mask[data.edge_label_index[0]] & train_mask[data.edge_label_index[1]]
    test_pos_mask = test_mask[data.edge_label_index[0]] & test_mask[data.edge_label_index[1]]
    
    train_pos_edges = data.edge_label_index[:, train_pos_mask]
    test_pos_edges = data.edge_label_index[:, test_pos_mask]

    # 4. Stitch Candidates & Weights (Structural Input Graph)
    train_edge_mask = train_mask[data.edge_index[0]] & train_mask[data.edge_index[1]]
    test_edge_mask = test_mask[data.edge_index[0]] & test_mask[data.edge_index[1]]
    
    train_edge_index = data.edge_index[:, train_edge_mask]
    test_edge_index = data.edge_index[:, test_edge_mask]
    
    # Safely slice weights if they exist
    train_edge_attr = data.edge_attr[train_edge_mask] if data.edge_attr is not None else None
    test_edge_attr = data.edge_attr[test_edge_mask] if data.edge_attr is not None else None

    # 5. Build and Save Clean PyG Subgraph Objects
    train_data = Data(x=data.x, edge_index=train_edge_index, edge_label_index=train_pos_edges, edge_attr=train_edge_attr)
    test_data = Data(x=data.x, edge_index=test_edge_index, edge_label_index=test_pos_edges, edge_attr=test_edge_attr)
    
    torch.save(train_data, os.path.join(CACHE_DIR, "train_data.pt"))
    torch.save(test_data, os.path.join(CACHE_DIR, "test_data.pt"))
    
    print(f"  -> Train Subgraph: {train_data.edge_index.shape[1]} structural edges | {train_data.edge_label_index.shape[1]} true synapses")
    print(f"  -> Test Subgraph: {test_data.edge_index.shape[1]} structural edges | {test_data.edge_label_index.shape[1]} true synapses")
    print("--- Spatial Splitting & Stitching Complete! ---\n")