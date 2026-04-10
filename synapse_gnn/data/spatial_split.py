import os
import json
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

def generate_spatial_masks_and_stitch(config):
    graph_filename = config["paths"]["input_nx_graph"]
    CACHE_DIR = config["paths"]["data_dir"]
    load_weights = config.get("graph_generation", {}).get("load_edge_weights", False)
    
    SPATIAL_OFFSET = config["graph_generation"]["spatial_threshold_nm"]
    TRAIN_SIZE = config["spatial_split"]["train_cluster_size"]
    TEST_SIZE = config["spatial_split"]["test_cluster_size"]
    
    print(f"\n--- Generating {SPATIAL_OFFSET}nm Spatial Split & Stitching ---")
    
    # 1. Load Prerequisites (including the base tensors we just extracted)
    with open(os.path.join(CACHE_DIR, "node_mapping.json"), 'r') as f:
        id_to_idx = {str(k): v for v, k in enumerate(json.load(f))}
    with open(os.path.join(CACHE_DIR, "synapses.json"), 'r') as f:
        syn_data = json.load(f)
        
    x_features = torch.load(os.path.join(CACHE_DIR, "x_features.pt"), weights_only=False)
    coords = x_features[:, [11, 12, 13]].numpy() # XYZ coordinates
    
    base_edges = torch.load(os.path.join(CACHE_DIR, "base_edges.pt"), weights_only=False)
    if load_weights:
        base_weights = torch.load(os.path.join(CACHE_DIR, "base_weights.pt"), weights_only=False)

    # 2. Generate Spatial Masks
    syn_locations = []
    for val in syn_data.values():
        src_id, dst_id = str(val[0][0]), str(val[0][1])
        if src_id in id_to_idx and dst_id in id_to_idx:
            midpoint = (coords[id_to_idx[src_id]] + coords[id_to_idx[dst_id]]) / 2
            syn_locations.append(midpoint)

    center_of_synapses = np.median(np.array(syn_locations), axis=0)
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

    # 3. Stitch Positives (True Synapses)
    train_pos_edges, test_pos_edges = [], []
    for val in syn_data.values():
        src_id, dst_id = str(val[0][0]), str(val[0][1])
        if src_id in id_to_idx and dst_id in id_to_idx:
            s, d = id_to_idx[src_id], id_to_idx[dst_id]
            if train_mask[s] and train_mask[d]: train_pos_edges.append([s, d])
            if test_mask[s] and test_mask[d]: test_pos_edges.append([s, d])

    if train_pos_edges:
        torch.save(torch.tensor(train_pos_edges, dtype=torch.long).t().contiguous(), os.path.join(CACHE_DIR, "graph_train_edges.pt"))
    if test_pos_edges:
        torch.save(torch.tensor(test_pos_edges, dtype=torch.long).t().contiguous(), os.path.join(CACHE_DIR, "graph_test_edges.pt"))

    # 4. Stitch Candidates & Weights (Filtering base_edges with the masks)
    train_edge_mask = train_mask[base_edges[0]] & train_mask[base_edges[1]]
    test_edge_mask = test_mask[base_edges[0]] & test_mask[base_edges[1]]

    torch.save(base_edges[:, train_edge_mask].contiguous(), os.path.join(CACHE_DIR, "graph_train_spatial_candidates.pt"))
    torch.save(base_edges[:, test_edge_mask].contiguous(), os.path.join(CACHE_DIR, "graph_test_spatial_candidates.pt"))

    if load_weights:
        torch.save(base_weights[train_edge_mask].contiguous(), os.path.join(CACHE_DIR, "graph_train_spatial_weights.pt"))
        torch.save(base_weights[test_edge_mask].contiguous(), os.path.join(CACHE_DIR, "graph_test_spatial_weights.pt"))
    else:
        # Purge stale ADP weights if we switched back to Euclidean
        for f in ["graph_train_spatial_weights.pt", "graph_test_spatial_weights.pt"]:
            path = os.path.join(CACHE_DIR, f)
            if os.path.exists(path): os.remove(path)
            
    print("--- Spatial Splitting & Stitching Complete! ---\n")