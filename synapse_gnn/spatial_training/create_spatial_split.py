import os
import argparse
import json
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

# --- CONFIGURATION LOADER ---
def parse_args():
    parser = argparse.ArgumentParser(description="Config-Driven Spatial Splitter")
    parser.add_argument('--config', type=str, default="config.json", help="Path to the JSON configuration file")
    return parser.parse_args()

def load_config(config_path):
    print(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

# Centroid X, Y, Z are at indices 11, 12, 13
COORD_COLS = [11, 12, 13] 

def generate_spatial_split():
    args = parse_args()
    config = load_config(args.config)
    
    CACHE_DIR = config["paths"]["data_dir"]
    TRAIN_SIZE = config["spatial_split"]["train_cluster_size"]
    TEST_SIZE = config["spatial_split"]["test_cluster_size"]
    
    # 1. Load Data
    x_features = torch.load(os.path.join(CACHE_DIR, "x_features.pt"), weights_only=False)
    coords = x_features[:, COORD_COLS].numpy()
    
    with open(os.path.join(CACHE_DIR, "node_mapping.json"), 'r') as f:
        node_map = json.load(f)
    # Map Bio ID -> Index in x_features
    id_to_idx = {str(bio_id): i for i, bio_id in enumerate(node_map)}

    with open(os.path.join(CACHE_DIR, "synapses.json"), 'r') as f:
        syn_data = json.load(f)

    # 2. Extract Coordinates by looking up Neuron Positions
    syn_locations = []
    print("Mapping synapse IDs to physical coordinates...")
    
    for val in syn_data.values():
        # Get the two neuron IDs: [86469113..., 86469113...]
        pair = val[0]
        src_id, dst_id = str(pair[0]), str(pair[1])
        
        # If both neurons exist in our volume, use their midpoint as the synapse location
        if src_id in id_to_idx and dst_id in id_to_idx:
            src_pos = coords[id_to_idx[src_id]]
            dst_pos = coords[id_to_idx[dst_id]]
            midpoint = (src_pos + dst_pos) / 2
            syn_locations.append(midpoint)

    syn_locations = np.array(syn_locations)
    if len(syn_locations) == 0:
        print("CRITICAL ERROR: No synapses matched the neurons in your x_features!")
        return

    # 3. Calculate "Hot Zone" Seeds
    center_of_synapses = np.median(syn_locations, axis=0)
    print(f"Synapse Center of Mass (Midpoints): {center_of_synapses}")

    # 4. Grow Clusters (The rest of the logic remains the same)
    train_seed = center_of_synapses.copy()
    train_seed[0] -= 10000  # Shift 10um Left
    test_seed = center_of_synapses.copy()
    test_seed[0] += 10000   # Shift 10um Right

    nbrs = NearestNeighbors(n_neighbors=max(TRAIN_SIZE, TEST_SIZE)).fit(coords)
    _, train_indices = nbrs.kneighbors(train_seed.reshape(1, -1))
    _, test_indices = nbrs.kneighbors(test_seed.reshape(1, -1))

    train_mask = torch.zeros(coords.shape[0], dtype=torch.bool)
    test_mask = torch.zeros(coords.shape[0], dtype=torch.bool)
    train_mask[train_indices[0][:TRAIN_SIZE]] = True
    test_mask[test_indices[0][:TEST_SIZE]] = True

    # Save and Validate
    torch.save({'train_mask': train_mask, 'test_mask': test_mask}, 
               os.path.join(CACHE_DIR, "spatial_split_masks.pt"))
    print(f"Success! Found {len(syn_locations)} synapses to center the split.")


    
if __name__ == "__main__":
    generate_spatial_split()