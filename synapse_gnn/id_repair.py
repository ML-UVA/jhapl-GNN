import torch
import json
import numpy as np
import os

# CONFIGURATION
# 1. Your original ground truth file (Source of truth for ID order)
JSON_PATH = "synapses.json" 
# 2. The indices you saved during preprocessing
VALID_INDICES_PATH = "cache_data/valid_indices.pt" 
# 3. Where to save the FIXED mapping
OUTPUT_MAPPING_PATH = "cache_data/id_mapping.pt"

def repair_ids():
    print("--- Starting ID Repair ---")
    
    if not os.path.exists(JSON_PATH):
        print(f"Error: Could not find {JSON_PATH}")
        return

    # 1. Load Original Data to get the Master List of IDs
    print(f"Loading {JSON_PATH}...")
    with open(JSON_PATH, "r") as f:
        true_synapses = json.load(f)
    
    # Reconstruct the exact sorted list used in preprocessing
    edges = []
    for syn_id, content in true_synapses.items():
        connection_pair = content[0] 
        if -1 not in connection_pair: 
            edges.append(connection_pair)
    
    edges_np = np.array(edges)
    # np.unique sorts the IDs. This matches how preprocessing assigned indices 0, 1, 2...
    all_neuron_ids = np.unique(edges_np) 
    print(f"Total Original IDs: {len(all_neuron_ids)}")

    # 2. Load the 'valid_indices' (The small integers you saw)
    if not os.path.exists(VALID_INDICES_PATH):
        print(f"Error: {VALID_INDICES_PATH} not found.")
        return

    print(f"Loading {VALID_INDICES_PATH}...")
    valid_indices = torch.load(VALID_INDICES_PATH, weights_only=False)
    
    # Handle tensor vs list
    if torch.is_tensor(valid_indices):
        valid_indices = valid_indices.tolist()
        
    print(f"Valid Indices Count: {len(valid_indices)}")
    
    # 3. Create the Critical Map
    # Logic: If valid_indices[k] == 5, it refers to all_neuron_ids[5]
    # We map: all_neuron_ids[5] (Real ID) -> k (Row in Feature Matrix)
    
    real_id_to_feature_row = {}
    
    print("Building ID Map...")
    for row_idx, original_list_idx in enumerate(valid_indices):
        real_id = all_neuron_ids[original_list_idx]
        
        # Ensure it's the same type as the edge list (uint64)
        real_id_int = int(real_id) 
        
        real_id_to_feature_row[real_id_int] = row_idx
        
    print(f"Mapped {len(real_id_to_feature_row)} neurons.")
    
    # 4. Save
    print(f"Saving mapping to {OUTPUT_MAPPING_PATH}...")
    torch.save(real_id_to_feature_row, OUTPUT_MAPPING_PATH)
    print("Done. You can now run the Stitcher.")

if __name__ == "__main__":
    repair_ids()