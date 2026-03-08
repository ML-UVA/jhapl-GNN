import torch
import numpy as np
import os
import concurrent.futures
from tqdm import tqdm
from datasci_tools import system_utils as su
from neuron_morphology_tools import neuron_nx_utils as nxu
from neuron_morphology_tools import neuron_nx_stats as nxs


# ---------------------------------------------------------
# HELPER: Get IDs from folder
# ---------------------------------------------------------
def get_neuron_ids_from_folder(graph_dir):
    """Scans the folder and returns a list of neuron IDs based on filenames."""
    if not os.path.exists(graph_dir):
        return []
    
    files = [f for f in os.listdir(graph_dir) if f.endswith(".pbz2")]
    # Assuming filename format: "864691134884749562_0_auto_proof_v7_proofread.pbz2"
    ids = [f.split("_")[0] for f in files]
    return sorted(list(set(ids)))

# ---------------------------------------------------------
# HELPER: Soma Detection
# ---------------------------------------------------------
def get_valid_soma_id(G):
    if "S0" in G.nodes(): return "S0"
    for n, data in G.nodes(data=True):
        if data.get("compartment") == "soma": return n
    return None

# ---------------------------------------------------------
# WORKER: Process One Neuron (Updated for Macro Features)
# ---------------------------------------------------------
def process_single_neuron(args):
    i, n_id, graph_dir = args
    filename = f"{n_id}_0_auto_proof_v7_proofread.pbz2" 
    path = os.path.join(graph_dir, filename)
    
    if not os.path.exists(path): return None

    try:
        G = su.decompress_pickle(path)
        soma_id = get_valid_soma_id(G)
        if soma_id is None: return None
        
        # 1. Base Soma Features
        soma_data = G.nodes[soma_id]
        soma_center = soma_data.get('mesh_center', [0,0,0])
        soma_vol = soma_data.get('mesh_volume', 0.0)
        
        # 2. Initialize Macro Feature Aggregators
        axon_len = 0.0
        apical_len = 0.0
        basal_len = 0.0
        
        max_axon_reach = 0.0
        max_dendrite_reach = 0.0
        
        total_spines = 0
        total_spine_volume = 0.0
        
        # 3. Loop and Aggregate Biologically
        for node, data in G.nodes(data=True):
            # Skip the soma node for these aggregations
            if node == soma_id: continue
                
            comp = str(data.get('compartment', '')).lower()
            length = data.get('skeletal_length', 0.0)
            
            # Using euclidean distance from soma for reach
            dist_from_soma = data.get('soma_distance_euclidean', 0.0) 
            
            # Aggregate Lengths & Reach by Compartment
            if 'axon' in comp:
                axon_len += length
                if dist_from_soma > max_axon_reach:
                    max_axon_reach = dist_from_soma
                    
            elif 'apical' in comp:
                apical_len += length
                if dist_from_soma > max_dendrite_reach:
                    max_dendrite_reach = dist_from_soma
                    
            elif 'basal' in comp or 'dendrite' in comp:
                basal_len += length
                if dist_from_soma > max_dendrite_reach:
                    max_dendrite_reach = dist_from_soma
            
            # Aggregate Structural Capacity (Spines)
            total_spines += data.get('n_spines', 0)
            
            # Sum up spine volume if the data is available
            spine_data = data.get('spine_data', [])
            if spine_data:
                for spine in spine_data:
                    total_spine_volume += spine.get('volume', 0.0)

        # 4. Centroid / Spatial Metadata (Kept for distance checking, NOT for model input!)
        try:
            from neuron_morphology_tools import neuron_nx_utils as nxu
            all_skeleton_verts = nxu.skeleton_nodes(G)
            if len(all_skeleton_verts) > 0:
                centroid = np.mean(all_skeleton_verts, axis=0)
            else:
                centroid = soma_center
        except:
            centroid = soma_center

        # 5. Compile the New Feature Vector (14 Features total)
        feats = [
            # --- MODEL INPUTS (Indices 0 to 7) ---
            soma_vol,             # 0: Size of cell body
            axon_len,             # 1: Extent of output cables
            basal_len,            # 2: Extent of local input cables
            apical_len,           # 3: Extent of distant input cables
            max_axon_reach,       # 4: How far the cell projects
            max_dendrite_reach,   # 5: How wide the cell listens
            total_spines,         # 6: Receptiveness to excitatory input
            total_spine_volume,   # 7: Total volume of excitatory targets
            
            # --- METADATA (DO NOT FEED TO GNN - Indices 8 to 13) ---
            soma_center[0],       # 8: Soma X
            soma_center[1],       # 9: Soma Y
            soma_center[2],       # 10: Soma Z
            centroid[0],          # 11: Centroid X
            centroid[1],          # 12: Centroid Y
            centroid[2]           # 13: Centroid Z
        ]
        
        return (i, feats)
        
    except Exception as e:
        # print(f"Failed on {n_id}: {e}") 
        return None
# ---------------------------------------------------------
# MAIN BUILDER
# ---------------------------------------------------------
def build_node_features(neuron_ids, graph_dir, num_workers=None):
    tasks = [(i, n_id, graph_dir) for i, n_id in enumerate(neuron_ids)]
    
    print(f"Extracting features for {len(tasks)} neurons (No Synapse Counts)...")
    
    features_list = []
    valid_indices = [] 
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_single_neuron, task) for task in tasks]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(tasks)):
            result = future.result()
            if result is not None:
                idx, feats = result
                features_list.append(feats)
                valid_indices.append(idx)

    if not features_list:
        return None, None

    # Convert to Tensor
    # We Sort by original index to keep alignment with the ID list
    # (Though typically we just use valid_indices to map back)
    combined = sorted(zip(valid_indices, features_list), key=lambda x: x[0])
    sorted_indices = [x[0] for x in combined]
    sorted_features = [x[1] for x in combined]

    x = torch.tensor(sorted_features, dtype=torch.float)
        
    # --- FIX: Only normalize the biological features (columns 0 to 7) ---
    features_to_normalize = x[:, 0:8]
    metadata_to_keep_raw = x[:, 8:14]
    
    mean = features_to_normalize.mean(dim=0)
    std = features_to_normalize.std(dim=0)
    
    normalized_features = (features_to_normalize - mean) / (std + 1e-6)
    
    # Stitch them back together
    x_final = torch.cat([normalized_features, metadata_to_keep_raw], dim=1)
    # --------------------------------------------------------------------
    
    return x_final, sorted_indices

# ---------------------------------------------------------
# EXECUTION BLOCK
# ---------------------------------------------------------
if __name__ == "__main__":
    import json
    
    # --- Configuration ---
    # Update this if your folder of pbz2 files is located somewhere else
    GRAPH_DIR = "./graph_exports" 
    
    # Where to save the output for the GNN
    CACHE_DIR = "./cache_spatial"
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    OUTPUT_TENSOR_PATH = os.path.join(CACHE_DIR, "x_features.pt")
    OUTPUT_MAPPING_PATH = os.path.join(CACHE_DIR, "node_mapping.json")

    # 1. Find all neurons
    print(f"Scanning '{GRAPH_DIR}' for neuron graphs...")
    neuron_ids = get_neuron_ids_from_folder(GRAPH_DIR)
    
    if not neuron_ids:
        print(f"CRITICAL ERROR: No .pbz2 files found in '{GRAPH_DIR}'. Please check the path.")
        exit()
        
    print(f"Found {len(neuron_ids)} unique neurons.")

    # 2. Extract Features
    # num_workers=None automatically uses all available CPU cores. 
    # If it crashes your computer, change it to num_workers=4 or 8.
    x_features, valid_indices = build_node_features(neuron_ids, GRAPH_DIR, num_workers=None)

    # 3. Save Outputs
    if x_features is not None:
        print(f"\nExtraction complete! Final tensor shape: {x_features.shape}")
        
        # Save the PyTorch Tensor
        torch.save(x_features, OUTPUT_TENSOR_PATH)
        print(f"Saved features to: {OUTPUT_TENSOR_PATH}")
        
        # Save a mapping so you know which row in the tensor corresponds to which neuron ID
        valid_neuron_ids = [neuron_ids[i] for i in valid_indices]
        with open(OUTPUT_MAPPING_PATH, 'w') as f:
            json.dump(valid_neuron_ids, f)
        print(f"Saved ID mapping to: {OUTPUT_MAPPING_PATH}")
        
        print("\nPreprocessing finished successfully. You are ready to build the adjacency graph.")
    else:
        print("\nPreprocessing failed. No features were extracted.")