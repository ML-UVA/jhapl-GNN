import torch
import numpy as np
import os
import argparse
import json
import concurrent.futures
from tqdm import tqdm
from datasci_tools import system_utils as su
from neuron_morphology_tools import neuron_nx_utils as nxu
from neuron_morphology_tools import neuron_nx_stats as nxs

# --- CONFIGURATION LOADER ---
def parse_args():
    parser = argparse.ArgumentParser(description="Extract morphological features from raw graphs")
    parser.add_argument('--config', type=str, default="synapse_gnn/config.json", help="Path to the JSON configuration file")
    return parser.parse_args()

def load_config(config_path):
    print(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

# ---------------------------------------------------------
# HELPER: Get IDs from folder
# ---------------------------------------------------------
def get_neuron_ids_from_folder(neurons_directory):
    print(f"Scanning directory for neuron files: {neurons_directory}")
    valid_ids = set()
    
    for filename in os.listdir(neurons_directory):
        if filename.endswith(".pbz2"):
            # THE FIX: Split by "_auto_proof" to drop the suffix.
            clean_id = filename.split('_auto_proof')[0]
            valid_ids.add(clean_id)
            
    sorted_valid_ids = sorted(list(valid_ids))
    
    print(f"Found {len(sorted_valid_ids):,} unique neurons (Split indices isolated).")
    return sorted_valid_ids

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
    
    # THE FIX: Dynamically reconstruct the filename using the clean ID
    filename = f"{n_id}_auto_proof_v7_proofread.pbz2" 
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
            if node == soma_id: continue
                
            comp = str(data.get('compartment', '')).lower()
            length = data.get('skeletal_length', 0.0)
            
            dist_from_soma = data.get('soma_distance_euclidean', 0.0) 
            
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
            
            total_spines += data.get('n_spines', 0)
            
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
            soma_vol, axon_len, basal_len, apical_len, 
            max_axon_reach, max_dendrite_reach, total_spines, total_spine_volume,
            soma_center[0], soma_center[1], soma_center[2], 
            centroid[0], centroid[1], centroid[2]
        ]
        
        return (i, feats)
        
    except Exception as e:
        print(f"Failed on {n_id}: {e}") 
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

    combined = sorted(zip(valid_indices, features_list), key=lambda x: x[0])
    sorted_indices = [x[0] for x in combined]
    sorted_features = [x[1] for x in combined]

    x = torch.tensor(sorted_features, dtype=torch.float)
        
    # Only normalize the biological features (columns 0 to 7)
    features_to_normalize = x[:, 0:8]
    metadata_to_keep_raw = x[:, 8:14]
    
    mean = features_to_normalize.mean(dim=0)
    std = features_to_normalize.std(dim=0)
    
    normalized_features = (features_to_normalize - mean) / (std + 1e-6)
    x_final = torch.cat([normalized_features, metadata_to_keep_raw], dim=1)
    
    return x_final, sorted_indices

# ---------------------------------------------------------
# EXECUTION BLOCK
# ---------------------------------------------------------
def main(config_path=None):
    # 1. Load the Config
    if config_path is None:
        args = parse_args()
        config_path = args.config
        
    config = load_config(config_path)
    from config import RAW_DATA_DIR
    GRAPH_DIR = config.get("raw_data", {}).get("neurons_directory") or str(RAW_DATA_DIR)
    # 2. Extract Paths Dynamically
    CACHE_DIR = config["paths"]["data_dir"]
    
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    OUTPUT_TENSOR_PATH = os.path.join(CACHE_DIR, "x_features.pt")
    OUTPUT_MAPPING_PATH = os.path.join(CACHE_DIR, "node_mapping.json")

    # 3. Find all neurons
    print(f"Scanning '{GRAPH_DIR}' for neuron graphs...")
    neuron_ids = get_neuron_ids_from_folder(GRAPH_DIR)
    
    if not neuron_ids:
        print(f"CRITICAL ERROR: No .pbz2 files found in '{GRAPH_DIR}'. Please check the path.")
        exit()
        
    print(f"Found {len(neuron_ids)} unique neurons.")

    # 4. Extract Features
    x_features, valid_indices = build_node_features(neuron_ids, GRAPH_DIR, num_workers=None)

    # 5. Save Outputs
    if x_features is not None:
        print(f"\nExtraction complete! Final tensor shape: {x_features.shape}")
        
        torch.save(x_features, OUTPUT_TENSOR_PATH)
        print(f"Saved features to: {OUTPUT_TENSOR_PATH}")
        
        valid_neuron_ids = [neuron_ids[i] for i in valid_indices]
        with open(OUTPUT_MAPPING_PATH, 'w') as f:
            json.dump(valid_neuron_ids, f)
        print(f"Saved ID mapping to: {OUTPUT_MAPPING_PATH}")
        
        print("\nPreprocessing finished successfully. You are ready to build the adjacency graph.")
    else:
        print("\nPreprocessing failed. No features were extracted.")

if __name__ == "__main__":
    main()