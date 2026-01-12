import torch
import numpy as np
import os
import concurrent.futures
from tqdm import tqdm
from datasci_tools import system_utils as su
from neuron_morphology_tools import neuron_nx_utils as nxu

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
# WORKER: Process One Neuron
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
        
        # Basic Features
        soma_data = G.nodes[soma_id]
        soma_center = soma_data.get('mesh_center', [0,0,0])
        soma_vol = soma_data.get('mesh_volume', 0)
        
        total_vol = 0
        total_len = 0
        # synapse_count = 0  <-- REMOVED
        
        for node, data in G.nodes(data=True):
            total_vol += data.get('mesh_volume', 0)
            total_len += data.get('skeletal_length', 0)
            # if 'synapse_data' in data:                 <-- REMOVED
            #    synapse_count += len(data['synapse_data']) <-- REMOVED

        # --- NEW: CENTROID CALCULATION ---
        # Get all skeleton points to calculate true center of mass
        try:
            all_skeleton_verts = nxu.skeleton_nodes(G)
            if len(all_skeleton_verts) > 0:
                centroid = np.mean(all_skeleton_verts, axis=0)
            else:
                centroid = soma_center
        except:
            centroid = soma_center

        feats = [
            soma_vol, 
            total_vol, 
            total_len, 
            # synapse_count, <-- REMOVED (This used to be index 3)
            soma_center[0], 
            soma_center[1], 
            soma_center[2],
            # NEW COLUMNS (Now Indices 6, 7, 8)
            centroid[0], 
            centroid[1], 
            centroid[2]
        ]
        
        return (i, feats)
        
    except Exception as e:
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
    
    # Normalize
    mean = x.mean(dim=0)
    std = x.std(dim=0)
    x_normalized = (x - mean) / (std + 1e-6)
    
    return x_normalized, sorted_indices