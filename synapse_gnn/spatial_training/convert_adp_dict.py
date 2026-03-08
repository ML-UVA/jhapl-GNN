import os
import json
import argparse
import pickle
import torch
import numpy as np

# --- 1. CONFIGURATION LOADER ---
def parse_args():
    parser = argparse.ArgumentParser(description="Convert Teammate's ADP Dict directly to PyTorch Tensor (RAM-Safe)")
    parser.add_argument('--config', type=str, default="config.json", help="Path to the JSON configuration file")
    parser.add_argument('--dict_path', type=str, required=True, help="Path to adp_data.pkl (converted teammate dict)")
    return parser.parse_args()

def load_config(config_path):
    print(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

# --- 2. MAIN CONVERSION LOGIC ---
def convert_adp_directly_to_pytorch():
    args = parse_args()
    config = load_config(args.config)
    
    # Extract paths from config
    CACHE_DIR = config["paths"]["data_dir"]
    PATH_INDICES = os.path.join(CACHE_DIR, "node_mapping.json")
    # This matches the input name in your split_and_stitch script
    OUTPUT_PT_PATH = os.path.join(CACHE_DIR, "adp_base_edges.pt")

    print(f"\n--- Bypassing NetworkX: Converting ADP Directly to PyTorch Tensor ---")
    
    # Load ID Map
    if not os.path.exists(PATH_INDICES):
        print(f"Error: Could not find {PATH_INDICES}.")
        return
        
    with open(PATH_INDICES, 'r') as f:
        valid_ids_list = json.load(f)
        
    # Standard mapping for long biological IDs to GNN indices
    id_to_idx = {str(bio_id): int(pt_idx) for pt_idx, bio_id in enumerate(valid_ids_list)}
    print(f"Loaded mapping for {len(valid_ids_list):,} neurons.")

    # Load the teammate's converted biological ID dict
    print(f"Loading ADP Dictionary from {args.dict_path}...")
    if not os.path.exists(args.dict_path):
        print(f"Error: {args.dict_path} does not exist.")
        return

    with open(args.dict_path, 'rb') as f:
        adp_dict = pickle.load(f)

    # STEP A: Estimate total edges for pre-allocation to save RAM
    # This prevents the memory-doubling spike of Python lists
    total_est_edges = sum(len(axons) for axons in adp_dict.values())
    print(f"Allocating space for ~{total_est_edges:,} potential edges...")
    
    # Pre-allocate fixed-size NumPy arrays (Uses 8 bytes per integer strictly)
    src_arr = np.zeros(total_est_edges, dtype=np.int64)
    dst_arr = np.zeros(total_est_edges, dtype=np.int64)
    
    idx = 0
    skipped_edges = 0

    # STEP B: Fill arrays (Axon -> Dendrite)
    print("Parsing dictionary and mapping IDs...")
    for dst_bio_id, axons in adp_dict.items(): 
        # Biological IDs from teammate often have _0 or _axon suffixes
        dst_str = str(dst_bio_id).split('_')[0]
        
        if dst_str in id_to_idx:
            dst_pt_idx = id_to_idx[dst_str]
            
            for src_bio_id in axons: 
                src_str = str(src_bio_id).split('_')[0]
                
                if src_str in id_to_idx:
                    src_arr[idx] = id_to_idx[src_str]
                    dst_arr[idx] = dst_pt_idx
                    idx += 1
                else:
                    skipped_edges += 1
        else:
            skipped_edges += len(axons)

    print(f"\nData parsed successfully:")
    print(f"  -> Total Aligned Edges: {idx:,}")
    print(f"  -> Ignored (Out of Volume): {skipped_edges:,}")

    # STEP C: Final Conversion and Save
    print("\nConverting to PyTorch Tensor...")
    # Trim the pre-allocated array to the actual number of valid edges found
    edge_index = torch.from_numpy(np.vstack([src_arr[:idx], dst_arr[:idx]]))
    
    # Delete the large arrays to free up RAM before saving
    del src_arr
    del dst_arr
    
    print(f"Saving highly compressed tensor to {OUTPUT_PT_PATH}...")
    torch.save(edge_index, OUTPUT_PT_PATH)
    print("Done! You can now proceed to split_and_stitch.")

if __name__ == "__main__":
    convert_adp_directly_to_pytorch()