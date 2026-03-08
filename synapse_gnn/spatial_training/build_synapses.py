import os
import json
import bz2
import pickle
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Step 0: Build Connectome by Matching Synapse IDs")
    parser.add_argument('--config', type=str, default="config.json", help="Path to config file")
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def build_connectome():
    args = parse_args()
    config = load_config(args.config)
    
    print("\n--- STEP 0: Reverse-Engineering the Connectome ---")
    
    NEURONS_DIR = config["raw_data"]["neurons_directory"]
    CACHE_DIR = config["paths"]["data_dir"]
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    OUTPUT_SYNAPSES = os.path.join(CACHE_DIR, "synapses_reverse_engineered.json")
    
    # This registry will hold the half-connections until they find their pair
    # Format: { syn_id: {"pre_id": None, "post_id": None, "attributes": None} }
    synapse_registry = {}

    file_list = [f for f in os.listdir(NEURONS_DIR) if f.endswith(".pbz2")]
    print(f"Scanning {len(file_list)} neurons to match pre- and post-synaptic IDs...")

    # ==========================================
    # PASS 1: MINE EVERY SYNAPSE IN THE DATASET
    # ==========================================
    for filename in tqdm(file_list, desc="Mining Synapses"):
        # Extract the biological integer ID (e.g., 864691135730442937)
        neuron_id = int(filename.split('_')[0])
        filepath = os.path.join(NEURONS_DIR, filename)
        
        try:
            with bz2.BZ2File(filepath, 'rb') as f:
                G = pickle.load(f)
        except Exception:
            continue

        # Iterate through all branches (nodes) in the neuron
        for node, data in G.nodes(data=True):
            syn_list = data.get("synapse_data", [])
            
            for syn in syn_list:
                syn_id = syn["syn_id"]
                syn_type = syn["syn_type"]
                
                # If we haven't seen this synapse before, create a slot for it
                if syn_id not in synapse_registry:
                    synapse_registry[syn_id] = {"pre_id": None, "post_id": None, "attributes": None}
                
                # Log the Pre-Synaptic side
                if syn_type == "presyn":
                    synapse_registry[syn_id]["pre_id"] = neuron_id
                    
                # Log the Post-Synaptic side & grab its morphological attributes
                elif syn_type == "postsyn":
                    synapse_registry[syn_id]["post_id"] = neuron_id
                    
                    # Convert numpy types to standard python types for JSON serialization
                    synapse_registry[syn_id]["attributes"] = {
                        "upstream_dist": float(syn.get("upstream_dist", 0.0)),
                        "head_neck_shaft": str(syn.get("head_neck_shaft", "unknown")),
                        "syn_id": int(syn_id),
                        "volume": int(syn.get("volume", 0)),
                        "syn_type": "postsyn"
                    }

    # ==========================================
    # PASS 2: STITCH VALID PAIRS TOGETHER
    # ==========================================
    print("\nStitching matched pairs together...")
    final_synapses = {}
    valid_count = 0
    orphan_count = 0

    for syn_id, info in synapse_registry.items():
        # A synapse is only valid if BOTH neurons exist in your 70,000 files
        if info["pre_id"] is not None and info["post_id"] is not None and info["attributes"] is not None:
            # Format to exactly match your friend's JSON structure
            final_synapses[str(syn_id)] = [
                [info["pre_id"], info["post_id"]],
                info["attributes"]
            ]
            valid_count += 1
        else:
            orphan_count += 1

    # Save the final exact JSON format
    with open(OUTPUT_SYNAPSES, 'w') as f:
        # We don't use indent=4 here because your friend's file was compressed (minified)
        json.dump(final_synapses, f)
        
    print(f"\nSuccess! Reconstructed the global connectome.")
    print(f"Matched Valid Synapses: {valid_count:,}")
    print(f"Ignored Orphan Synapses (target missing from dataset): {orphan_count:,}")
    print(f"Exported to: {OUTPUT_SYNAPSES}")

if __name__ == "__main__":
    build_connectome()