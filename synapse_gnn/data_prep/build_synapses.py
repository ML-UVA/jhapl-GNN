import os
import json
import argparse
from pathlib import Path
import numpy as np

# Specific JHU APL imports required for decompression
from datasci_tools import system_utils as su

# --- CONFIGURATION LOADER ---
def parse_args():
    parser = argparse.ArgumentParser(description="Extract ground-truth synapses from raw graphs")
    parser.add_argument('--config', type=str, default="config.json", help="Path to the JSON configuration file")
    return parser.parse_args()

def load_config(config_path):
    print(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def extract_synapses(graph_dir="./graph_exports/", output_file="synapses.json"):
    print(f"Scanning {graph_dir} for synapses...")
    synapses = dict()
    
    for filename in os.listdir(graph_dir):
        if not filename.endswith(".pbz2"):
            continue
            
        name = os.path.join(graph_dir, filename)
        
        # Keep the split index! (e.g., "86469113..._0")
        if "_auto_proof" in filename:
            fileid = filename.split("_auto_proof")[0]
        else:
            fileid = filename.split(".")[0]
            
        if not Path(name).exists():
            raise Exception(f"File not found: {name}")
            
        try:
            G = su.decompress_pickle(name)
        except Exception as e:
            print(f"Could not decompress {filename}: {e}")
            continue
            
        for node in G.nodes:
            if "synapse_data" in G.nodes[node]:
                for data in G.nodes[node]["synapse_data"]:
                    data['upstream_dist'] = float(data['upstream_dist'])
                    data['syn_id'] = int(data['syn_id'])
                    data['volume'] = int(data['volume'])
                    
                    pos = 0 if data["syn_type"] == "presyn" else 1
                    
                    if data["syn_id"] not in synapses:
                        # [-1, -1] holds the [presynaptic_neuron_id, postsynaptic_neuron_id]
                        synapses[data["syn_id"]] = [[-1, -1], data]
                        
                    if synapses[data["syn_id"]][0][pos] == -1:
                        synapses[data["syn_id"]][0][pos] = fileid
                    else:
                        pass # Synapse already mapped

    print("Filtering for complete pairs...")
    true_synapses = dict()
    for syn in synapses:
        # Only keep synapses where BOTH the pre and post neurons were found
        if -1 not in synapses[syn][0]:
            true_synapses[int(syn)] = synapses[syn]

    print(f"Found {len(true_synapses)} complete ground-truth synapses.")
    
    with open(output_file, "w") as f:
        json.dump(true_synapses, f, indent=4)
    print(f"Saved successfully to {output_file}")

def main(config_path=None):
    if config_path is None:
        args = parse_args()
        config_path = args.config
        
    config = load_config(config_path)
    
    # Extract cache directory from config to ensure consistency across the pipeline
    CACHE_DIR = config["paths"]["data_dir"]
    output_json = os.path.join(CACHE_DIR, "synapses.json")
    
    # Assuming graph_exports is the standard directory name
    GRAPH_DIR = config["raw_data"]["neurons_directory"]
    
    extract_synapses(graph_dir=GRAPH_DIR, output_file=output_json)

if __name__ == "__main__":
    main()