import os
import json
import argparse
from pathlib import Path
import torch
from datasci_tools import system_utils as su

def parse_args():
    parser = argparse.ArgumentParser(description="Extract ground-truth synapses")
    parser.add_argument('--config', type=str, default="synapse_gnn/config.json")
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def extract_synapses(graph_dir, output_file):
    print(f"Extracting ground-truth synapses from raw .pbz2 graphs in {graph_dir}...")
    synapses = dict()
    
    for filename in os.listdir(graph_dir):
        if not filename.endswith(".pbz2"):
            continue
            
        name = os.path.join(graph_dir, filename)
        fileid = filename.split("_auto_proof")[0] if "_auto_proof" in filename else filename.split(".")[0]
            
        try:
            G = su.decompress_pickle(name)
        except Exception:
            continue
            
        for node in G.nodes:
            if "synapse_data" in G.nodes[node]:
                for data in G.nodes[node]["synapse_data"]:
                    pos = 0 if data["syn_type"] == "presyn" else 1
                    syn_id = int(data['syn_id'])
                    
                    if syn_id not in synapses:
                        synapses[syn_id] = [-1, -1]
                        
                    if synapses[syn_id][pos] == -1:
                        synapses[syn_id][pos] = fileid

    print("Filtering for complete pairs...")
    true_synapses = {k: v for k, v in synapses.items() if -1 not in v}

    # ---------------------------------------------------------
    # NEW PYTORCH EXPORT FORMAT
    # ---------------------------------------------------------
    print("Converting ground-truth synapses to PyTorch tensors...")
    
    # 1. Create the Node ID Map (Alphabetically sorted)
    unique_strings = set()
    for syn_id, nodes in true_synapses.items():
        unique_strings.add(nodes[0])
        unique_strings.add(nodes[1])
        
    node_ids = sorted(list(unique_strings))
    id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
    
    # 2. Extract edges to integers
    sources = [id_to_idx[nodes[0]] for nodes in true_synapses.values()]
    targets = [id_to_idx[nodes[1]] for nodes in true_synapses.values()]
        
    # 3. Create the PyTorch tensor
    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    
    # 4. Save the uniform dictionary
    graph_dict = {
        'edge_index': edge_index,
        'node_ids': node_ids
    }
    
    pt_output = output_file.replace('.json', '.pt')
    os.makedirs(os.path.dirname(pt_output), exist_ok=True)
    torch.save(graph_dict, pt_output)
    
    print(f"Saved successfully to {pt_output}")
    print(f"  -> Pos Edge Index Shape: {edge_index.shape}")

def main(config_path=None):
    if config_path is None:
        args = parse_args()
        config_path = args.config
        
    from config import INTERMEDIATE_DIR, RAW_DATA_DIR
    config = load_config(config_path)

    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    output_pt = str(INTERMEDIATE_DIR / "synapses.pt")
    GRAPH_DIR = config.get("raw_data", {}).get("neurons_directory") or str(RAW_DATA_DIR)

    extract_synapses(graph_dir=GRAPH_DIR, output_file=output_pt)

if __name__ == "__main__":
    main()