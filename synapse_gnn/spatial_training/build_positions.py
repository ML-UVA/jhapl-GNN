import os
import json
import concurrent.futures
from tqdm import tqdm
from datasci_tools import system_utils as su
import numpy as np

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
RAW_GRAPH_DIR = "graph_exports/"
OUTPUT_FILE = "positions.json"

def get_preferred_files(graph_dir):
    """
    Scans directory and selects the best file for each neuron ID.
    Priority: _0_ file > _1_ file
    """
    all_files = [f for f in os.listdir(graph_dir) if f.endswith(".pbz2")]
    neuron_files = {}

    print("Indexing files...")
    for f in all_files:
        # Filename format: "864691135986925571_0_auto..."
        parts = f.split("_")
        n_id = parts[0]
        version = parts[1] # "0" or "1"
        
        if n_id not in neuron_files:
            neuron_files[n_id] = []
        neuron_files[n_id].append((version, f))

    # Select preferred file
    final_tasks = []
    duplicates = 0
    
    for n_id, file_list in neuron_files.items():
        # Sort by version (0 comes before 1)
        file_list.sort() 
        
        # Pick the first one (which will be '0' if it exists)
        best_file = file_list[0][1]
        
        if len(file_list) > 1:
            duplicates += 1
            
        final_tasks.append(best_file)
        
    print(f"Found {len(neuron_files)} unique neurons.")
    print(f"Resolved {duplicates} duplicate cases (chose _0 over _1).")
    return final_tasks

def process_neuron_file(filename):
    """
    Extracts Soma Center.
    """
    if not filename.endswith(".pbz2"): return None
    
    neuron_id = filename.split("_")[0]
    file_path = os.path.join(RAW_GRAPH_DIR, filename)

    try:
        G = su.decompress_pickle(file_path)
        
        # Method 1: Check S0 node directly
        if "S0" in G.nodes:
            center = G.nodes["S0"].get("mesh_center")
            if center is not None:
                return neuron_id, list(center)
        
        # Method 2: Scan for compartment="soma"
        for n, data in G.nodes(data=True):
            if data.get("compartment") == "soma":
                center = data.get("mesh_center")
                if center is not None:
                    return neuron_id, list(center)
                    
        return None 
    except:
        return None

def main():
    if not os.path.exists(RAW_GRAPH_DIR):
        print("Directory not found.")
        return

    # 1. Get Clean List of Files (Handling the _0 vs _1 logic)
    files_to_process = get_preferred_files(RAW_GRAPH_DIR)

    positions = {}

    # 2. Parallel Extraction
    print(f"Extracting positions from {len(files_to_process)} files...")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_neuron_file, files_to_process), total=len(files_to_process)))

    # 3. Build Dictionary
    for res in results:
        if res is not None:
            n_id, center = res
            positions[n_id] = [float(c) for c in center]

    # 4. Save
    print(f"Saving {len(positions)} positions to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w") as f:
        json.dump(positions, f)
    print("Done.")

if __name__ == "__main__":
    main()