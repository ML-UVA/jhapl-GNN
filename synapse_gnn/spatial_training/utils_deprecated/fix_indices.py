import torch
import preprocessing
import os

# CONFIGURATION
CACHE_DIR = "cache_spatial"
RAW_GRAPH_DIR = "graph_exports/" # Ensure this matches your folder

def fix_indices():
    print("--- Fixing ID Mismatch ---")
    
    # 1. Load the "Bad" indices (Integers like 0, 1, 5...)
    path_indices = os.path.join(CACHE_DIR, "valid_indices.pt")
    if not os.path.exists(path_indices):
        print("Error: valid_indices.pt not found.")
        return

    indices = torch.load(path_indices, weights_only=False)
    print(f"Loaded {len(indices)} valid indices (Integers).")

    # 2. Get the Master List of IDs (Strings like '864...')
    # We re-scan the folder to get the exact same list 'main.py' saw.
    print(f"Scanning {RAW_GRAPH_DIR} to recover IDs...")
    all_neuron_ids = preprocessing.get_neuron_ids_from_folder(RAW_GRAPH_DIR)
    
    # 3. Map Integer -> Real ID
    # This recovers the specific IDs that successfully passed preprocessing
    valid_real_ids = [all_neuron_ids[i] for i in indices]
    
    print(f"Recovered {len(valid_real_ids)} Real IDs (e.g., {valid_real_ids[0]})")

    # 4. Save the "Good" file
    torch.save(valid_real_ids, path_indices)
    print(f"Success! Overwrote {path_indices} with Real IDs.")

if __name__ == "__main__":
    fix_indices()