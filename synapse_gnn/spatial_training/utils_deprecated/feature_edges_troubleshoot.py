import torch
import numpy as np
import os

# PATHS
VALID_INDICES_PATH = "cache_data/valid_indices.pt"
FIRST_CHUNK_PATH = "processed_edges/edges_dict_part_0.npy"

def diagnose_mismatch():
    print("--- ID DIAGNOSTIC ---")
    
    # 1. Inspect Valid Neurons (Features)
    if os.path.exists(VALID_INDICES_PATH):
        valid_ids = torch.load(VALID_INDICES_PATH, weights_only=False)
        print(f"\n[Feature Set] Count: {len(valid_ids)}")
        print(f"Type: {type(valid_ids)}")
        
        # Convert to list if tensor for cleaner printing
        if torch.is_tensor(valid_ids):
            sample = valid_ids[:5].tolist()
        else:
            sample = valid_ids[:5]
            
        print(f"Sample IDs: {sample}")
        print(f"Sample ID Type: {type(sample[0])}")
    else:
        print("Error: Valid indices file not found.")

    # 2. Inspect Raw Edges (Adjacency)
    if os.path.exists(FIRST_CHUNK_PATH):
        edges = np.load(FIRST_CHUNK_PATH)
        print(f"\n[Edge Chunk 0] Shape: {edges.shape}")
        print(f"Type: {edges.dtype}")
        
        # Get unique sources and targets from first 10 edges
        sample_edges = edges[:5]
        print(f"Sample Edges (Src -> Dst):\n{sample_edges}")
        
        sample_src = sample_edges[0][0]
        print(f"Sample Source ID Type: {type(sample_src)}")
        
        # 3. Check for ANY overlap in this small sample
        # Convert feature IDs to a set for fast lookup
        feature_id_set = set(valid_ids if not torch.is_tensor(valid_ids) else valid_ids.tolist())
        
        # Check the first 10,000 edges for any match
        print("\n--- Checking first 10,000 edges for overlap ---")
        hits = 0
        scanned = 0
        for src, dst in edges[:10000]:
            if src in feature_id_set and dst in feature_id_set:
                hits += 1
            scanned += 1
            
        print(f"Scanned: {scanned}")
        print(f"Matches found: {hits}")
        
        if hits == 0:
            print("\nVERDICT: Zero overlap. The IDs are definitely different.")
            # Heuristic check: Are the edge IDs much larger/smaller?
            feat_min = min(sample)
            edge_min = sample_src
            print(f"Feature ID range start: {feat_min}")
            print(f"Edge ID range start:    {edge_min}")
    else:
        print("Error: Edge chunk not found.")

if __name__ == "__main__":
    diagnose_mismatch()