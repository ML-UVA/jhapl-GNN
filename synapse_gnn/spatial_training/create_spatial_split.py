import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
import os

# CONFIGURATION
CACHE_DIR = "cache_spatial"  # UPDATED FOLDER
PATH_X = os.path.join(CACHE_DIR, "x_features.pt")
OUTPUT_MASK = os.path.join(CACHE_DIR, "spatial_split_masks.pt")

# UPDATED INDICES: Centroid X, Y, Z are now at indices 7, 8, 9
COORD_COLS = [7, 8, 9] 

# HYPERPARAMETERS
TRAIN_CLUSTER_SIZE = 15000 
TEST_CLUSTER_SIZE  = 3000   
BUFFER_DISTANCE    = 50000 

def generate_spatial_split():
    print("--- Generating Spatial Split (Centroid Based) ---")
    
    if not os.path.exists(PATH_X):
        print(f"Error: {PATH_X} not found. Run main.py first to generate features.")
        return

    x_features = torch.load(PATH_X, weights_only=False)
    coords = x_features[:, COORD_COLS].numpy()
    num_nodes = coords.shape[0]
    
    print(f"Loaded {num_nodes} neurons.")
    
    # Sort by X to find Left/Right sides
    sort_idx_x = np.argsort(coords[:, 0])
    
    # Pick Seeds (25% and 75% along X-axis)
    train_seed_idx = sort_idx_x[int(num_nodes * 0.25)]
    train_seed_pos = coords[train_seed_idx].reshape(1, -1)
    
    test_seed_idx  = sort_idx_x[int(num_nodes * 0.75)]
    test_seed_pos  = coords[test_seed_idx].reshape(1, -1)
    
    # Grow Clusters (k-NN)
    print("Growing dense clusters...")
    nbrs = NearestNeighbors(n_neighbors=num_nodes, algorithm='ball_tree').fit(coords)
    
    distances_tr, indices_tr = nbrs.kneighbors(train_seed_pos)
    train_indices = indices_tr[0][:TRAIN_CLUSTER_SIZE]
    
    distances_te, indices_te = nbrs.kneighbors(test_seed_pos)
    test_indices = indices_te[0][:TEST_CLUSTER_SIZE]
    
    # Create Masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask  = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[train_indices] = True
    test_mask[test_indices] = True
    
    # Verify Overlap
    overlap = (train_mask & test_mask).sum().item()
    if overlap > 0:
        print(f"WARNING: {overlap} nodes overlap! Increase separation.")
    else:
        print("Success: No overlap.")
        
    torch.save({'train_mask': train_mask, 'test_mask': test_mask}, OUTPUT_MASK)
    print(f"Saved masks to {OUTPUT_MASK}")

if __name__ == "__main__":
    generate_spatial_split()