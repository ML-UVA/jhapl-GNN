import sys
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# --- 1. SYSTEM PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get parent (.../spatial_training)
parent_dir = os.path.dirname(current_dir)
# Get grandparent (.../synapse_gnn) -> This is the project root
grandparent_dir = os.path.dirname(parent_dir)

# Add paths so Python can find 'spatial_training'
if grandparent_dir not in sys.path:
    sys.path.append(grandparent_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import spatial_training.gnn as gnn
from spatial_training.main import get_random_subgraph

# --- 2. CONFIGURATION ---
CACHE_DIR = os.path.join(grandparent_dir, "cache_spatial")
OUTPUT_FOLDER = os.path.join(grandparent_dir, "saved_models_spatial")

PATH_X = os.path.join(CACHE_DIR, "x_features.pt")
PATH_TEST_EDGES = os.path.join(CACHE_DIR, "graph_test_edges.pt")
MODEL_PATH = os.path.join(OUTPUT_FOLDER, "best_model.pth")

# --- 3. LOAD DATA (3-Feature Mode) ---
print(f"Loading data from {CACHE_DIR}...")
x_global_raw = torch.load(PATH_X, weights_only=False)
test_edges = torch.load(PATH_TEST_EDGES, weights_only=False)

# Keep ONLY Centroids (Original indices 6, 7, 8)
features_to_keep = [6, 7, 8]
x_global = x_global_raw[:, features_to_keep]

print(f"Data loaded. Shape: {x_global.shape} (Should be N x 3)")

# --- 4. LOAD MODEL (3-Feature Mode) ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = gnn.SynapsePredictor(in_channels=3, hidden_channels=128).to(device)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=False))
    print("Model loaded successfully.")
else:
    print(f"CRITICAL ERROR: Model not found at {MODEL_PATH}")
    exit()
model.eval()

# --- HELPER FUNCTION: Get Spatial Cluster ---
def get_spatial_subgraph(num_nodes=300):
    print(f"Selecting a spatial cluster of {num_nodes} neurons...")
    
    # In x_global (3 features), coordinates are now indices 0, 1, 2
    all_coords = x_global[:, 0:3] 
    best_indices = None
    max_edges = -1
    
    # Try 10 times to find a dense cluster
    for i in range(10):
        center_idx = torch.randint(0, x_global.size(0), (1,)).item()
        center_coord = all_coords[center_idx].unsqueeze(0)
        
        dists = torch.cdist(all_coords, center_coord).squeeze()
        _, indices = torch.topk(dists, k=num_nodes, largest=False)
        indices = indices.sort().values 
        
        mask = torch.zeros(x_global.size(0), dtype=torch.bool)
        mask[indices] = True
        row, col = test_edges
        edge_mask = mask[row] & mask[col]
        edge_count = edge_mask.sum().item()
        
        if best_indices is None or edge_count > max_edges:
            max_edges = edge_count
            best_indices = indices
            print(f"  Attempt {i+1}: Found cluster with {edge_count} edges (New Best)")
            
    print(f"Selected final cluster with {max_edges} internal edges.")
    
    # Create the subgraph mapping
    mask = torch.zeros(x_global.size(0), dtype=torch.bool)
    mask[best_indices] = True
    row, col = test_edges
    edge_mask = mask[row] & mask[col]
    
    mapping = torch.full((x_global.size(0),), -1, dtype=torch.long)
    mapping[best_indices] = torch.arange(len(best_indices))
    
    local_edge_index = torch.stack([
        mapping[test_edges[0][edge_mask]], 
        mapping[test_edges[1][edge_mask]]
    ], dim=0)
    
    return local_edge_index, best_indices

# --- VISUALIZATION 1: ACCURATE PLOTS ---
def generate_accurate_plots():
    local_edge_index, node_indices = get_spatial_subgraph(num_nodes=300)
    
    batch_x = x_global[node_indices].to(device)
    local_edge_index = local_edge_index.to(device)
    
    # Coordinates are 0, 1, 2 in the new 3-feature tensor
    coords = batch_x[:, 0:3].cpu().numpy()
    
    # Calculate tight bounds
    margin = 0.05
    x_min, x_max = coords[:,0].min() - margin, coords[:,0].max() + margin
    y_min, y_max = coords[:,1].min() - margin, coords[:,1].max() + margin
    z_min, z_max = coords[:,2].min() - margin, coords[:,2].max() + margin

    with torch.no_grad():
        z = model.encode(batch_x, local_edge_index)
        src, dst = local_edge_index[0], local_edge_index[1]
        scores = (z[src] * z[dst]).sum(dim=1).sigmoid().cpu().numpy()
    
    mask = scores > 0.75
    src_np = src.cpu().numpy()[mask]
    dst_np = dst.cpu().numpy()[mask]
    
    print(f"Plotting {len(src_np)} predicted synapses...")

    # --- PLOT 3D ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(coords[:,0], coords[:,1], coords[:,2], s=20, c='blue', alpha=0.15, label="Neurons")
    
    for i in range(len(src_np)):
        start = coords[src_np[i]]
        end = coords[dst_np[i]]
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
                c='green', alpha=0.6, linewidth=1.0)

    ax.set_title("3D View: 300-Neuron Spatial Subgraph (3 Features)", fontsize=14)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max); ax.set_zlim(z_min, z_max)
    ax.view_init(elev=25, azim=135)
    plt.tight_layout()
    plt.savefig("demo_visual_accurate_3d.png", dpi=300)
    print("Saved 'demo_visual_accurate_3d.png'")
    
    # --- PLOT 2D ---
    plt.figure(figsize=(8, 8))
    plt.scatter(coords[:,0], coords[:,1], s=25, c='blue', alpha=0.2, label="Neurons")
    
    for i in range(len(src_np)):
        start = coords[src_np[i]]
        end = coords[dst_np[i]]
        plt.plot([start[0], end[0]], [start[1], end[1]], c='green', alpha=0.5, linewidth=1.0)
        
    plt.title("2D Projection (3 Features)", fontsize=14)
    plt.xlabel("X"); plt.ylabel("Y")
    plt.xlim(x_min, x_max); plt.ylim(y_min, y_max)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig("demo_visual_accurate_2d.png", dpi=300)
    print("Saved 'demo_visual_accurate_2d.png'")

# --- VISUALIZATION 2: HISTOGRAM ---
def plot_confidence_histogram():
    print("Generating Confidence Histogram...")
    
    # Sample random subgraph
    local_edge_index, node_indices = get_random_subgraph(test_edges, x_global.size(0), sample_size=8000)
    
    batch_x = x_global[node_indices].to(device)
    local_edge_index = local_edge_index.to(device)
    
    with torch.no_grad():
        z = model.encode(batch_x, local_edge_index)
        
        # Real Synapses
        src, dst = local_edge_index[0], local_edge_index[1]
        pos_scores = (z[src] * z[dst]).sum(dim=1).sigmoid().cpu().numpy()
        
        # Hard Negatives
        num_pos = len(pos_scores)
        neg_src = torch.randint(0, batch_x.size(0), (num_pos*5,), device=device)
        neg_dst = torch.randint(0, batch_x.size(0), (num_pos*5,), device=device)
        cand_scores = (z[neg_src] * z[neg_dst]).sum(dim=1).sigmoid()
        hard_neg_scores, _ = torch.topk(cand_scores, k=num_pos)
        hard_neg_scores = hard_neg_scores.cpu().numpy()

    plt.figure(figsize=(10, 6))
    sns.histplot(pos_scores, color="green", label="True Synapses", kde=True, bins=30, alpha=0.6)
    sns.histplot(hard_neg_scores, color="red", label="Hard Negatives (Impostors)", kde=True, bins=30, alpha=0.6)
    
    plt.title("Model Confidence: True Synapses vs. Hard Impostors", fontsize=14)
    plt.xlabel("Predicted Probability", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.axvline(0.73, color='black', linestyle='--', label="Optimal Threshold (0.73)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("demo_visual_histogram.png", dpi=300)
    print("Saved 'demo_visual_histogram.png'")

if __name__ == "__main__":
    generate_accurate_plots()
    plot_confidence_histogram()