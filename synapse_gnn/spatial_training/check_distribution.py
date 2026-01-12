import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import main  # Imports your setup
import numpy as np

# Load the latest best model
model = main.gnn.SynapsePredictor(10, 128).to(main.device)
model.load_state_dict(torch.load("saved_models_spatial/best_model.pth"))
model.eval()

# Load data
x = torch.load(main.PATH_X)
edges = torch.load(main.PATH_TEST_EDGES)

# Sample 
local_edge_index, node_indices = main.get_random_subgraph(edges, x.size(0), 5000)
local_edge_index = local_edge_index.to(main.device)
batch_x = x[node_indices].to(main.device)

# Get Scores
with torch.no_grad():
    z = model.encode(batch_x, local_edge_index)
    pos = (z[local_edge_index[0]] * z[local_edge_index[1]]).sum(dim=1).sigmoid().cpu().numpy()
    
    # Random negatives
    neg_idx = torch.randint(0, 5000, (len(pos), 2))
    neg = (z[neg_idx[:,0]] * z[neg_idx[:,1]]).sum(dim=1).sigmoid().cpu().numpy()

# Plot
plt.hist(pos, bins=50, alpha=0.5, label='Real Synapses', color='green')
plt.hist(neg, bins=50, alpha=0.5, label='Random Negatives', color='red')
plt.legend()
plt.title("Score Distribution (Validation)")
plt.savefig("score_dist_check.png")
print("Saved plot to score_dist_check.png")