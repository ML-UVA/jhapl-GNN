import torch
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from synapse_gnn.models.gnn import SynapsePredictor

# --- CONFIGURATION ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Update this list to match the exact order of features in your data.x tensor!
FEATURE_NAMES = [
    "Soma Vol", "Axon Len", "Basal Len", "Apical Len", 
    "Max Axon Reach", "Max Dendrite Reach", "Total Spines", "Total Spine Vol"
]

def generate_pdp(model, data, target_edges, explicit_weights, feature_idx, feature_name, grid_resolution=50):
    """
    Generates data for a Partial Dependence Plot for a specific feature.
    """
    model.eval()
    
    # 1. Find the min and max values of the target feature in the dataset
    orig_feature_vals = data.x[:, feature_idx].cpu().numpy()
    val_min, val_max = np.percentile(orig_feature_vals, 1), np.percentile(orig_feature_vals, 99) # 1st to 99th percentile to avoid crazy outliers
    
    # 2. Create the grid of values we want to sweep across
    feature_grid = np.linspace(val_min, val_max, grid_resolution)
    avg_predictions = []
    
    print(f"Sweeping '{feature_name}' from {val_min:.2f} to {val_max:.2f}...")

    with torch.no_grad():
        for val in feature_grid:
            # 3. Clone the data so we don't permanently overwrite the real features
            temp_data = deepcopy(data).to(device)
            
            # 4. Magically force ALL nodes to have this specific feature value
            temp_data.x[:, feature_idx] = val
            
            # 5. Run the full forward pass (Encode -> Decode)
            z = model.encode(temp_data.x, temp_data.edge_index)
            preds = model.decode(z, target_edges.to(device), explicit_weight=explicit_weights)
            
            # 6. Get the average predicted probability across all evaluated edges
            probs = torch.sigmoid(preds).cpu().numpy()
            avg_predictions.append(np.mean(probs))
            
    return feature_grid, avg_predictions

def plot_pdp(feature_name, grid, predictions, save_path):
    """Plots and saves the PDP."""
    plt.figure(figsize=(8, 5), dpi=300)
    plt.plot(grid, predictions, color='#4B0082', linewidth=3) # Indigo line
    plt.fill_between(grid, predictions, color='#4B0082', alpha=0.1)
    
    plt.title(f"Partial Dependence Plot: {feature_name}", fontsize=14)
    plt.xlabel(f"{feature_name} Value", fontsize=12)
    plt.ylabel("Average Predicted Probability", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved PDP to {save_path}")

def main():
    # 1. Load Model
    model = SynapsePredictor(in_channels=8, hidden_channels=128).to(device)