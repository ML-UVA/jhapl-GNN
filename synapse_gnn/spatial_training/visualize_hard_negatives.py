import matplotlib
matplotlib.use('Agg') # Force non-interactive backend (Works on SSH/tmux)
import matplotlib.pyplot as plt
import torch
import main
import numpy as np

def run_analysis():
    print("Loading model and data...")
    device = main.device
    
    # 1. Load Model
    model = main.gnn.SynapsePredictor(10, 128).to(device)
    try:
        model.load_state_dict(torch.load("saved_models_spatial/best_model.pth", weights_only=True))
    except:
        print("Error: Could not find 'saved_models_spatial/best_model.pth'")
        return
    model.eval()

    # 2. Load Data
    x = torch.load(main.PATH_X, weights_only=False)
    edges = torch.load(main.PATH_TEST_EDGES, weights_only=False)
    
    # 3. Mine for Hard Negatives
    print("Mining for hard negatives (this may take a moment)...")
    
    # We sample a large batch to find rare "hard" cases
    sample_size = 10000 
    local_edge_index, node_indices = main.get_random_subgraph(edges, x.size(0), sample_size)
    
    batch_x = x[node_indices].to(device)
    
    with torch.no_grad():
        z = model.encode(batch_x, local_edge_index.to(device))
        
        # A. Get Positive Scores (Real Synapses)
        pos_src, pos_dst = local_edge_index[0], local_edge_index[1]
        pos_scores = (z[pos_src] * z[pos_dst]).sum(dim=1).sigmoid()
        
        # B. Generate Random Candidates
        # We try 50,000 random pairs to find the ones that look like connections
        cand_src = torch.randint(0, sample_size, (50000,), device=device)
        cand_dst = torch.randint(0, sample_size, (50000,), device=device)
        
        cand_scores = (z[cand_src] * z[cand_dst]).sum(dim=1).sigmoid()
        
        # C. Filter: Find High-Scoring Negatives
        # We want pairs with Score > 0.8 (Model is very confident)
        hard_mask = cand_scores > 0.80
        hard_indices = torch.nonzero(hard_mask).squeeze()
        
        print(f"Scanned 50,000 random pairs. Found {len(hard_indices)} 'Hard Negatives' (Score > 0.8).")
        
        if len(hard_indices) == 0:
            print("Model is too good! No negatives found with score > 0.8. Lowering threshold to top 5...")
            values, hard_indices = torch.topk(cand_scores, 5)

    # 4. Visualization Function
    def plot_neuron_pair(ax, idx1, idx2, title, color_pair):
        # Indices in batch_x
        n1 = batch_x[idx1].cpu().numpy()
        n2 = batch_x[idx2].cpu().numpy()
        
        # Extract Coords (Indices 4,5,6 are Soma; 7,8,9 are Centroid)
        # Note: These are Z-Score normalized! We plot them as-is.
        soma1 = n1[4:7]
        cent1 = n1[7:10]
        
        soma2 = n2[4:7]
        cent2 = n2[7:10]
        
        # Plot Neuron 1
        ax.plot([soma1[0], cent1[0]], [soma1[1], cent1[1]], 'o-', color=color_pair[0], label='Neuron A')
        # Plot Neuron 2
        ax.plot([soma2[0], cent2[0]], [soma2[1], cent2[1]], 'o-', color=color_pair[1], label='Neuron B')
        
        # Draw "Connection" line (Dashed)
        ax.plot([cent1[0], cent2[0]], [cent1[1], cent2[1]], '--', color='gray', alpha=0.5)
        
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Norm. X Coord")
        ax.set_ylabel("Norm. Y Coord")

    # 5. Generate Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot 1: A Real Synapse (True Positive)
    # Find a high-scoring positive
    best_pos_idx = torch.argmax(pos_scores)
    p_score = pos_scores[best_pos_idx].item()
    p_src = pos_src[best_pos_idx].item()
    p_dst = pos_dst[best_pos_idx].item()
    
    plot_neuron_pair(axes[0], p_src, p_dst, 
                     f"True Positive (Real Synapse)\nModel Score: {p_score:.4f}", 
                     ['green', 'lime'])

    # Plot 2: The Hardest Negative
    # Pick the highest scoring negative
    if hard_indices.dim() == 0: hard_indices = hard_indices.unsqueeze(0)
    worst_idx = hard_indices[0] 
    
    n_score = cand_scores[worst_idx].item()
    n_src = cand_src[worst_idx].item()
    n_dst = cand_dst[worst_idx].item()
    
    plot_neuron_pair(axes[1], n_src, n_dst, 
                     f"Hard Negative (No Synapse)\nModel Score: {n_score:.4f}", 
                     ['red', 'orange'])
    
    plt.tight_layout()
    plt.savefig("hard_negative_analysis.png")
    print("\n[Success] Visualization saved to 'hard_negative_analysis.png'")
    print("Use 'ls -lh' to confirm.")

if __name__ == "__main__":
    run_analysis()