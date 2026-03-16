import os
import json
import matplotlib.pyplot as plt

def plot_both_sweeps():
    model_dir = "./saved_models_spatial"
    
    euc_results = []
    adp_results = []
    
    print(f"Scanning {model_dir} for metrics...")
    
    for filename in os.listdir(model_dir):
        if not filename.endswith(".json"): continue
            
        filepath = os.path.join(model_dir, filename)
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        graph_type = data.get("graph_type", "")
        thresh = data.get("spatial_threshold_nm", 0)
        auc = data.get("roc_auc", 0.0)
        
        # Sort the results into their respective lists, converting nm to microns
        if "adp" in graph_type.lower():
            adp_results.append((thresh / 1000.0, auc))
        elif "euc" in graph_type.lower():
            euc_results.append((thresh / 1000.0, auc))

    # Sort sequentially by threshold
    euc_results.sort(key=lambda x: x[0])
    adp_results.sort(key=lambda x: x[0])
    
    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    
    if euc_results:
        thresholds = [x[0] for x in euc_results]
        aucs = [x[1] for x in euc_results]
        plt.plot(thresholds, aucs, marker='o', linewidth=3, markersize=8, 
                 color='#4b0082', label='Euclidean Graph (Distance)')
                 
    if adp_results:
        thresholds = [x[0] for x in adp_results]
        aucs = [x[1] for x in adp_results]
        plt.plot(thresholds, aucs, marker='s', linewidth=3, markersize=8, 
                 color='#d2691e', label='ADP Graph (Physical Touch)')
        
    plt.title("Model Robustness: Euclidean vs. ADP Connectivity", fontsize=16, fontweight='bold')
    plt.xlabel("Spatial Quarantine Distance (Microns)", fontsize=14)
    plt.ylabel("ROC-AUC Score", fontsize=14)
    
    # Zoom in to perfectly highlight the 0.75 - 0.85 stability band
    plt.ylim(0.70, 0.85)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right', fontsize=12)
    
    out_path = os.path.join(model_dir, "final_combined_sweep_trend.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"Success! Saved combined trend plot to {out_path}")

if __name__ == "__main__":
    plot_both_sweeps()