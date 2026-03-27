import os
import json
import matplotlib.pyplot as plt

def plot_both_sweeps():
    base_dir = "." # Scans the entire project starting from the current directory
    
    euc_results = []
    adp_results = []
    
    print(f"Scanning project directories for metrics...")
    
    # Recursively hunt through all folders (evals, saved_models, etc.)
    for root, dirs, files in os.walk(base_dir):
        for filename in files:
            # Only look at our specific metric files
            if filename.startswith("metrics_") and filename.endswith(".json"):
                filepath = os.path.join(root, filename)
                
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    
                graph_type = data.get("graph_type", "")
                thresh = data.get("spatial_threshold_nm", 0)
                auc = data.get("roc_auc", 0.0)
                
                # Filter 1: Grab the NEW Version 2 ADP metrics
                if "adp" in graph_type.lower() and "added_adp_weights" in filename:
                    adp_results.append((thresh / 1000.0, auc))
                    
                # Filter 2: Grab the Version 1 Euclidean baseline metrics
                elif "euc" in graph_type.lower():
                    euc_results.append((thresh / 1000.0, auc))

    # Sort sequentially by threshold
    euc_results.sort(key=lambda x: x[0])
    adp_results.sort(key=lambda x: x[0])
    
    print(f" -> Found {len(euc_results)} Euclidean baseline metrics.")
    print(f" -> Found {len(adp_results)} Version 2 ADP metrics.")

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    
    if euc_results:
        thresholds = [x[0] for x in euc_results]
        aucs = [x[1] for x in euc_results]
        plt.plot(thresholds, aucs, marker='o', linewidth=3, markersize=8, 
                 color='#4b0082', label='Version 1: Euclidean Baseline')
                 
    if adp_results:
        thresholds = [x[0] for x in adp_results]
        aucs = [x[1] for x in adp_results]
        plt.plot(thresholds, aucs, marker='s', linewidth=3, markersize=8, 
                 color='#d2691e', label='Version 2: Continuous ADP Weights')
        
    plt.title("Model Robustness: Euclidean vs. Continuous ADP Weights", fontsize=16, fontweight='bold')
    plt.xlabel("Spatial Quarantine Distance (Microns)", fontsize=14)
    plt.ylabel("ROC-AUC Score", fontsize=14)
    
    # Zoom in to perfectly highlight the stability band
    plt.ylim(0.70, 0.99)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right', fontsize=12)
    
    # Save to the root directory so it is incredibly easy to find
    out_path = "./final_combined_sweep_trend.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"\nSuccess! Saved the final trend showdown to: {out_path}")

if __name__ == "__main__":
    plot_both_sweeps()