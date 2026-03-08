import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Configuration
MODEL_DIR = "./saved_models_spatial"
JSON_FILE = "metrics_cache_spatial.json"
OUTPUT_PLOT = "performance_metrics.png"

def plot_metrics():
    file_path = os.path.join(MODEL_DIR, JSON_FILE)
    
    if not os.path.exists(file_path):
        print(f"Error: Could not find {file_path}. Is the training finished?")
        return

    # Load the JSON data
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Extract the core metrics
    metrics = {
        'ROC-AUC': data['roc_auc'],
        'F1 Score': data['best_f1'],
        'Recall': data['recall'],
        'Precision': data['precision'],
        'Accuracy': data['accuracy']
    }

    names = list(metrics.keys())
    values = list(metrics.values())

    # Create the Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, values, color=['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974'])

    # Formatting
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title(f'GNN Performance: {data["graph_type"]} Graph\n(Optimal Threshold: {data["optimal_threshold"]:.2f})', 
                 fontsize=14, fontweight='bold', pad=15)
    
    # Add the exact numbers on top of the bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.02, 
                f'{yval:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add a grid for readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    # Save and show
    plt.tight_layout()
    save_path = os.path.join(MODEL_DIR, OUTPUT_PLOT)
    plt.savefig(save_path, dpi=300)
    print(f"Success! Metrics chart saved to: {save_path}")

if __name__ == "__main__":
    plot_metrics()