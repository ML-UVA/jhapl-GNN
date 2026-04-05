"""
Main pipeline script for neural network motif analysis.

This script runs the complete analysis pipeline with configurable options for
null models, metrics, and visualizations.

CONFIGURATION
=============
Edit the CONFIG section below to customize which analyses to run.
"""

import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Data paths (relative to project root)
    'data_dir': PROJECT_ROOT / 'data' / 'processed',
    'output_dir': PROJECT_ROOT / 'outputs',
    
    # Which null models to run
    'null_models': [
        'ER',
        'configuration',
        'BA',
        'smallworld',
        'spatial_null',  # Requires bin_model
    ],
    
    # Which metrics to compute
    'metrics': [
        'gini',
        'coef_variation',
        'mean_deg',
        'clustering',
        'transitivity',
        'triangles',
    ],
    
    # Which visualizations to generate
    'visualizations': [
        'motif_comparison',      # Requires: null models, triadic Census
        'subgraph',              # Requires: positions
        'metric_summary_table',  # Requires: metrics
    ],
    
    # Analysis parameters
    'n_null_samples': 10,        # Samples per null model
    'n_motif_samples': 10,       # Samples for motif analysis
    'n_bins': 20,                # Feature bins for spatial null
    'spatial_radius': 50000,     # Spatial filtering radius
    
    # Reproducibility
    'random_seed': 42,
}

# ============================================================================
# IMPORTS
# ============================================================================

from graph_io import read_json, build_synapse_digraph
from spatial_analysis import filter_neurons, build_partial_graph, decompose, plot_vis
from binning.compute_bins import compute_bins, BinModel
from null_models.wrappers import get_null_model, NULL_MODELS
from metrics.count_metrics import count_tri, generate_motif_df, plot_summary
from metrics.hub_spoke_metrics import (
    gini, coef_variation, mean_deg, max_deg, deg_assortativity
)
from metrics.clustering_metrics import clustering, transitivity, triangles
from metrics.generators import run_null_models, summarize_results
from config import N_BINS, N_NULLS, RANDOM_SEED

# ============================================================================
# METRIC REGISTRY
# ============================================================================

METRIC_FUNCTIONS = {
    'gini': gini,
    'coef_variation': coef_variation,
    'mean_deg': mean_deg,
    'max_deg': max_deg,
    'deg_assortativity': deg_assortativity,
    'clustering': clustering,
    'transitivity': transitivity,
    'triangles': triangles,
}

def get_metric(name):
    """Get a metric function by name."""
    if name not in METRIC_FUNCTIONS:
        raise KeyError(f"Unknown metric: {name}. Options: {list(METRIC_FUNCTIONS.keys())}")
    return METRIC_FUNCTIONS[name]

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Run the complete analysis pipeline."""
    
    print("=" * 80)
    print("JHU-GNN: Graph Analysis Pipeline")
    print("=" * 80)
    
    # Create output directory
    output_path = Path(CONFIG['output_dir'])
    output_path.mkdir(exist_ok=True)
    print(f"\nOutput directory: {output_path.absolute()}")
    
    # ========================================================================
    # 1. LOAD DATA
    # ========================================================================
    print("\n[1] Loading data...")
    try:
        data_path = Path(CONFIG['data_dir'])
        synapses = read_json(str(data_path / 'synapses.json'))
        positions = read_json(str(data_path / 'positions.json'))
        print(f"  ✓ Loaded {len(synapses)} synapses")
        print(f"  ✓ Loaded positions for {len(positions)} neurons")
    except Exception as e:
        print(f"  ✗ Error loading data: {e}")
        return
    
    # Build ground truth graph
    GT = build_synapse_digraph(synapses)
    print(f"  ✓ Ground truth graph: {GT.number_of_nodes()} nodes, {GT.number_of_edges()} edges")
    
    # Prepare coordinate array
    neuron_ids = list(positions.keys())
    coords = np.array([positions[nid] for nid in neuron_ids])
    print(f"  ✓ Coordinates shape: {coords.shape}")
    
    # ========================================================================
    # 2. PREPARE SPATIAL FEATURES (for spatial null model)
    # ========================================================================
    print("\n[2] Preparing spatial features...")
    
    # Compute pairwise distances
    n = coords.shape[0]
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(coords[i] - coords[j])
            distances[i, j] = dist
            distances[j, i] = dist
    print(f"  ✓ Distance matrix computed: {distances.shape}")
    
    # Create edge indicator and feature list
    edge_list = []
    edge_indicator = []
    feature_list = []
    
    for i, u in enumerate(neuron_ids):
        for j, v in enumerate(neuron_ids):
            if i < j:
                has_edge = GT.has_edge(u, v)
                edge_list.append((u, v))
                edge_indicator.append(1 if has_edge else 0)
                feature_list.append(distances[i, j])
    
    edge_indicator = np.array(edge_indicator)
    feature_list = np.array(feature_list)
    print(f"  ✓ Edge indicator: {len(edge_list)} pairs, {edge_indicator.sum()} edges")
    print(f"  ✓ Feature range: [{feature_list.min():.2f}, {feature_list.max():.2f}]")
    
    # ========================================================================
    # 3. COMPUTE BINNING MODEL (for spatial null)
    # ========================================================================
    if 'spatial_null' in CONFIG['null_models']:
        print("\n[3] Computing binning model...")
        try:
            bin_model = compute_bins(
                feature_list,
                edge_indicator,
                n_bins=CONFIG['n_bins'],
                method='quantile'
            )
            print(f"  ✓ Binning model created with {len(bin_model.bin_probs)} bins")
        except Exception as e:
            print(f"  ✗ Error creating binning model: {e}")
            bin_model = None
    else:
        bin_model = None
    
    # ========================================================================
    # 4. COMPUTE GROUND TRUTH METRICS
    # ========================================================================
    print("\n[4] Computing ground truth metrics...")
    gt_metrics = {}
    for metric_name in CONFIG['metrics']:
        try:
            metric_fn = get_metric(metric_name)
            gt_metrics[metric_name] = metric_fn(GT)
            print(f"  ✓ {metric_name}: {gt_metrics[metric_name]:.4f}")
        except Exception as e:
            print(f"  ✗ Error computing {metric_name}: {e}")
    
    # ========================================================================
    # 5. MOTIF ANALYSIS
    # ========================================================================
    if 'motif_comparison' in CONFIG['visualizations']:
        print("\n[5] Computing triadic motifs...")
        try:
            # Get null model functions for motif comparison
            null_fns = [get_null_model(name) for name in CONFIG['null_models'] 
                       if name != 'spatial_null']  # Skip spatial for motif (no interface match yet)
            
            motif_summary = generate_motif_df(
                GT,
                null_fns,
                n=CONFIG['n_motif_samples']
            )
            
            # Save motif summary
            motif_path = output_path / 'motif_summary.csv'
            motif_summary.to_csv(motif_path)
            print(f"  ✓ Motif summary saved to {motif_path}")
            
            # Plot motif comparison
            motif_plot_path = output_path / 'motif_comparison.png'
            plt.figure(figsize=(14, 6))
            plot_summary(motif_summary, null_fns)
            plt.savefig(motif_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Motif plot saved to {motif_plot_path}")
            
        except Exception as e:
            print(f"  ✗ Error in motif analysis: {e}")
    
    # ========================================================================
    # 6. NULL MODEL METRICS COMPARISON
    # ========================================================================
    print("\n[6] Running null models and metrics...")
    try:
        # Get metric functions
        metric_fns = [get_metric(name) for name in CONFIG['metrics']]
        
        # Get null model functions (exclude spatial for now)
        null_fns = [get_null_model(name) for name in CONFIG['null_models'] 
                   if name != 'spatial_null']
        
        if null_fns:
            results = run_null_models(null_fns, metric_fns, GT, N=CONFIG['n_null_samples'])
            summary = summarize_results(GT, results, metric_fns)
            
            # Save summary
            summary_path = output_path / 'metric_summary.csv'
            summary.to_csv(summary_path, index=False)
            print(f"  ✓ Metric summary saved to {summary_path}")
    except Exception as e:
        print(f"  ✗ Error running null models: {e}")
    
    # ========================================================================
    # 7. VISUALIZATION: SUBGRAPH
    # ========================================================================
    if 'subgraph' in CONFIG['visualizations']:
        print("\n[7] Creating subgraph visualization...")
        try:
            # Filter neurons by spatial proximity
            sub_neurons, sub_coords = filter_neurons(
                neuron_ids, coords, CONFIG['spatial_radius']
            )
            
            if len(sub_neurons) > 0:
                # Build subgraph edges
                edges_as_tuples = [(u, v) for u, v in GT.edges()]
                sub_edges = build_partial_graph(sub_neurons, edges_as_tuples)
                
                # Decompose to 2D
                decomp = decompose(sub_coords)
                
                # Plot
                plt.figure(figsize=(8, 8))
                plot_vis(sub_neurons, sub_edges, decomp)
                subgraph_path = output_path / 'subgraph_visualization.png'
                plt.savefig(subgraph_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"  ✓ Subgraph visualization saved to {subgraph_path}")
                print(f"    Filtered to {len(sub_neurons)} neurons within radius {CONFIG['spatial_radius']}")
        except Exception as e:
            print(f"  ✗ Error in subgraph visualization: {e}")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("Pipeline complete!")
    print("=" * 80)
    print(f"\nOutput files in: {output_path.absolute()}")
    print("\nFiles generated:")
    for f in sorted(output_path.glob('*')):
        print(f"  - {f.name}")
    print()

if __name__ == '__main__':
    main()
