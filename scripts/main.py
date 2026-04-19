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
import argparse
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path

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
    'n_bins': 5,                # Feature bins for spatial null
    'spatial_radius': 50000,     # Spatial filtering radius
    'distance_metric': 'euclidean',  # 'euclidean' or 'adp'
    
    # Reproducibility
    'random_seed': 42,
}

# ============================================================================
# IMPORTS
# ============================================================================

from data_prep.graph_io import (
    load_synapses_from_pt, load_positions_from_pt,
    build_synapse_digraph, export_graph_to_pt, export_positions_to_pt
)
from data_prep.spatial_analysis import filter_neurons, build_partial_graph, decompose, plot_vis
from null_analysis.binning.compute_bins import compute_bins, BinModel
from null_analysis.null_models.wrappers import get_null_model, NULL_MODELS
from null_analysis.metrics.count_metrics import count_tri, generate_motif_df, plot_summary
from null_analysis.metrics.hub_spoke_metrics import (
    gini, coef_variation, mean_deg, max_deg, deg_assortativity
)
from null_analysis.metrics.clustering_metrics import clustering, transitivity, triangles
from null_analysis.metrics.generators import run_null_models, summarize_results
from null_analysis.config import N_BINS, N_NULLS, RANDOM_SEED

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
# COMMAND-LINE ARGUMENTS
# ============================================================================

def parse_arguments():
    """Parse command-line arguments for custom data paths."""
    parser = argparse.ArgumentParser(
        description='JHU-GNN: Graph Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(f'''
        Examples:
          Run with default data paths:
            python3 main.py
          
          Use custom synapse and positions .pt files:
            python3 main.py --synapses custom_synapses.pt --positions custom_positions.pt
          
          Use paths relative to project root:
            python3 main.py --synapses data/demo/demo_synapses.pt --positions data/demo/demo_positions.pt
        ''')
    )
    
    parser.add_argument(
        '--synapses', '-s',
        type=str,
        default=None,
        help='Path to synapses .pt file (default: data/processed/synapses.pt)'
    )
    
    parser.add_argument(
        '--positions', '-p',
        type=str,
        default=None,
        help='Path to positions .pt file (default: data/processed/positions.pt)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output directory (default: outputs/)'
    )
    
    parser.add_argument(
        '--distance-metric', '-d',
        type=str,
        default=None,
        choices=['euclidean', 'adp'],
        help='Distance metric for binning: euclidean or adp (default: euclidean)'
    )
    
    return parser.parse_args()

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Run the complete analysis pipeline."""
    
    # Parse command-line arguments
    args = parse_arguments()
    
    # Apply command-line overrides to CONFIG
    if args.distance_metric:
        CONFIG['distance_metric'] = args.distance_metric
    
    print("=" * 80)
    print("JHU-GNN: Graph Analysis Pipeline")
    print("=" * 80)
    
    # ========================================================================
    # HANDLE CUSTOM DATA PATHS
    # ========================================================================
    
    # Resolve synapse and position file paths
    if args.synapses:
        synapses_path = Path(args.synapses)
        # If relative path, resolve relative to project root
        if not synapses_path.is_absolute():
            synapses_path = PROJECT_ROOT / synapses_path
    else:
        synapses_path = Path(CONFIG['data_dir']) / 'synapses.pt'
    
    if args.positions:
        positions_path = Path(args.positions)
        # If relative path, resolve relative to project root
        if not positions_path.is_absolute():
            positions_path = PROJECT_ROOT / positions_path
    else:
        positions_path = Path(CONFIG['data_dir']) / 'positions.pt'
    
    # Override output directory if provided
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
    else:
        output_path = Path(CONFIG['output_dir'])
    
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_path.absolute()}")
    print(f"Synapses file: {synapses_path.absolute()}")
    print(f"Positions file: {positions_path.absolute()}")
    
    # ========================================================================
    # 1. LOAD DATA
    # ========================================================================
    print("\n[1] Loading data...")
    try:
        synapses = load_synapses_from_pt(str(synapses_path))
        positions_data = load_positions_from_pt(str(positions_path))
        print(f"  ✓ Loaded {len(synapses)} synapses")
        print(f"  ✓ Loaded positions for {len(positions_data)} neurons")
    except Exception as e:
        print(f"  ✗ Error loading data: {e}")
        return
    
    # Build ground truth graph
    GT = build_synapse_digraph(synapses)
    print(f"  ✓ Ground truth graph: {GT.number_of_nodes()} nodes, {GT.number_of_edges()} edges")
    
    # Prepare coordinate array
    neuron_ids = list(positions_data.keys())
    coords = np.array([positions_data[nid] for nid in neuron_ids])
    print(f"  ✓ Coordinates shape: {coords.shape}")
    
    # ========================================================================
    # 2. PREPARE SPATIAL FEATURES (for spatial null model)
    # ========================================================================
    print("\n[2] Preparing spatial features...")
    
    # Load or compute distance matrix and feature list
    if CONFIG['distance_metric'] == 'adp':
        import pickle
        adp_path = Path(CONFIG['data_dir']) / 'adp_data.pkl'
        try:
            with open(adp_path, 'rb') as f:
                adp_data = pickle.load(f)
            
            # Extract ALL nodes from ADP data (much larger than neuron_ids)
            all_adp_nodes = set(adp_data.keys())
            for nbrs in adp_data.values():
                all_adp_nodes |= set(nbrs.keys())
            all_adp_nodes = list(all_adp_nodes)
            print(f"  ✓ Extracted {len(all_adp_nodes)} nodes from ADP data")
            
            # Build feature_list and edge_indicator from ALL ADP pairs
            # This matches the notebook's approach for proper probability calibration
            edge_list = []
            edge_indicator = []
            feature_list = []
            
            for i, u in enumerate(all_adp_nodes):
                for j, v in enumerate(all_adp_nodes):
                    if i < j:
                        # Get ADP value (check both directions)
                        adp_val = None
                        if u in adp_data and v in adp_data[u]:
                            adp_val = adp_data[u][v]
                        elif v in adp_data and u in adp_data[v]:
                            adp_val = adp_data[v][u]
                        
                        # Only include if ADP value exists
                        if adp_val is not None:
                            edge_list.append((u, v))
                            # Check if edge exists in GT (check both directions since GT is directed)
                            has_edge = GT.has_edge(u, v) or GT.has_edge(v, u)
                            edge_indicator.append(1 if has_edge else 0)
                            feature_list.append(adp_val)
            
            edge_indicator = np.array(edge_indicator)
            feature_list = np.array(feature_list)
            print(f"  ✓ ADP feature list: {len(edge_list)} pairs, {edge_indicator.sum()} edges")
            print(f"  ✓ Feature range: [{feature_list.min():.2f}, {feature_list.max():.2f}]")
            
            # Store all_adp_nodes and feature info for spatial null
            spatial_null_nodes = all_adp_nodes
            
        except Exception as e:
            print(f"  ✗ Error loading ADP data: {e}. Falling back to euclidean.")
            CONFIG['distance_metric'] = 'euclidean'
            spatial_null_nodes = None
    
    # Fallback to euclidean if ADP not selected or failed
    if CONFIG['distance_metric'] == 'euclidean':
        # Compute pairwise euclidean distances
        n = len(neuron_ids)
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(coords[i] - coords[j])
                distances[i, j] = dist
                distances[j, i] = dist
        
        # Create edge indicator and feature list from neuron_ids only
        edge_list = []
        edge_indicator = []
        feature_list = []
        
        for i, u in enumerate(neuron_ids):
            for j, v in enumerate(neuron_ids):
                if i < j:
                    # Check both directions since GT is a directed graph
                    has_edge = GT.has_edge(u, v) or GT.has_edge(v, u)
                    edge_list.append((u, v))
                    edge_indicator.append(1 if has_edge else 0)
                    feature_list.append(distances[i, j])
        
        edge_indicator = np.array(edge_indicator)
        feature_list = np.array(feature_list)
        print(f"  ✓ Euclidean: examined {len(edge_list)} pair combinations")
        print(f"  ✓ Edge indicator: {edge_indicator.sum()} edges out of {len(edge_list)} pairs")
        print(f"  ✓ Feature range: [{feature_list.min():.2f}, {feature_list.max():.2f}]")
        spatial_null_nodes = None
    
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
            # Convert to undirected graph for triangles metric (not implemented for directed)
            if metric_name == 'triangles':
                gt_metrics[metric_name] = metric_fn(GT.to_undirected())
            else:
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
            # Build null model functions for motif comparison
            # Convert spatial_null output to directed using to_directed()
            motif_null_fns = []
            for model_name in CONFIG['null_models']:
                if model_name == 'spatial_null':
                    # Convert spatial null (undirected) to directed for motif analysis
                    spatial_null_fn = get_null_model(model_name)
                    if bin_model:
                        # Use prepared pair_features (includes all ADP pairs if ADP metric)
                        pair_features = []
                        for idx, (u, v) in enumerate(edge_list):
                            pair_features.append((u, v, feature_list[idx]))
                        # Wrapper converts undirected spatial_null to directed
                        wrapped_spatial_null = lambda G, sp_null=spatial_null_fn, bm=bin_model, pf=pair_features: sp_null(G, bm, pf).to_directed()
                        wrapped_spatial_null.__name__ = 'spatial_null'
                        motif_null_fns.append(wrapped_spatial_null)
                else:
                    motif_null_fns.append(get_null_model(model_name))
            
            motif_summary = generate_motif_df(
                GT,
                motif_null_fns,
                n=CONFIG['n_motif_samples']
            )
            
            # Save motif summary
            motif_path = output_path / 'motif_summary.csv'
            motif_summary.to_csv(motif_path)
            print(f"  ✓ Motif summary saved to {motif_path}")
            
            # Plot motif comparison
            motif_plot_path = output_path / 'motif_comparison.png'
            plt.figure(figsize=(18, 7))
            plot_summary(motif_summary, motif_null_fns)
            plt.tight_layout()
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
        
        # Build null model functions list
        null_fns = []
        for model_name in CONFIG['null_models']:
            if model_name == 'spatial_null':
                # Create a wrapper that binds bin_model and pair_features to spatial null
                spatial_null_fn = get_null_model(model_name)
                if bin_model:
                    # Use already-prepared pair_features (from ADP or euclidean)
                    # For ADP: includes all available pairs from ADP data
                    # For euclidean: includes all neuron_id pairs
                    pair_features = []
                    for idx, (u, v) in enumerate(edge_list):
                        pair_features.append((u, v, feature_list[idx]))
                    wrapped_spatial_null = lambda G, sp_null=spatial_null_fn, bm=bin_model, pf=pair_features: sp_null(G, bm, pf)
                    wrapped_spatial_null.__name__ = 'spatial_null'
                    null_fns.append(wrapped_spatial_null)
                else:
                    print(f"  ⚠ Skipping spatial_null: bin_model not available")
            else:
                null_fns.append(get_null_model(model_name))
        
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
    # 8. EXPORT RESULTS TO PYTORCH FORMAT
    # ========================================================================
    print("\n[8] Exporting results to PyTorch format...")
    try:
        # Export ground truth graph
        gt_export_path = output_path / 'gt_graph.pt'
        positions_dict = {nid: positions_data[nid] for nid in neuron_ids}
        export_graph_to_pt(GT, gt_export_path, node_positions=positions_dict)
        
        # Export positions separately
        positions_export_path = output_path / 'positions.pt'
        export_positions_to_pt(positions_dict, positions_export_path)
        
        print(f"  ✓ PyTorch exports saved to output directory")
    except Exception as e:
        print(f"  ✗ Error exporting to PyTorch format: {e}")
    
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
    
    # Print metric summary if available
    summary_path = output_path / 'metric_summary.csv'
    if summary_path.exists():
        print("\n" + "=" * 80)
        print("Metric Summary")
        print("=" * 80)
        df = pd.read_csv(summary_path)
        df = df.round(6)
        # Remove stdev columns
        df = df.drop(columns=[col for col in df.columns if 'stdev' in col])
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.width', None)
        print(df.to_string(index=False))
        pd.reset_option('display.max_columns')
        pd.reset_option('display.max_colwidth')
        pd.reset_option('display.width')
    print()

if __name__ == '__main__':
    main()
