"""
Entry point for the null_analysis motif pipeline.

Runs the complete analysis (null models, metrics, visualizations) when invoked
as `python -m null_analysis` from the project root. Edit the CONFIG dict below
to customize which analyses to run.
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
        # 'subgraph',            # Requires: positions (load separately if needed)
        # 'metric_summary_table',  # Requires: metrics
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
        description='JHU-GNN: Null Model Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(f'''
        Examples:
          Run with default data paths (synapses + distance_graph from data/processed/):
            python3 -m null_analysis
          
          Use custom files:
            python3 -m null_analysis --synapses custom_synapses.pt --distance-graph custom_graph.pt
          
          Use paths relative to project root:
            python3 -m null_analysis --synapses data/demo/demo_synapses.pt --distance-graph data/demo/demo_graph.pt
          
          Note: Pre-compute distance graph first using:
            python -m data_prep.compute_distance_graph --positions data/processed/positions.pt \\
              --output data/processed/distance_graph.pt
        ''')
    )
    
    parser.add_argument(
        '--synapses', '-s',
        type=str,
        default=None,
        help='Path to synapses .pt file (default: data/processed/synapses.pt)'
    )
    
    parser.add_argument(
        '--distance-graph', '-g',
        type=str,
        default=None,
        help='Path to pre-computed distance_graph.pt file (default: data/processed/distance_graph.pt)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output directory (default: outputs/)'
    )
    
    return parser.parse_args()

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Run the complete analysis pipeline."""
    
    # Parse command-line arguments
    args = parse_arguments()
    
    print("=" * 80)
    print("JHU-GNN: Null Model Analysis Pipeline")
    print("=" * 80)
    
    # ========================================================================
    # HANDLE CUSTOM DATA PATHS
    # ========================================================================
    
    # Resolve synapse and distance graph file paths
    if args.synapses:
        synapses_path = Path(args.synapses)
        # If relative path, resolve relative to project root
        if not synapses_path.is_absolute():
            synapses_path = PROJECT_ROOT / synapses_path
    else:
        synapses_path = Path(CONFIG['data_dir']) / 'synapses.pt'
    
    if args.distance_graph:
        distance_graph_path = Path(args.distance_graph)
        # If relative path, resolve relative to project root
        if not distance_graph_path.is_absolute():
            distance_graph_path = PROJECT_ROOT / distance_graph_path
    else:
        distance_graph_path = Path(CONFIG['data_dir']) / 'distance_graph.pt'
    
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
    print(f"Distance graph file: {distance_graph_path.absolute()}")
    
    # ========================================================================
    # 1. LOAD DATA
    # ========================================================================
    print("\n[1] Loading data...")
    try:
        synapses = load_synapses_from_pt(str(synapses_path))
        print(f"  ✓ Loaded {len(synapses)} synapses")
    except Exception as e:
        print(f"  ✗ Error loading synapses: {e}")
        return
    
    # Build ground truth graph from synapses
    GT = build_synapse_digraph(synapses)
    print(f"  ✓ Ground truth graph: {GT.number_of_nodes()} nodes, {GT.number_of_edges()} edges")
    
    # Load pre-computed distance graph (contains all spatial information)
    try:
        import torch
        graph_data = torch.load(distance_graph_path, weights_only=False)
        distance_nodes = graph_data['nodes']
        print(f"  ✓ Loaded distance graph: {len(distance_nodes)} neurons, {len(graph_data['edges'])} distance pairs")
    except Exception as e:
        print(f"  ✗ Error loading distance graph: {e}")
        return
    
    # ========================================================================
    # 2. PREPARE SPATIAL FEATURES (for spatial null model)
    # ========================================================================
    # ========================================================================
    # 2. PREPARE SPATIAL FEATURES FOR NULL MODELS
    # ========================================================================
    print("\n[2] Preparing pair features from distance graph...")
    
    # Extract edge weights (distances) from pre-computed distance graph
    # These are used for spatial null model binning
    edge_list = []  # (u, v) pairs
    edge_indicator = []  # 1 if edge exists in GT, 0 otherwise
    feature_list = []  # distance values
    
    for (u, v), dist in zip(graph_data['edges'], graph_data['edge_weights']):
        
        # Check if edge exists in empirical graph (both directions)
        has_edge = GT.has_edge(u, v) or GT.has_edge(v, u)
        
        edge_list.append((u, v))
        edge_indicator.append(1 if has_edge else 0)
        feature_list.append(float(dist))
    
    edge_indicator = np.array(edge_indicator)
    feature_list = np.array(feature_list)
    
    print(f"  ✓ Extracted {len(edge_list)} distance pairs")
    print(f"  ✓ {edge_indicator.sum()} pairs have edges in empirical graph")
    print(f"  ✓ Distance range: [{np.min(feature_list):.2f}, {np.max(feature_list):.2f}]")
    
    # ========================================================================
    # 3. COMPUTE BINNING MODEL (for spatial null)
    # ========================================================================
    # Binning model learns P(edge | distance_bin) from empirical graph
    # Used to generate spatial null models
    if 'spatial_null' in CONFIG['null_models']:
        print("\n[3] Computing binning model for spatial null...")
        try:
            bin_model = compute_bins(
                feature_list,
                edge_indicator,
                n_bins=CONFIG['n_bins'],
                method='quantile'
            )
            print(f"  ✓ Binning model: {len(bin_model.bin_probs)} bins")
            # Print bin statistics
            for i, (prob, center) in enumerate(zip(bin_model.bin_probs, bin_model.bin_centers)):
                print(f"    Bin {i}: P(edge|bin)={prob:.3f}, feature_center={center:.2f}")
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
                    wrapped_spatial_null = lambda G, sp_null=spatial_null_fn, bm=bin_model, pf=pair_features: sp_null(G, bm, pf)
                    wrapped_spatial_null.__name__ = 'spatial_null'
                    null_fns.append(wrapped_spatial_null)
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
    # 8. EXPORT RESULTS
    # ========================================================================
    # Summary of results exported to output directory:
    # 
    # - metric_summary.csv: Quantitative comparison of ground truth vs. null models
    # - motif_summary.csv: Triadic motif count (if motif_comparison enabled)
    # - *_visualization.png: Plots for motif comparison and subgraph views
    #
    print("\n[8] Results summary")
    print("  Analysis complete. Results saved to output directory:")
    
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
