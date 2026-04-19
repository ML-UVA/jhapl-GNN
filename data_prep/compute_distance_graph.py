"""
Compute and save distance graphs between neurons.

A distance graph is an undirected weighted graph where:
- Nodes are neuron IDs
- Edges connect all neuron pairs (complete graph)
- Edge weights are distances between neurons

The distance metric can be any feature: euclidean distance, ADP distance, etc.
The distance graph is used as input to spatial null models and should be pre-computed
and saved before running the analysis pipeline.

Usage as module:
    from data_prep.compute_distance_graph import compute_distance_graph
    distance_graph = compute_distance_graph(synapses, positions, metric='euclidean')

Usage as script:
    python -m data_prep.compute_distance_graph \\
        --synapses data/processed/synapses.pt \\
        --positions data/processed/positions.pt \\
        --output data/processed/distance_graph.pt \\
        --metric euclidean
"""

import torch
import numpy as np
import networkx as nx
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse


def load_positions_from_pt(filename: str) -> Dict:
    """Load neuron positions from PyTorch .pt file."""
    from .graph_io import load_positions_from_pt
    return load_positions_from_pt(filename)


def load_synapses_from_pt(filename: str) -> Dict:
    """Load synapses from PyTorch .pt file."""
    from .graph_io import load_synapses_from_pt
    return load_synapses_from_pt(filename)


def compute_euclidean_distances(
    neuron_ids: List,
    positions: Dict
) -> Tuple[np.ndarray, List]:
    """
    Compute pairwise Euclidean distances between neurons.

    Parameters
    ----------
    neuron_ids : list
        List of neuron IDs.
    positions : dict
        Dict mapping neuron_id -> [x, y, z] coordinates.

    Returns
    -------
    distances : np.ndarray
        (n_neurons, n_neurons) symmetric distance matrix
    edge_list : list of tuples
        List of (neuron_i, neuron_j, distance) for all pairs i < j
    """
    n = len(neuron_ids)
    distances = np.zeros((n, n))
    edge_list = []

    coords = np.array([positions[nid] for nid in neuron_ids])

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(coords[i] - coords[j])
            distances[i, j] = dist
            distances[j, i] = dist
            edge_list.append((neuron_ids[i], neuron_ids[j], dist))

    return distances, edge_list


def compute_adp_distances(
    neuron_ids: List,
    adp_data: Dict
) -> Tuple[np.ndarray, List]:
    """
    Compute pairwise ADP distances between neurons.

    Parameters
    ----------
    neuron_ids : list
        List of neuron IDs to include.
    adp_data : dict
        ADP distance dictionary loaded from adp_data.pkl

    Returns
    -------
    distances : np.ndarray
        (n_neurons, n_neurons) symmetric distance matrix, NaN where no ADP value
    edge_list : list of tuples
        List of (neuron_i, neuron_j, distance) for all pairs with ADP values
    """
    n = len(neuron_ids)
    distances = np.full((n, n), np.nan)
    edge_list = []

    for i in range(n):
        for j in range(i + 1, n):
            u, v = neuron_ids[i], neuron_ids[j]

            # Try both directions
            adp_val = None
            if u in adp_data and v in adp_data[u]:
                adp_val = adp_data[u][v]
            elif v in adp_data and u in adp_data[v]:
                adp_val = adp_data[v][u]

            if adp_val is not None:
                distances[i, j] = adp_val
                distances[j, i] = adp_val
                edge_list.append((u, v, adp_val))

    return distances, edge_list


def build_distance_graph(
    neuron_ids: List,
    edge_list: List[Tuple]
) -> nx.Graph:
    """
    Build a NetworkX graph from distance edges.

    Parameters
    ----------
    neuron_ids : list
        List of all neuron IDs (nodes).
    edge_list : list of tuples
        List of (neuron_i, neuron_j, distance) tuples.

    Returns
    -------
    G : nx.Graph
        Undirected weighted graph with 'distance' attribute on edges.
    """
    G = nx.Graph()
    G.add_nodes_from(neuron_ids)

    for u, v, dist in edge_list:
        G.add_edge(u, v, distance=dist)

    return G


def compute_distance_graph(
    positions: Dict,
    metric: str = 'euclidean',
    adp_data: Optional[Dict] = None,
    verbose: bool = True
) -> nx.Graph:
    """
    Compute a complete distance graph between all neurons.

    This is a pre-processing step: the distance graph is computed once and saved,
    then loaded and used by the analysis pipeline. It contains all spatial/distance
    information needed by null models, so positions data is not needed by the main
    pipeline.

    Parameters
    ----------
    positions : dict
        Dict mapping neuron_id -> [x, y, z] coordinates.
    metric : str, default 'euclidean'
        Distance metric: 'euclidean' or 'adp'.
    adp_data : dict, optional
        ADP data dictionary (required if metric='adp'). Loaded from adp_data.pkl.
    verbose : bool
        Print progress messages.

    Returns
    -------
    G : nx.Graph
        Complete undirected weighted graph where edge weights are distances.
        Nodes are neuron IDs, edges have 'distance' attribute.
    """
    if verbose:
        print("=" * 80)
        print(f"Computing distance graph (metric: {metric})")
        print("=" * 80)

    neuron_ids = list(positions.keys())
    print(f"\n  Neurons: {len(neuron_ids)}")

    if metric == 'euclidean':
        if verbose:
            print(f"  Computing Euclidean distances...")
        distances, edge_list = compute_euclidean_distances(neuron_ids, positions)
        print(f"  ✓ Computed {len(edge_list)} pairwise distances")
        print(f"  ✓ Distance range: [{np.min(distances[distances > 0]):.2f}, {np.max(distances):.2f}]")

    elif metric == 'adp':
        if adp_data is None:
            raise ValueError("ADP data required for metric='adp'")
        if verbose:
            print(f"  Computing ADP distances...")
        distances, edge_list = compute_adp_distances(neuron_ids, adp_data)
        valid_count = np.isfinite(distances).sum() // 2
        print(f"  ✓ Found {len(edge_list)} ADP distances")
        valid_vals = distances[np.isfinite(distances)]
        if len(valid_vals) > 0:
            print(f"  ✓ ADP range: [{valid_vals.min():.2f}, {valid_vals.max():.2f}]")

    else:
        raise ValueError(f"Unknown metric: {metric}. Options: 'euclidean', 'adp'")

    # Build graph
    G = build_distance_graph(neuron_ids, edge_list)
    print(f"\n  ✓ Distance graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    return G


def save_distance_graph(
    G: nx.Graph,
    output_path: Path,
    verbose: bool = True
) -> None:
    """
    Save distance graph to PyTorch .pt file.

    Parameters
    ----------
    G : nx.Graph
        Distance graph to save.
    output_path : Path or str
        Output .pt file path.
    verbose : bool
        Print progress messages.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to serializable format
    graph_data = {
        'nodes': list(G.nodes()),
        'edges': [],
        'edge_weights': []
    }

    for u, v, dist in G.edges(data='distance', default=1.0):
        graph_data['edges'].append((u, v))
        graph_data['edge_weights'].append(float(dist) if dist is not None else 1.0)

    graph_data['edges'] = [tuple(e) for e in graph_data['edges']]

    torch.save(graph_data, output_path)

    if verbose:
        print(f"\n✓ Distance graph saved to {output_path}")
        print(f"  Nodes: {len(graph_data['nodes'])}")
        print(f"  Edges: {len(graph_data['edges'])}")


def load_distance_graph(graph_path: Path, verbose: bool = False) -> nx.Graph:
    """
    Load distance graph from PyTorch .pt file.

    Parameters
    ----------
    graph_path : Path or str
        Path to saved distance graph .pt file.
    verbose : bool
        Print progress messages.

    Returns
    -------
    G : nx.Graph
        Reconstructed distance graph.
    """
    graph_path = Path(graph_path)
    graph_data = torch.load(graph_path, weights_only=False)

    G = nx.Graph()
    G.add_nodes_from(graph_data['nodes'])

    for (u, v), dist in zip(graph_data['edges'], graph_data['edge_weights']):
        G.add_edge(u, v, distance=dist)

    if verbose:
        print(f"Loaded distance graph from {graph_path}")
        print(f"  Nodes: {G.number_of_nodes()}")
        print(f"  Edges: {G.number_of_edges()}")

    return G


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """Command-line interface for computing distance graphs."""
    parser = argparse.ArgumentParser(
        description='Compute and save distance graphs for neural network analysis.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m data_prep.compute_distance_graph \\
    --synapses data/processed/synapses.pt \\
    --positions data/processed/positions.pt \\
    --output data/processed/distance_graph_euclidean.pt \\
    --metric euclidean
    
  # or with ADP metric
  python -m data_prep.compute_distance_graph \\
    --synapses data/processed/synapses.pt \\
    --positions data/processed/positions.pt \\
    --output data/processed/distance_graph_adp.pt \\
    --metric adp
        """
    )

    parser.add_argument(
        '--synapses', '-s',
        required=True,
        help='Path to synapses.pt file'
    )
    parser.add_argument(
        '--positions', '-p',
        required=True,
        help='Path to positions.pt file'
    )
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output path for distance_graph.pt'
    )
    parser.add_argument(
        '--metric', '-m',
        default='euclidean',
        choices=['euclidean', 'adp'],
        help='Distance metric (default: euclidean)'
    )
    parser.add_argument(
        '--adp-data',
        help='Path to adp_data.pkl (required if metric=adp)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )

    args = parser.parse_args()

    # Load data
    positions = load_positions_from_pt(args.positions)

    adp_data = None
    if args.metric == 'adp':
        if not args.adp_data:
            raise ValueError("--adp-data required when metric=adp")
        import pickle
        with open(args.adp_data, 'rb') as f:
            adp_data = pickle.load(f)

    # Compute distance graph
    G = compute_distance_graph(
        positions,
        metric=args.metric,
        adp_data=adp_data,
        verbose=args.verbose
    )

    # Save
    save_distance_graph(G, args.output, verbose=args.verbose)


if __name__ == '__main__':
    main()
