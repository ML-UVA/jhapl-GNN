"""
filter_graph.py

Builds an in-memory ``networkx.DiGraph`` of the connectome from
``synapses_with_features.pt`` + ``positions.pt``, optionally restricted
to a spatial bounding box. Returns the graph directly — no intermediate
adjlist is written.

The ``--use_existing`` path reads a pre-computed edge CSV
(``data/top5_k1.csv``) instead of the .pt pipeline.

USAGE

    python -m motifs.models.filter_graph \\
        --x_min 800000 --x_max 1000000 \\
        --y_min 700000 --y_max 900000
"""

import argparse
import os

import networkx as nx
import pandas as pd
import torch


def _canonical_coords(positions_path):
    """Load ``positions.pt`` and dedupe by stripping the ``_<N>`` suffix.

    First occurrence wins. Returns ``dict[int_neuron_id, (x, y, z)]``.
    """
    data = torch.load(positions_path, weights_only=False)
    node_ids = data['node_ids']
    positions = data['positions']

    coords = {}
    for i, key in enumerate(node_ids):
        numeric_id = int(str(key).split('_')[0])
        if numeric_id not in coords:
            coords[numeric_id] = tuple(positions[i].tolist())
    return coords


def _canonical_edges(synapses_path):
    """Load ``synapses_with_features.pt`` and return ``list[(pre_int, post_int)]``."""
    data = torch.load(synapses_path, weights_only=False)
    edge_index = data['edge_index']
    node_ids = data['node_ids']
    id_table = [int(str(k).split('_')[0]) for k in node_ids]

    pre = edge_index[0].tolist()
    post = edge_index[1].tolist()
    return [(id_table[u], id_table[v]) for u, v in zip(pre, post)]


def build_graph(
    use_existing=False,
    existing_csv='data/top5_k1.csv',
    synapses_path=None,
    positions_path=None,
    x_min=None, x_max=None,
    y_min=None, y_max=None,
    z_min=None, z_max=None,
):
    from config import INTERMEDIATE_DIR
    if synapses_path is None:
        synapses_path = str(INTERMEDIATE_DIR / 'synapses_with_features.pt')
    if positions_path is None:
        positions_path = str(INTERMEDIATE_DIR / 'positions.pt')
    if use_existing:
        if not os.path.exists(existing_csv):
            raise FileNotFoundError(f"Existing CSV not found: {existing_csv}")
        df = pd.read_csv(existing_csv)
        G = nx.DiGraph()
        for _, row in df.iterrows():
            G.add_edge(int(row['pre_id']), int(row['post_id']))
        print(f"[filter_graph] Loaded existing graph: {len(G.nodes()):,} nodes, {len(G.edges()):,} edges")
        return G

    print(f"[filter_graph] Loading positions from: {positions_path}")
    coords = _canonical_coords(positions_path)
    print(f"[filter_graph] Neurons with coordinates: {len(coords):,}")

    any_threshold = any(v is not None for v in [x_min, x_max, y_min, y_max, z_min, z_max])
    if not any_threshold:
        print("\nWARNING: No thresholds specified — all neurons will be kept.")

    kept_ids = set()
    for neuron_id, (x, y, z) in coords.items():
        if x_min is not None and x < x_min: continue
        if x_max is not None and x > x_max: continue
        if y_min is not None and y < y_min: continue
        if y_max is not None and y > y_max: continue
        if z_min is not None and z < z_min: continue
        if z_max is not None and z > z_max: continue
        kept_ids.add(neuron_id)

    print(f"[filter_graph] Neurons before filter: {len(coords):,}")
    print(f"[filter_graph] Neurons after filter:  {len(kept_ids):,}")

    if len(kept_ids) == 0:
        raise ValueError("No neurons survived the spatial filter. Check your thresholds.")

    print(f"[filter_graph] Loading synapses from: {synapses_path}")
    edges = _canonical_edges(synapses_path)
    G = nx.DiGraph()
    for pre_id, post_id in edges:
        if pre_id in kept_ids and post_id in kept_ids:
            G.add_edge(pre_id, post_id)

    print(f"[filter_graph] Graph: {len(G.nodes()):,} nodes, {len(G.edges()):,} edges")
    if len(G.edges()) == 0:
        raise ValueError("No edges survived the spatial filter.")
    return G


if __name__ == '__main__':
    from config import INTERMEDIATE_DIR
    parser = argparse.ArgumentParser(description="Filter connectome graph by spatial bounding box")
    parser.add_argument('--use_existing',   action='store_true')
    parser.add_argument('--existing_csv',   type=str, default='data/top5_k1.csv')
    parser.add_argument('--synapses_path',  type=str, default=str(INTERMEDIATE_DIR / 'synapses_with_features.pt'))
    parser.add_argument('--positions_path', type=str, default=str(INTERMEDIATE_DIR / 'positions.pt'))
    parser.add_argument('--x_min', type=float, default=None)
    parser.add_argument('--x_max', type=float, default=None)
    parser.add_argument('--y_min', type=float, default=None)
    parser.add_argument('--y_max', type=float, default=None)
    parser.add_argument('--z_min', type=float, default=None)
    parser.add_argument('--z_max', type=float, default=None)
    args = parser.parse_args()
    build_graph(**vars(args))
