"""
motifs package entry point.

Run from the repository root:

    python -m motifs --use_existing

Orchestrates the full connectome GNN pipeline:

  1. filter_graph.build_graph                 (synapses_with_features.pt + positions.pt -> nx.DiGraph)
  2. graphnodeshapley.train_and_compute_shapley   (GAE + In-Run Data Shapley)
  3. normalize.normalize_shapley
  4. motifs.extract_and_visualize
"""

import argparse
import os

from .models.filter_graph import build_graph
from .models.graphnodeshapley import train_and_compute_shapley
from .models.normalize import normalize_shapley
from .models.motifs import extract_and_visualize

parser = argparse.ArgumentParser(description="Connectome GNN Pipeline")

parser.add_argument('--use_existing',   action='store_true',
                    help='Use existing CSV instead of building a new graph')
parser.add_argument('--existing_csv',   type=str, default='data/top5_k1.csv')
parser.add_argument('--synapses_path',  type=str, default='data/processed/synapses_with_features.pt')
parser.add_argument('--positions_path', type=str, default='data/processed/positions.pt')
parser.add_argument('--config',         type=str, default='synapse_gnn/config.json',
                    help='Config with raw_data.neurons_directory; used to regenerate .pt files if missing')
parser.add_argument('--x_min',          type=float, default=None)
parser.add_argument('--x_max',          type=float, default=None)
parser.add_argument('--y_min',          type=float, default=None)
parser.add_argument('--y_max',          type=float, default=None)
parser.add_argument('--z_min',          type=float, default=None)
parser.add_argument('--z_max',          type=float, default=None)

# gae parameters
parser.add_argument('--feature_path',   type=str, default='data/neuron_features.pt')
parser.add_argument('--variational',    action='store_true')
parser.add_argument('--linear',         action='store_true')
parser.add_argument('--epochs',         type=int,   default=200)
parser.add_argument('--latent_dim',     type=int,   default=16)
parser.add_argument('--lr',             type=float, default=0.01)
parser.add_argument('--save_interval',  type=int,   default=50)

# motifs
parser.add_argument('--motif_sizes',    type=int, nargs='+', default=[5, 10, 20])
parser.add_argument('--top_k',          type=int, default=3)

# output
parser.add_argument('--output_dir',     type=str, default='results')

args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

shapley_path = os.path.join(args.output_dir, 'graph_node_shapley.value')
norm_path    = os.path.join(args.output_dir, 'graph_node_shapley_normalized.value')


def _ensure_synapses_pt(path, config_path):
    if os.path.exists(path):
        return
    print(f"[motifs] {path} not found — regenerating from raw .pbz2 exports...")
    from data_prep.build_synapses_with_features import main as build_syn
    build_syn(config_path=config_path, output=path)


def _ensure_positions_pt(path, config_path):
    if os.path.exists(path):
        return
    print(f"[motifs] {path} not found — regenerating from raw .pbz2 exports...")
    import json
    from pathlib import Path
    from data_prep.compute_positions import compute_positions_and_distances
    with open(config_path) as f:
        cfg = json.load(f)
    path_p = Path(path)
    compute_positions_and_distances(
        synapses_file=None,
        graph_dir=Path(cfg['raw_data']['neurons_directory']),
        positions_file=path_p,
        distance_graph_file=path_p.with_name('distance_graph.gml'),
        verbose=False,
    )


def _ensure_features_pt(path, synapses_path, positions_path):
    if os.path.exists(path):
        return
    print(f"[motifs] {path} not found — regenerating from {synapses_path} + {positions_path}...")
    from .models.data.makefeatures import build_features
    build_features(synapses_path, positions_path, path)


_ensure_synapses_pt(args.synapses_path, args.config)
_ensure_positions_pt(args.positions_path, args.config)
_ensure_features_pt(args.feature_path, args.synapses_path, args.positions_path)

print("Step 1: Build Graph")
G = build_graph(
    use_existing=args.use_existing,
    existing_csv=args.existing_csv,
    synapses_path=args.synapses_path,
    positions_path=args.positions_path,
    x_min=args.x_min, x_max=args.x_max,
    y_min=args.y_min, y_max=args.y_max,
    z_min=args.z_min, z_max=args.z_max,
)

print("Step 2: GAE Training + In-Run Data Shapley")
train_and_compute_shapley(
    G=G,
    feature_path=args.feature_path,
    output_path=shapley_path,
    epochs=args.epochs,
    latent_dim=args.latent_dim,
    lr=args.lr,
    save_interval=args.save_interval,
    variational=args.variational,
    linear=args.linear,
)

print("Step 3: Normalizing Shapley Values")
normalize_shapley(
    input_path=shapley_path,
    output_path=norm_path,
)

print("Step 4: Motif Extraction and Visualization")
extract_and_visualize(
    G=G,
    shapley_path=norm_path,
    motif_sizes=args.motif_sizes,
    top_k=args.top_k,
    output_dir=args.output_dir,
)

print(f"All outputs saved to: {args.output_dir}/")
