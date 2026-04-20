"""
main.py

Orchestrates the full connectome GNN pipeline by calling
the existing scripts in order:

  1. filter_graph.py    
  2. graphnodeshapley.py (which trains GAE + computes Shapley values)
  3. normalize.py        (normalizes Shapley values)
  4. motifs.py           (extract and visualize most significant motifs)

Example Runs: 

Use existing graph:
    python main.py --use_existing

Build a new graph with desired threshold:
    python main.py \
        --x_min 800000 --x_max 1000000 \
        --y_min 700000 --y_max 900000

Full custom run:
    python main.py \
        --use_existing \
        --epochs 200 \
        --latent_dim 16 \
        --motif_sizes 5 10 20 \
        --top_k 3 \
        --output_dir results
"""

import argparse
import subprocess
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from filter_graph import build_graph
from graphnodeshapley import train_and_compute_shapley
from normalize import normalize_shapley
from motifs import extract_and_visualize

parser = argparse.ArgumentParser(description="Connectome GNN Pipeline")

parser.add_argument('--use_existing',   action='store_true',
                    help='Use existing CSV instead of building a new graph')
parser.add_argument('--existing_csv',   type=str, default='data/top5_k1.csv')
parser.add_argument('--synapses_path',  type=str, default='synapses.json')
parser.add_argument('--coords_path',    type=str, default='data/neuron_coords.json')
parser.add_argument('--x_min',          type=float, default=None)
parser.add_argument('--x_max',          type=float, default=None)
parser.add_argument('--y_min',          type=float, default=None)
parser.add_argument('--y_max',          type=float, default=None)
parser.add_argument('--z_min',          type=float, default=None)
parser.add_argument('--z_max',          type=float, default=None)

#gae parameters
parser.add_argument('--feature_path',   type=str, default='data/neuron_features.pt')
parser.add_argument('--variational',    action='store_true')
parser.add_argument('--linear',         action='store_true')
parser.add_argument('--epochs',         type=int,   default=200)
parser.add_argument('--latent_dim',     type=int,   default=16)
parser.add_argument('--lr',             type=float, default=0.01)
parser.add_argument('--save_interval',  type=int,   default=50)

#motifs
parser.add_argument('--motif_sizes',    type=int, nargs='+', default=[5, 10, 20])
parser.add_argument('--top_k',          type=int, default=3)

#output
parser.add_argument('--output_dir',     type=str, default='results')

args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

#paths that get passed between steps
#csv_path      = os.path.join(args.output_dir, 'filtered_graph.csv')
#shapley_path  = os.path.join(args.output_dir, 'graph_node_shapley.value')
#norm_path     = os.path.join(args.output_dir, 'graph_node_shapley_normalized.value')
graph_path   = os.path.join(args.output_dir, 'filtered_graph.adjlist')
shapley_path = os.path.join(args.output_dir, 'graph_node_shapley.value')
norm_path    = os.path.join(args.output_dir, 'graph_node_shapley_normalized.value')

G = build_graph(
    use_existing=args.use_existing,
    existing_csv=args.existing_csv,
    synapses_path=args.synapses_path,
    coords_path=args.coords_path,
    x_min=args.x_min, x_max=args.x_max,
    y_min=args.y_min, y_max=args.y_max,
    z_min=args.z_min, z_max=args.z_max,
    output=graph_path,
)

#Step2: GAE TRAINING + SHAPLEY


print("Step 2: GAE Training + In-Run Data Shapley")

train_and_compute_shapley(
    graph_path=graph_path,
    feature_path=args.feature_path,
    output_path=shapley_path,
    epochs=args.epochs,
    latent_dim=args.latent_dim,
    lr=args.lr,
    save_interval=args.save_interval,
    variational=args.variational,
    linear=args.linear,
)

print("STEP 3: Normalizing Shapley Values")
normalize_shapley(
    input_path=shapley_path,
    output_path=norm_path,
)
#motifs

print("STEP 4: Motif Extraction and Visualization")

extract_and_visualize(
    graph_path=graph_path,
    shapley_path=norm_path,
    motif_sizes=args.motif_sizes,
    top_k=args.top_k,
    output_dir=args.output_dir,
)
 


print(f"All outputs saved to: {args.output_dir}/")
