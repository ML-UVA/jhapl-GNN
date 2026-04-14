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
csv_path      = os.path.join(args.output_dir, 'filtered_graph.csv')
shapley_path  = os.path.join(args.output_dir, 'graph_node_shapley.value')
norm_path     = os.path.join(args.output_dir, 'graph_node_shapley_normalized.value')

def run(cmd):
    """Run a command and exit if it fails."""
    print(f"\n>>> {' '.join(cmd)}\n")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\n[ERROR] Command failed with return code {result.returncode}")
        sys.exit(result.returncode)

#graph construction
print("Step 1: Graph Construction")

if args.use_existing:
    cmd = [
        sys.executable, 'filter_graph.py',
        '--use_existing',
        '--existing_csv', args.existing_csv,
        '--output',       csv_path,
    ]
else:
    cmd = [
        sys.executable, 'filter_graph.py',
        '--synapses_path', args.synapses_path,
        '--coords_path',   args.coords_path,
        '--output',        csv_path,
    ]
    if args.x_min is not None: cmd += ['--x_min', str(args.x_min)]
    if args.x_max is not None: cmd += ['--x_max', str(args.x_max)]
    if args.y_min is not None: cmd += ['--y_min', str(args.y_min)]
    if args.y_max is not None: cmd += ['--y_max', str(args.y_max)]
    if args.z_min is not None: cmd += ['--z_min', str(args.z_min)]
    if args.z_max is not None: cmd += ['--z_max', str(args.z_max)]

run(cmd)


#Step2: GAE TRAINING + SHAPLEY


print("Step 2: GAE Training + In-Run Data Shapley")


cmd = [
    sys.executable, 'graphnodeshapley.py',
    '--csv_path',      csv_path,
    '--feature_path',  args.feature_path,
    '--epochs',        str(args.epochs),
    '--latent_dim',    str(args.latent_dim),
    '--lr',            str(args.lr),
    '--save_interval', str(args.save_interval),
    '--output_path',   shapley_path,
]
if args.variational: cmd.append('--variational')
if args.linear:      cmd.append('--linear')

run(cmd)

#normalize
print("\n" + "="*60)
print("STEP 3: Normalizing Shapley Values")
print("="*60)

cmd = [
    sys.executable, 'normalize.py',
    '--input_path',  shapley_path,
    '--output_path', norm_path,
]

run(cmd)

#motifs

print("STEP 4: Motif Extraction and Visualization")

cmd = [
    sys.executable, 'motifs.py',
    '--csv_path',     csv_path,
    '--shapley_path', norm_path,
    '--top_k',        str(args.top_k),
    '--output_dir',   args.output_dir,
] + ['--motif_sizes'] + [str(s) for s in args.motif_sizes]

run(cmd)


print(f"All outputs saved to: {args.output_dir}/")
