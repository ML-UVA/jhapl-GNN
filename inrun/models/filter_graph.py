"""
filter_graph.py

Takes synapses.json and neuron_coords.json and produces a filtered
edge CSV containing only synapses where BOTH neurons fall within
the specified x/y/z spatial bounding box.

The output CSV is a drop-in replacement for top5_k1.csv and can be
passed directly to graphnodeshapley.py via --csv_path.

USAGE EXAMPLES
--------------
# Use the existing top5_k1.csv as-is (no spatial filtering):
    python filter_graph.py --use_existing

# Filter by spatial thresholds:
    python filter_graph.py \
        --x_min 800000 --x_max 1000000 \
        --y_min 700000 --y_max 900000 \
        --output data/filtered_graph.csv

# Filter on just one axis:
    python filter_graph.py --x_min 900000 --output data/filtered_x.csv

# Use a different synapses file:
    python filter_graph.py \
        --synapses_path data/synapses.json \
        --coords_path   data/neuron_coords.json \
        --x_min 800000 \
        --output data/filtered_graph.csv

INPUT FILES

synapses.json   : {syn_id: [[pre_id, post_id], {attrs}], ...}
neuron_coords.json : {"neuron_id_0": [x, y, z], ...}
                     (keys have _0/_1 suffix, neuron IDs in synapses do not)

OUTPUT

CSV with columns [pre_id, post_id] — same format as top5_k1.csv
Only edges where both pre_id and post_id pass the spatial filter.
"""

import argparse
import json
import os
import pandas as pd
import networkx as nx 
def build_graph(
    use_existing=False,
    existing_csv='data/top5_k1.csv',
    synapses_path='synapses.json',
    coords_path='data/neuron_coords.json',
    x_min=None, x_max=None,
    y_min=None, y_max=None,
    z_min=None, z_max=None,
    output='results/filtered_graph.adjlist',
):
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
 
    if use_existing:
        if not os.path.exists(existing_csv):
            raise FileNotFoundError(f"Existing CSV not found: {existing_csv}")
        df = pd.read_csv(existing_csv)
        G = nx.DiGraph()
        for _, row in df.iterrows():
            G.add_edge(int(row['pre_id']), int(row['post_id']))
        print(f"[filter_graph] Loaded existing graph: {len(G.nodes()):,} nodes, {len(G.edges()):,} edges")
        nx.write_adjlist(G, output)
        print(f"[filter_graph] Saved -> {output}")
        return G
    print(f"[filter_graph] Loading coordinates from: {coords_path}")
    with open(coords_path) as f:
        raw_coords = json.load(f)
    coords = {}
    for key, xyz in raw_coords.items():
        numeric_id = int(key.split('_')[0])
        if numeric_id not in coords:
            coords[numeric_id] = xyz
    print(f"[filter_graph] Neurons with coordinates: {len(coords):,}")
    any_threshold = any(v is not None for v in [
        x_min, x_max, y_min, y_max, z_min, z_max
    ])
 
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
    with open(synapses_path) as f:
        raw = json.load(f)
    G = nx.DiGraph()
    for syn_id, val in raw.items():
        try:
            pre_id, post_id = int(val[0][0]), int(val[0][1])
        except Exception:
            continue
        if pre_id in kept_ids and post_id in kept_ids:
            G.add_edge(pre_id, post_id)
 
    print(f"[filter_graph] Graph: {len(G.nodes()):,} nodes, {len(G.edges()):,} edges")
 
    if len(G.edges()) == 0:
        raise ValueError("No edges survived the spatial filter.")
    nx.write_adjlist(G, output)
    print(f"[filter_graph] Saved -> {output}")
    return G
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Filter connectome graph by spatial bounding box")
    parser.add_argument('--use_existing',   action='store_true')
    parser.add_argument('--existing_csv',   type=str, default='data/top5_k1.csv')
    parser.add_argument('--synapses_path',  type=str, default='synapses.json')
    parser.add_argument('--coords_path',    type=str, default='data/neuron_coords.json')
    parser.add_argument('--x_min',          type=float, default=None)
    parser.add_argument('--x_max',          type=float, default=None)
    parser.add_argument('--y_min',          type=float, default=None)
    parser.add_argument('--y_max',          type=float, default=None)
    parser.add_argument('--z_min',          type=float, default=None)
    parser.add_argument('--z_max',          type=float, default=None)
    parser.add_argument('--output',         type=str, default='results/filtered_graph.adjlist')
    args = parser.parse_args()
 
    build_graph(
        use_existing=args.use_existing,
        existing_csv=args.existing_csv,
        synapses_path=args.synapses_path,
        coords_path=args.coords_path,
        x_min=args.x_min, x_max=args.x_max,
        y_min=args.y_min, y_max=args.y_max,
        z_min=args.z_min, z_max=args.z_max,
        output=args.output,
    )
"""
parser = argparse.ArgumentParser(
    description="Filter connectome graph by spatial bounding box"
)

parser.add_argument(
    '--use_existing',
    action='store_true',
    help='Skip filtering and just use the existing top5_k1.csv'
)
parser.add_argument(
    '--existing_csv',
    type=str,
    default='data/top5_k1.csv',
    help='Path to existing CSV to use when --use_existing is set'
)

parser.add_argument(
    '--synapses_path',
    type=str,
    default='synapses.json',
    help='Path to synapses.json'
)
parser.add_argument(
    '--coords_path',
    type=str,
    default='data/neuron_coords.json',
    help='Path to neuron_coords.json produced by extract_coordinates.py'
)

parser.add_argument('--x_min', type=float, default=None, help='Keep neurons with x >= x_min (nm)')
parser.add_argument('--x_max', type=float, default=None, help='Keep neurons with x <= x_max (nm)')
parser.add_argument('--y_min', type=float, default=None, help='Keep neurons with y >= y_min (nm)')
parser.add_argument('--y_max', type=float, default=None, help='Keep neurons with y <= y_max (nm)')
parser.add_argument('--z_min', type=float, default=None, help='Keep neurons with z >= z_min (nm)')
parser.add_argument('--z_max', type=float, default=None, help='Keep neurons with z <= z_max (nm)')

parser.add_argument(
    '--output',
    type=str,
    default='data/filtered_graph.csv',
    help='Output edge CSV path'
)

args = parser.parse_args()


if args.use_existing:
    if not os.path.exists(args.existing_csv):
        raise FileNotFoundError(f"Existing CSV not found: {args.existing_csv}")
    df = pd.read_csv(args.existing_csv)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"[use_existing] Copied {args.existing_csv} -> {args.output}")
    print(f"  Edges : {len(df):,}")
    print(f"  Nodes : {len(pd.unique(df[['pre_id','post_id']].values.ravel())):,}")
    raise SystemExit(0)


print(f"Loading coordinates from: {args.coords_path}")
with open(args.coords_path) as f:
    raw_coords = json.load(f)

#strip _0/_1 suffix from keys
coords = {}
for key, xyz in raw_coords.items():
    numeric_id = int(key.split('_')[0])
    # If a neuron has both _0 and _1 splits, _0 takes priority
    if numeric_id not in coords:
        coords[numeric_id] = xyz

print(f"  Neurons with coordinates: {len(coords):,}")

#apply filtering based on threshold

any_threshold = any(v is not None for v in [
    args.x_min, args.x_max, args.y_min, args.y_max, args.z_min, args.z_max
])

if not any_threshold:
    print("\nWARNING: No thresholds specified — all neurons will be kept.")
    print("Use --x_min, --x_max, --y_min, --y_max, --z_min, --z_max to filter.")

print("\nApplying spatial filter:")
if args.x_min: print(f"  x >= {args.x_min:,.0f} nm")
if args.x_max: print(f"  x <= {args.x_max:,.0f} nm")
if args.y_min: print(f"  y >= {args.y_min:,.0f} nm")
if args.y_max: print(f"  y <= {args.y_max:,.0f} nm")
if args.z_min: print(f"  z >= {args.z_min:,.0f} nm")
if args.z_max: print(f"  z <= {args.z_max:,.0f} nm")

kept_ids = set()
for neuron_id, (x, y, z) in coords.items():
    if args.x_min is not None and x < args.x_min: continue
    if args.x_max is not None and x > args.x_max: continue
    if args.y_min is not None and y < args.y_min: continue
    if args.y_max is not None and y > args.y_max: continue
    if args.z_min is not None and z < args.z_min: continue
    if args.z_max is not None and z > args.z_max: continue
    kept_ids.add(neuron_id)

print(f"\n  Neurons before filter : {len(coords):,}")
print(f"  Neurons after filter  : {len(kept_ids):,}")

if len(kept_ids) == 0:
    raise ValueError(
        "No neurons survived the spatial filter. "
        "Check your threshold values against the coordinate ranges printed "
        "at the end of extract_coordinates.py output."
    )


print(f"\nLoading synapses from: {args.synapses_path}")
with open(args.synapses_path) as f:
    raw = json.load(f)

print(f"  Total synapses: {len(raw):,}")

rows = []
for syn_id, val in raw.items():
    try:
        pre_id, post_id = int(val[0][0]), int(val[0][1])
    except Exception:
        continue
    if pre_id in kept_ids and post_id in kept_ids:
        rows.append({'pre_id': pre_id, 'post_id': post_id})

df = pd.DataFrame(rows)

print(f"  Edges before filter   : {len(raw):,}")
print(f"  Edges after filter    : {len(df):,}")

if len(df) == 0:
    raise ValueError(
        "No edges survived the spatial filter. "
        "The filtered region may be too small or the neurons may not be connected."
    )

unique_nodes = len(pd.unique(df[['pre_id', 'post_id']].values.ravel()))
print(f"  Unique neurons in output: {unique_nodes:,}")


os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
df.to_csv(args.output, index=False)

print(f"\nSaved filtered graph -> {args.output}")

kept_coords = [coords[n] for n in kept_ids if n in coords]
if kept_coords:
    xs = [c[0] for c in kept_coords]
    ys = [c[1] for c in kept_coords]
    zs = [c[2] for c in kept_coords]
    print(f"\nCoordinate ranges of kept neurons (nm):")
    print(f"  x: [{min(xs):,.1f},  {max(xs):,.1f}]")
    print(f"  y: [{min(ys):,.1f},  {max(ys):,.1f}]")
    print(f"  z: [{min(zs):,.1f},  {max(zs):,.1f}]")
"""
