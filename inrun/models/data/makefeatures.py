import pandas as pd
import torch
import argparse
import json
#csv_path = "top5_k1.csv"
#out_path = "neuron_features.pt"
#csv_path = "synapses_as_csv.csv"   
#out_path = "demo_neuron_features.pt"

parser = argparse.ArgumentParser()
parser.add_argument('--synapses_path', type=str, default='data/synapses.json')
parser.add_argument('--output',        type=str, default='data/demo_neuron_features.pt')
args = parser.parse_args()
 
print(f"Loading synapses from: {args.synapses_path}")
with open(args.synapses_path) as f:
    raw = json.load(f)
 
# Build rows from synapses.json
rows = []
for syn_id, val in raw.items():
    try:
        pre_id, post_id = int(val[0][0]), int(val[0][1])
        attrs = val[1]
        rows.append({
            'pre_id':         pre_id,
            'post_id':        post_id,
            'volume':         attrs.get('volume', 0) or 0,
            'upstream_dist':  attrs.get('upstream_dist', 0) or 0,
            'head_neck_shaft': attrs.get('head_neck_shaft', 'no_head') or 'no_head',
        })
    except Exception:
        continue
 
df = pd.DataFrame(rows)
print(f"Loaded {len(df):,} synapses")

neurons = pd.unique(df[["pre_id", "post_id"]].values.ravel())
neuron_index = {n: i for i, n in enumerate(neurons)}

num_nodes = len(neurons)
X = torch.zeros((num_nodes, 10), dtype=torch.float)

out_deg = df["pre_id"].value_counts()
in_deg = df["post_id"].value_counts()

for neuron, idx in neuron_index.items():
    o = out_deg.get(neuron, 0)
    i = in_deg.get(neuron, 0)
    X[idx, 0] = o
    X[idx, 1] = i
    X[idx, 2] = o + i

grouped = df.groupby("pre_id")

for neuron, g in grouped:
    idx = neuron_index.get(neuron)
    if idx is None:
        continue

    X[idx, 3] = g["volume"].mean()
    X[idx, 4] = g["volume"].std() if len(g) > 1 else 0.0
    X[idx, 5] = g["upstream_dist"].mean()
    X[idx, 6] = g["upstream_dist"].std() if len(g) > 1 else 0.0

    total = len(g)
    X[idx, 7] = (g["head_neck_shaft"] == "head").sum() / total
    X[idx, 8] = (g["head_neck_shaft"] == "shaft").sum() / total
    X[idx, 9] = (g["head_neck_shaft"] == "no_head").sum() / total

torch.save(X, args.output)

print("saved features", X.shape)
