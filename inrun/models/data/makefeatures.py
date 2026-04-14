import pandas as pd
import torch

csv_path = "top5_k1.csv"
out_path = "neuron_features.pt"

df = pd.read_csv(csv_path)

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

torch.save(X, out_path)

print("saved features", X.shape)
