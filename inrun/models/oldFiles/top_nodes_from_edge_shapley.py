import pickle
import numpy as np
import pandas as pd
from collections import defaultdict

# =====================
# Config
# =====================
EDGE_SHAPLEY_PATH = "graph_edge_shapley.value"
CSV_PATH = "data/top5_k1.csv"
TOP_K_EDGES = 20
TOP_K_NODES = 20

# =====================
# Load CSV to rebuild mapping
# =====================
df = pd.read_csv(CSV_PATH)
neuron_ids = pd.unique(df[['pre_id', 'post_id']].values.ravel())
idx_to_neuron = {i: int(nid) for i, nid in enumerate(neuron_ids)}

# =====================
# Load edge Shapley
# =====================
r = pickle.load(open(EDGE_SHAPLEY_PATH, "rb"))
edge_index = r["edge_index"]  # [2, E]
shapley_raw = np.array(r["First-order In-Run Edge Shapley"])

# =====================
# Aggregate undirected edges
# =====================
edge_scores = defaultdict(float)

for i in range(edge_index.shape[1]):
    u_idx = int(edge_index[0, i])
    v_idx = int(edge_index[1, i])

    u = idx_to_neuron[u_idx]
    v = idx_to_neuron[v_idx]

    key = tuple(sorted((u, v)))
    edge_scores[key] += shapley_raw[i]

# Normalize
total = sum(edge_scores.values())
for k in edge_scores:
    edge_scores[k] /= total

# =====================
# Get top-K edges
# =====================
top_edges = sorted(
    edge_scores.items(),
    key=lambda x: x[1],
    reverse=True
)[:TOP_K_EDGES]

print(f"\nTop {TOP_K_EDGES} edges:")
for (u, v), s in top_edges:
    print(f"{u} -- {v} | shapley = {s:.6e}")

# =====================
# Aggregate edge Shapley onto nodes
# =====================
node_scores = defaultdict(float)

for (u, v), s in top_edges:
    node_scores[u] += s
    node_scores[v] += s

# =====================
# Rank nodes
# =====================
top_nodes = sorted(
    node_scores.items(),
    key=lambda x: x[1],
    reverse=True
)[:TOP_K_NODES]

print(f"\nTop {TOP_K_NODES} nodes induced by edge Shapley:")
for i, (n, s) in enumerate(top_nodes, 1):
    print(f"{i:02d}. Neuron {n} | score = {s:.6e}")
