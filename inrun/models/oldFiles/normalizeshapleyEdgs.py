import pickle
import numpy as np
import pandas as pd
from collections import defaultdict

INPUT_FILE = "graph_edge_shapley.value"
OUTPUT_PKL = "graph_edge_shapley_undirected_normalized.pkl"
OUTPUT_CSV = "graph_edge_shapley_undirected_normalized.csv"

# --------------------------------------------------
# Load
# --------------------------------------------------
with open(INPUT_FILE, "rb") as f:
    record = pickle.load(f)

edge_index = record["edge_index"]              # shape (2, E)
shapley = np.array(record["First-order In-Run Edge Shapley"])  # shape (E,)

assert edge_index.shape[1] == shapley.shape[0]

# --------------------------------------------------
# Collapse directed edges → undirected edges
# --------------------------------------------------
edge_scores = defaultdict(float)

for i in range(edge_index.shape[1]):
    u = int(edge_index[0, i])
    v = int(edge_index[1, i])
    key = tuple(sorted((u, v)))   # undirected edge
    edge_scores[key] += shapley[i]

# Convert to arrays
edges = np.array(list(edge_scores.keys()))        # shape (E_undir, 2)
scores = np.array(list(edge_scores.values()))     # shape (E_undir,)

# --------------------------------------------------
# Normalize (recommended: sum-to-one)
# --------------------------------------------------
score_sum = scores.sum()
if score_sum > 0:
    scores_norm = scores / score_sum
else:
    scores_norm = scores

# Optional alternatives (comment out if undesired)
# scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
# scores_norm = scores / scores.mean()

# --------------------------------------------------
# Save pickle (clean + reusable)
# --------------------------------------------------
out_record = {
    "edge_index": edges,                 # undirected edges
    "edge_shapley_raw": scores,
    "edge_shapley_normalized": scores_norm,
}

with open(OUTPUT_PKL, "wb") as f:
    pickle.dump(out_record, f)

# --------------------------------------------------
# Save CSV (human-friendly)
# --------------------------------------------------
df = pd.DataFrame({
    "src": edges[:, 0],
    "dst": edges[:, 1],
    "shapley_raw": scores,
    "shapley_normalized": scores_norm,
})

df.sort_values("shapley_normalized", ascending=False).to_csv(
    OUTPUT_CSV, index=False
)

print(f"Saved normalized undirected edge Shapley to:")
print(f"  {OUTPUT_PKL}")
print(f"  {OUTPUT_CSV}")
