import argparse
import pickle
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--input_path',  type=str, default='results/graph_node_shapley.value')
parser.add_argument('--output_path', type=str, default='results/graph_node_shapley_normalized.value')
args = parser.parse_args()

with open(args.input_path, "rb") as f:
    record = pickle.load(f)

shapley    = np.array(record["First-order In-Run Data Shapley"])
node_index = record["node_index"]

total      = shapley.sum()
normalized = 100.0 * shapley / total

out = {
    "node_index":             node_index,
    "Normalized Shapley (%)": normalized.tolist(),
    "Raw Shapley":            shapley.tolist()
}

with open(args.output_path, "wb") as f:
    pickle.dump(out, f)

print("Saved:", args.output_path)
print("Nodes:", len(shapley))
print("Sum:",   normalized.sum())

top = np.argsort(normalized)[-10:][::-1]
for i in top:
    print(f"Node {node_index[i]} | {normalized[i]:.6f}% | {shapley[i]:.6f}")
