import pickle
import numpy as np

input_path = "graph_node_shapley.value"
output_path = "graph_node_shapley_normalized.value"

with open(input_path, "rb") as f:
    record = pickle.load(f)

shapley = np.array(record["First-order In-Run Data Shapley"])
node_index = record["node_index"]

total = shapley.sum()
normalized = 100.0 * shapley / total

out = {
    "node_index": node_index,
    "Normalized Shapley (%)": normalized.tolist(),
    "Raw Shapley": shapley.tolist()
}

with open(output_path, "wb") as f:
    pickle.dump(out, f)

print("Saved:", output_path)
print("Nodes:", len(shapley))
print("Sum:", normalized.sum())

top = np.argsort(normalized)[-10:][::-1]
for i in top:
    print(
        f"Node {node_index[i]} | "
        f"{normalized[i]:.6f}% | "
        f"{shapley[i]:.6f}"
    )
