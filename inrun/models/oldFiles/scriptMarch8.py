import pickle
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default="graph_node_shapley.value")
parser.add_argument("--output", type=str, default="graph_node_shapley_normalized.value")
args = parser.parse_args()

input_path = args.input
output_path = args.output

print("Loading:", input_path)

with open(input_path, "rb") as f:
    record = pickle.load(f)

# ---------------------------------------------------
# Handle different possible key names
# ---------------------------------------------------

if "First-order In-Run Data Shapley" in record:
    shapley = np.array(record["First-order In-Run Data Shapley"])
elif "node_shapley" in record:
    shapley = np.array(record["node_shapley"])
else:
    raise KeyError("Could not find Shapley values in record.")

node_index = record["node_index"]

# ---------------------------------------------------
# Normalize safely
# ---------------------------------------------------

total = shapley.sum()

if total == 0:
    raise ValueError("Sum of Shapley values is zero — cannot normalize.")

normalized = 100.0 * shapley / total

# ---------------------------------------------------
# Save
# ---------------------------------------------------

out = {
    "node_index": node_index,
    "Normalized Shapley (%)": normalized.tolist(),
    "Raw Shapley": shapley.tolist()
}

with open(output_path, "wb") as f:
    pickle.dump(out, f)

print("\nSaved:", output_path)
print("Nodes:", len(shapley))
print("Sum of normalized values:", normalized.sum())

# ---------------------------------------------------
# Print Top 10
# ---------------------------------------------------

top = np.argsort(normalized)[-10:][::-1]

print("\nTop 10 Nodes:")
for i in top:
    print(
        f"Node {node_index[i]} | "
        f"{normalized[i]:.6f}% | "
        f"Raw: {shapley[i]:.6f}"
    )
