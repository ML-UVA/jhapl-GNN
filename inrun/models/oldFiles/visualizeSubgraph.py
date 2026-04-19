import pickle
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

print("loading shapley values")
shap = pickle.load(open("graph_node_shapley_normalized.value", "rb"))

shap_vals = shap["Normalized Shapley (%)"]

print("loading graph csv")
df = pd.read_csv("data/top5_k1.csv")

print("building node index mapping")
neuron_ids = pd.unique(df[["pre_id", "post_id"]].values.ravel())

neuron_to_idx = {}
idx_to_neuron = {}

for i in range(len(neuron_ids)):
    neuron_to_idx[neuron_ids[i]] = i
    idx_to_neuron[i] = neuron_ids[i]

print("building graph")
G = nx.Graph()

for i in range(len(df)):
    u = neuron_to_idx[df.pre_id.iloc[i]]
    v = neuron_to_idx[df.post_id.iloc[i]]
    G.add_edge(u, v)

print("graph nodes", G.number_of_nodes())
print("graph edges", G.number_of_edges())

print("starting greedy expansion")

start_node = int(shap_vals.index(max(shap_vals)))

subgraph_nodes = [start_node]
used = set(subgraph_nodes)

target_size = 5

while len(subgraph_nodes) < target_size:
    best_node = None
    best_value = -1

    for u in subgraph_nodes:
        for v in G.neighbors(u):
            if v in used:
                continue

            val = shap_vals[v]
            if val > best_value:
                best_value = val
                best_node = v

    if best_node is None:
        break

    subgraph_nodes.append(best_node)
    used.add(best_node)

print("greedy subgraph size", len(subgraph_nodes))
print("node indices (last 10 digits):", [str(idx_to_neuron[n])[-10:] for n in subgraph_nodes])

# Calculate and print total normalized Shapley value
total_shapley = sum(shap_vals[node] for node in subgraph_nodes)
print(f"Total normalized Shapley value: {total_shapley:.4f}")

H = G.subgraph(subgraph_nodes).copy()

node_colors = []
node_sizes = []

for n in H.nodes():
    node_colors.append(shap_vals[n])
    node_sizes.append(800 + 5000 * shap_vals[n])  # Bigger dots: base 800, scale up to 5800

print("drawing subgraph")

fig, ax = plt.subplots(figsize=(10, 10))  # Slightly larger figure

pos = nx.spring_layout(H, seed=1, k=2)  # Increased k for better spacing with bigger nodes

nx.draw(
    H,
    pos,
    node_color=node_colors,
    node_size=node_sizes,
    cmap="viridis",
    with_labels=True,
    font_size=8,  # Smaller font to fit labels
    font_weight='bold',
    ax=ax
)

sm = plt.cm.ScalarMappable(
    cmap="viridis",
    norm=plt.Normalize(
        vmin=min(node_colors),
        vmax=max(node_colors)
    )
)
sm.set_array([])

fig.colorbar(sm, ax=ax, label="Normalized Shapley (%)")

ax.set_title("Greedy Shapley Subgraph")

out_file = "greedy_shapley_subgraph.png"
plt.savefig(out_file, dpi=300, bbox_inches="tight")
plt.close()

print("saved", out_file)
