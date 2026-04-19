import pickle
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

print("loading greedy edge results")
results = pickle.load(open("edge_greedy_subgraphs.pkl", "rb"))
top = results[0]

print("loading edge shapley")
edge_data = pickle.load(open("graph_edge_shapley.value", "rb"))
edge_index = edge_data["edge_index"]
shapley_raw = edge_data["First-order In-Run Edge Shapley"]

print("loading graph csv")
df = pd.read_csv("data/top5_k1.csv")

print("building neuron index mapping")
neuron_ids = pd.unique(df[["pre_id", "post_id"]].values.ravel())

neuron_to_idx = {}
idx_to_neuron = {}

for i in range(len(neuron_ids)):
    neuron_to_idx[int(neuron_ids[i])] = i
    idx_to_neuron[i] = int(neuron_ids[i])

print("aggregating shapley per undirected edge")

edge_scores = defaultdict(float)

for i in range(edge_index.shape[1]):
    u_idx = int(edge_index[0, i])
    v_idx = int(edge_index[1, i])

    u = idx_to_neuron[u_idx]
    v = idx_to_neuron[v_idx]

    key = tuple(sorted((u, v)))
    edge_scores[key] += shapley_raw[i]

print("building top subgraph in index space")

H = nx.Graph()

for (u_raw, v_raw) in top["edges"]:
    u = neuron_to_idx[u_raw]
    v = neuron_to_idx[v_raw]
    H.add_edge(u, v)

print("subgraph nodes", H.number_of_nodes())
print("subgraph edges", H.number_of_edges())

# Edge colors + widths
edge_colors = []
edge_widths = []

for (u, v) in H.edges():
    raw_u = idx_to_neuron[u]
    raw_v = idx_to_neuron[v]
    score = edge_scores[tuple(sorted((raw_u, raw_v)))]

    edge_colors.append(score)
    edge_widths.append(1 + 5 * score / max(edge_scores.values()))

print("drawing")

fig, ax = plt.subplots(figsize=(8, 8))

pos = nx.spring_layout(H, seed=1)

nx.draw_networkx_nodes(H, pos, node_size=300, node_color="lightgray", ax=ax)

edges = nx.draw_networkx_edges(
    H,
    pos,
    edge_color=edge_colors,
    edge_cmap=plt.cm.viridis,
    width=edge_widths,
    ax=ax
)

nx.draw_networkx_labels(H, pos, font_size=6, ax=ax)

sm = plt.cm.ScalarMappable(
    cmap="viridis",
    norm=plt.Normalize(
        vmin=min(edge_colors),
        vmax=max(edge_colors)
    )
)
sm.set_array([])

fig.colorbar(sm, ax=ax, label="Edge Shapley Value")

ax.set_title("Top Edge-Shapley Greedy Subgraph")

out_file = "top_edge_shapley_subgraph.png"
plt.savefig(out_file, dpi=300, bbox_inches="tight")
plt.close()

print("saved", out_file)
