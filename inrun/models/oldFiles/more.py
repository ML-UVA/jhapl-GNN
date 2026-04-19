import pickle
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

print("loading shapley")
shap = pickle.load(open("graph_node_shapley_normalized.value", "rb"))

node_ids = shap["node_index"]
node_vals = shap["Normalized Shapley (%)"]

shapley_map = {}
for i in range(len(node_ids)):
    shapley_map[node_ids[i]] = node_vals[i]

print("loading graph")
df = pd.read_csv("data/top5_k1.csv")

neuron_ids = pd.unique(df[["pre_id", "post_id"]].values.ravel())
neuron_map = {}
rev_map = {}

for i, nid in enumerate(neuron_ids):
    neuron_map[nid] = i
    rev_map[i] = nid

G = nx.Graph()

for i in range(len(neuron_ids)):
    G.add_node(i)

for _, row in df.iterrows():
    u = neuron_map[row["pre_id"]]
    v = neuron_map[row["post_id"]]
    G.add_edge(u, v)

print("graph nodes", G.number_of_nodes())
print("graph edges", G.number_of_edges())

colors = []
for n in G.nodes():
    if n in shapley_map:
        colors.append(shapley_map[n])
    else:
        colors.append(0.0)

plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G, seed=0)

nx.draw_networkx_nodes(
    G,
    pos,
    node_size=8,
    node_color=colors,
    cmap="viridis"
)

nx.draw_networkx_edges(
    G,
    pos,
    alpha=0.05,
    width=0.3
)

plt.title("Full Neuron Graph Colored by Normalized Shapley Value")
plt.axis("off")

sm = plt.cm.ScalarMappable(
    cmap="viridis",
    norm=plt.Normalize(vmin=min(colors), vmax=max(colors))
)
sm.set_array([])
plt.colorbar(sm, shrink=0.7, label="Normalized Shapley (%)")

plt.savefig("full_graph_shapley.png", dpi=300, bbox_inches="tight")
print("saved full_graph_shapley.png")
