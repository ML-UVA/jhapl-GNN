import pickle
import networkx as nx
import matplotlib.pyplot as plt

# Load results
results = pickle.load(open("edge_greedy_subgraphs.pkl", "rb"))

# Top subgraph (already sorted by score)
top = results[0]

print("Top seed:", top["seed_edge"])
print("Score:", top["score"])
print("Edges:", len(top["edges"]))
print("Nodes:", len(top["nodes"]))

# Build graph from that subgraph
H = nx.Graph()
H.add_edges_from(top["edges"])

# Draw
plt.figure(figsize=(8, 8))
pos = nx.spring_layout(H, seed=42)

nx.draw(
    H,
    pos,
    node_size=200,
    with_labels=False,
    edge_color="black"
)

plt.title("Top Edge-Shapley Subgraph")
plt.show()
