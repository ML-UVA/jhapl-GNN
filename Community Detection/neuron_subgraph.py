import pickle
import statistics
from collections import Counter

import networkx as nx
import igraph as ig
import leidenalg
import matplotlib.pyplot as plt

# Load neuron-level connectivity graph

with open("data/neuron_graph.gpickle", "rb") as f:
    G_nx = pickle.load(f)

print(f"Loaded NetworkX graph with {G_nx.number_of_nodes()} nodes "
      f"and {G_nx.number_of_edges()} edges")

# Convert NetworkX → igraph
# (Leiden runs on igraph graphs)


# Ensure nodes are contiguous integers (important!)
node_mapping = {node: i for i, node in enumerate(G_nx.nodes())}
edges_igraph = [(node_mapping[u], node_mapping[v]) for u, v in G_nx.edges()]

g = ig.Graph(
    n=len(node_mapping),
    edges=edges_igraph,
    directed=True
)

print(f"Converted to igraph with {g.vcount()} vertices")


# --------------------------------------------------
# Run Leiden across resolution parameters
# --------------------------------------------------

resolutions = [0.002, 0.005, 0.0075, 0.01, 0.02, 0.05,
               0.075, 0.1, 0.2, 0.5, 1, 2, 5, 7.5, 10]

median_sizes = []
num_communities = []

for res in resolutions:
    print(f"\nResolution = {res}")

    partition = leidenalg.find_partition(
        g,
        leidenalg.CPMVertexPartition,
        resolution_parameter=res
    )

    # Community sizes
    sizes = Counter(partition.membership).values()

    median_size = statistics.median(sizes)
    median_sizes.append(median_size)

    n_comms = len(sizes)
    num_communities.append(n_comms)

    print(f"  communities: {n_comms}")
    print(f"  median size: {median_size}")

print("\nMedian community sizes:")
print(median_sizes)

print("\nNumber of communities:")
print(num_communities)

plt.figure(figsize=(7, 5))
plt.plot(resolutions, median_sizes, marker="o", label="Median community size")
plt.plot(resolutions, num_communities, marker="s", label="Number of communities")
plt.xscale("log")
plt.xlabel("Resolution parameter (log scale)")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.show()