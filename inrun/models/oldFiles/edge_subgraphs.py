import pickle
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict

CSV_PATH = "data/top5_k1.csv"
EDGE_SHAPLEY_PATH = "graph_edge_shapley.value"
IDX_TO_NEURON_PATH = "idx_to_neuron.pkl"

TOP_K = [10, 20]
GREEDY_EDGE_LIMIT = 20

OUT_EDGE_CSV = "edge_shapley_ranked.csv"
OUT_PKL = "edge_greedy_subgraphs.pkl"


df = pd.read_csv(CSV_PATH)

G = nx.Graph()
for row in df.itertuples(index=False):
    G.add_edge(int(row.pre_id), int(row.post_id))

print(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")


#idx_to_neuron = pickle.load(open(IDX_TO_NEURON_PATH, "rb"))
neuron_ids = pd.unique(df[['pre_id', 'post_id']].values.ravel())
idx_to_neuron = {i: int(nid) for i, nid in enumerate(neuron_ids)}

r = pickle.load(open(EDGE_SHAPLEY_PATH, "rb"))
edge_index = r["edge_index"]  # [2, E]
shapley_raw = np.array(r["First-order In-Run Edge Shapley"])

# =====================
# Aggregate undirected edges (RAW neuron IDs)
# =====================
edge_scores = defaultdict(float)

for i in range(edge_index.shape[1]):
    u_idx = int(edge_index[0, i])
    v_idx = int(edge_index[1, i])

    u = idx_to_neuron[u_idx]
    v = idx_to_neuron[v_idx]

    key = tuple(sorted((u, v)))
    edge_scores[key] += shapley_raw[i]

edges, scores = zip(*edge_scores.items())
scores = np.array(scores)

# Normalize
scores = scores / scores.sum()

edge_df = pd.DataFrame({
    "src": [e[0] for e in edges],
    "dst": [e[1] for e in edges],
    "shapley": scores
}).sort_values("shapley", ascending=False).reset_index(drop=True)

edge_df.to_csv(OUT_EDGE_CSV, index=False)
print("Saved ranked edges to:", OUT_EDGE_CSV)

# =====================
# Print top-k edges
# =====================
for k in TOP_K:
    print(f"\nTop {k} edges (raw neuron IDs):")
    for i in range(k):
        r = edge_df.iloc[i]
        print(f"{int(r.src)} -- {int(r.dst)} | score = {r.shapley:.6e}")

# =====================
# Greedy edge subgraph
# =====================
def greedy_edge_subgraph(seed_edge, k):
    picked_edges = set([seed_edge])
    picked_nodes = set(seed_edge)

    frontier = set()

    for u in seed_edge:
        for v in G.neighbors(u):
            e = tuple(sorted((u, v)))
            if e in edge_scores:
                frontier.add(e)

    while len(picked_edges) < k and frontier:
        best = max(frontier, key=lambda e: edge_scores[e])
        frontier.remove(best)

        if best in picked_edges:
            continue

        picked_edges.add(best)
        picked_nodes.update(best)

        for u in best:
            for v in G.neighbors(u):
                e = tuple(sorted((u, v)))
                if e not in picked_edges and e in edge_scores:
                    frontier.add(e)

    return picked_edges, picked_nodes

# =====================
# Build greedy subgraphs
# =====================
greedy_results = []
SEED_LIMIT = 100
for i in range(min(SEED_LIMIT, len(edge_df))):
    seed = (int(edge_df.src[i]), int(edge_df.dst[i]))

    edges_subg, nodes_subg = greedy_edge_subgraph(seed, GREEDY_EDGE_LIMIT)
    score = sum(edge_scores[e] for e in edges_subg)

    greedy_results.append({
        "seed_edge": seed,
        "score": score,
        "num_edges": len(edges_subg),
        "num_nodes": len(nodes_subg),
        "edges": list(edges_subg),
        "nodes": list(nodes_subg),
    })

greedy_results.sort(key=lambda x: x["score"], reverse=True)

# =====================
# Print best subgraphs
# =====================
print("\nTop greedy edge-based subgraphs:")
for i in range(5):
    g = greedy_results[i]
    print(
        f"Seed {g['seed_edge']} | "
        f"score={g['score']:.6e} | "
        f"edges={g['num_edges']} | "
        f"nodes={g['num_nodes']}"
    )
    print(" Nodes:", g["nodes"])
    print(" Edges:", g["edges"])
    print()

# =====================
# Save
# =====================
pickle.dump(greedy_results, open(OUT_PKL, "wb"))
print("Saved greedy subgraphs to:", OUT_PKL)
