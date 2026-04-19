import pickle
import pandas as pd
import networkx as nx
import numpy as np
import os

############################################
# SETTINGS
############################################

REAL_CSV = "data/top5_k1.csv"
SHAPLEY_PATH = "graph_node_shapley.value"

NULL_DIR = "data/null_graphs"
NULL_FILES = [
    "null_graph_0.csv",
    "null_graph_1.csv",
    "null_graph_2.csv",
    "null_graph_3.csv"
]

SUBGRAPH_SIZE = 5
TOP_K_SEEDS = 50
TOP_SUBGRAPHS = 5

############################################
# LOAD SHAPLEY
############################################

shapley = pickle.load(open(SHAPLEY_PATH, "rb"))
vals = np.array(shapley["First-order In-Run Data Shapley"])

############################################
# HELPER FUNCTIONS
############################################

def build_graph_from_csv(path):
    df = pd.read_csv(path)

    # handle real vs null graphs
    if "pre_id" in df.columns:
        edges = zip(df["pre_id"], df["post_id"])
    else:
        edges = zip(df["source"], df["target"])

    G = nx.Graph()
    for u, v in edges:
        G.add_edge(u, v)

    return G

def map_shapley_to_nodes(G, vals):
    nodes = list(G.nodes())
    node_val = {}

    for i in range(min(len(nodes), len(vals))):
        node_val[nodes[i]] = vals[i]

    return node_val

def greedy_subgraph(G, node_val, start, k):
    picked = set([start])
    frontier = set(G.neighbors(start))

    while len(picked) < k and frontier:
        best = None
        best_val = -1

        for n in frontier:
            if n in node_val and node_val[n] > best_val:
                best_val = node_val[n]
                best = n

        if best is None:
            break

        frontier.remove(best)
        picked.add(best)

        for nb in G.neighbors(best):
            if nb not in picked:
                frontier.add(nb)

    return G.subgraph(picked).copy()

def score(subg, node_val):
    return sum(node_val.get(n, 0) for n in subg.nodes())

def get_top_subgraphs(G, node_val):
    sorted_nodes = sorted(node_val.items(), key=lambda x: x[1], reverse=True)
    seeds = [n for n, _ in sorted_nodes[:TOP_K_SEEDS]]

    results = []

    for s in seeds:
        sg = greedy_subgraph(G, node_val, s, SUBGRAPH_SIZE)
        sc = score(sg, node_val)
        results.append((s, sc, list(sg.nodes())))

    results.sort(key=lambda x: x[1], reverse=True)

    return results[:TOP_SUBGRAPHS]

############################################
# RUN ON REAL GRAPH
############################################

print("\n=== REAL GRAPH ===")

G_real = build_graph_from_csv(REAL_CSV)
node_val_real = map_shapley_to_nodes(G_real, vals)

real_top = get_top_subgraphs(G_real, node_val_real)

for i, (seed, sc, nodes) in enumerate(real_top):
    print(f"[REAL {i}] seed={seed} score={sc:.4f} nodes={nodes}")

############################################
# RUN ON NULL GRAPHS
############################################

null_results = {}

for fname in NULL_FILES:
    print(f"\n=== NULL GRAPH: {fname} ===")

    path = os.path.join(NULL_DIR, fname)
    G_null = build_graph_from_csv(path)

    node_val_null = map_shapley_to_nodes(G_null, vals)

    top_subs = get_top_subgraphs(G_null, node_val_null)

    null_results[fname] = top_subs

    for i, (seed, sc, nodes) in enumerate(top_subs):
        print(f"[{fname} {i}] seed={seed} score={sc:.4f} nodes={nodes}")

############################################
# SAVE EVERYTHING
############################################

out = {
    "real": real_top,
    "null": null_results
}

pickle.dump(out, open("subgraph_results.pkl", "wb"))

print("\nSaved to subgraph_results.pkl")
