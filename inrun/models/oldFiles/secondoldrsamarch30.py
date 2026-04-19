import pickle
import pandas as pd
import networkx as nx
import os

############################################
# PATHS
############################################

REAL_CSV = "data/top5_k1.csv"

NULL_DIR = "data/null_graphs"
NULL_FILES = [
    "null_graph_0.csv",
    "null_graph_1.csv",
    "null_graph_2.csv",
    "null_graph_3.csv"
]

SUBGRAPH_RESULTS = "subgraph_results.pkl"

############################################
# LOAD GRAPH
############################################

def build_graph_from_csv(path):
    df = pd.read_csv(path)

    if "pre_id" in df.columns:
        edges = zip(df["pre_id"], df["post_id"])
    else:
        edges = zip(df["source"], df["target"])

    G = nx.Graph()
    for u, v in edges:
        G.add_edge(u, v)

    return G

############################################
# COUNT SUBGRAPH OCCURRENCES
############################################

def count_subgraph_occurrences(G, pattern_nodes, pattern_edges):
    """
    Count how many times a subgraph shape appears in G
    using isomorphism matching
    """

    pattern = nx.Graph()
    pattern.add_nodes_from(pattern_nodes)
    pattern.add_edges_from(pattern_edges)

    matcher = nx.algorithms.isomorphism.GraphMatcher(G, pattern)

    count = 0
    for _ in matcher.subgraph_isomorphisms_iter():
        count += 1

    return count

############################################
# MAIN
############################################

# Load results
data = pickle.load(open(SUBGRAPH_RESULTS, "rb"))

real_subgraphs = data["real"]

# Build real graph
G_real = build_graph_from_csv(REAL_CSV)

print("\n=== REAL GRAPH COUNTS ===")

pattern_list = []

for i, (seed, score, nodes) in enumerate(real_subgraphs):
    subg = G_real.subgraph(nodes)

    edges = list(subg.edges())

    pattern_list.append((nodes, edges))

    count = count_subgraph_occurrences(G_real, nodes, edges)

    print(f"[REAL motif {i}] count = {count}")

############################################
# NULL GRAPHS
############################################

for fname in NULL_FILES:
    print(f"\n=== NULL GRAPH: {fname} ===")

    path = os.path.join(NULL_DIR, fname)
    G_null = build_graph_from_csv(path)

    for i, (nodes, edges) in enumerate(pattern_list):
        count = count_subgraph_occurrences(G_null, nodes, edges)
        print(f"[{fname} motif {i}] count = {count}")
