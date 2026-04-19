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
    print(f"Loading graph from {path}", flush=True)

    df = pd.read_csv(path)

    if "pre_id" in df.columns:
        edges = zip(df["pre_id"], df["post_id"])
    else:
        edges = zip(df["source"], df["target"])

    G = nx.Graph()
    for u, v in edges:
        G.add_edge(u, v)

    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges", flush=True)
    return G

############################################
# COUNT SUBGRAPH OCCURRENCES
############################################

def count_subgraph_occurrences(G, pattern_edges, max_matches=1000000000):
    """
    Count how many times a subgraph shape appears in G
    using isomorphism matching
    """

    pattern = nx.Graph()
    pattern.add_edges_from(pattern_edges)

    matcher = nx.algorithms.isomorphism.GraphMatcher(G, pattern)

    count = 0

    for _ in matcher.subgraph_isomorphisms_iter():
        count += 1

        # safety break (optional)
        if count >= max_matches:
            print("Reached max_matches limit, stopping early...", flush=True)
            break

    return count

############################################
# MAIN
############################################

print("Loading subgraph results...", flush=True)

data = pickle.load(open(SUBGRAPH_RESULTS, "rb"))
real_subgraphs = data["real"]

# Build real graph
G_real = build_graph_from_csv(REAL_CSV)

print("\n=== REAL GRAPH COUNTS ===", flush=True)

pattern_list = []

for i, (seed, score, nodes) in enumerate(real_subgraphs):
    print(f"\nProcessing REAL motif {i}", flush=True)

    subg = G_real.subgraph(nodes)
    edges = list(subg.edges())

    pattern_list.append(edges)

    print(f"Counting occurrences...", flush=True)
    count = count_subgraph_occurrences(G_real, edges)

    print(f"[REAL motif {i}] count = {count}", flush=True)

############################################
# NULL GRAPHS
############################################

for fname in NULL_FILES:
    print(f"\n=== NULL GRAPH: {fname} ===", flush=True)

    path = os.path.join(NULL_DIR, fname)
    G_null = build_graph_from_csv(path)

    for i, edges in enumerate(pattern_list):
        print(f"Counting motif {i} in {fname}...", flush=True)

        count = count_subgraph_occurrences(G_null, edges)

        print(f"[{fname} motif {i}] count = {count}", flush=True)

print("\nDONE", flush=True)
