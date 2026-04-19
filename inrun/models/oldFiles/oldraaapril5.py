import pickle
import pandas as pd
import networkx as nx
import os

from dotmotif import Motif
from dotmotif.executors import GrandIsoExecutor


REAL_CSV = "data/top5_k1.csv"
NULL_DIR = "data/null_graphs"
NULL_FILES = [
    "null_graph_0.csv",
    "null_graph_1.csv",
    "null_graph_2.csv",
    "null_graph_3.csv"
]
SUBGRAPH_RESULTS = "subgraph_results.pkl"



def build_graph_from_csv(path):
    print(f"Loading graph from {path}", flush=True)
    
    df = pd.read_csv(path)
    
    if "pre_id" in df.columns:
        edges = zip(df["pre_id"], df["post_id"])
    else:
        edges = zip(df["source"], df["target"])
    
    G = nx.DiGraph()
    
    for u, v in edges:
        G.add_edge(u, v)
    
    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges", flush=True)
    return G


def edges_to_motif(edges):

    
    node_map = {}
    counter = 0
    
    lines = []
    for u, v in edges:
        if u not in node_map:
            node_map[u] = f"N{counter}"
            counter += 1
        if v not in node_map:
            node_map[v] = f"N{counter}"
            counter += 1
        
        lines.append(f"{node_map[u]} -> {node_map[v]}")
    
    return "\n".join(lines)

def count_with_dotmotif(executor, motif_str):
    motif = Motif(motif_str)
    results = executor.find(motif)
    return len(results)

print("Loading subgraph results...", flush=True)
data = pickle.load(open(SUBGRAPH_RESULTS, "rb"))
real_subgraphs = data["real"]

G_real = build_graph_from_csv(REAL_CSV)
executor_real = GrandIsoExecutor(graph=G_real)

print("\n=== REAL GRAPH COUNTS ===", flush=True)
pattern_list = []

for i, (seed, score, nodes) in enumerate(real_subgraphs):
    print(f"\nProcessing REAL motif {i}", flush=True)
    
    subg = G_real.subgraph(nodes)
    edges = list(subg.edges())
    
    if len(edges) < 4:
        print("Skipping trivial motif", flush=True)
        continue
    
    motif_str = edges_to_motif(edges)
    pattern_list.append(motif_str)
    
    print("Motif definition:")
    print(motif_str)
    
    print("Counting occurrences...", flush=True)
    count = count_with_dotmotif(executor_real, motif_str)
    
    print(f"[REAL motif {i}] count = {count}", flush=True)


for fname in NULL_FILES:
    print(f"\n=== NULL GRAPH: {fname} ===", flush=True)
    
    path = os.path.join(NULL_DIR, fname)
    G_null = build_graph_from_csv(path)
    executor_null = GrandIsoExecutor(graph=G_null)
    
    for i, motif_str in enumerate(pattern_list):
        print(f"\nCounting motif {i} in {fname}...", flush=True)
        
        count = count_with_dotmotif(executor_null, motif_str)
        
        print(f"[{fname} motif {i}] count = {count}", flush=True)

print("\nDONE", flush=True)
