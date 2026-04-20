"""y
motifs.py

Extracts and visualizes top-k motifs from the connectome graph
based on normalized In-Run Data Shapley values.

Can be run standalone or imported as a module.

USAGE:
    python motifs.py \
        --graph_path    results/filtered_graph.adjlist \
        --shapley_path  results/graph_node_shapley_normalized.value \
        --motif_sizes   5 10 20 \
        --top_k         3 \
        --output_dir    results
"""

import argparse
import os
import pickle
import networkx as nx 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



def extract_and_visualize(
    graph_path,
    shapley_path,
    motif_sizes=[5, 10, 20],
    top_k=3,
    output_dir='results',
):

    os.makedirs(output_dir, exist_ok=True)


    print("[motifs] Loading graph...")
    G = nx.read_adjlist(graph_path, create_using=nx.MultiDiGraph(), nodetype=int)
    print(f"[motifs] Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    print("[motifs] Loading normalized Shapley values...")
    with open(shapley_path, "rb") as f:
        shap_data = pickle.load(f)

    shap_vals = shap_data["Normalized Shapley (%)"]


    neuron_ids = list(G.nodes())
    neuron_to_idx = {nid: i for i, nid in enumerate(neuron_ids)}

    print(f"[motifs] Shapley length: {len(shap_vals)}, Mapping length: {len(neuron_to_idx)}")

    def motif_score(nodes):
        return sum(shap_vals[neuron_to_idx[n]] for n in nodes if n in neuron_to_idx)

    def directed_density(H):
        n = len(H.nodes())
        if n <= 1:
            return 0
        return H.number_of_edges() / (n * (n - 1))

    def extract_motifs(size, top_k=3):
        used_nodes = set()
        motifs = []
        valid_nodes = [n for n in G.nodes() if n in neuron_to_idx]
        sorted_nodes = sorted(valid_nodes,
                              key=lambda n: shap_vals[neuron_to_idx[n]],
                              reverse=True)
        for seed in sorted_nodes:
            if seed in used_nodes:
                continue
            motif_nodes = {seed}
            frontier = {n for n in (set(G.successors(seed)) | set(G.predecessors(seed)))
                        if n in neuron_to_idx and n not in used_nodes}
            while len(motif_nodes) < size and frontier:
                best = max(frontier, key=lambda n: shap_vals[neuron_to_idx[n]])
                motif_nodes.add(best)
                new_neighbors = {n for n in (set(G.successors(best)) | set(G.predecessors(best)))
                                 if n in neuron_to_idx and n not in used_nodes}
                frontier |= new_neighbors
                frontier -= motif_nodes
            if len(motif_nodes) == size:
                motifs.append(motif_nodes)
                used_nodes |= motif_nodes
            if len(motifs) == top_k:
                break
        return motifs

    def visualize_motif(nodes, size, motif_id):
        H = G.subgraph(nodes).copy()
        print(f"\nMotif Size {size} - #{motif_id}")
        print("Total Shapley:", motif_score(nodes))
        print("Directed Density:", directed_density(H))
        print("Nodes:", list(nodes))

        H_simple = nx.DiGraph()
        for u, v in H.edges():
            if H_simple.has_edge(u, v):
                H_simple[u][v]["weight"] += 1
            else:
                H_simple.add_edge(u, v, weight=1)

        node_colors  = [shap_vals[neuron_to_idx[n]] for n in H_simple.nodes()]
        edge_weights = [H_simple[u][v]["weight"] for u, v in H_simple.edges()]
        pos          = nx.spring_layout(H_simple, k=1.8, iterations=100, seed=1)
        labels       = {n: str(n)[-6:] for n in H_simple.nodes()}

        fig, ax = plt.subplots(figsize=(10, 10))
        nx.draw(H_simple, pos, node_color=node_colors, cmap="viridis",
                node_size=900, with_labels=False,
                width=[w * 0.7 for w in edge_weights],
                edge_color="black", arrowsize=20, ax=ax)
        nx.draw_networkx_labels(H_simple, pos, labels=labels,
                                font_size=9, font_color="black")
        sm = plt.cm.ScalarMappable(cmap="viridis",
                                   norm=plt.Normalize(vmin=min(shap_vals),
                                                      vmax=max(shap_vals)))
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="Normalized Shapley (%)")
        ax.set_title(f"Motif Size {size} - #{motif_id}")
        plt.savefig(os.path.join(output_dir, f"motif_size{size}_{motif_id}_graph.png"), dpi=300)
        plt.close()

        node_list = list(nodes)
        adj = np.zeros((len(node_list), len(node_list)))
        for u, v in H.edges():
            i = node_list.index(u)
            j = node_list.index(v)
            adj[i][j] += 1
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        sns.heatmap(adj, cmap="magma", square=True, ax=ax2)
        ax2.set_title(f"Adjacency Heatmap Size {size} - #{motif_id}")
        plt.savefig(os.path.join(output_dir, f"motif_size{size}_{motif_id}_heatmap.png"), dpi=300)
        plt.close()

    for size in motif_sizes:
        print(f"\n[motifs] Extracting motifs of size {size}...")
        motifs = extract_motifs(size, top_k=top_k)
        for i, nodes in enumerate(motifs):
            visualize_motif(nodes, size, i + 1)

    print("\n[motifs] Done.")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_path',   type=str,           default='results/filtered_graph.adjlist')  
    parser.add_argument('--shapley_path', type=str,           default='results/graph_node_shapley_normalized.value')
    parser.add_argument('--motif_sizes',  type=int, nargs='+', default=[5, 10, 20])
    parser.add_argument('--top_k',        type=int,           default=3)
    parser.add_argument('--output_dir',   type=str,           default='results')
    args = parser.parse_args()

    extract_and_visualize(
        graph_path=args.graph_path,
        shapley_path=args.shapley_path,
        motif_sizes=args.motif_sizes,
        top_k=args.top_k,
        output_dir=args.output_dir,
    )
