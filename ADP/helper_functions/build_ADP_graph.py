import pickle
import argparse
import networkx as nx
import os
import torch
import numpy as np


def build_graph(base_path, threshold):
    neuron_ids_path = os.path.join(base_path, "neuron_ids.pkl")
    with open(neuron_ids_path, "rb") as f:
        neuron_ids = pickle.load(f)

    input_file = os.path.join(base_path, "adp_data.pkl")

    edges = [[], []]
    edge_values = []
    index_values = {}

    output_file = os.path.join(
        base_path,
        f"adp_graph_threshold_{threshold}.pt"
    )

    with open(input_file, "rb") as f:
        adp_dict = pickle.load(f)

    # map neuron_id -> index
    for i, neuron_id in enumerate(neuron_ids):
        index_values[neuron_id] = i

    print(f"Loaded ADP dictionary from {input_file}")

    # ===== COLLECT ALL ADP VALUES (for distribution stats) =====
    all_adp_values = []

    total_possible_edges = 0

    for a, targets in adp_dict.items():
        for b, adp in targets.items():
            total_possible_edges += 1
            all_adp_values.append(adp)

            if adp >= threshold:
                a_index = index_values[a]
                b_index = index_values[b]
                edges[0].append(a_index)
                edges[1].append(b_index)
                edge_values.append(adp)

    all_adp_values = np.array(all_adp_values)
    edge_values_np = np.array(edge_values)

    # ===== TORCH =====
    torch_edges = torch.tensor(edges, dtype=torch.long)
    torch_edge_values = torch.tensor(edge_values, dtype=torch.float)

    output = {
        'edge_index': torch_edges,
        'edge_attr': torch_edge_values,
        'node_ids': neuron_ids
    }

    # ==========================================================
    # ===================== STATISTICS ==========================
    # ==========================================================

    print("\n========== ADP DISTRIBUTION (RAW) ==========")
    print("Total pairs:", total_possible_edges)
    print("Min:", np.min(all_adp_values))
    print("Max:", np.max(all_adp_values))
    print("Mean:", np.mean(all_adp_values))
    print("Std:", np.std(all_adp_values))
    print("Median:", np.median(all_adp_values))
    print("90th percentile:", np.percentile(all_adp_values, 90))
    print("99th percentile:", np.percentile(all_adp_values, 99))

    print("\n========== THRESHOLD EFFECT ==========")
    print("Threshold:", threshold)
    print("Edges kept:", len(edge_values))
    print("Edges removed:", total_possible_edges - len(edge_values))
    print("Keep ratio:", len(edge_values) / total_possible_edges)

    print("\n========== GRAPH STATS ==========")
    num_nodes = len(neuron_ids)
    num_edges = len(edge_values)

    print("Nodes:", num_nodes)
    print("Edges:", num_edges)

    if num_nodes > 1:
        density = num_edges / (num_nodes * (num_nodes - 1))
        print("Density:", density)

    if num_edges > 0:
        print("Edge weight stats (AFTER threshold):")
        print("  Min:", np.min(edge_values_np))
        print("  Max:", np.max(edge_values_np))
        print("  Mean:", np.mean(edge_values_np))
        print("  Std:", np.std(edge_values_np))

    # ===== NETWORKX ANALYSIS (important structure insight) =====
    print("\n========== NETWORK STRUCTURE ==========")

    G = nx.DiGraph()
    for i in range(num_nodes):
        G.add_node(i)

    for i in range(num_edges):
        G.add_edge(edges[0][i], edges[1][i], weight=edge_values[i])

    if num_edges > 0:
        degrees = [d for _, d in G.degree()]
        print("Avg degree:", np.mean(degrees))
        print("Max degree:", np.max(degrees))
        print("Min degree:", np.min(degrees))

        components = list(nx.weakly_connected_components(G))
        print("Connected components:", len(components))
        print("Largest component size:", max(len(c) for c in components))

    torch.save(output, output_file)
    print(f"\nGraph saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Build ADP graph with thresholding")

    parser.add_argument(
        "base_path",
        help="Directory containing adp_data.pkl"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Minimum ADP value to include edge"
    )

    args = parser.parse_args()

    build_graph(args.base_path, args.threshold)


if __name__ == "__main__":
    main()