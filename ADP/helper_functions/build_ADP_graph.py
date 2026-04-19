import pickle
import argparse
import networkx as nx
import os


def build_graph(base_path, threshold):
    input_file = os.path.join(base_path, "adp_data.pkl")

    output_file = os.path.join(
        base_path,
        f"adp_graph_threshold_{threshold}.pkl"
    )

    with open(input_file, "rb") as f:
        adp_dict = pickle.load(f)

    print(f"Loaded ADP dictionary from {input_file}")

    G = nx.DiGraph()

    for a, targets in adp_dict.items():
        for b, adp in targets.items():
            if adp >= threshold:
                G.add_edge(b, a, adp=adp)

    print("Graph built")
    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())

    with open(output_file, "wb") as f:
        pickle.dump(G, f)

    print(f"Graph saved to {output_file}")


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
        help="Minimum ADP value to include edge (default: 0)"
    )

    args = parser.parse_args()

    build_graph(args.base_path, args.threshold)


if __name__ == "__main__":
    main()