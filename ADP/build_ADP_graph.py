import pickle
import argparse
import networkx as nx


def build_graph(input_file, output_file, threshold):
    with open(input_file, "rb") as f:
        adp_dict = pickle.load(f)

    print("Loaded ADP dictionary")

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

    parser.add_argument("input_file", help="Path to ADP dictionary pickle file")
    parser.add_argument("output_file", help="Output graph pickle file")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Minimum ADP value to include edge (default: 0)"
    )

    args = parser.parse_args()

    build_graph(args.input_file, args.output_file, args.threshold)


if __name__ == "__main__":
    main()