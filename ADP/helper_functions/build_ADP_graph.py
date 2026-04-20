import pickle
import argparse
import networkx as nx
import os
import torch


def build_graph(base_path, threshold, output_dir=None):
    """Build the thresholded ADP graph.

    Reads ``neuron_ids.pkl`` and ``adp_data.pkl`` from ``base_path``. The
    resulting graph is written to ``output_dir`` (defaults to ``base_path``
    for backwards compatibility).
    """
    neuron_ids_path = os.path.join(base_path, "neuron_ids.pkl")
    with open(neuron_ids_path,"rb") as f:
        neuron_ids = pickle.load(f)
    input_file = os.path.join(base_path, "adp_data.pkl")
    edges = [[],[]]
    edge_values = []
    index_values = {}

    if output_dir is None:
        output_dir = base_path
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(
        output_dir,
        f"adp_graph_threshold_{threshold}.pt"
    )

    with open(input_file, "rb") as f:
        adp_dict = pickle.load(f)
    

    for i,neuron_id in enumerate(neuron_ids):
        index_values[neuron_id]=i

    print(f"Loaded ADP dictionary from {input_file}")

    for a, targets in adp_dict.items():
        for b, adp in targets.items():
            a_index = index_values[a]
            b_index = index_values[b]
            if adp >= threshold:
                edges[0].append(a_index)
                edges[1].append(b_index)
                edge_values.append(adp)
    
    torch_edges=torch.tensor(edges,dtype=torch.long)
    torch_edge_values = torch.tensor(edge_values,dtype=torch.float)
    
    output = {}
    output['edge_index'] = torch_edges
    output['edge_attr']=torch_edge_values
    output['node_ids'] = neuron_ids


    print("Graph built")
    print("Nodes:", len(neuron_ids))
    print("Edges:", len(edge_values))

    torch.save(output,output_file)
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