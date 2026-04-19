import numpy as np
import networkx as nx


def generate_spatial_null(
    nodes,
    pair_features,
    bin_model,
    target_edges=None,
    seed=None
):
    """
    Generate a spatial null graph using empirical P(edge | feature).

    Parameters
    ----------
    nodes : list
        List of node identifiers.

    pair_features : list of tuples
        [(i, j, feature_value), ...] for all measurable pairs.

    bin_model : BinModel
        Output of compute_bins().

    target_edges : int or None
        If provided, prune edges until this count is reached.

    seed : int or None
        Random seed.

    Returns
    -------
    networkx.Graph
    """

    rng = np.random.default_rng(seed)

    G = nx.Graph()
    G.add_nodes_from(nodes)

    # sample edges
    for i, j, feature in pair_features:

        p = bin_model.lookup_prob(feature)

        if rng.random() < p:
            G.add_edge(i, j)

    # match observed edge count if requested
    if target_edges is not None:

        edges = list(G.edges())

        if len(edges) > target_edges:

            remove_n = len(edges) - target_edges
            remove_idx = rng.choice(len(edges), size=remove_n, replace=False)

            for idx in remove_idx:
                G.remove_edge(*edges[idx])

    return G