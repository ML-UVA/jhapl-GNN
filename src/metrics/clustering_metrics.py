"""
Local and global clustering metrics.

Computes clustering and transitivity statistics that characterize the
prevalence of triangles and local closure in the network.
"""

import networkx as nx


def clustering(G):
    """
    Compute average clustering coefficient.

    For each node, clustering coefficient is the fraction of possible edges
    between its neighbors that actually exist. Returns simple average across
    all nodes. Ranges from 0 (no triangles) to 1 (complete closure).

    Parameters
    ----------
    G : networkx.Graph or networkx.DiGraph
        Input graph.

    Returns
    -------
    float
        Average clustering coefficient.
    """
    return nx.average_clustering(G)


def transitivity(G):
    """
    Compute global transitivity (clustering from a global perspective).

    Ratio of actual triangles to all connected triples of nodes.
    More robust to isolated nodes than average clustering coefficient.

    Parameters
    ----------
    G : networkx.Graph or networkx.DiGraph
        Input graph.

    Returns
    -------
    float
        Global transitivity (ratio in range [0, 1]).
    """
    return nx.transitivity(G)


def triangles(G):
    """
    Compute mean number of triangles per node.

    For each node, counts triangles containing that node, then averages
    across all nodes. Captures local clustering density.

    Parameters
    ----------
    G : networkx.Graph or networkx.DiGraph
        Input graph.

    Returns
    -------
    float
        Average triangle count per node.
    """
    ttt = nx.triangles(G)
    s = 0
    for n in ttt:
        s += ttt[n]
    return s / len(ttt)
