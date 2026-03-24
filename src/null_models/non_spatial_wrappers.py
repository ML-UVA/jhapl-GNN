"""
Wrapper functions for null model generators.

Parameterizes non-spatial random graph generators based on properties
of an empirical ground truth graph, providing consistent null models
for comparative analysis.
"""

from null_models.non_spatial_null_models import (
    erdos_renyi_directed,
    configuration_model_directed,
    barabasi_albert_directed,
    watts_strogatz_directed
)
import networkx as nx
from config import RANDOM_SEED


def ER(GT: nx.Graph):
    """
    Generate Erdős-Rényi null model matching edge density of ground truth.

    Computes the edge probability p from GT's density and generates
    a random graph with same number of nodes.

    Parameters
    ----------
    GT : networkx.Graph or networkx.DiGraph
        Ground truth graph to match.

    Returns
    -------
    dict
        Erdős-Rényi adjacency list.
    """
    n = GT.number_of_nodes()
    m = GT.number_of_edges()
    p = (2 * m) / (n * (n - 1))
    return erdos_renyi_directed(n, p, RANDOM_SEED)


def configuration(GT: nx.Graph):
    """
    Generate Configuration Model matching degree sequence of ground truth.

    Preserves the degree distribution while randomizing connections.
    In/out degrees are identical (symmetric degree sequence).

    Parameters
    ----------
    GT : networkx.Graph or networkx.DiGraph
        Ground truth graph to match.

    Returns
    -------
    dict
        Configuration model adjacency list.
    """
    degree_seq = [d for _, d in GT.degree()]
    return configuration_model_directed(degree_seq, degree_seq, RANDOM_SEED)


def BA(GT: nx.Graph):
    """
    Generate Barabási-Albert scale-free null model.

    Creates preferential attachment graph with same number of nodes
    and edges as ground truth.

    Parameters
    ----------
    GT : networkx.Graph or networkx.DiGraph
        Ground truth graph to match.

    Returns
    -------
    dict
        Barabási-Albert adjacency list.
    """
    n = GT.number_of_nodes()
    m = GT.number_of_edges()
    return barabasi_albert_directed(n, m, RANDOM_SEED)


def smallworld(GT: nx.Graph):
    """
    Generate Watts-Strogatz small-world null model.

    Creates small-world network with same number of nodes and derived
    parameters from ground truth edge density and degree.

    Parameters
    ----------
    GT : networkx.Graph or networkx.DiGraph
        Ground truth graph to match.

    Returns
    -------
    dict
        Watts-Strogatz adjacency list.

    Notes
    -----
    The k parameter is derived from GT.degree() count, which may not be
    appropriate for all graph types. Consider alternative parameterization
    for directed graphs or graphs with highly variable degrees.
    """
    n = GT.number_of_nodes()
    m = GT.number_of_edges()
    p = (2 * m) / (n * (n - 1))
    k = len(GT.degree())
    return watts_strogatz_directed(n, k, p, RANDOM_SEED)