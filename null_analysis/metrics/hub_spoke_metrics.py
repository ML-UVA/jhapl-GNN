"""
Hub-spoke and degree heterogeneity metrics.

Computes graph-level statistics that characterize the degree distribution
and structure: Gini coefficient, coefficient of variation, degree assortativity,
and extremes (mean/max degree).
"""

import numpy as np
import networkx as nx


def gini(G):
    """
    Compute the Gini coefficient of the degree distribution.

    Measures degree inequality: 0 = uniform distribution, 1 = maximum inequality
    (all edges on one node). Quantifies hub-spoke structure.

    Parameters
    ----------
    G : networkx.Graph or networkx.DiGraph
        Input graph.

    Returns
    -------
    float
        Gini coefficient, in range [0, 1].
    """
    x = np.array([d for _, d in G.degree()])
    x = np.sort(np.array(x))
    n = len(x)
    return (2 * np.sum((np.arange(n) + 1) * x) / (n * np.sum(x))) - (n + 1) / n


def coef_variation(G):
    """
    Compute the coefficient of variation of degree distribution.

    Ratio of standard deviation to mean degree. Measures relative variability:
    higher values indicate more heterogeneous degree distribution.

    Parameters
    ----------
    G : networkx.Graph or networkx.DiGraph
        Input graph.

    Returns
    -------
    float
        Coefficient of variation (standard deviation / mean).
    """
    x = np.array([d for _, d in G.degree()])
    return x.std() / x.mean()


def mean_deg(G):
    """
    Compute mean node degree.

    Parameters
    ----------
    G : networkx.Graph or networkx.DiGraph
        Input graph.

    Returns
    -------
    float
        Average degree across all nodes.
    """
    x = np.array([d for _, d in G.degree()])
    return np.mean(x)


def max_deg(G):
    """
    Compute maximum node degree.

    Parameters
    ----------
    G : networkx.Graph or networkx.DiGraph
        Input graph.

    Returns
    -------
    int
        Highest degree in the network.
    """
    x = np.array([d for _, d in G.degree()])
    return np.max(x)


def deg_assortativity(G):
    """
    Compute degree assortativity coefficient.

    Measures tendency of high-degree nodes to connect to other high-degree nodes:
    positive = assortative (hubs cluster), negative = disassortative (hubs avoid each other),
    zero = no correlation.

    Parameters
    ----------
    G : networkx.Graph or networkx.DiGraph
        Input graph.

    Returns
    -------
    float
        Assortativity coefficient in range [-1, 1].
    """
    return nx.degree_assortativity_coefficient(G)