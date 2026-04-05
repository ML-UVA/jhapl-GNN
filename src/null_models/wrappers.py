"""
Unified null model interface.

All wrapper functions follow the standard interface:
    wrapper(GT: nx.Graph, bin_model=None, **kwargs) -> nx.Graph

This allows flexible iteration over multiple null models with a consistent API.
"""

import networkx as nx
from .non_spatial_null_models import (
    erdos_renyi_directed,
    configuration_model_directed,
    barabasi_albert_directed,
    watts_strogatz_directed
)
from .spatial_null_model import generate_spatial_null

try:
    from config import RANDOM_SEED
except ImportError:
    from ..config import RANDOM_SEED


# ============================================================================
# Non-Spatial Null Models
# ============================================================================

def ER(GT: nx.Graph, bin_model=None, **kwargs):
    """
    Erdős-Rényi null model matching edge density of ground truth.

    Computes the edge probability p from GT's density and generates
    a random graph with the same number of nodes.

    Parameters
    ----------
    GT : networkx.Graph or networkx.DiGraph
        Ground truth graph to match.

    bin_model : None
        Unused (for interface compatibility).

    **kwargs
        Unused additional arguments.

    Returns
    -------
    networkx.Graph
        Erdős-Rényi random graph.
    """
    n = GT.number_of_nodes()
    m = GT.number_of_edges()
    p = (2 * m) / (n * (n - 1)) if n > 1 else 0
    adj = erdos_renyi_directed(n, p, RANDOM_SEED)
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for u in adj:
        for v in adj[u]:
            G.add_edge(u, v)
    return G


def configuration(GT: nx.Graph, bin_model=None, **kwargs):
    """
    Configuration Model null matching degree sequence of ground truth.

    Preserves the degree distribution while randomizing connections.
    In/out degrees are identical (symmetric degree sequence).

    Parameters
    ----------
    GT : networkx.Graph or networkx.DiGraph
        Ground truth graph to match.

    bin_model : None
        Unused (for interface compatibility).

    **kwargs
        Unused additional arguments.

    Returns
    -------
    networkx.Graph
        Configuration model random graph.
    """
    degree_seq = [d for _, d in GT.degree()]
    adj = configuration_model_directed(degree_seq, degree_seq, RANDOM_SEED)
    
    n = GT.number_of_nodes()
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for u in adj:
        for v in adj[u]:
            G.add_edge(u, v)
    return G


def BA(GT: nx.Graph, bin_model=None, **kwargs):
    """
    Barabási-Albert scale-free null model.

    Creates preferential attachment graph with same number of nodes
    and edges as ground truth.

    Parameters
    ----------
    GT : networkx.Graph or networkx.DiGraph
        Ground truth graph to match.

    bin_model : None
        Unused (for interface compatibility).

    **kwargs
        Unused additional arguments.

    Returns
    -------
    networkx.Graph
        Barabási-Albert random graph.
    """
    n = GT.number_of_nodes()
    m = GT.number_of_edges()
    m_param = max(1, m // n)  # Edges per new node
    adj = barabasi_albert_directed(n, m_param, RANDOM_SEED)
    
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for u in adj:
        for v in adj[u]:
            G.add_edge(u, v)
    return G


def smallworld(GT: nx.Graph, bin_model=None, **kwargs):
    """
    Watts-Strogatz small-world null model.

    Creates small-world network with same number of nodes and derived
    parameters from ground truth edge density and degree.

    Parameters
    ----------
    GT : networkx.Graph or networkx.DiGraph
        Ground truth graph to match.

    bin_model : None
        Unused (for interface compatibility).

    **kwargs
        Unused additional arguments.

    Returns
    -------
    networkx.Graph
        Watts-Strogatz random graph.
    """
    n = GT.number_of_nodes()
    m = GT.number_of_edges()
    p = (2 * m) / (n * (n - 1)) if n > 1 else 0
    k = max(2, min(n - 1, int(2 * m / n)))
    if k % 2 != 0:
        k -= 1  # Ensure k is even
    
    adj = watts_strogatz_directed(n, k, p, RANDOM_SEED)
    
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for u in adj:
        for v in adj[u]:
            G.add_edge(u, v)
    return G


# ============================================================================
# Spatial Null Model
# ============================================================================

def spatial_null(GT: nx.Graph, bin_model, pair_features, **kwargs):
    """
    Spatial null model using empirical P(edge | feature).

    Preserves spatial structure by sampling edges based on a binning model
    of edge probabilities conditioned on a spatial feature (e.g., distance).

    Parameters
    ----------
    GT : networkx.Graph or networkx.DiGraph
        Ground truth graph (used for node list and edge count).

    bin_model : BinModel
        Binning model from compute_bins() with empirical P(edge | feature).

    pair_features : list of tuple
        [(node1, node2, feature_value), ...] for all node pairs.

    **kwargs
        Optional:
        - target_edges (int): Match this edge count instead of GT's
        - seed (int): Random seed for reproducibility

    Returns
    -------
    networkx.Graph
        Spatial null model graph.
    """
    target_edges = kwargs.get('target_edges', GT.number_of_edges())
    seed = kwargs.get('seed', RANDOM_SEED)
    
    G = generate_spatial_null(
        nodes=list(GT.nodes()),
        pair_features=pair_features,
        bin_model=bin_model,
        target_edges=target_edges,
        seed=seed
    )
    return G


# ============================================================================
# Wrapper Registry
# ============================================================================

NULL_MODELS = {
    'ER': ER,
    'configuration': configuration,
    'BA': BA,
    'smallworld': smallworld,
    'spatial_null': spatial_null,
}

def get_null_model(name):
    """
    Get a null model wrapper by name.

    Parameters
    ----------
    name : str
        Name of null model: 'ER', 'configuration', 'BA', 'smallworld', or 'spatial_null'.

    Returns
    -------
    callable
        Null model function.

    Raises
    ------
    KeyError
        If name not in registered null models.
    """
    if name not in NULL_MODELS:
        raise KeyError(f"Unknown null model: {name}. Options: {list(NULL_MODELS.keys())}")
    return NULL_MODELS[name]
