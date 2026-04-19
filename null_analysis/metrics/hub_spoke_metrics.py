"""
Hub-spoke and degree heterogeneity metrics.

Computes graph-level statistics that characterize the degree distribution
and structure: Gini coefficient, coefficient of variation, degree assortativity,
extremes (mean/max degree), and synapse type ratios.
"""

import numpy as np
import networkx as nx
from typing import Dict, Optional


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


def synapse_type_ratio(
    G: nx.DiGraph,
    neuron_types: Dict,
    synapses: Optional[Dict] = None
) -> Dict:
    """
    Compute ratio of synapse types: exc-inh, exc-exc, inh-inh.

    Quantifies the breakdown of synapse connectivity by neuron type (excitatory/inhibitory).
    Values are reported as fractions of total synapses.

    Parameters
    ----------
    G : networkx.DiGraph
        Ground truth directed graph.
    neuron_types : dict
        Mapping from neuron_id to neuron_type ('E' for excitatory, 'I' for inhibitory).
        Neurons not in this dict are treated as unknown type.
    synapses : dict, optional
        Synapse data dict mapping syn_id -> [[pre_id, post_id], synapse_data].
        If provided, uses actual synapse counts; otherwise uses edge counts from G.

    Returns
    -------
    dict
        Ratios keyed by synapse type:
        - 'exc_inh': Exc→Inh synapses / total
        - 'exc_exc': Exc→Exc synapses / total
        - 'inh_inh': Inh→Inh synapses / total
        - 'inh_exc': Inh→Exc synapses / total
        - 'unknown': Unknown type synapses / total
    """
    counts = {
        'exc_inh': 0,   # Exc→Inh
        'exc_exc': 0,   # Exc→Exc
        'inh_inh': 0,   # Inh→Inh
        'inh_exc': 0,   # Inh→Exc
        'unknown': 0,   # Unknown
    }

    if synapses is not None:
        # Count by actual synapses
        total = 0
        for syn_id, syn_data in synapses.items():
            pre_id, post_id = syn_data[0]
            if pre_id == -1 or post_id == -1:
                total += 1  # Incomplete synapse
                counts['unknown'] += 1
                continue

            pre_type = neuron_types.get(pre_id, '?')
            post_type = neuron_types.get(post_id, '?')

            key = None
            if pre_type == 'E' and post_type == 'E':
                key = 'exc_exc'
            elif pre_type == 'E' and post_type == 'I':
                key = 'exc_inh'
            elif pre_type == 'I' and post_type == 'I':
                key = 'inh_inh'
            elif pre_type == 'I' and post_type == 'E':
                key = 'inh_exc'
            else:
                key = 'unknown'

            counts[key] += 1
            total += 1
    else:
        # Count by graph edges (single edge per pair)
        total = G.number_of_edges()
        for u, v in G.edges():
            u_type = neuron_types.get(u, '?')
            v_type = neuron_types.get(v, '?')

            key = None
            if u_type == 'E' and v_type == 'E':
                key = 'exc_exc'
            elif u_type == 'E' and v_type == 'I':
                key = 'exc_inh'
            elif u_type == 'I' and v_type == 'I':
                key = 'inh_inh'
            elif u_type == 'I' and v_type == 'E':
                key = 'inh_exc'
            else:
                key = 'unknown'

            counts[key] += 1

    # Convert to ratios
    result = {}
    if total > 0:
        for key in counts:
            result[key] = counts[key] / total
    else:
        result = {key: 0.0 for key in counts}

    return result
