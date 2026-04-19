"""
Non-spatial random graph generators.

Implements classical random graph models commonly used as null models:
Erdős-Rényi, Configuration Model, Barabási-Albert, and Watts-Strogatz.
All functions return directed graphs as adjacency lists.
"""

import random


def erdos_renyi_directed(n, p, seed=None, self_loops=False):
    """
    Generate Erdős-Rényi random directed graph.

    Each possible directed edge exists independently with probability p.
    Produces graphs with Poisson degree distributions, uniformly random structure.

    Parameters
    ----------
    n : int
        Number of nodes.

    p : float
        Edge probability, in range [0, 1].

    seed : int, optional
        Random seed for reproducibility.

    self_loops : bool, optional
        If True, allow self-loops. Default: False.

    Returns
    -------
    dict
        Adjacency list: {node_id: [list of neighbors]}.
    """
    if seed is not None:
        random.seed(seed)

    adj = {i: [] for i in range(n)}

    for i in range(n):
        for j in range(n):
            if not self_loops and i == j:
                continue
            if random.random() < p:
                adj[i].append(j)

    return adj


def configuration_model_directed(in_degrees, out_degrees, seed=None):
    """
    Generate directed graph with specified degree sequence.

    Randomly assigns edges while respecting the given in/out degree requirements.
    Preserves degree distribution while randomizing topology (null model).

    Parameters
    ----------
    in_degrees : array-like
        Target in-degree for each node.

    out_degrees : array-like
        Target out-degree for each node.

    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict
        Adjacency list: {node_id: [list of neighbors]}.

    Raises
    ------
    ValueError
        If sum of in-degrees does not equal sum of out-degrees.
    """
    if seed is not None:
        random.seed(seed)

    if sum(in_degrees) != sum(out_degrees):
        raise ValueError("In-degree sum must equal out-degree sum")

    n = len(in_degrees)
    in_stubs = []
    out_stubs = []

    for i in range(n):
        in_stubs.extend([i] * in_degrees[i])
        out_stubs.extend([i] * out_degrees[i])

    random.shuffle(in_stubs)
    random.shuffle(out_stubs)

    adj = {i: [] for i in range(n)}

    for u, v in zip(out_stubs, in_stubs):
        adj[u].append(v)

    return adj


def barabasi_albert_directed(n, m, seed=None):
    """
    Generate Barabási-Albert preferential attachment graph.

    Starts with m fully connected nodes, then adds nodes one at a time,
    each connecting to m existing nodes preferentially by degree.
    Produces power-law degree distributions (scale-free property).

    Parameters
    ----------
    n : int
        Total number of nodes in the final graph.

    m : int
        Number of edges per new node (also initial clique size).

    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict
        Adjacency list: {node_id: [list of neighbors]}.
    """
    if seed is not None:
        random.seed(seed)

    adj = {i: [] for i in range(m)}

    for i in range(m):
        for j in range(m):
            if i != j:
                adj[i].append(j)

    in_degree_list = []
    for i in range(m):
        in_degree_list.extend([i] * len(adj[i]))

    for new_node in range(m, n):
        adj[new_node] = []

        targets = set()
        while len(targets) < m:
            targets.add(random.choice(in_degree_list))

        for t in targets:
            adj[new_node].append(t)

        in_degree_list.extend(list(targets))
        in_degree_list.extend([new_node] * m)

    return adj


def watts_strogatz_directed(n, k, p, seed=None):
    """
    Generate Watts-Strogatz small-world network.

    Starts with a ring of k nearest neighbors, then rewires edges with
    probability p. Produces networks with high clustering and short path lengths.

    Parameters
    ----------
    n : int
        Number of nodes arranged in a ring.

    k : int
        Each node initially connects to k nearest neighbors (k/2 on each side).
        Must be even.

    p : float
        Rewiring probability for each edge, in range [0, 1].

    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict
        Adjacency list: {node_id: [list of neighbors]}.

    Raises
    ------
    ValueError
        If k is odd.
    """
    if seed is not None:
        random.seed(seed)

    if k % 2 != 0:
        raise ValueError("k must be even")

    adj = {i: [] for i in range(n)}

    # Initial ring lattice with k nearest neighbors
    for i in range(n):
        for j in range(1, k // 2 + 1):
            adj[i].append((i + j) % n)

    # Rewire edges
    for i in range(n):
        for j in list(adj[i]):
            if random.random() < p:
                adj[i].remove(j)
                new = random.choice(
                    [x for x in range(n) if x != i and x not in adj[i]]
                )
                adj[i].append(new)

    return adj
