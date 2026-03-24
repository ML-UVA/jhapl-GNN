"""
Graph visualization and spatial filtering utilities.

Functions for filtering neurons by spatial proximity, constructing
subgraph views, and visualizing networks with PCA decomposition.
"""

import numpy as np
from sklearn.decomposition import PCA


def filter_neurons(neurons, coords, R):
    """
    Filter neurons by spatial distance from the region centroid.

    Selects neurons within radius R of the centroid of the coordinate cloud.
    Useful for extracting local subnetworks in spatially embedded neural systems.

    Parameters
    ----------
    neurons : list
        Neuron identifiers.

    coords : np.ndarray
        Array of shape (n_neurons, 3) with 3D coordinates for each neuron.

    R : float
        Spatial radius threshold for filtering.

    Returns
    -------
    tuple
        - rsub_neurons : list of filtered neuron identifiers
        - sub_coords : np.ndarray of filtered coordinates (subset of input)
    """
    center = coords.mean(axis=0)
    dists = np.linalg.norm(coords - center, axis=1)
    keep = dists < R
    rsub_neurons = [neurons[i] for i in np.where(keep)[0]]
    sub_coords = coords[keep]
    return rsub_neurons, sub_coords


def build_partial_graph(sub_neurons, edgeset):
    """
    Extract subgraph edges connecting a subset of neurons.

    Given a set of neurons and a global edge list, returns only edges
    where both endpoints are in the neuron subset.

    Parameters
    ----------
    sub_neurons : list
        Subset of neuron identifiers to retain.

    edgeset : list of tuple
        All edges as (source, target) tuples (strings or ints).

    Returns
    -------
    list of tuple
        Edges where both endpoints are in sub_neurons.
    """
    sub_set = set(sub_neurons)
    sub_edges = [(u, v) for (u, v) in edgeset
                 if str(u) in sub_set and str(v) in sub_set]
    return sub_edges


def decompose(coords):
    """
    Reduce coordinates to 2D using Principal Component Analysis.

    Projects high-dimensional coordinates onto the first two principal
    components for visualization purposes.

    Parameters
    ----------
    coords : np.ndarray
        Array of shape (n, d) where d >= 2.

    Returns
    -------
    np.ndarray
        Array of shape (n, 2) with 2D projected coordinates.
    """
    pca = PCA(n_components=2)
    xy = pca.fit_transform(coords)
    return xy


def plot_vis(sub_neurons, sub_edges, decomp):
    """
    Visualize network structure in 2D space using PCA coordinates.

    Plots neurons as scatter points and edges as connecting lines with
    transparency. Useful for visualizing small spatial subgraphs.

    Parameters
    ----------
    sub_neurons : list
        Neuron identifiers in the subgraph.

    sub_edges : list of tuple
        Edges as (source, target) pairs.

    decomp : np.ndarray
        Array of shape (n, 2) with 2D coordinates (output of decompose()).

    Returns
    -------
    None
        Displays matplotlib figure.
    """
    import matplotlib.pyplot as plt
    idx = {n: i for i, n in enumerate(sub_neurons)}

    plt.figure(figsize=(6, 6))

    for u, v in sub_edges:
        i, j = idx[str(u)], idx[str(v)]
        plt.plot([decomp[i, 0], decomp[j, 0]],
                 [decomp[i, 1], decomp[j, 1]],
                 alpha=0.25)
    plt.scatter(decomp[:, 0], decomp[:, 1], s=12, c="black")
    plt.axis("equal")
    plt.title("Local neuron subgraph")
    plt.show()