"""
Graph I/O and construction utilities.

Functions for loading synapse data from JSON files and constructing
undirected and directed graph representations.
"""

import json
import networkx as nx


def read_json(filename):
    """
    Load JSON data from file.

    Parameters
    ----------
    filename : str
        Path to JSON file.

    Returns
    -------
    dict
        Parsed JSON object.
    """
    with open(filename, 'r') as file:
        data = json.load(file)
    return data


def build_synapse_graph():
    """
    Construct undirected graph from synapse data.

    Loads synapse JSON file, extracts all unique nodes and edges,
    and builds an undirected NetworkX graph where nodes are neurons
    and edges represent synaptic connections (directionality ignored).

    Expected JSON structure: {synapse_id: [[source, target], ...], ...}

    Returns
    -------
    networkx.Graph
        Undirected graph of neural connections.

    Notes
    -----
    File path is hardcoded to "../../processed/raw/synapses.json"
    relative to this module's location.
    """
    data = read_json("../../processed/raw/synapses.json")
    nodes = set()
    edges = set()
    for synid in data:
        nodes.add(data[synid][0][0])
        nodes.add(data[synid][0][1])
        edges.add((data[synid][0][0], data[synid][0][1]))
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return G


def build_synapse_digraph(data):
    """
    Construct directed graph from synapse data.

    Loads synapse JSON file, extracts all unique nodes and directed edges,
    and builds a directed NetworkX graph where edges preserve synaptic
    directionality from source to target neuron.

    Expected JSON structure: {synapse_id: [[source, target], ...], ...}

    Returns
    -------
    networkx.DiGraph
        Directed graph of neural connections preserving source→target orientation.

    Notes
    -----
    File path is hardcoded to "../../processed/raw/synapses.json"
    relative to this module's location.
    """
    nodes = set()
    edges = set()
    for synid in data:
        nodes.add(data[synid][0][0])
        nodes.add(data[synid][0][1])
        edges.add((data[synid][0][0], data[synid][0][1]))
    G = nx.DiGraph(edges)
    return G
