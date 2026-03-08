import json
import networkx as nx

def read_json(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

def build_synapse_graph():
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

def build_synapse_digraph():
    data = read_json("../../processed/raw/synapses.json")
    nodes = set()
    edges = set()
    for synid in data:
        nodes.add(data[synid][0][0])
        nodes.add(data[synid][0][1])
        edges.add((data[synid][0][0], data[synid][0][1]))
    G = nx.DiGraph(edges)
    return G
