"""
Graph I/O and construction utilities.

Functions for loading synapse and position data from PyTorch .pt files,
constructing directed and undirected graph representations, and exporting
graphs to PyTorch format.
"""

import torch
import numpy as np
import networkx as nx
from pathlib import Path


def load_pt(filename):
    """
    Load data from a PyTorch .pt file.

    Parameters
    ----------
    filename : str or Path
        Path to .pt file.

    Returns
    -------
    dict
        Loaded PyTorch dictionary.
    """
    return torch.load(filename, weights_only=False)


def load_synapses_from_pt(filename):
    """
    Load synapse data from PyTorch .pt file.

    Expects the `{edge_index: LongTensor[2, N], node_ids: list[str]}` schema
    produced by `data_prep.build_synapses`, and returns a dict
    `{synapse_index: [[source, target]]}` for callers that want the
    legacy shape (e.g. `build_synapse_digraph`).
    """
    data = load_pt(filename)
    edge_index = data['edge_index']
    node_ids = data['node_ids']
    return {
        i: [[node_ids[int(src)], node_ids[int(tgt)]]]
        for i, (src, tgt) in enumerate(edge_index.t().tolist())
    }


def load_positions_from_pt(filename):
    """
    Load neuron positions from PyTorch .pt file.

    Expected .pt file contains a dict with 'positions' (tensor) and 'node_ids' (list) keys.

    Parameters
    ----------
    filename : str or Path
        Path to positions .pt file.

    Returns
    -------
    dict
        Positions dictionary mapping neuron IDs to coordinate lists.
    """
    data = load_pt(filename)
    
    # Handle new .pt format with tensor + node_ids
    if isinstance(data, dict) and 'positions' in data and 'node_ids' in data:
        positions_tensor = data['positions']
        node_ids = data['node_ids']
        
        # Convert tensor to numpy array if needed
        if isinstance(positions_tensor, torch.Tensor):
            positions_array = positions_tensor.numpy()
        else:
            positions_array = np.array(positions_tensor)
        
        # Reconstruct dict: {node_id: [x, y, z], ...}
        return {node_ids[i]: positions_array[i].tolist() 
                for i in range(len(node_ids))}
    
    # Fallback: assume it's already a dict
    elif isinstance(data, dict) and 'positions' in data:
        positions_dict = data['positions']
        # Convert numpy arrays back to lists if needed
        if isinstance(positions_dict, dict):
            return {k: v.tolist() if isinstance(v, np.ndarray) else list(v) 
                    for k, v in positions_dict.items()}
        return positions_dict
    
    # Failsafe: return as dict if possible
    return data


def build_synapse_digraph(data):
    """
    Construct directed graph from synapse data.

    Loads synapse data, extracts all unique nodes and directed edges,
    and builds a directed NetworkX graph where edges preserve synaptic
    directionality from source to target neuron.

    Expected data structure: {synapse_key: [[source, target], ...], ...}
    (as produced by `load_synapses_from_pt`).

    Returns
    -------
    networkx.DiGraph
        Directed graph of neural connections preserving source→target orientation.
    """
    nodes = set()
    edges = set()
    for synid in data:
        nodes.add(data[synid][0][0])
        nodes.add(data[synid][0][1])
        edges.add((data[synid][0][0], data[synid][0][1]))
    G = nx.DiGraph(edges)
    return G


def export_graph_to_pt(G, output_filepath, node_positions=None, edge_features=None):
    """
    Export a NetworkX graph to PyTorch .pt format compatible with PyTorch Geometric.

    Parameters
    ----------
    G : networkx.Graph or networkx.DiGraph
        The graph to export.
    output_filepath : str or Path
        Output path for the .pt file.
    node_positions : dict, optional
        Mapping of node IDs to 3D coordinates. If provided, stored as 'pos' key.
    edge_features : np.ndarray or torch.Tensor, optional
        Edge feature matrix. If provided, stored as 'edge_attr' key.

    Returns
    -------
    None
        Saves graph to .pt file.
    """
    print(f"\nConverting NetworkX graph to PyTorch tensors...")
    
    # Create stable mapping from biological String IDs to integer indices
    node_ids = sorted(list(G.nodes()))
    id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
    
    # Extract edges and convert to integer indices
    sources = []
    targets = []
    
    for u, v in G.edges():
        sources.append(id_to_idx[u])
        targets.append(id_to_idx[v])
        
        # For undirected graphs, add reverse connection
        if not G.is_directed():
            sources.append(id_to_idx[v])
            targets.append(id_to_idx[u])
    
    # Create PyTorch edge_index tensor [2, num_edges]
    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    
    # Build dictionary
    graph_dict = {
        'edge_index': edge_index,
        'node_ids': node_ids,  # CRITICAL: Maps integer indices back to biological IDs
        'num_nodes': len(node_ids),
    }
    
    # Add optional fields
    if node_positions is not None:
        # Convert positions to tensor
        pos_array = np.array([node_positions[nid] for nid in node_ids])
        graph_dict['pos'] = torch.tensor(pos_array, dtype=torch.float)
    
    if edge_features is not None:
        if not isinstance(edge_features, torch.Tensor):
            edge_features = torch.tensor(edge_features, dtype=torch.float)
        graph_dict['edge_attr'] = edge_features
    
    # Save
    torch.save(graph_dict, output_filepath)
    print(f"✓ Saved PyTorch graph to: {output_filepath}")
    print(f"  -> Nodes: {len(node_ids)}")
    print(f"  -> Edge index shape: {edge_index.shape}")
    if node_positions is not None:
        print(f"  -> Node positions shape: {graph_dict['pos'].shape}")
    if edge_features is not None:
        print(f"  -> Edge features shape: {graph_dict['edge_attr'].shape}")


def export_positions_to_pt(positions_dict, output_filepath):
    """
    Export neuron positions to PyTorch .pt format.

    Parameters
    ----------
    positions_dict : dict
        Mapping of neuron IDs to 3D coordinates [x, y, z].
    output_filepath : str or Path
        Output path for the .pt file.

    Returns
    -------
    None
        Saves positions to .pt file.
    """
    # Convert to tensor format
    node_ids = sorted(list(positions_dict.keys()))
    positions_array = np.array([positions_dict[nid] for nid in node_ids])
    positions_tensor = torch.tensor(positions_array, dtype=torch.float)
    
    positions_data = {
        'positions': positions_tensor,
        'node_ids': node_ids,
    }
    
    torch.save(positions_data, output_filepath)
    print(f"✓ Saved positions to: {output_filepath}")
    print(f"  -> Shape: {positions_tensor.shape}")


def export_synapses_to_pt(synapses_dict, output_filepath):
    """
    Export synapse data to PyTorch .pt format.

    Parameters
    ----------
    synapses_dict : dict
        Synapse data {synapse_id: [[source, target], ...], ...}
    output_filepath : str or Path
        Output path for the .pt file.

    Returns
    -------
    None
        Saves synapses to .pt file.
    """
    synapses_data = {
        'synapses': synapses_dict,
    }
    
    torch.save(synapses_data, output_filepath)
    print(f"✓ Saved synapses to: {output_filepath}")
    print(f"  -> Total synapses: {len(synapses_dict)}")
