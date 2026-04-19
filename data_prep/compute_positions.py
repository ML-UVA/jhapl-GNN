"""
Extract neuron positions and compute pairwise distance graph.

This script:
1. Discovers all neuron IDs from graph_exports folder (by scanning .pbz2 files)
2. Loads positions from pickle files for each neuron
3. Saves positions to positions.pt (PyTorch format)
4. Computes pairwise distances between all neurons
5. Saves pairwise distances to distances.pt (PyTorch format, same format as compute_distance_graph.py)

Usage:
    # Process all neurons in graph_exports folder (with defaults)
    python3 compute_positions.py
    
    # Specify custom graph_exports folder
    python3 compute_positions.py ../../demo_graph_exports
    
    # Custom output paths
    python3 compute_positions.py ../../demo_graph_exports --positions custom_positions.pt --distances custom_distances.pt
"""

import sys
import json
import torch
import math
import argparse
import numpy as np
import networkx as nx
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_SYNAPSES_FILE = Path(__file__).parent.parent / 'data' / 'processed' / 'synapses.pt'
DEFAULT_GRAPH_DIR = Path(__file__).parent.parent / 'data' / 'raw' / 'graph_exports'
DEFAULT_POSITIONS_FILE = Path(__file__).parent.parent / 'data' / 'processed' / 'positions.pt'
DEFAULT_DISTANCE_GRAPH_FILE = Path(__file__).parent.parent / 'data' / 'processed' / 'distance_graph.gml'

# Distance threshold for creating edges (in spatial units)
DISTANCE_THRESHOLD = 1e6


# ============================================================================
# UTILITIES
# ============================================================================

def decompress_pickle(filename: str):
    """
    Load a compressed pickle file (.pbz2).

    Parameters
    ----------
    filename : str or Path
        Path to compressed pickle file.

    Returns
    -------
    object
        Decompressed object (typically a NetworkX graph).

    Raises
    ------
    FileNotFoundError
        If file doesn't exist.
    Exception
        If decompression fails.
    """
    import bz2
    import pickle

    filename = str(filename)
    if not filename.endswith('.pbz2'):
        filename += '.pbz2'

    if not Path(filename).exists():
        raise FileNotFoundError(f"File not found: {filename}")

    try:
        with bz2.BZ2File(filename, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        raise Exception(f"Failed to decompress {filename}: {e}")


def discover_neuron_ids(graph_dir: Path) -> List[int]:
    """
    Discover all neuron IDs in the graph_exports folder.

    Scans for .pbz2 files and extracts neuron IDs from filenames.

    Parameters
    ----------
    graph_dir : Path
        Directory containing pickle files.

    Returns
    -------
    list of int
        Sorted list of unique neuron IDs.
    """
    neuron_ids = set()
    
    for pkl_file in sorted(graph_dir.glob('*.pbz2')):
        try:
            # Extract neuron ID from filename (first part before '_')
            neuron_id = pkl_file.name.split('_')[0]+"_"+pkl_file.name.split('_')[1]
            neuron_ids.add(neuron_id)
        except (ValueError, IndexError):
            continue
    
    return sorted(list(neuron_ids))


def load_synapses_neuron_ids(synapses_file: Path) -> set:
    """
    Extract neuron IDs from synapses.json.

    Parameters
    ----------
    synapses_file : Path
        Path to synapses.json file.

    Returns
    -------
    set of int
        Neuron IDs that appear in synapses.
    """
    try:
        with open(synapses_file, 'r') as f:
            synapses_data = json.load(f)
        
        neuron_ids = set()
        for syn_id, syn_data in synapses_data.items():
            pre_id, post_id = syn_data[0]
            if pre_id != -1:
                neuron_ids.add(pre_id)
            if post_id != -1:
                neuron_ids.add(post_id)
        
        return neuron_ids
    except Exception as e:
        print(f"✗ Error loading synapses: {e}")
        return set()


def extract_neuron_position(graph_dir: Path, neuron_id: int, verbose: bool = False) -> Optional[Tuple[float, float, float]]:
    """
    Extract 3D position (mesh_center) from a neuron's pickle file.

    Attempts to load pickle file with variations in filename (_0_, _1_).

    Parameters
    ----------
    graph_dir : Path
        Directory containing pickle files.

    neuron_id : int
        Neuron ID to look up.

    verbose : bool
        Print debug messages.

    Returns
    -------
    tuple of (float, float, float) or None
        3D coordinates (x, y, z) or None if not found/failed.
    """
    # Try common filename patterns
    patterns = [
        f"{neuron_id}_auto_proof_v7_proofread.pbz2",
    ]

    for pattern in patterns:
        pkl_file = graph_dir / pattern
        try:
            if pkl_file.exists():
                if verbose:
                    print(f"  Loading {pattern}...", end=" ")
                
                graph = decompress_pickle(str(pkl_file))
                
                # Extract mesh_center from the "S0" node
                if "S0" in graph.nodes and "mesh_center" in graph.nodes["S0"]:
                    pos = graph.nodes["S0"]["mesh_center"]
                    if verbose:
                        print(f"✓ Position: {pos}")
                    return tuple(pos)
                else:
                    if verbose:
                        print("✗ No mesh_center at S0")
        except Exception as e:
            if verbose:
                print(f"✗ Error: {e}")
            continue

    if verbose:
        print(f"  ✗ Could not find position for neuron {neuron_id}")
    
    return None


def euclidean_distance(pos1: Tuple[float, float, float], 
                       pos2: Tuple[float, float, float]) -> float:
    """
    Compute Euclidean distance between two 3D points.

    Parameters
    ----------
    pos1 : tuple of (float, float, float)
        First 3D position (x1, y1, z1).

    pos2 : tuple of (float, float, float)
        Second 3D position (x2, y2, z2).

    Returns
    -------
    float
        Euclidean distance.
    """
    x1, y1, z1 = pos1
    x2, y2, z2 = pos2
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)


def compute_pairwise_distances(positions: Dict[int, Tuple[float, float, float]]) -> np.ndarray:
    """
    Compute pairwise distance matrix.

    Parameters
    ----------
    positions : dict
        {neuron_id: (x, y, z), ...}

    Returns
    -------
    np.ndarray
        Distance matrix of shape (n_neurons, n_neurons).
    """
    neuron_ids = sorted(positions.keys())
    n = len(neuron_ids)
    distances = np.zeros((n, n))

    for i, node_i in enumerate(neuron_ids):
        for j, node_j in enumerate(neuron_ids):
            if i >= j:
                continue
            dist = euclidean_distance(positions[node_i], positions[node_j])
            distances[i, j] = dist
            distances[j, i] = dist

    return distances, neuron_ids


def build_distance_graph(positions: Dict[int, Tuple[float, float, float]],
                         distance_threshold: float = DISTANCE_THRESHOLD) -> nx.Graph:
    """
    Build NetworkX graph where edge weight = Euclidean distance.

    Edges are created for all pairs within distance_threshold.

    Parameters
    ----------
    positions : dict
        {neuron_id: (x, y, z), ...}

    distance_threshold : float
        Maximum distance for creating an edge. Default: 1e6.

    Returns
    -------
    networkx.Graph
        Undirected graph with neurons as nodes and distances as edge weights.
    """
    G = nx.Graph()

    # Add nodes with positions as lists (not tuples, for GraphML compatibility)
    for neuron_id in positions:
        G.add_node(neuron_id, pos=list(positions[neuron_id]))

    # Add edges with distance weights
    neuron_ids = sorted(positions.keys())
    n = len(neuron_ids)

    edges_added = 0
    edges_skipped = 0

    for i, node_i in enumerate(neuron_ids):
        for j, node_j in enumerate(neuron_ids):
            if i >= j:
                continue

            dist = euclidean_distance(positions[node_i], positions[node_j])

            if dist <= distance_threshold:
                G.add_edge(node_i, node_j, weight=float(dist), distance=float(dist))
                edges_added += 1
            else:
                edges_skipped += 1

    print(f"\nDistance graph edges:")
    print(f"  - Within threshold ({distance_threshold}): {edges_added}")
    print(f"  - Beyond threshold: {edges_skipped}")

    return G


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def compute_positions_and_distances(synapses_file: Path, 
                                    graph_dir: Path,
                                    positions_file: Path,
                                    distance_graph_file: Path,
                                    distances_file: Path = None,
                                    verbose: bool = True) -> Tuple[Dict, nx.Graph]:
    """
    Extract positions and compute distance graph.

    Parameters
    ----------
    synapses_file : Path
        Path to synapses.json (unused, for compatibility).

    graph_dir : Path
        Directory containing pickle files.

    positions_file : Path
        Output file for positions.pt.

    distance_graph_file : Path
        Deprecated/unused parameter (for backward compatibility).

    distances_file : Path, optional
        Output file for pairwise distances matrix (default: distances.pt in same dir as graph_dir).

    verbose : bool
        Print progress messages.

    Returns
    -------
    tuple
        (positions_dict, networkx_graph)
    """
    print("=" * 80)
    print("Neural Network Position & Distance Extraction")
    print("=" * 80)

    # Validate inputs
    if synapses_file:
        synapses_file = Path(synapses_file)
    graph_dir = Path(graph_dir)

    if not graph_dir.exists():
        print(f"✗ Graph directory not found: {graph_dir.absolute()}")
        return {}, nx.Graph()

    print(f"\nInputs:")
    print(f"  Graph directory: {graph_dir.absolute()}")
    print()

    # ========================================================================
    # 1. DISCOVER ALL NEURON IDS
    # ========================================================================
    print("[1] Discovering neurons in graph_exports...")
    
    neuron_ids = discover_neuron_ids(graph_dir)
    print(f"  ✓ Found {len(neuron_ids)} neurons to process")

    # ========================================================================
    # 2. EXTRACT POSITIONS FROM PICKLE FILES
    # ========================================================================
    print(f"\n[2] Extracting positions from {len(neuron_ids)} neurons...")

    positions = {}
    successful = 0
    failed = 0

    for i, neuron_id in enumerate(neuron_ids, 1):
        try:
            if verbose:
                print(f"  [{i}/{len(neuron_ids)}] Neuron {neuron_id}...", end=" ")

            pos = extract_neuron_position(graph_dir, neuron_id, verbose=False)

            if pos is not None:
                positions[neuron_id] = pos
                if verbose:
                    print(f"✓")
                successful += 1
            else:
                if verbose:
                    print(f"✗ (no mesh_center)")
                failed += 1

        except Exception as e:
            if verbose:
                print(f"✗ ({str(e)[:40]})")
            failed += 1
            continue

    print(f"\n  Summary:")
    print(f"    - Successful: {successful}")
    print(f"    - Failed: {failed}")

    if successful == 0:
        print("✗ No positions extracted!")
        return {}, nx.Graph()

    # ========================================================================
    # 3. SAVE POSITIONS TO PYTORCH FORMAT
    # ========================================================================
    print(f"\n[3] Saving positions...")

    positions_file.parent.mkdir(parents=True, exist_ok=True)

    # Convert to PyTorch tensor format
    node_ids = sorted(list(positions.keys()))
    positions_array = np.array([positions[nid] for nid in node_ids])
    positions_tensor = torch.tensor(positions_array, dtype=torch.float)
    
    positions_data = {
        'positions': positions_tensor,
        'node_ids': node_ids,
    }
    
    torch.save(positions_data, positions_file)
    print(f"  ✓ Saved {len(positions)} positions to {positions_file.absolute()}")
    print(f"    - Shape: {positions_tensor.shape}")

    # ========================================================================
    # 4. COMPUTE PAIRWISE DISTANCES
    # ========================================================================
    print(f"\n[4] Computing pairwise distances...")

    distances, neuron_list = compute_pairwise_distances(positions)

    print(f"  ✓ Distance matrix: {distances.shape}")
    print(f"    - Min distance: {distances[distances > 0].min():.2f}")
    print(f"    - Max distance: {distances.max():.2f}")

    # ========================================================================
    # 5. BUILD DISTANCE GRAPH
    # ========================================================================
    print(f"\n[5] Building distance graph...")

    G = build_distance_graph(positions, DISTANCE_THRESHOLD)

    print(f"\nDistance graph:")
    print(f"  - Nodes: {G.number_of_nodes()}")
    print(f"  - Edges: {G.number_of_edges()}")
    print(f"  - Density: {nx.density(G):.4f}")

    # ========================================================================
    # 6. SAVE PAIRWISE DISTANCES TO PYTORCH FORMAT
    # ========================================================================
    if distances_file is None:
        distances_file = Path(positions_file).parent / 'distances.pt'
    
    print(f"\n[6] Saving pairwise distances...")
    
    distances_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save distances matrix in same format as compute_distance_graph.py
    # (nodes, edges, edge_weights)
    distances_data = {
        'nodes': neuron_list,
        'edges': [],
        'edge_weights': []
    }
    
    # Convert distance matrix to edge list format
    for i, nid_i in enumerate(neuron_list):
        for j, nid_j in enumerate(neuron_list):
            if i < j:
                dist = float(distances[i, j])
                distances_data['edges'].append((nid_i, nid_j))
                distances_data['edge_weights'].append(dist)
    
    distances_data['edges'] = [tuple(e) for e in distances_data['edges']]
    
    torch.save(distances_data, distances_file)
    print(f"  ✓ Saved distances to {distances_file.absolute()}")
    print(f"    - Neurons: {len(distances_data['nodes'])}")
    print(f"    - Edges: {len(distances_data['edges'])}")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("Extraction Complete!")
    print("=" * 80)
    print(f"\nOutput files:")
    print(f"  - Positions: {positions_file.absolute()}")
    print(f"  - Pairwise distances: {distances_file.absolute()}")
    print()

    return positions, G


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='Extract neuron positions from graph_exports folder',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Process all neurons in graph_exports:
    python3 compute_positions.py ../../demo_graph_exports
  
  Custom output paths:
    python3 compute_positions.py ../../demo_graph_exports --positions data/custom_positions.json
        """
    )
    
    parser.add_argument(
        'graph_dir',
        nargs='?',
        default=None,
        help='Path to graph_exports folder (default: data/raw/graph_exports)'
    )
    
    parser.add_argument(
        '--positions', '-p',
        type=str,
        default=None,
        help='Output path for positions.pt'
    )
    
    parser.add_argument(
        '--distance-graph', '-d',
        type=str,
        default=None,
        help='Output path for distance graph'
    )
    
    parser.add_argument(
        '--distances', '-D',
        type=str,
        default=None,
        help='Output path for pairwise distances matrix (default: data/processed/distances.pt)'
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    graph_dir = Path(args.graph_dir) if args.graph_dir else DEFAULT_GRAPH_DIR
    positions_file = Path(args.positions) if args.positions else DEFAULT_POSITIONS_FILE
    distances_file = Path(args.distances) if args.distances else (positions_file.parent / 'distances.pt')
    
    compute_positions_and_distances(
        None,
        graph_dir,
        positions_file,
        None,  # distance_graph_file (deprecated)
        distances_file,
        verbose=True
    )


if __name__ == '__main__':
    main()
