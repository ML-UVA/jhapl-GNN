"""
Extract neural network synapses from compressed pickle files.

This module processes NetworkX neuron graphs and extracts synapse connectivity
information. Can be used as an importable module or run as a script.

Usage as module:
    from data_prep.extract_synapses import extract_synapses
    synapses = extract_synapses(graph_dir, distance_graph_pt, output_file)

Usage as script:
    python -m data_prep.extract_synapses ../data/raw/graph_exports \\
        ../data/processed/distance_graph.pt ../data/processed/synapses.pt
"""

import sys
import torch
import bz2
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

# ============================================================================
# CONFIGURATION
# ============================================================================

# Default paths (relative to project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_GRAPH_DIR = PROJECT_ROOT / 'data' / 'raw' / 'graph_exports'
DEFAULT_DISTANCE_GRAPH = PROJECT_ROOT / 'data' / 'processed' / 'distance_graph.pt'
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / 'data' / 'processed' / 'synapses.pt'


# ============================================================================
# UTILITIES
# ============================================================================

def decompress_pickle(filename: str) -> Any:
    """
    Load a compressed pickle file (.pbz2).

    Parameters
    ----------
    filename : str or Path
        Path to compressed pickle file.

    Returns
    -------
    Any
        Decompressed object (typically a NetworkX graph).

    Raises
    ------
    FileNotFoundError
        If file doesn't exist.
    Exception
        If decompression fails.
    """
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


def extract_neuron_id(filename: str) -> str:
    """
    Extract neuron ID from filename.

    Assumes filename format: {neuron_id}_*.pbz2
    e.g., "864691135740581995_0_auto_proof_v7_proofread.pbz2"

    Parameters
    ----------
    filename : str
        Filename of pickle file.

    Returns
    -------
    str
        Neuron ID parsed from filename.
    """
    return filename.split("_")[0] + "_" + filename.split("_")[1]


def extract_synapses_from_graph(G, neuron_id: str) -> Dict[int, List]:
    """
    Extract synapse data from a neuron graph.

    For each node with synapse_data, creates synapse records mapping
    pre/post-synaptic partners to their IDs.

    Parameters
    ----------
    G : networkx.Graph
        Neuron morphology graph with synapse annotations.

    neuron_id : str
        ID of neuron this graph represents.

    Returns
    -------
    dict
        Synapse records: {syn_id: [[pre_id, post_id], synapse_data], ...}
    """
    synapses = {}

    if not hasattr(G, 'nodes'):
        return synapses

    for node in G.nodes():
        node_data = G.nodes[node]

        if 'synapse_data' not in node_data:
            continue

        for syn_data in node_data['synapse_data']:
            try:
                # Normalize data types
                syn_data['upstream_dist'] = float(syn_data.get('upstream_dist', 0))
                syn_data['syn_id'] = int(syn_data['syn_id'])
                syn_data['volume'] = int(syn_data.get('volume', 0))

                syn_id = int(syn_data['syn_id'])

                # Determine if pre-synaptic (pos=0) or post-synaptic (pos=1)
                is_presynaptic = syn_data.get('syn_type') == 'presyn'
                pos = 0 if is_presynaptic else 1

                # Initialize synapse entry
                if syn_id not in synapses:
                    synapses[syn_id] = [[-1, -1], syn_data]

                # Add neuron ID to appropriate position
                if synapses[syn_id][0][pos] == -1:
                    synapses[syn_id][0][pos] = neuron_id
                else:
                    # Log if duplicate (shouldn't happen in well-formed data)
                    print(f"  ⚠ Synapse {syn_id} already has "
                          f"{'pre' if pos == 0 else 'post'}-synaptic neuron "
                          f"{synapses[syn_id][0][pos]}, skipping neuron {neuron_id}")

            except (KeyError, ValueError, TypeError) as e:
                print(f"  ✗ Error processing synapse data: {e}")
                continue

    return synapses


def merge_synapses(all_synapses: List[Dict]) -> Dict:
    """
    Merge synapse data from multiple neurons.

    Parameters
    ----------
    all_synapses : list of dict
        List of synapse dicts from individual neurons.

    Returns
    -------
    dict
        Merged synapse dict with complete (pre and post) synapses only.
    """
    merged = {}

    for syn_dict in all_synapses:
        for syn_id, syn_data in syn_dict.items():
            if syn_id not in merged:
                merged[syn_id] = syn_data
            else:
                # Check if we can complete this synapse
                pre_id, post_id = merged[syn_id][0]
                new_pre_id, new_post_id = syn_data[0]

                if pre_id == -1 and new_pre_id != -1:
                    merged[syn_id][0][0] = new_pre_id
                if post_id == -1 and new_post_id != -1:
                    merged[syn_id][0][1] = new_post_id

    # Filter to complete synapses only
    complete_synapses = {
        syn_id: syn_data
        for syn_id, syn_data in merged.items()
        if -1 not in syn_data[0]
    }

    return complete_synapses


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def extract_synapses(
    graph_dir: Path,
    distance_graph_pt: Optional[Path] = None,
    output_file: Optional[Path] = None,
    verbose: bool = True
) -> int:
    """
    Extract synapses from all pickle files in a directory.

    Parameters
    ----------
    graph_dir : Path
        Directory containing .pbz2 pickle files.

    distance_graph_pt : Path, optional
        Path to distance graph .pt file (for validation/filtering).
        If None, skipped.

    output_file : Path, optional
        Path to write synapses.pt. If None, returns in-memory dict only.

    verbose : bool
        Print progress messages. Default: True.

    Returns
    -------
    int
        Number of complete synapses extracted.
    """
    print("=" * 80)
    print("Neural Network Synapse Extraction")
    print("=" * 80)

    # Validate input directory
    graph_dir = Path(graph_dir)
    if not graph_dir.exists():
        print(f"✗ Graph directory not found: {graph_dir.absolute()}")
        return 0

    if not graph_dir.is_dir():
        print(f"✗ Not a directory: {graph_dir.absolute()}")
        return 0

    # Load distance graph if provided
    distance_graph = None
    if distance_graph_pt is not None:
        distance_graph_pt = Path(distance_graph_pt)
        if distance_graph_pt.exists():
            try:
                distance_graph_data = torch.load(distance_graph_pt, weights_only=False)
                distance_graph = distance_graph_data
                if verbose:
                    print(f"Loaded distance graph from {distance_graph_pt.absolute()}")
            except Exception as e:
                print(f"⚠ Warning: Could not load distance graph: {e}")

    # Find all pickle files
    pickle_files = sorted(graph_dir.glob('*.pbz2'))
    if not pickle_files:
        print(f"✗ No .pbz2 files found in {graph_dir.absolute()}")
        return 0

    print(f"\nFound {len(pickle_files)} pickle files")
    print(f"Graph directory: {graph_dir.absolute()}")
    if output_file:
        print(f"Output file: {output_file.absolute()}")
    print()

    # Process each pickle file
    all_synapses: List[Dict] = []
    successful = 0
    failed = 0

    for i, pkl_file in enumerate(pickle_files, 1):
        try:
            if verbose:
                print(f"[{i}/{len(pickle_files)}] Processing {pkl_file.name}...", end=" ")

            neuron_id = extract_neuron_id(pkl_file.name)
            graph = decompress_pickle(str(pkl_file))

            synapses = extract_synapses_from_graph(graph, neuron_id)
            all_synapses.append(synapses)

            if verbose:
                print(f"✓ ({len(synapses)} synapses)")

            successful += 1

        except Exception as e:
            if verbose:
                print(f"✗ Error: {e}")
            failed += 1
            continue

    # Merge and filter synapses
    print(f"\nMerging synapses from {successful} neurons...")
    complete_synapses = merge_synapses(all_synapses)

    print(f"Complete synapses (pre & post identified): {len(complete_synapses)}")

    # Write output in PyTorch format if output file specified
    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        synapses_data = {
            'synapses': complete_synapses,
        }

        torch.save(synapses_data, output_file)
        print(f"\n✓ Synapses written to {output_file.absolute()}")

    print(f"\nSummary:")
    print(f"  - Processed: {successful} neurons")
    print(f"  - Failed: {failed} neurons")
    print(f"  - Complete synapses: {len(complete_synapses)}")
    print("=" * 80)

    return len(complete_synapses)


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """Command-line interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Extract synapses from compressed pickle graph files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m data_prep.extract_synapses ../data/raw/graph_exports
  python -m data_prep.extract_synapses \\
    ../data/raw/graph_exports \\
    --distance-graph ../data/processed/distance_graph.pt \\
    --output ../data/processed/synapses.pt
        """
    )

    parser.add_argument(
        'graph_dir',
        nargs='?',
        default=DEFAULT_GRAPH_DIR,
        help=f'Directory containing .pbz2 files (default: {DEFAULT_GRAPH_DIR})'
    )
    parser.add_argument(
        '--distance-graph',
        default=DEFAULT_DISTANCE_GRAPH,
        help=f'Distance graph .pt file (default: {DEFAULT_DISTANCE_GRAPH})'
    )
    parser.add_argument(
        '--output',
        default=DEFAULT_OUTPUT_FILE,
        help=f'Output synapses.pt file (default: {DEFAULT_OUTPUT_FILE})'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Print progress messages'
    )

    args = parser.parse_args()

    extract_synapses(
        graph_dir=args.graph_dir,
        distance_graph_pt=args.distance_graph,
        output_file=args.output,
        verbose=args.verbose
    )


if __name__ == '__main__':
    main()
