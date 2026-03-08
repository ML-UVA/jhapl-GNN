import json
import os
import numpy as np
from scipy.spatial import KDTree

# --- CONFIGURATION ---
DISTANCE_THRESHOLD_NM = 100_000 
CACHE_DIR = "cache_spatial" # Points to your symlink

def main():
    print("Loading synapses and positions...")
    
    # 1. Load Synapses
    synapses_path = os.path.join(CACHE_DIR, 'synapses.json')
    with open(synapses_path, 'r') as file:
        synapse_data = json.load(file)

    nodes_in_synapses = set()
    for synid in synapse_data:
        nodes_in_synapses.add(str(synapse_data[synid][0][0]))
        nodes_in_synapses.add(str(synapse_data[synid][0][1]))

    # 2. Load Positions
    positions_path = os.path.join(CACHE_DIR, 'positions.json')
    with open(positions_path, "r") as f:
        all_positions = json.load(f)

    # 3. Filter valid nodes
    valid_nodes = [n for n in nodes_in_synapses if n in all_positions]
    print(f"Total valid neurons for spatial graph: {len(valid_nodes)}")

    # 4. Build KD-Tree
    print(f"Building KD-Tree (Threshold: {DISTANCE_THRESHOLD_NM / 1000} µm)...")
    coords = np.array([all_positions[n] for n in valid_nodes])
    tree = KDTree(coords)

    # 5. Query pairs
    pairs = tree.query_pairs(DISTANCE_THRESHOLD_NM)

    # 6. Build Adjacency Dictionary
    adjacency = {n: [] for n in valid_nodes}
    for i, j in pairs:
        node_i = valid_nodes[i]
        node_j = valid_nodes[j]
        adjacency[node_i].append(node_j)
        adjacency[node_j].append(node_i)

    # 7. Print Stats and Save
    total_possible_edges = (len(valid_nodes) * (len(valid_nodes) - 1)) // 2
    edges_kept = len(pairs)
    negative = total_possible_edges - edges_kept

    print(f"Edges kept (Candidates for synapses): {edges_kept:,}")
    print(f"Edges rejected (Too far apart): {negative:,}")

    # Save directly to the cache folder
    output_path = os.path.join(CACHE_DIR, "adjacency.json")
    print(f"Saving to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(adjacency, f)
        
    print("Done! You are ready to run your chunking script.")

if __name__ == "__main__":
    main()