import pickle
import networkx as nx
import os
import numpy as np
from scipy.spatial import KDTree
from collections import defaultdict
import time

def neighbor_blocks(bid):
    x,y,z = bid
    for dx in (-1,0,1):
        for dy in (-1,0,1):
            for dz in (-1,0,1):
                yield (x+dx, y+dy, z+dz)
def int_dict():
    return defaultdict(int)

def calc_adp(data_path,r=5000.0):
    ADP = defaultdict(int_dict)

    axon_tree_path = os.path.join(data_path, "axon_kd_tree.pkl")
    axon_owner_path = os.path.join(data_path, "axon_owner_dict.pkl")

    dend_tree_path = os.path.join(data_path, "dendrite_kd_tree.pkl")
    dend_owner_path = os.path.join(data_path, "dendrite_owner_dict.pkl")

    neuron_to_idx_path = os.path.join(data_path, "neuron_to_idx.pkl")
    idx_to_neuron_path = os.path.join(data_path, "idx_to_neuron.pkl")

    ADP_path = os.path.join(data_path,"adp_dict.pkl")

    
    with open(axon_tree_path, "rb") as f:
        axon_blocks = pickle.load(f)

    with open(axon_owner_path, "rb") as f:
        axon_owners = pickle.load(f)

    with open(dend_tree_path, "rb") as f:
        dend_blocks = pickle.load(f)

    with open(dend_owner_path, "rb") as f:
        dend_owners = pickle.load(f)
    with open(neuron_to_idx_path, "rb") as f:
        neuron_to_idx = pickle.load(f)
    with open(idx_to_neuron_path, "rb") as f:
        idx_to_neuron = pickle.load(f)

    time_start = time.time()
    
    for i, (bid, dend_tree) in enumerate(dend_blocks.items(),start=1):
        if i%100 == 0:
            print(f"iteration number: {i}")
            print(f"time taken for last iteration batch: {time.time()-time_start}")
            time_start = time.time()
        if dend_tree.n == 0:
            continue
        dend_owner_list = dend_owners[bid]

        seen_per_dendrite = {
                d_idx: set() for d_idx in range(dend_tree.n)
        }

        for neighbor in neighbor_blocks(bid):
            if neighbor not in axon_blocks:
                continue

            axon_tree = axon_blocks[neighbor]
            axon_owner_list = axon_owners[neighbor]

            hits = dend_tree.query_ball_tree(axon_tree, r)
            

            for d_idx, axon_hit_indices in enumerate(hits):
                if not axon_hit_indices:
                    continue

                d_neuron = dend_owner_list[d_idx]

                # get unique axon neurons hit by this dendrite point
                for a_idx in axon_hit_indices:
                    a_neuron = axon_owner_list[a_idx]
                    
                    if a_neuron == d_neuron:
                        continue
                    if a_neuron in seen_per_dendrite[d_idx]:
                        continue
                    seen_per_dendrite[d_idx].add(a_neuron)    
                    ADP[d_neuron][a_neuron] += 1
    ADP = {k: dict(v) for k, v in ADP.items()}
                       
    print(len(ADP))
    with open(ADP_path, "wb") as f:
        pickle.dump(ADP, f, protocol=pickle.HIGHEST_PROTOCOL)
        