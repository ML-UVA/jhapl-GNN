import pickle
import networkx as nx
import os
import numpy as np
from scipy.spatial import cKDTree
from collections import defaultdict

def generate_dendrite_tree(KD_tree_data, data_path, neuron_to_idx, idx_to_neuron, block_id):

    Dendrite_Tree_List = {}

    blocks = defaultdict(list)
    block_owner = defaultdict(list)

    Dendrite_tree_path = os.path.join(data_path, "dendrite_kd_tree.pkl")
    Dendrite_owner_path = os.path.join(data_path, "dendrite_owner_dict.pkl")

    for i, (neuron, KD_data) in enumerate(KD_tree_data.items(), start=1):

        idx = neuron_to_idx[neuron]
        for point in KD_data['dendrite'].data:
            bid = block_id(point)
            blocks[bid].append(point)
            block_owner[bid].append(idx)

    for block, points in blocks.items():
        np_points = np.array(points)
        Dendrite_Tree_List[block] = cKDTree(np_points)

    #print(Dendrite_Tree_List)

    with open(Dendrite_tree_path, "wb") as f:
        pickle.dump(Dendrite_Tree_List, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(Dendrite_owner_path, "wb") as f:
        pickle.dump(block_owner, f, protocol=pickle.HIGHEST_PROTOCOL)


def generate_axon_tree(KD_tree_data, data_path, neuron_to_idx,idx_to_neuron,block_id):

    Axon_Tree_List = {}

    blocks = defaultdict(list)
    block_owner = defaultdict(list)

    Axon_tree_path = os.path.join(data_path, "axon_kd_tree.pkl")
    Axon_owner_path = os.path.join(data_path, "axon_owner_dict.pkl")

    for i, (neuron, KD_data) in enumerate(KD_tree_data.items(),start=1):

        idx = neuron_to_idx[neuron]
        for point in KD_data['axon'].data:
            bid = block_id(point)
            blocks[bid].append(point)
            block_owner[bid].append(idx)
    for block, points in blocks.items():
        np_points = np.array(points)
        Axon_Tree_List[block] = cKDTree(np_points)
    
    #print(Axon_Tree_List)

    with open(Axon_tree_path, "wb") as f:
        pickle.dump(Axon_Tree_List, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(Axon_owner_path, "wb") as f:
        pickle.dump(block_owner, f, protocol=pickle.HIGHEST_PROTOCOL)

def build_global_kd_trees(data_path,r=5000.0):



    KD_tree_data_path = os.path.join(data_path, "kd_tree_data_simple.pkl")
    neuron_to_idx_path = os.path.join(data_path, "neuron_to_idx.pkl")
    idx_to_neuron_path = os.path.join(data_path, "idx_to_neuron.pkl")


    with open(KD_tree_data_path, "rb") as f:
        KD_tree_data = pickle.load(f)

    neuron_ids = list(KD_tree_data.keys())
    neuron_to_idx = {nid: i for i, nid in enumerate(neuron_ids)}
    idx_to_neuron = {i: nid for nid, i in neuron_to_idx.items()}

    mins = np.array([np.inf, np.inf, np.inf])
    maxs = np.array([-np.inf, -np.inf, -np.inf])

    for neuron, data in KD_tree_data.items():
        for part in ("axon", "dendrite"):
            pts = data[part].data   # (N,3)
            if pts.size == 0:
                continue
            mins = np.minimum(mins, pts.min(axis=0))
            maxs = np.maximum(maxs, pts.max(axis=0))

    print("mins:", mins)
    print("maxs:", maxs)
    print("range:", maxs - mins)
    ranges = maxs - mins
    block_size = r

    extent = maxs - mins
    nblocks = np.ceil(extent / block_size).astype(int)
    print(f"number of blocks: {nblocks}")

    block_id = make_block_id(mins, block_size)

    generate_axon_tree(KD_tree_data,data_path, neuron_to_idx,idx_to_neuron,block_id)
    generate_dendrite_tree(KD_tree_data,data_path, neuron_to_idx,idx_to_neuron,block_id)

    with open(neuron_to_idx_path, "wb") as f:
        pickle.dump(neuron_to_idx, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(idx_to_neuron_path, "wb") as f:
        pickle.dump(idx_to_neuron, f, protocol=pickle.HIGHEST_PROTOCOL)


    
def make_block_id(mins,block_size):
    def block_id(p):
        return tuple(((p - mins) // block_size).astype(np.int32))
    return block_id