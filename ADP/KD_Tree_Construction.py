import pickle
import os
import time
import numpy as np
from scipy.spatial import KDTree

def build_KD_trees(data_path):
    """
    Build KD-tree strucutres for skeletonization data and save them to disk.

    Each neuron contains multiple branch skeletons; for each branch,
    a KD-tree is constructed.
    """

    skeletonization_path = os.path.join(data_path, "skeletonization_data.pkl")
    output_path = os.path.join(data_path, "KD_tree_data.pkl")

    with open(skeletonization_path, "rb") as f:
        skeletonization_data = pickle.load(f)

    time_start = time.time()

    KD_tree_dict = {}
    for i, (neuron, skel_dict) in enumerate(skeletonization_data.items(),start=1):
        
        KD_tree_dict[neuron] = {}
        for node, data in skel_dict.items():
            KD_tree_dict[neuron][node] = [data[0],KDTree(data[1])]
        if i%1000==0:
            print(f"Number of Neuron KD Tree Data Generated: {i}")
            print(f"Time iterated for 1000 neurons = {time.time()-time_start}")
            time_start = time.time()
    print(f"KD_tree_dict length: {len(KD_tree_dict)}")
    print(KD_tree_dict)
    with open(output_path, "wb") as f:
        pickle.dump(KD_tree_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
