import pickle
import os
import time
import numpy as np
from scipy.spatial import cKDTree

def build_kd_trees(data_path):
    """
    Build KD-tree strucutres for skeletonization data and save them to disk.

    Each neuron contains multiple branch skeletons; for each branch,
    a KD-tree is constructed.
    """

    skeletonization_path = os.path.join(data_path, "skeletonization_data_simple.pkl")
    output_path = os.path.join(data_path, "kd_tree_data_simple.pkl")

    with open(skeletonization_path, "rb") as f:
        skeletonization_data = pickle.load(f)

    time_start = time.time()

    KD_tree_dict = {}
    for i, (neuron, skel_dict) in enumerate(skeletonization_data.items(),start=1):
        
        KD_tree_dict[neuron] = {}
        axon_pts = skeletonization_data[neuron]["axon"].astype(np.float32, copy=False)
        den_pts  = skeletonization_data[neuron]["dendrite"].astype(np.float32, copy=False)


        KD_tree_dict[neuron]["axon"] = cKDTree(axon_pts)
        KD_tree_dict[neuron]["dendrite"] = cKDTree(den_pts)

    print(f"KD_tree_dict length: {len(KD_tree_dict)}")
    with open(output_path, "wb") as f:
        pickle.dump(KD_tree_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
