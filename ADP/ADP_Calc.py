import pickle
import networkx as nx
import os
import numpy as np
from scipy.spatial import KDTree

def find_point_pairs(ADP_dict, KD_tree_data, neuron1, neuron2, r):
    ADP_dict.setdefault(neuron1, {})

    # find axon tree for neuron2

    axon_tree = None
    for node, data in KD_tree_data[neuron2].items():
        if data[0] == "axon":
            axon_tree = data[1]
            break
    if axon_tree is None:
        return

    # iterate neuron1 non-axon structures

    total_co_travel_dist = 0
    per_node_close_points = {}

    for node, (label,points) in KD_tree_data[neuron1].items():
        if label == "axon":
            continue

        neighbors = points.query_ball_tree(axon_tree, r)

        close_points_list = []

        for idx, hits in enumerate(neighbors):
            if hits:  # at least one axon point nearby
                close_points_list.append(points.data[idx])

        if close_points_list:
            close_points = np.array(close_points_list)
            per_node_close_points[node] = close_points
            total_co_travel_dist += close_points.shape[0]
    
    if per_node_close_points:
        ADP_dict[neuron1][neuron2] = {
            "total_co_travel_dist": total_co_travel_dist,
            "close_points": per_node_close_points,
        }


def calc_ADP(data_path):

    KD_tree_path = os.path.join(data_path, "KD_tree_data.pkl")


    
    with open(KD_tree_path, "rb") as f:
        KD_tree_data = pickle.load(f)

    
    print(len(KD_tree_data))


    ADP_dict = {}

    for i, dendritic_neuron in enumerate(KD_tree_data):
        print(i)
        ADP_dict[dendritic_neuron] = {}
        for j, axonal_neuron in enumerate(KD_tree_data):
            if dendritic_neuron==axonal_neuron:
                continue
            find_point_pairs(ADP_dict,KD_tree_data,dendritic_neuron,axonal_neuron,5.0)
        if i %100:
            print(i)
    with open(data_path, "wb") as f:
        pickle.dump(ADP_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
