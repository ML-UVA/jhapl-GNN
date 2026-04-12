import pickle
import networkx as nx
import os
import numpy as np
from scipy.spatial import KDTree
from collections import defaultdict
import time

def convert_adp(data_path):
    

    idx_to_neuron_path = os.path.join(data_path, "idx_to_neuron.pkl")

    adp_dict_path = os.path.join(data_path, "adp_dict.pkl")

    adp_data = {}

    with open(adp_dict_path, "rb") as f:
        adp_dict = pickle.load(f)
    with open(idx_to_neuron_path, "rb") as f:
        idx_to_neuron = pickle.load(f)

    for dend_idx in adp_dict:
        adp_data[idx_to_neuron[dend_idx]] = {}
        for axon_idx in adp_dict[dend_idx]:
            adp_data[idx_to_neuron[dend_idx]][idx_to_neuron[axon_idx]] = adp_dict[dend_idx][axon_idx]
    
    out_path = os.path.join(data_path, "adp_data.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(adp_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    

