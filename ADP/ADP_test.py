import networkx as nx
import bz2
import pickle
import os
import sys

import time
import torch
import numpy as np

neuron_directory = "../../../demo_graph_exports"

rel_data_directory = "../demo_data/adp_data.pkl"
neuron_ids_path = "../demo_data/neuron_ids.pkl"
g_path = "../demo_data/adp_graph_threshold_0.pt"

with open(rel_data_directory, 'rb') as f:
    data = pickle.load(f)

with open(neuron_ids_path, 'rb') as f:
    neuron_ids = pickle.load(f)
g= torch.load(g_path)


print(neuron_ids)
#print(data)
print(len(data))
print(len(neuron_ids))

print(g)

files = [f for f in os.listdir(neuron_directory) if os.path.isfile(os.path.join(neuron_directory, f)) and f.endswith(".pbz2")]

print(len(files))

for (i,file) in enumerate(files,start=1):

    path = os.path.join(neuron_directory, file)
    fname = os.path.basename(path)
    neuron_id = "_".join(fname.split("_")[0:2])
    print(neuron_id)

    with bz2.open(path, 'rb') as f:
        G = pickle.load(f)
        axon_list = []
        dendrite_list = []

        skeleton_dict = {"axon":np.empty((0,3)),"dendrite":np.empty((0,3))}

        G_dict = dict(list(G.nodes(data=True)))
        print(sys.getsizeof(G_dict))

        for node in G_dict:
            if "axon_compartment" in G_dict[node]:
                if G_dict[node]['axon_compartment'] == "axon":
                    axon_list.append(G_dict[node]["skeleton_data"])
                else:
                    dendrite_list.append(G_dict[node]["skeleton_data"])
        if len(axon_list)>0:
            skeleton_dict["axon"] = np.vstack(axon_list)
        if len(dendrite_list)>0:
            skeleton_dict["dendrite"] = np.vstack(dendrite_list)
    break