"""
Generates and saves skeletonization data for each neuron.

This includes saving the skeletonization data for each branch off the soma
of each neuron in a dictionary.

Note: Last number of neuron_id corresponds to split_index and
is not part of the actual neuron_id.

"""
import networkx as nx
import bz2
import pickle
import os

import time
import numpy as np




def generate_skeleton_data(neuron_directory):
    """Generate skeletonization data from neuron graph exports."""

    files = [f for f in os.listdir(neuron_directory) if os.path.isfile(os.path.join(neuron_directory, f)) and f.endswith(".pbz2")]


    final_skeletonization_dict = {}


    time_start = time.time()

    for (i,file) in enumerate(files,start=1):

        path = os.path.join(neuron_directory, file)
        fname = os.path.basename(path)
        neuron_id = "_".join(fname.split("_")[0:2])

        with bz2.open(path, 'rb') as f:
            G = pickle.load(f)
            axon_list = []
            dendrite_list = []

            skeleton_dict = {"axon":np.empty((0,3)),"dendrite":np.empty((0,3))}

            G_dict = dict(list(G.nodes(data=True)))

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
            

            final_skeletonization_dict[neuron_id] = skeleton_dict
        if i%1000 ==0:
            print(f"Number of Neuron Skeletonization Data Generated: {i}")
            print(f"Time iterated for 1000 neurons = {time.time()-time_start}")
            time_start = time.time()

    print(f"Total number of neurons in skeletonization data = {len(final_skeletonization_dict)}")


    out_path = os.path.join("data", "skeletonization_data_simple.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(final_skeletonization_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    