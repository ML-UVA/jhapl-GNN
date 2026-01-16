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


BASE_DIR = os.path.dirname(__file__)


def generate_skeleton_data(neuron_directory):
    """Generate skeletonization data from neuron graph exports."""

    files = [f for f in os.listdir(neuron_directory) if os.path.isfile(os.path.join(neuron_directory, f))]

    print(f"number of neurons: {len(files)}")
    print(f"first 10 files: {files[0:10]}")

    
    # Final skeletonization data that is indexed by the neuron_id and split index
    # ex: final_skeletonization_dict["86491135657800322_0"]

    final_skeletonization_dict = {}


    time_start = time.time()

    for (i,file) in enumerate(files,start=1):

        path = os.path.join(neuron_directory, file)
        fname = os.path.basename(path)
        neuron_id = "_".join(fname.split("_")[0:2])

        with bz2.open(path, 'rb') as f:
            G = pickle.load(f)

            final_branch_dict={}
            
            first_child_list=[]
            
            for edge in G.edges:
                if edge[0] == "S0":
                    first_child_list.append(edge[1])

            for node in first_child_list:
                downstream_connections = nx.descendants(G, node)
                final_branch_dict[node]=[node]
                final_branch_dict[node]+= list(downstream_connections)
                

            G_dict = dict(list(G.nodes(data=True)))

            skeleton_dict = {}
            
            for node in final_branch_dict:
                if not G_dict[node]:
                    continue
                node_list = [node] + final_branch_dict[node]
                skeleton_dict[node] = [G_dict[node]['labels'][0],np.empty((0,3))]
                for child_node in final_branch_dict[node]:
                    if not G_dict[child_node]:
                        continue
                    node_data = G_dict[child_node]
                    points = node_data['skeleton_data']
                    skeleton_dict[node][1] = np.vstack((skeleton_dict[node][1], points))

            final_skeletonization_dict[neuron_id] = skeleton_dict

        if i%1000 ==0:
            print(f"Number of Neuron Skeletonization Data Generated: {i}")
            print(f"Time iterated for 1000 neurons = {time.time()-time_start}")
            time_start = time.time()

    print(f"Total number of neurons in skeletonization data = {len(final_skeletonization_dict)}")


    out_path = os.path.join(BASE_DIR, "data", "skeletonization_data.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(final_skeletonization_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
