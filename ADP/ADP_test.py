import pickle
import networkx as nx
import os
import numpy as np
from scipy.spatial import KDTree
import time
import bz2

# KD_tree_path = "data/KD_tree_data.pkl"

# with open(KD_tree_path, "rb") as f:
#     KD_tree_data = pickle.load(f)

# for i in KD_tree_data:
#     print(KD_tree_data[i])
#     break

graph_directory = "data/graph_exports"

files = [f for f in os.listdir(graph_directory)
         if os.path.isfile(os.path.join(graph_directory, f))]

print(f"First 10 files: {files[0:10]}")
print(f"Number of files: {len(files)}")

i=0
n=0
n1=0
c = {}
j={}
m={}
for file in files[0:1000]:

    # TIME: LOADING (bz2 + pickle)
    t_load_start = time.time()

    path = os.path.join(graph_directory, file)
    fname = os.path.basename(path)
    neuron_id = int(fname.split("_")[0])

    with bz2.open(path, 'rb') as f:
        G = pickle.load(f)

    t_load_end = time.time()
    load_time = t_load_end - t_load_start

    # TIME: ITERATION OVER NODES
    t_iter_start = time.time()

    for node in G.nodes():
        attr = G.nodes[node]
        if 'compartment' not in attr:
            continue
        if attr['compartment']!="soma":
            #print(attr['compartment'],attr['axon_compartment'])
            c[attr['compartment']] = c.get(attr['compartment'],0)+1
            j[attr['axon_compartment']] = j.get(attr['axon_compartment'],0)+1
            m[attr['labels'][0]] = m.get(attr['labels'][0],0)+1
            
        if 'axon_compartment' in attr and attr['axon_compartment']=="axon":
            #print(attr['skeleton_data'].shape)
            #print(attr['skeleton_data'])
            n+=attr['skeleton_data'].shape[0]
            #print("found_axon")
            continue
        if 'axon_compartment' in attr:
            n1+=attr['skeleton_data'].shape[0]
    if i %100==0:
        print(i)
    i+=1
print(c)
print(j)
print(m)
print(n/1000)
print(n1/1000)