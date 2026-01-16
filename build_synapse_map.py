import networkx as nx
import bz2
import pickle
import os
import time
import numpy as np

# Note: Current file is outdated and treats split indices as one neuron: will be fixed soon

graph_directory = "data/graph_exports"
output_directory = "data"

files = [f for f in os.listdir(graph_directory)
         if os.path.isfile(os.path.join(graph_directory, f))]

print(f"First 10 files: {files[0:10]}")
print(f"Number of files: {len(files)}")

max_synapse_id = 600000000  # rough approximation

# Note: 600000000 was determined by iterating and finding highest synapse id in the dataset.
# This was determined to be slightly more memory efficient in certain cases

post_syn_dict = np.zeros(max_synapse_id, dtype=np.int64)
# index it by synapse id post_syn_dict[syn_id] = neuron_id

pre_syn_dict = np.zeros(max_synapse_id, dtype=np.int64)
# index it by synapse id pre_syn_dict[syn_id] = neuron_id

i = 0
number_of_synapses = 0

for file in files:

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
        # num_synapses += attr['n_synapses']
        syn_data = attr.get("synapse_data")
        if syn_data:
            for syn in syn_data:
                sid = int(syn['syn_id'])
                if sid >= max_synapse_id:
                    continue

                if syn['syn_type'] == "presyn":
                    number_of_synapses += 1
                    pre_syn_dict[sid] = neuron_id

                if syn['syn_type'] == "postsyn":
                    number_of_synapses += 1
                    post_syn_dict[sid] = neuron_id

    t_iter_end = time.time()
    iter_time = t_iter_end - t_iter_start

    # print per-file timings
    i += 1
    if i % 100 == 0:
        print(f"{file}: load={load_time:.4f}s, iterate={iter_time:.4f}s")
        print(f"Number of files iterated: {i}")
        print(f"number of synapses: {number_of_synapses}")

np.save(output_directory + "/pre_syn_dict.npy", pre_syn_dict)
np.save(output_directory + "/post_syn_dict.npy", post_syn_dict)
