import networkx as nx   
import pickle
import numpy as np
from pathlib import Path

# --------------------------------------------------
# Load synapse → neuron lookup tables
# pre_syn_dict[synapse_id]  = presynaptic neuron_id
# post_syn_dict[synapse_id] = postsynaptic neuron_id
# --------------------------------------------------

required = ["data/pre_syn_dict.npy", "data/post_syn_dict.npy"]

missing = [f for f in required if not Path(f).is_file()]

if missing:
    raise FileNotFoundError(f"Missing required assets: {missing}")

pre_syn_dict = np.load("data/pre_syn_dict.npy")
post_syn_dict = np.load("data/post_syn_dict.npy")

pre_nonzero = np.count_nonzero(pre_syn_dict)
post_nonzero = np.count_nonzero(post_syn_dict)

print("pre_syn non-zero =", pre_nonzero)
print("post_syn non-zero =", post_nonzero)

G = nx.DiGraph()

#Add neurons and synaptic connections to directed graph

for pre, post in zip(pre_syn_dict, post_syn_dict):
    if pre != 0 and post != 0:   # skip missing connections
        G.add_edge(int(pre), int(post))


# Save neuron connectivity graph to disk

with open("neuron_graph.gpickle", "wb") as f:
    pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

