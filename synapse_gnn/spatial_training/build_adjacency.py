import json
import math
import os
from datasci_tools import system_utils as su


with open('synapses.json', 'r') as file:
    data = json.load(file)
print(len(data))
nodes = set()
edges = set()
for synid in data:
    nodes.add(data[synid][0][0])
    nodes.add(data[synid][0][1])
    edges.add((data[synid][0][0], data[synid][0][1]))
position = dict()
for node in nodes:
    try:
        G = su.decompress_pickle(f"./graph_exports/{node}_0_auto_proof_v7_proofread.pbz2")
    except:
        G = su.decompress_pickle(f"./graph_exports/{node}_1_auto_proof_v7_proofread.pbz2")
    position[node] = G.nodes["S0"]["mesh_center"]
for node in nodes:
    try:
        G = su.decompress_pickle(f"./graph_exports/{node}_0_auto_proof_v7_proofread.pbz2")
    except:
        G = su.decompress_pickle(f"./graph_exports/{node}_1_auto_proof_v7_proofread.pbz2")
    print(G.nodes["S0"])
    break
names = set(os.listdir("./graph_exports/"))
print(os.listdir("/p/mlatuva/jhu-graph/avery/jhapl-GNN/"))
for node in nodes:
    tot = 0
    tot += int(f"./graph_exports/{node}_0_auto_proof_v7_proofread.pbz2" in names)
    tot += int(f"./graph_exports/{node}_1_auto_proof_v7_proofread.pbz2" in names)
    if tot == 1:
        print(node)
with open("positions.json", "r") as f:
    position = json.load(f)
def distance(x1, y1, z1, x2, y2, z2):
    def d2(a, b):
        return (a-b)**2
    return math.sqrt(d2(x1, x2)+d2(y1, y2)+d2(z1, z2))
adjacency = {x: [] for x in nodes}
negative = 0
nodes = list(nodes)
print(nodes)
for i, node in enumerate(nodes):
    for j in range(i+1, len(nodes)):
        # Checks if distance is less than or equal to 1 million nanometers(default units of the dataset) which is 1mm.
        if distance(*position[str(node)], *position[str(nodes[j])]) <= 1e6:
            adjacency[node].append(nodes[j])
            adjacency[nodes[j]].append(node)
        else:
            negative += 1
print(negative)
with open("adjacency.json", "w") as f:
    json.dump(adjacency, f)