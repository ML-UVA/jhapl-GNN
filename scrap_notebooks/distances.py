import json

with open('synapses.json', 'r') as file:
    data = json.load(file)
print(len(data))
nodes = set()
edges = set()
for synid in data:
    nodes.add(data[synid][0][0])
    nodes.add(data[synid][0][1])
    edges.add((data[synid][0][0], data[synid][0][1]))
import compat_numpy
from datasci_tools import system_utils as su
position = dict()
for node in nodes:
    try:
        G = su.decompress_pickle(f"./graph_exports/{node}_0_auto_proof_v7_proofread.pbz2")
    except:
        G = su.decompress_pickle(f"./graph_exports/{node}_1_auto_proof_v7_proofread.pbz2")
    position[node] = G.nodes["S0"]["mesh_center"]
import math
def distance(x1, y1, z1, x2, y2, z2):
    def d2(a, b):
        return (a-b)**2
    return math.sqrt(d2(x1, x2)+d2(y1, y2)+d2(z1, z2))
adjacency = {x: [] for x in nodes}
negative = 0
nodes = list(nodes)
for i, node in enumerate(nodes):
    for j in range(i+1, len(nodes)):
        if distance(*position[node], *position[nodes[j]]) <= 1e6:
            adjacency[node].append(nodes[j])
            adjacency[nodes[j]].append(node)
        else:
            negative += 1
print(negative)
import json
with open("data.json", "w") as f:
    json.dump(adjacency, f)