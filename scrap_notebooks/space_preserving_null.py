import json
with open('positions.json', 'r') as file:
    data = json.load(file)
print(len(data))
nodes = list(data.keys())
import math
def distance(x1, y1, z1, x2, y2, z2):
    def d2(a, b):
        return (a-b)**2
    return math.sqrt(d2(x1, x2)+d2(y1, y2)+d2(z1, z2))
import numpy as np
N = len(nodes)
#distances = np.zeros((N, N))
#for i in range(N):
#    for j in range(N):
#        distances[i,j] = distance(*data[nodes[i]], *data[nodes[j]])
#np.save("distances.json", distances)
distances = np.load("distances.npy")
with open('synapses.json', 'r') as file:
    data = json.load(file)
print(len(data))
nodes = set()
edges = set()
for synid in data:
    nodes.add(data[synid][0][0])
    nodes.add(data[synid][0][1])
    edges.add((data[synid][0][0], data[synid][0][1]))
nodes = list(nodes)
A = np.zeros((N, N))
idxMap = {nodes[i]:i for i in range(len(nodes))}
for x, y in edges:
    A[idxMap[x]][idxMap[y]] = 1
    A[idxMap[y]][idxMap[x]] = 1

triu_idx = np.triu_indices(N, k=1)
dists = distances[triu_idx]
edges = A[triu_idx]

bins = np.linspace(dists.min(), dists.max(), 21)
bin_centers = 0.5 * (bins[:-1] + bins[1:])

p_empirical = []
for k in range(len(bins)-1):
    idx = (dists >= bins[k]) & (dists < bins[k+1])
    if np.sum(idx) > 0:
        p_empirical.append(edges[idx].mean())
    else:
        p_empirical.append(np.nan)
p_empirical = np.array(p_empirical)
from scipy.optimize import curve_fit

def exp_decay(d, lam):
    return np.exp(-d/lam)

# Remove bins with no data
valid = ~np.isnan(p_empirical) & (p_empirical>0)
print(p_empirical[valid])
p_fit = p_empirical[valid]
d_fit = bin_centers[valid]

# Fit lambda
lam_opt, _ = curve_fit(exp_decay, d_fit, p_fit)
lambda_est = lam_opt[0]
print("Estimated lambda:", lambda_est)

import networkx as nx

G_null = nx.Graph()
G_null.add_nodes_from(range(N))

for i in range(N):
    for j in range(i+1, N):
        p = np.exp(-distances[i,j]/lambda_est)
        if np.random.rand() < p:
            G_null.add_edge(i, j)

nx.write_graphml(G_null, "my_graph.graphml")