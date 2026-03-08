import numpy as np
from sklearn.decomposition import PCA

def filter_neurons(neurons, coords, R):

    center = coords.mean(axis=0)
    R = 40000

    dists = np.linalg.norm(coords - center, axis=1)
    keep = dists < R
    rsub_neurons = [neurons[i] for i in np.where(keep)[0]]
    sub_coords = coords[keep]
    return rsub_neurons, sub_coords

def build_partial_graph(sub_neurons, edgeset):
    sub_set = set(sub_neurons)
    sub_edges = [(u, v) for (u, v) in edgeset
                if str(u) in sub_set and str(v) in sub_set]
    return sub_edges

def decompose(coords):
    pca = PCA(n_components=2)
    xy = pca.fit_transform(coords)
    return xy

def plot_vis(sub_neurons, sub_edges, decomp):
    import matplotlib.pyplot as plt
    idx = {n:i for i,n in enumerate(sub_neurons)}

    plt.figure(figsize=(6,6))

    for u, v in sub_edges:
        i, j = idx[str(u)], idx[str(v)]
        plt.plot([decomp[i,0], decomp[j,0]],
                [decomp[i,1], decomp[j,1]],
                alpha=0.25)
    plt.scatter(decomp[:,0], decomp[:,1], s=12, c="black")
    plt.axis("equal")
    plt.title("Local neuron subgraph")
    plt.show()