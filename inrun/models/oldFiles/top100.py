import pickle
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# ---------- load shapley ----------
shap = pickle.load(open("graph_node_shapley_normalized.value", "rb"))
node_ids = np.array(shap["node_index"])
shap_vals = np.array(shap["Normalized Shapley (%)"])

# ---------- load graph ----------
df = pd.read_csv("data/top5_k1.csv")
G = nx.Graph()
G.add_edges_from(zip(df.pre_id, df.post_id))

# ---------- select top 100 ----------
order = np.argsort(shap_vals)[::-1]
K = 100

top_indices = order[:K]
top_nodes = node_ids[top_indices]
top_shap = shap_vals[top_indices]

# ---------- build heatmap ----------
H = np.zeros((K, K))

for i in range(K):
    for j in range(K):
        u = top_nodes[i]
        v = top_nodes[j]
        if G.has_edge(u, v):
            # symmetric, interpretable weight
            H[i, j] = top_shap[i] + top_shap[j]

# optional log compression (recommended)
H = np.log1p(H)

# ---------- plot ----------
plt.figure(figsize=(9, 8))
plt.imshow(H, cmap="hot", interpolation="nearest")
plt.colorbar(label="log(Shapleyᵢ + Shapleyⱼ)")
plt.title("Connectivity Between Top 100 Neurons by Shapley Value")
plt.xlabel("Neuron rank (by Shapley)")
plt.ylabel("Neuron rank (by Shapley)")
plt.tight_layout()

out = "top100_shapley_connectivity_heatmap.png"
plt.savefig(out, dpi=300)
plt.close()

print("saved", out)
