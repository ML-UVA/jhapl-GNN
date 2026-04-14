import numpy as np
import pandas as pd
import pickle
import bz2
import os

print("Loading normalized node Shapley values...")
shap = pickle.load(open("graph_node_shapley_normalized.value", "rb"))
shap_vals = shap["Normalized Shapley (%)"]

print("Loading graph CSV to build index mapping...")
df = pd.read_csv("data/top5_k1.csv")

neuron_ids = pd.unique(df[["pre_id", "post_id"]].values.ravel())

neuron_to_idx = {}
idx_to_neuron = {}

for i in range(len(neuron_ids)):
    neuron_to_idx[int(neuron_ids[i])] = i
    idx_to_neuron[i] = int(neuron_ids[i])

print("Loading manually extended neurons from txt...")
manual_set = set()

with open("manaully_extended_neurons.txt", "r") as f:
    for line in f:
        manual_set.add(line.strip())

print("Total manual neurons listed:", len(manual_set))

print("Building segment_id → split_index mapping from graph files...")

graph_folder = "data"
segment_split_map = {}

for file in os.listdir(graph_folder):
    if file.endswith(".pbz2"):
        path = os.path.join(graph_folder, file)
        with bz2.BZ2File(path, "rb") as f:
            G = pickle.load(f)

        segment_id = int(G.graph["segment_id"])
        split_index = int(G.graph["split_index"])

        segment_split_map[segment_id] = split_index

manual_shap = []
non_manual_shap = []

print("Matching Shapley values to manual status...")

for idx in range(len(shap_vals)):
    segment_id = idx_to_neuron[idx]
    split_index = segment_split_map.get(segment_id, 0)

    key = f"{segment_id}_{split_index}"

    if key in manual_set:
        manual_shap.append(shap_vals[idx])
    else:
        non_manual_shap.append(shap_vals[idx])

manual_shap = np.array(manual_shap)
non_manual_shap = np.array(non_manual_shap)

print("\nRESULTS")
print("Manual count:", len(manual_shap))
print("Non-manual count:", len(non_manual_shap))

print("\nAverage normalized Shapley (manual):", manual_shap.mean())
print("Average normalized Shapley (non-manual):", non_manual_shap.mean())

print("\nMedian normalized Shapley (manual):", np.median(manual_shap))
print("Median normalized Shapley (non-manual):", np.median(non_manual_shap))
