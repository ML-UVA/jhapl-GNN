from Skeletonization_Gen import generate_skeleton_data
from KD_Tree_Construction import build_KD_trees
from ADP_Calc import calc_ADP

# Graph path relative to where ADP_pipeline.py is ran
rel_neuron_graph_path = "data/graph_exports"
rel_data_checkpoint_path = "data"


# Generates and saves skeletonization data in data/skeletonization_data.pkl using data/graph_exports of neurons
generate_skeleton_data(rel_neuron_graph_path) 

# Generates and saves KD trees from skeletonization data in data/skeletonization_data.pkl
build_KD_trees(rel_data_checkpoint_path)

# Generates and saves ADP calculations from data/KD_tree_data.pkl
calc_ADP(rel_data_checkpoint_path)

