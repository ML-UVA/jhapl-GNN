Spatial GNN for Connectome Edge Classification
A scalable Graph Neural Network (GNN) pipeline designed to predict synaptic connections in massive biological connectomes. This project implements a Spatial Holdout strategy to prevent data leakage and utilizes GraphSAINT sampling to train on billion-scale graphs using standard consumer hardware (2GB VRAM).

🚀 Key Features
Spatial Validation: Implements a k-Nearest Neighbor (Ball Tree) split to strictly separate Training and Testing data by physical brain region, ensuring the model learns general connectivity rules rather than memorizing local neighborhoods.

True Centroid Calculation: Extracts the geometric center of mass from neuron skeletons (via preprocessing.py) rather than just soma location for precise spatial clustering.

Memory Efficient: Uses a custom implementation of GraphSAINT (Random Subgraph Sampling) to train on graphs with hundreds of millions of edges without requiring torch-sparse or massive RAM (reduced footprint from ~500GB to ~2GB).

Self-Healing Pipeline: The orchestrator (main.py) automatically detects missing features, generates them from raw data, and manages caching.

📊 Performance
Evaluated on a spatially separated holdout region (Testing Set N≈1,000,000 edges):

Metric	Score
ROC AUC	0.9996
F1 Score	0.9996
Precision	0.9992
Recall	1.0000


Note: The high performance indicates the model successfully distinguishes real proximal synapses from random distant pairs based on morphological features.

pip install -r requirements.txt
📂 Data Structure
The pipeline expects raw neuron graph files (pickled NetworkX objects) in a specific directory. Update RAW_GRAPH_DIR in main.py to point to your data.

⚙️ Data Processing Pipeline
Before training the GNN, the raw spatial data must be converted into a structured graph format.

1. Neuron Position Extraction (build_positions.py)
Maps every neuron ID to its 3D spatial coordinates (Mesh Center).

Input: graph_exports/*.pbz2 (Raw neuron files).

Output: positions.json (Dictionary of id: [x, y, z]).

2. Spatial Adjacency Construction (build_adjacency.py)
Defines the search space for the model. Instead of evaluating every possible pair (N 
2), we build a Radius Graph connecting only neurons that are spatially proximal.

Input: synapses.json, positions.json.

Algorithm: Calculates Euclidean distance. Pairs > 1mm apart are discarded as biologically implausible.

Output: adjacency.json (Sparse adjacency list).

3. Adjacency Chunking (converting_adjacency.py)
Optimizes data loading for PyTorch Geometric by converting the adjacency list into binary chunks.

Process: Converts JSON list → Coordinate Format (COO) matrix → Split .npy files.

Benefit: Enables out-of-core loading during GraphSAINT sampling, preventing OOM errors.

🖥️ Usage Pipeline
The pipeline runs in sequential stages. Run these commands from the spatial_training folder

Step 1: Feature Extraction
Scans raw data, calculates centroids, and generates the feature matrix (x_features.pt).

'''bash
python main.py
'''

Note: The script will pause here if edge lists are not found yet.

Step 2: Generate Spatial Split
Creates masks for "Left Brain" (Train) and "Right Brain" (Test) using k-NN clustering to define holdout regions.

'''bash
python create_spatial_split.py
'''

Step 3: Stitch and Filter Edges
Processes raw edge chunks, applies spatial masks, and removes "cross-edges" to strictly enforce the split.

'''bash
python split_and_stitch.py
'''

Step 4: Train the Model
Run the main script again. It will detect the newly created edge files and automatically begin the GraphSAINT training loop.

'''bash
python main.py
'''

Step 5: Final Evaluation
Calculates AUC, determines the optimal decision threshold, and generates the confusion matrix on the unseen test set.

'''bash
python evaluate_model.py
'''