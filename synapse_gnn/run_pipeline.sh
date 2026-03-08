#!/bin/bash

# Exit immediately if any command exits with a non-zero status
set -e



# # Step 1: Euclidean Graph Generation
# echo -e "\n[Step 1/6] Building Config-Driven Euclidean Graph..."
# python spatial_training/build_euc_graph.py

# # Step 2: PyG Conversion
# echo -e "\n[Step 2/6] Converting NetworkX Graph to PyTorch..."
# python spatial_training/networkx_to_pyg.py

# Step 3: Geographic Splitting
echo -e "\n[Step 3/6] Growing Spatial Training/Testing Clusters..."
python spatial_training/create_spatial_split.py

# Step 4: Edge Masking
echo -e "\n[Step 4/6] Splitting and Stitching Ground Truth..."
python spatial_training/split_and_stitch_edges.py

# Step 5: Model Training
echo -e "\n[Step 5/6] Training GraphSAGE Model & Evaluating..."
python spatial_training/train_and_eval.py


echo -e "\n========================================================"
echo "   PIPELINE COMPLETE. Check saved_models_spatial/    "
echo "========================================================"