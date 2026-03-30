#!/bin/bash

# The full spatial decay test
THRESHOLDS=(10000 15000 20000 25000 30000)

# Set the graph type to EUC
GRAPH_TYPE="euc_graph"

for THRESH in "${THRESHOLDS[@]}"
do
    echo "======================================================"
    echo " STARTING RUN: ${GRAPH_TYPE} @ ${THRESH}nm (MLP Decoder)"
    echo "======================================================"

    # Safely update config.json using Python
    python3 -c "
import json

with open('config.json', 'r') as f:
    config = json.load(f)

graph_type = '${GRAPH_TYPE}'
thresh = ${THRESH}

# Update input graph path
if 'paths' not in config: config['paths'] = {}
config['paths']['input_nx_graph'] = f'{graph_type}.gpickle'

# Update the Spatial Threshold
if 'graph_generation' not in config: config['graph_generation'] = {}
config['graph_generation']['spatial_threshold_nm'] = thresh

# Update Output Names dynamically
if 'logging' not in config: config['logging'] = {}
config['logging']['log_file_name'] = f'training_log_{graph_type}_{thresh}nm.txt'

# Tagging this EUC run with 'mlp_decoder' so we can compare it fairly!
if 'adp' in graph_type.lower():
    config['paths']['visualization_output'] = f'evals_{graph_type}_added_adp_weights_mlp_decoder_{thresh}nm'
else:
    config['paths']['visualization_output'] = f'evals_{graph_type}_mlp_decoder_{thresh}nm'

with open('config.json', 'w') as f:
    json.dump(config, f, indent=4)
"
    echo "Successfully updated config.json for EUC ${THRESH}nm."

    # 1. Geographic Splitting
    echo "Growing Spatial Training/Testing Clusters..."
    python spatial_training/create_spatial_split.py --config config.json

    # 2. Edge Masking 
    echo "Splitting and Stitching EUC Ground Truth..."
    python spatial_training/split_and_stitch_edges.py --config config.json

    # 3. Model Training
    echo "Training GraphSAGE Model..."
    python spatial_training/train_and_eval.py --config config.json

    # 4. Visualizations
    echo "Generating evaluation plots..."
    python spatial_training/visualization_scripts/check_distribution.py --config config.json
    python spatial_training/visualization_scripts/generate_feature_analysis_mlp_decoder.py --config config.json
    
    echo "------------------------------------------------------"
    echo " Finished EUC ${THRESH}nm run. Results saved!"
    echo "------------------------------------------------------"
    echo ""
done

echo "ALL EUC MLP SWEEPS COMPLETED SUCCESSFULLY."