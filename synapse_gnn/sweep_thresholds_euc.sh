#!/bin/bash

# Define the thresholds you want to test (in nanometers)
THRESHOLDS=(10000 15000 20000 25000 30000)

# Set the graph type for the sweep
GRAPH_TYPE="euc_graph"

for THRESH in "${THRESHOLDS[@]}"
do
    echo "======================================================"
    echo " STARTING RUN: ${GRAPH_TYPE} @ ${THRESH}nm"
    echo "======================================================"

    # Safely update config.json using Python to avoid regex/sed corruption
    python3 -c "
import json

with open('config.json', 'r') as f:
    config = json.load(f)

# Use bash variables safely inside the Python script
graph_type = '${GRAPH_TYPE}'
thresh = ${THRESH}

# IMPORTANT: Update input graph path so train_and_eval.py parses 'graph_type' correctly!
if 'paths' not in config: config['paths'] = {}
config['paths']['input_nx_graph'] = f'{graph_type}.gpickle'

# Update the Spatial Threshold
if 'graph_generation' not in config: config['graph_generation'] = {}
config['graph_generation']['spatial_threshold_nm'] = thresh

# Update Output Names dynamically
if 'logging' not in config: config['logging'] = {}
config['logging']['log_file_name'] = f'training_log_{graph_type}_{thresh}nm.txt'

config['paths']['visualization_output'] = f'evals_{graph_type}_{thresh}nm'

with open('config.json', 'w') as f:
    json.dump(config, f, indent=4)
"
    echo "Successfully updated config.json for ${THRESH}nm."

    # 1. Run the pipeline!
    # (Ensure run_pipeline.sh is set up to run networkx_to_pyg.py for Euclidean graphs)
    ./run_pipeline.sh

    # 2. Run the visualizations
    echo "Generating evaluation plots..."
    python spatial_training/visualization_scripts/check_distribution.py --config config.json
    python spatial_training/visualization_scripts/generate_feature_analysis.py --config config.json
    
    echo "------------------------------------------------------"
    echo " Finished ${THRESH}nm run. Results saved to: evals_${GRAPH_TYPE}_${THRESH}nm/"
    echo "------------------------------------------------------"
    echo ""
done

echo "ALL SWEEPS COMPLETED SUCCESSFULLY."