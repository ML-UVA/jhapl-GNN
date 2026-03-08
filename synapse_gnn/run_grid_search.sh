#!/bin/bash

# Define the thresholds you want to test (in nanometers)
THRESHOLDS=(10000 15000 20000)

for THRESH in "${THRESHOLDS[@]}"
do
    echo "------------------------------------------------"
    echo "STARTING RUN WITH THRESHOLD: ${THRESH}nm"
    echo "------------------------------------------------"

    # 1. Update the threshold in config.json
    # Uses sed to find the line after "graph_generation" and replace the value
    sed -i "/\"graph_generation\": {/,/}/ s/\"spatial_threshold_nm\": [0-9]*/\"spatial_threshold_nm\": ${THRESH}/" config.json
    
    # 2. Update the log file name so we don't overwrite
    sed -i "s/\"log_file_name\": \".*\"/\"log_file_name\": \"training_log_euc_${THRESH}nm.txt\"/" config.json

    # 3. Update the visualization folder name
    sed -i "s/\"visualization_output\": \".*\"/\"visualization_output\": \"evals_euc_${THRESH}nm\"/" config.json

    # 4. Run the full pipeline
    ./run_pipeline.sh

    python spatial_training/visualization_scripts/check_distribution.py --config config.json
    echo "Finished ${THRESH}nm run. Results saved in evals_euc_${THRESH}nm"
done