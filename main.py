import argparse
import json
import torch
import os
import subprocess
import sys

from synapse_gnn.data.loader import load_graph_data
from synapse_gnn.models.gnn import SynapsePredictor
from synapse_gnn.training.train_engine import run_training
from synapse_gnn.evaluation.metrics import run_inductive_evaluation
from synapse_gnn.evaluation.visualizations import generate_all_visualizations
from synapse_gnn.data_prep import preprocessing
from synapse_gnn.data_prep import build_synapses
from synapse_gnn.data_prep import build_demo_euc_graph


def parse_args():
    parser = argparse.ArgumentParser(description="Config-Driven GraphSAGE Synapse Predictor Pipeline")
    parser.add_argument('--config', type=str, default="config.json", help="Path to the JSON configuration file")
    # NEW: Toggle to run the data generation scripts before the ML pipeline
    parser.add_argument('--build_data', action='store_true', help="Run preprocessing, ground truth, and graph building before training")
    return parser.parse_args()

def run_command(command):
    """Helper function to execute shell commands securely and halt on failure."""
    print(f"\n[ORCHESTRATOR] Running: {command}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"\n[CRITICAL ERROR] Command failed with exit code {result.returncode}. Halting pipeline.")
        sys.exit(1)

def main():
    args = parse_args()
    print(f"Loading configuration from: {args.config}")
    with open(args.config, 'r') as f:
        config = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using compute device: {device}")
    
# --- STAGE 0: DATA GENERATION ---
    if args.build_data:
        print("\n" + "="*50)
        print(" STAGE 0: BUILDING DATASET FROM SCRATCH")
        print("="*50)
        
        # 1. Extract Morphological Features
        preprocessing.main(config_path=args.config)
        
        # 2. Extract Ground Truth Synapses
        build_synapses.main(config_path=args.config)
        
        # 3. Build Demo Euclidean Graph
        build_demo_euc_graph.main(config_path=args.config)
        
        print("\n--- Data Generation Complete! Transitioning to ML Pipeline ---\n")
# --- STAGE 1: ML PIPELINE ---
    # Create required directories
    os.makedirs(config["paths"]["model_out"], exist_ok=True)
    os.makedirs(config["paths"]["visualization_output"], exist_ok=True)

    # 1. LOAD DATA
    data_dict = load_graph_data(config)
    
    use_weights = config["architecture"].get("use_edge_weights", True)
    print(f"Using edge weights: {use_weights}")
    
    # 2. INITIALIZE MODEL
    num_features = len(config["architecture"].get("selected_features", [0, 1, 2, 3, 4, 5, 6, 7]))
    model = SynapsePredictor(
        in_channels=num_features, 
        hidden_channels=config["architecture"]["hidden_dim"],
        use_edge_weights=use_weights
    ).to(device)

    # 3. TRAIN
    best_model_path = run_training(model, data_dict, config, device)

    # 4. EVALUATE
    run_inductive_evaluation(model, best_model_path, data_dict, config, device)

    # 5. VISUALIZE
    generate_all_visualizations(model, data_dict, config, device)

if __name__ == "__main__":
    main()