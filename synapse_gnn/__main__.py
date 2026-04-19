import argparse
import json
import torch
import os
import sys

from synapse_gnn.models.gnn import SynapsePredictor
from synapse_gnn.training.train_engine import run_training
from synapse_gnn.evaluation.metrics import run_inductive_evaluation
from synapse_gnn.evaluation.visualizations import generate_all_visualizations
from synapse_gnn.data_prep import preprocessing
from data_prep import build_synapses
from synapse_gnn.data_prep import build_demo_euc_graph
from synapse_gnn.data import spatial_split


def parse_args():
    parser = argparse.ArgumentParser(description="Config-Driven GraphSAGE Synapse Predictor Pipeline")
    parser.add_argument('--config', type=str, default="synapse_gnn/config.json", help="Path to the JSON configuration file")
    parser.add_argument('--build_data', action='store_true', help="Run preprocessing, ground truth, and graph building before training")
    return parser.parse_args()

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
        
        preprocessing.main(config_path=args.config)
        build_synapses.main(config_path=args.config)
        build_demo_euc_graph.main(config_path=args.config)
        
        # Add this line to stitch everything together!
        spatial_split.generate_spatial_masks_and_stitch(config)
        
        print("\n--- Data Generation Complete! Transitioning to ML Pipeline ---\n")
    # --- STAGE 1: ML PIPELINE ---
    os.makedirs(config["paths"]["model_out"], exist_ok=True)
    os.makedirs(config["paths"]["visualization_output"], exist_ok=True)

    # 1. LOAD PYG DATA OBJECTS directly from cache
    CACHE_DIR = config["paths"]["data_dir"]
    try:
        train_data = torch.load(os.path.join(CACHE_DIR, "train_data.pt"), weights_only=False)
        test_data = torch.load(os.path.join(CACHE_DIR, "test_data.pt"), weights_only=False)
        print("Successfully loaded PyG Spatial Subgraphs.")
    except FileNotFoundError:
        print(f"Error: train_data.pt or test_data.pt not found in {CACHE_DIR}. Run `python -m synapse_gnn --build_data` to generate the dataset.")
        sys.exit(1)
    
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
    best_model_path = run_training(model, train_data, test_data, config, device)

    # 4. EVALUATE
    run_inductive_evaluation(model, best_model_path, train_data, test_data, config, device)

    # 5. VISUALIZE
    generate_all_visualizations(model, train_data, test_data, config, device)

if __name__ == "__main__":
    main()