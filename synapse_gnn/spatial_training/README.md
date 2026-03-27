# Spatial Training Pipeline for Connectomics Edge Classification

This repository module provides a robust, spatially-aware training and evaluation pipeline for classifying synapses in large-scale connectomics volumes. It uses a GraphSAGE-based architecture to differentiate true biological synapses from structural false positives.

Unlike standard random node/edge splits, this pipeline utilizes **spatial geographic masking** to ensure that training and testing sets represent distinct, biologically dense, and physically separated regions of the brain volume. This guarantees the model learns universal biological morphological rules rather than memorizing local spatial neighborhoods (spatial leakage).

### Key Capabilities
* **Dual-Graph Architecture:** Supports training on both purely distance-based (Euclidean) graphs and physical-intersection (ADP) graphs.
* **Config-Driven Architecture:** The entire pipeline, from data sampling to neural network dimensions, is controlled via a single `config.json` file.
* **Terabyte-Scale Memory Mitigation:** Uses dynamic subgraph sampling, fixed-size array pre-allocation, and chunked tensor processing to evaluate 57+ million edges without causing Out-of-Memory (OOM) crashes.
* **Biological ID Preservation:** Successfully parses and maps 18-digit JHU APL node IDs while preserving `_0` and `_1` proofreading splits.
* **Automated Hyperparameter Sweeping:** Includes bash orchestration scripts to automatically sweep through spatial quarantine thresholds (10µm to 30µm), log metrics, and generate trend visualizations.

---

## 1. Data Preprocessing & Feature Extraction

Before generating any graphs, you must extract the biological features and ground-truth synapse mapping from the raw JHU APL `.pbz2` files.

### Step 1: Extract Morphological Features
```bash
python spatial_training/preprocessing.py
```
**Description:** Scans the raw graph exports, normalizes structural features (Axon Length, Spine Density, etc.), and retains unnormalized spatial metadata.
* **Outputs:** `x_features.pt` and `node_mapping.json`

### Step 2: Extract Ground-Truth Synapses
```bash
python spatial_training/build_synapses.py --config config.json
```
**Description:** Extracts synapse data while successfully preserving the `_0` and `_1` proofreading ID suffixes to ensure perfect biological alignment with the feature matrix.
* **Output:** `synapses.json`

---

## 2. Base Graph Generation (Choose Your Path)

The pipeline supports two distinct methods for defining "Negative Candidates" (False Positives). You must generate the base edge tensor for your desired approach.

### Path A: The Euclidean Baseline Graph (Distance-Based)
This approach connects neurons that are simply physically close to each other in 3D space.
```bash
# 1. Build the NetworkX graph based on the spatial threshold in config.json
python spatial_training/build_euc_graph.py --config config.json

# 2. Convert the generated .gpickle file into a PyTorch edge tensor
python spatial_training/networkx_to_pyg.py --config config.json
```
* **Outputs:** `euc_graph.gpickle` -> `base_edges.pt`

### Path B: The ADP Graph (Physical Touch)
This approach uses pre-computed mesh intersections (provided by Ben's pipeline) to define candidates that actually physically graze each other.
```bash
python spatial_training/adp_nx_to_pyg.py --config config.json --nx_path path/to/adp_graph_raw.pkl
```
* **Output:** `adp_base_edges.pt`

---
## 3. Core Training Pipeline Execution

Once the features and base edges are generated, the pipeline execution is identical for both graph types. *(Note: Ensure `paths.input_nx_graph` in your config points to your desired graph type before running).*

### Step 1: Generate the Spatial Split
```bash
python spatial_training/create_spatial_split.py --config config.json
```
**Description:** Calculates the physical "Center of Mass" of all ground truth synapses. It anchors the Train and Test seeds offset from this density center and uses k-NN to grow geographic masks, enforcing the `spatial_threshold_nm` quarantine zone.
* **Output:** `spatial_split_masks.pt`

### Step 2: Split and Stitch Edges
```bash
python spatial_training/split_and_stitch_edges.py --config config.json
```
**Description:** Applies the spatial masks to both the ground truth synapses and the structural candidates. It requires both the source and destination neurons of an edge to fall within the same spatial mask to be included, creating isolated positive and negative sets.
* **Outputs:** `graph_train_edges.pt`, `graph_test_edges.pt`, `graph_train_spatial_candidates.pt`, `graph_test_spatial_candidates.pt`

### Step 3: Model Training and Evaluation
```bash
python spatial_training/train_and_eval.py --config config.json
```
**Description:** Initializes the GraphSAGE model and dynamically samples subgraphs. Utilizes hard negative mining, forcing the model to distinguish between true synapses and structurally similar False Positives.
* **Outputs:** `best_model_[graph_type]_[nm].pth`, `metrics_[graph_type]_[nm].json`

### Step 4: Generate Visualizations
```bash
python spatial_training/visualization_scripts/check_distribution.py --config config.json
python spatial_training/visualization_scripts/generate_feature_analysis.py --config config.json
```
**Description:** Reads the saved model and test sets to dynamically generate publication-ready plots, including Score Distributions, Feature Importance Bar Charts, and Confidence vs. Distance Scatter Plots. 
* **Outputs:** Saved directly to the dynamically named `paths.visualization_output` directory.

---

## 4. Automated Hyperparameter Sweeping (Recommended)

To rigorously test the model against spatial leakage, the pipeline includes automated orchestration scripts. These scripts iteratively increase the spatial quarantine gap between the Train and Test sets (10µm -> 30µm), update the `config.json` dynamically, run the entire core ML pipeline (Split -> Train -> Visualize), and segregate the outputs.

**For Euclidean Distance Graphs:**
```bash
./sweep_thresholds_euc.sh
```

**For ADP (Physical Touch) Graphs:**
```bash
./sweep_thresholds_adp.sh
```

**Generate Combined Trend Plot:**
```bash
python spatial_training/visualization_scripts/plot_sweep_auc.py
```
*Scans all generated metric JSONs from both sweeps to produce a single side-by-side ROC-AUC trend line.*

---

## 5. Configuration (`config.json`)

**Sample Configuration:**
```jsonc
{
    "raw_data": {
        // Path to the raw .pbz2 files from JHU APL containing neuron morphology and connectivity.
        "neurons_directory": "/p/mlatuva/jhu-graph/graph_exports"
    },
    "paths": {
        // Where the preprocessed PyTorch tensors (e.g., base_edges.pt) are saved.
        "data_dir": "./cache_spatial",
        // Where the trained GraphSAGE model weights (.pth) and metrics (.json) are saved.
        "model_out": "./saved_models_spatial",
        // The base graph file to load (Toggle between "euc_graph.gpickle" and "adp_graph.gpickle").
        "input_nx_graph": "adp_graph.gpickle",
        // Directory for dynamically generated output plots (Distributions, Feature Importance, etc.).
        "visualization_output": "evals_adp_graph_10000nm"
    },
    "graph_generation": {
        // The quarantine distance (in nanometers) forced between the Train and Test geographic masks.
        "spatial_threshold_nm": 10000,
        // The minimum physical contact area required to form an edge in the ADP graph (0.0 = any physical touch).
        "adp_threshold": 0.0
    },
    "graph_conversion": {
        // Duplicates all directed edges in reverse (A->B and B->A) to allow bidirectional message passing in the GNN.
        "make_undirected": true
    },
    "spatial_split": {
        // Number of neurons to sample within the Training geographic quarantine zone.
        "train_cluster_size": 35000,
        // Number of neurons to sample within the Testing geographic quarantine zone.
        "test_cluster_size": 15000
    },
    "architecture": {
        // Number of biological input features per neuron (e.g., Axon Length, Spine Density).
        "in_channels": 8,
        // The size of the hidden representation layers inside the GraphSAGE model.
        "hidden_dim": 128
    },
    "training": {
        // Total number of full training loops over the data.
        "epochs": 50,
        // The step size for the Adam optimizer during backpropagation.
        "learning_rate": 0.001,
        // Number of subgraph batches processed per epoch to avoid RAM overload.
        "steps_per_epoch": 100,
        // Size of the random subgraph sampled during each training step.
        "train_node_sample_size": 8000,
        // How many validation runs to average together to get a stable validation metric.
        "validation_averaging_runs": 5,
        // Size of the random subgraph sampled during each validation step.
        "validation_node_sample_size": 8000,
        // File to append training metrics and epoch losses to.
        "log_file_name": "training_log_adp_sweep.txt"
    },
    "evaluation": {
        // Size of the massive subgraph used for the final, blinded inductive test.
        "test_node_sample_size": 25000,
        // How many independent test sets to evaluate and aggregate for the final metrics.
        "test_aggregation_runs": 10
    }
}```

---
