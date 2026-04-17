# jhapl-GNN: Graph Analysis Pipeline for Neural Network Motifs

A comprehensive Python pipeline for analyzing neural network structure through motif census, clustering coefficients, degree statistics, and spatial null model comparison.

## Overview

This project provides tools to:
- **Characterize network structure** through triadic motifs and network metrics
- **Generate null models** both spatial (preserving proximity) and non-spatial (random rewiring)
- **Compute metrics** including clustering, degree distribution, and motif z-scores
- **Visualize** local subgraphs and comparative analyses

### Data Format: PyTorch .pt Files

This codebase uses **PyTorch `.pt` format** for all neural network data files for improved performance and ML compatibility. See [PYTORCH_MIGRATION.md](PYTORCH_MIGRATION.md) for:
- Data format specifications
- Migration guide for existing JSON files
- Conversion utilities

## Project Structure

This project uses **flat layout** for the Python package. The main source code is in the `jhapl_gnn/` directory.

```
jhapl-GNN/
├── jhapl_gnn/              # Main package (flat layout)
│   ├── binning/            # Spatial feature binning
│   ├── metrics/            # Network analysis metrics
│   ├── null_models/        # Null model generators
│   ├── config.py           # Global configuration
│   ├── graph_io.py         # Graph I/O operations
│   └── spatial_analysis.py # Spatial utilities
├── scripts/                # Executable scripts
├── data/                   # Data files (input/processed)
└── notebooks/              # Jupyter notebooks
```

### `jhapl_gnn/binning/`
Spatial feature binning for edge probability estimation.

**Key module:** `compute_bins.py`
- `BinModel`: Class storing binned edge probabilities
- `compute_bins()`: Compute P(edge | feature) by quantizing continuous feature values
- `assign_bins()`: Assign feature values to bin indices
- Supports any spatial feature (distance, ADP, etc.) with quantile or uniform binning

### `jhapl_gnn/metrics/`
Network analysis metrics and comparative analysis.

**Modules:**
- `count_metrics.py`: Triadic motif census and z-score computation
  - `count_tri()`: Count 16 possible 3-node directed motifs
  - `generate_motif_df()`: Compare motif counts against null models with z-scores
  - `plot_summary()`: Visualize motif comparisons on log scale

- `hub_spoke_metrics.py`: Degree distribution statistics
  - `gini()`: Degree inequality coefficient
  - `coef_variation()`: Relative degree heterogeneity
  - `deg_assortativity()`: Correlation between node degrees

- `clustering_metrics.py`: Triangle and clustering statistics
  - `clustering()`: Average local clustering coefficient
  - `transitivity()`: Global clustering (triangle fraction)
  - `triangles()`: Mean triangle count per node

- `generators.py`: Pipeline for running metrics across null models
  - `run_null_models()`: Apply metrics to null model samples
  - `summarize_results()`: Compute mean/stdev comparisons

### `jhapl_gnn/null_models/`
Random graph generators and unified null model interface.

**Modules:**
- `non_spatial_null_models.py`: Classical random graph generators (implementation)
  - `erdos_renyi_directed()`: ER random graphs
  - `configuration_model_directed()`: Preserves degree sequence
  - `barabasi_albert_directed()`: Preferential attachment (scale-free)
  - `watts_strogatz_directed()`: Small-world networks

- `spatial_null_model.py`: Spatial null model (implementation)
  - `generate_spatial_null()`: Sample edges using empirical P(edge | feature)

- `wrappers.py`: **Unified null model interface**
  - All wrappers follow: `wrapper(GT: nx.Graph, bin_model=None, **kwargs) -> nx.Graph`
  - **Non-spatial:** `ER()`, `configuration()`, `BA()`, `smallworld()`
  - **Spatial:** `spatial_null()` (requires bin_model and pair_features)
  - Registry: `NULL_MODELS` dict and `get_null_model(name)` function
  - Allows flexible iteration over multiple null models with consistent API

### `jhapl_gnn/` Core modules
Core modules and utilities.

**Key modules:**
- `config.py`: Global configuration (N_BINS, N_NULLS, RANDOM_SEED)
- `graph_io.py`: Graph I/O operations (PyTorch .pt format)
  - `load_synapses_from_pt()`: Load synapse connectivity
  - `load_positions_from_pt()`: Load neuron coordinates
  - `build_synapse_graph()`: Construct undirected graph
  - `build_synapse_digraph()`: Construct directed graph

- `spatial_analysis.py`: Spatial utilities for filtering and visualization
  - `filter_neurons()`: Extract subgraph by spatial radius
  - `build_partial_graph()`: Get edges for neuron subset
  - `decompose()`: PCA decomposition to 2D
  - `plot_vis()`: Visualize network in 2D

## Data Requirements

The pipeline expects JSON files in `data/processed/`:

**`synapses.json`**: Edge information
```json
{
  "synapse_id_1": [[source_node, target_node], ...],
  "synapse_id_2": [[source_node, target_node], ...],
  ...
}
```

**`positions.json`**: 3D neuron coordinates
```json
{
  "neuron_1": [x, y, z],
  "neuron_2": [x, y, z],
  ...
}
```

**`adjacency.json`**: Adjacency information (optional)
```json
{
  "neuron_1": [list of connected neurons],
  ...
}
```

## Usage Guide

### 1. Basic Setup

```python
from graph_io import read_json, build_synapse_digraph
from binning.compute_bins import compute_bins
from metrics.clustering_metrics import clustering

# Load data
synapses = read_json("data/processed/synapses.json")
GT = build_synapse_digraph(synapses)
```

### 2. Compute Network Metrics

```python
from metrics.clustering_metrics import clustering, transitivity
from metrics.hub_spoke_metrics import gini, coef_variation

# Compute metrics on ground truth graph
c = clustering(GT)
t = transitivity(GT)
g = gini(GT)
```

### 3. Generate Spatial Null Model

```python
from binning.compute_bins import compute_bins
from null_models.spatial_null_model import generate_spatial_null

# Compute edge probability bins
bin_model = compute_bins(distances, edges, n_bins=20, method="quantile")

# Generate null model preserving spatial structure
G_null = generate_spatial_null(
    nodes=nodes,
    pair_features=[(u, v, dist) for ...],
    bin_model=bin_model,
    target_edges=GT.number_of_edges()
)
```

### 4. Compare Against Multiple Nulls (Unified Interface)

```python
from null_models.wrappers import get_null_model, NULL_MODELS
from metrics.generators import run_null_models, summarize_results

# Option 1: Use by name
metrics_list = [clustering, transitivity, gini]
null_model_names = ['ER', 'configuration', 'BA', 'smallworld']
null_fns = [get_null_model(name) for name in null_model_names]

# Run comparison
results = run_null_models(null_fns, metrics_list, GT, N=50)
summary = summarize_results(GT, results, metrics_list)

# Option 2: Iterate over all available
for name, null_fn in NULL_MODELS.items():
    if name != 'spatial_null':  # Skip spatial (special interface)
        G_null = null_fn(GT)
        c = clustering(G_null)

# Option 3: Generate spatial null model
from null_models.wrappers import spatial_null
from binning.compute_bins import compute_bins

bin_model = compute_bins(distances, edges, n_bins=20)
pair_features = [(u, v, dist) for u, v, dist in ...]  # All node pairs with features
G_spatial = spatial_null(GT, bin_model, pair_features)
```

### 5. Analyze Motifs

```python
from metrics.count_metrics import generate_motif_df, plot_summary

# Compute motifs with z-scores
motif_summary = generate_motif_df(GT, null_models, n=100)

# Plot comparison
plot_summary(motif_summary, null_models)
```

### 6. Visualize Local Subgraph

```python
from spatial_analysis import filter_neurons, decompose, plot_vis
import numpy as np

# Extract local region
sub_neurons, sub_coords = filter_neurons(neuron_ids, coords, R=50000)

# Decompose to 2D
xy = decompose(sub_coords)

# Plot
plot_vis(sub_neurons, sub_edges, xy)
```

## Running the Main Pipeline

The easiest way to run the complete analysis is with the main script:

```bash
cd scripts
python main.py
```

### Configuration

The script has a `CONFIG` dictionary at the top that controls what analyses to run:

```python
CONFIG = {
    # Data paths
    'data_dir': '../data/processed',
    'output_dir': '../outputs',
    
    # Which null models to run
    'null_models': [
        'ER',
        'configuration',
        'BA',
        'smallworld',
        'spatial_null',
    ],
    
    # Which metrics to compute
    'metrics': [
        'gini',
        'coef_variation',
        'mean_deg',
        'clustering',
        'transitivity',
        'triangles',
    ],
    
    # Which visualizations to generate
    'visualizations': [
        'motif_comparison',
        'subgraph',
        'metric_summary_table',
    ],
    
    # Analysis parameters
    'n_null_samples': 10,        # Samples per null model
    'n_motif_samples': 10,       # Samples for motif analysis
    'n_bins': 20,                # Feature bins for spatial null
    'spatial_radius': 50000,     # Spatial filtering radius
}
```

Edit these values to customize the analysis before running.

### Output

The script generates outputs in the `outputs/` directory (configurable):
- `motif_summary.csv` - Triadic motif counts with z-scores
- `motif_comparison.png` - Motif visualization
- `metric_summary.csv` - Mean/stdev metrics for each null model
- `subgraph_visualization.png` - Local network structure
- Plus any additional outputs based on CONFIG settings

## Example Notebook

See `notebooks/full_pipeline_example.ipynb` for a complete interactive walkthrough with detailed explanations.

## Configuration

Edit `src/config.py` to adjust:
- `N_BINS`: Number of bins for feature quantization (default: 20)
- `N_NULLS`: Number of null model samples (default: 100)
- `RANDOM_SEED`: Reproducibility seed (default: 42) 