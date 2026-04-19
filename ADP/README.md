# Axon–Dendrite Proximity and Co-Travel Distance

## Overview

This repository provides tools and pipelines for computing **Axon–Dendrite Proximity (ADP)** between neurons.

**ADP** is a spatial metric that quantifies how often an axon from one neuron comes within a specified distance of a dendrite from another neuron in 3D space. It is measured in terms of total co-travel distance.

These tools are designed to efficiently process large-scale neuron datasets.

---

## Data Requirements

The pipeline assumes neuron morphologies are stored as **NetworkX graph objects** located in a folder passed into CLI (--graph_path):

Each graph encodes:
- Neuron morphology (3D structure)
- Node-level features
- Edge-level features

Graphs are stored as compressed `.pbz2` files.

Example:
```
86491134110093308_0_auto_proof_v7_proofread.pbz2

86491134110093308: neuron_id
0: split_index
```

---

## Data Source

The datasets used for development and validation were sourced from:

- MICrONS project
- Processed using NEURD for automated proofreading and structural refinement

---

## Pipeline

The main entry point is:

```
ADP_Calculation_Full.py
```

This script provides a command-line interface (CLI) for running the full ADP pipeline.

## Example Usage

Run the full ADP pipeline from the project root:

```bash
python -m ADP.ADP_Calculation_Full \
  --graph_path ../../graph_exports \
  --data_path data/ \
  --radius 5000
  --threshold 10
```

---

## Inputs

- `--graph_path`  
  Path to neuron graph exports (e.g., `data/graph_exports`) relative to current working directory

- `--data_path`  
  Path where intermediate and final outputs will be stored (e.g., `data/checkpoints`) relative to current working directory

- `--radius`  
  Distance threshold (in nanometers) used for proximity calculations

- `--threshold`  
  Minimum distance threshold (in microns) to keep edges in generated networkx


---

## Processing Steps

The pipeline performs the following:

1. Skeletonization  
   Converts neuron graphs into skeletonized representations  
   (assumes 1000 nm spacing between points)

2. KD-Tree Construction  
   Builds KD-tree structures for each neuron to enable efficient spatial queries

3. Spatial Partitioning and Search  
   - Partitions 3D space into blocks  
   - Performs localized neighbor searches  
   - Computes proximity between axons and dendrites

4. ADP Computation  
   Produces a dictionary of the form:
   ```
   ADP[dendrite_neuron][axon_neuron] = proximity count
   ```

5. Output Generation  
   Saves intermediate structures and final results (e.g., pickle files) to the specified `data_path` (adp_data.pkl)

6. Graph Generation  
   Generates networkx bidirectional graph with ADP values as edge values

---
The primary output of the pipeline is a **NetworkX directed graph** representing axon–dendrite proximity relationships between neurons.

- **Nodes** represent neurons  
- **Directed edges** represent axon → dendrite proximity  
- **Edge weights (`adp`)** store total co-travel distance (in microns)

---

### Graph Construction

The graph is constructed from an intermediate ADP dictionary of the form:

```
{dendrite_neuron: {axon_neuron: value}}
```

For each entry:

```
ADP[A][B] = value
```

a directed edge is added:

```
B → A
```

if:

```
value >= threshold
```

---

### Interpretation

- Edge **B → A** means:
  - The **axon of neuron B** is within `{radius}` nm of  
  - The **dendrites of neuron A**

- The edge weight (`adp`) represents the total co-travel distance (in microns), computed from discrete points (~1000 nm spacing) along dendritic branches.

---

### Output

The resulting graph is saved as:

```
<data_path>/adp_graph_threshold_<threshold>.pkl
```

This file contains a **NetworkX `DiGraph`** object and can be loaded with:

```python
import pickle

with open("data/adp_graph_threshold_10.pkl", "rb") as f:
    G = pickle.load(f)
```
