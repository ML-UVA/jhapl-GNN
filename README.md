# Neuron and Morphology Graph Analyzation


## Setup
This repository contains tools and analyses for studying neuronal morphology using graph-based representations.

The analyses assume that neuron morphologies are stored as NetworkX graph objects located in data/graph_exports/. These graphs encode morphological structure along with additional node- and edge-level features. These graph objects were stored as .pbz2 files.

Ex: 86491134110093308_0_auto_proof_v7_proofread.pbz2

The datasets used to develop and validate these analyses were sourced from the MICrONS project and processed through NEURD for automated proofreading and structural refinement.

Each .py and .pynb file should be ran directly from the root directory. All data generated from the pipeline and analyzes are stored in the data folder.

## Project Structure:

```text
data/
├── ADP/    # Calculated Axon-Dendrite proximity between every pair of neuron
├── Community Detection/        # Detects communities in neuron graph
├── Visualization/          # Visualization Tools
├── datao      # Stores all data and checkpoint data
```