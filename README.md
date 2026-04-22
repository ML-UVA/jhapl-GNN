# Neuron and Morphology Graph Analysis Tools

[Slides with Results](https://myuva-my.sharepoint.com/:p:/g/personal/zcp3re_virginia_edu/IQD3fgMrGUVlQ4PfwbILxI9TAcTp0DtNCuD3Wiw_kSiv5tI?e=iyQpD1)

## Setup
This repository contains tools and analyses for studying neuronal morphology using graph-based representations.

The analyses assume that neuron morphologies are stored as NetworkX graph objects located in a graph_exports folder. These graphs encode morphological structure along with additional node- and edge-level features. These graph objects were stored as .pbz2 files.

Ex: 86491134110093308_0_auto_proof_v7_proofread.pbz2

The datasets used to develop and validate these analyses were sourced from the MICrONS project and processed through NEURD for automated proofreading and structural refinement.

All data generated from the pipeline and analyzes are stored in the data folder.

## Paths

Global paths live in [`config.py`](config.py):

- `RAW_DATA_DIR` — `../../demo_graph_exports` (raw `.pbz2` morphologies)
- `INTERMEDIATE_DIR` — `data/processed/` (shared `.pt` / `.pkl` artifacts)
- `OUTPUT_DIR` — `outputs/` (per-subpackage results live under `outputs/<subpkg>/`)

## Entry points

Run from the repo root.

```bash
python -m synapse_gnn --build_data           # GraphSAGE GNN synapse predictor (first run: build dataset, then train)
python -m synapse_gnn                        # Retrain GNN from cached dataset

python -m null_analysis                      # Motif / null-model analysis (auto-regenerates missing synapses.pt / positions.pt)

python -m motifs                             # GAE + In-Run Data Shapley motif discovery (auto-regenerates missing synapses_with_features.pt / positions.pt / neuron_features.pt)
python -m motifs --use_existing              # ... from the pre-built top5_k1.csv graph

python -m ADP                                # Axon-dendrite proximity pipeline

python -m data_prep.build_synapses                # Build data/processed/synapses.pt from raw .pbz2
python -m data_prep.build_synapses_with_features  # Build data/processed/synapses_with_features.pt (adds volume / upstream_dist / head_neck_shaft)
python -m data_prep.compute_positions             # Build data/processed/positions.pt + distance graph
```