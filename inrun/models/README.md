# JHU APL Connectome GNN: Motif Discovery Pipeline

This pipeline performs unsupervised motif discovery on connectome graphs using Graph Autoencoders (GAE) and In-Run Data Shapley values. Given a graph of neurons and synapses, it identifies structurally important subgraphs by measuring each neuron's contribution to the model's learned representation.

---

## Repository Structure

```
inrun/
├── main.py  
├── run_entire_pipeline.slurm
│
├── models/
│   ├── filter_graph.py  
│   ├── graphnodeshapley.py
│   ├── normalize.py       
│   └── motifs.py          
│
└── ghostEngines/
    ├── __init__.py
    ├── graddotprod_engine.py  
    ├── autograd_grad_sample_dotprod.py
    ├── supported_layers_grad_samplers_dotprod.py
    └── engine_manager.py
```

---

## Usage

The pipeline is run through a single entry point. All steps, graph construction, GAE training, Shapley computation, normalization, and motif visualization, are handled automatically.

### Use the existing pre-built graph:
```bash
python main.py --use_existing
```

### Build a new graph from a spatial region of the brain:
```bash
python main.py \
    --synapses_path synapses.json \
    --coords_path   data/neuron_coords.json \
    --x_min 800000 --x_max 1000000 \
    --y_min 700000 --y_max 900000
```

### Submit as a SLURM job:
```bash
sbatch run_entire_pipeline.slurm
```

Override SLURM config without editing the file:
```bash
sbatch --export=ALL,EPOCHS=500,USE_EXISTING=false,X_MIN=800000 run_pipeline.sh
```

---

## Configuration
 
All arguments are passed to `main.py`.
 
For the graph source, pass `--use_existing` to run on the default graph, or pass spatial thresholds like `--x_min`, `--x_max`, `--y_min`, `--y_max`, `--z_min`, `--z_max` (in nanometers) to filter a new one from `synapses.json`. The coordinates file at `data/neuron_coords.json` is required for spatial filtering — see the setup section below for how to generate it.
 
For the model, the default is a standard 2-layer GCN autoencoder. Add `--variational` to use a VGAE instead, which adds a KL divergence loss term and produces smoother embeddings. Add `--linear` to use a single-layer linear encoder. These flags can be combined. Other model arguments are `--epochs` (default 200), `--latent_dim` (default 16), and `--lr` (default 0.01).
 
For motifs, `--motif_sizes` controls which subgraph sizes to extract (default: 5 10 20), `--top_k` controls how many motifs to extract per size (default: 3), and `--output_dir` sets where all outputs are saved (default: results/).

---

## Outputs

All outputs are saved to `--output_dir` (default: `results/`):

| File | Description |
|---|---|
| `filtered_graph.csv` | Graph used for training |
| `graph_node_shapley.value` | Raw per-node Shapley values (pickle) |
| `graph_node_shapley_normalized.value` | Normalized Shapley values as % of total (pickle) |
| `graph_node_shapley_model.pt` | Trained model checkpoint |
| `grad_dotprods/` | Gradient dot-product logs |
| `motif_size{N}_{K}_graph.png` | Motif graph colored by Shapley value |
| `motif_size{N}_{K}_heatmap.png` | Adjacency heatmap for each motif |


---

