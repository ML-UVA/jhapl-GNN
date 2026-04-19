# jhapl-GNN

Motif, clustering, degree, and spatial-null analysis of a directed synapse graph.

## Layout

```
data_prep/              # Shared I/O
  graph_io.py           # Load/save .pt, build NetworkX graphs
  spatial_analysis.py   # Filter, PCA, plot
  extract_synapses.py   # .pbz2 -> synapses.pt
null_analysis/          # Motif / null-model pipeline
  __main__.py           # Entry point
  config.py             # N_BINS, N_NULLS, RANDOM_SEED
  binning/              # BinModel, P(edge | feature)
  metrics/              # Motifs, clustering, degree stats
  null_models/          # ER / configuration / BA / smallworld / spatial
scripts/                # Preprocessing CLIs
ADP/                    # Separate: axonal-dendritic proximity
synapse_gnn/            # Separate: GNN training
data/ notebooks/
```

## Inputs

- `data/processed/synapses.pt` — `{'synapses': {syn_id: [[pre, post], meta]}}`
- `data/processed/positions.pt` — `{'positions': tensor[N,3], 'node_ids': [...]}`
- `data/processed/adp_data.pkl` (ADP mode only) — `{u: {v: adp_value}}`

Produce from raw `.pbz2`:

```
python -m data_prep.extract_synapses
python scripts/compute_positions.py
```

## Run

```
python -m null_analysis [-s SYN.pt] [-p POS.pt] [-o OUTDIR] [-d euclidean|adp]
```

Defaults: synapses/positions from `data/processed/`, output to `outputs/`,
euclidean. Tune null models, metrics, sample counts, `n_bins`, `spatial_radius`
via the `CONFIG` dict at the top of [null_analysis/__main__.py](null_analysis/__main__.py).

## Outputs (in `--output` dir)

- `motif_summary.csv` — triadic motif counts with z-scores
- `metric_summary.csv` — mean/stdev per null model
- `motif_comparison.png`, `subgraph_visualization.png`
- `gt_graph.pt`, `positions.pt`

## Library use

```python
from data_prep.graph_io import load_synapses_from_pt, build_synapse_digraph
from null_analysis.null_models.wrappers import get_null_model
from null_analysis.metrics.clustering_metrics import clustering
from null_analysis.metrics.generators import run_null_models, summarize_results

GT = build_synapse_digraph(load_synapses_from_pt('data/processed/synapses.pt'))
nulls = [get_null_model(n) for n in ['ER', 'configuration', 'BA', 'smallworld']]
results = run_null_models(nulls, [clustering], GT, N=50)
summary = summarize_results(GT, results, [clustering])
```

Spatial null:

```python
from null_analysis.binning.compute_bins import compute_bins
from null_analysis.null_models.wrappers import spatial_null

bin_model = compute_bins(features, edge_indicator, n_bins=20, method='quantile')
G_null = spatial_null(GT, bin_model, [(u, v, feat), ...])
```

Full worked example in [null_analysis/__main__.py](null_analysis/__main__.py).

