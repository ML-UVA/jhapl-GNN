# jhapl-GNN

Connectomics pipelines over shared `.pbz2` morphology graphs. Run from repo root.

## Entry points

```bash
python -m synapse_gnn --build_data   # GraphSAGE synapse predictor (first run)
python -m synapse_gnn                # retrain from cached dataset
python -m null_analysis              # null-model clustering analysis
python -m ADP --graph_path <pbz2_dir> --data_path <ckpt_dir> --radius <nm>
```

Per-workstream docs: [synapse_gnn/README.md](synapse_gnn/README.md), [null_analysis/README.md](null_analysis/README.md).
