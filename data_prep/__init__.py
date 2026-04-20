"""
data_prep: Data I/O and preparation for the jhapl-GNN connectome.

Modules:
- graph_io: Load/save synapse and position data in PyTorch .pt format; build NetworkX graphs.
- spatial_analysis: Spatial filtering, 2D decomposition, and visualization helpers.
- build_synapses: Extract ground-truth synapses from raw .pbz2 files and save them as a PyG-ready `{edge_index, node_ids}` .pt file.
- build_synapses_with_features: Same as build_synapses but additionally captures per-synapse `volume`, `upstream_dist`, and `head_neck_shaft`.
- compute_positions: Extract neuron `S0.mesh_center` positions and build a pairwise distance graph from raw .pbz2 files.
"""
