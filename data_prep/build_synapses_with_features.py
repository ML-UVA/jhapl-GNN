"""
Extract ground-truth synapses *with per-synapse features* from raw .pbz2
graph exports.

Same endpoint-pairing logic as ``data_prep.build_synapses`` but additionally
captures the ``volume``, ``upstream_dist`` and ``head_neck_shaft`` fields
from each ``synapse_data`` entry.

Output schema (``data/processed/synapses_with_features.pt``)::

    {
        'edge_index':      LongTensor[2, N_edges]   # indices into node_ids
        'node_ids':        list[str]                # neuron fileids (with _0/_1 suffix)
        'volume':          FloatTensor[N_edges]
        'upstream_dist':   FloatTensor[N_edges]
        'head_neck_shaft': list[str]                # per-edge, one of {'head','shaft','no_head'}
    }
"""

import os
import json
import argparse
import torch
from datasci_tools import system_utils as su


def parse_args():
    from config import INTERMEDIATE_DIR
    parser = argparse.ArgumentParser(
        description="Extract ground-truth synapses with per-synapse features"
    )
    parser.add_argument('--config', type=str, default="synapse_gnn/config.json")
    parser.add_argument('--output', type=str,
                        default=str(INTERMEDIATE_DIR / "synapses_with_features.pt"))
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)


def extract_synapses_with_features(graph_dir, output_file):
    print(f"Extracting ground-truth synapses with features from raw .pbz2 graphs in {graph_dir}...")

    # syn_id -> {'endpoints': [pre_fileid, post_fileid],
    #            'volume': float, 'upstream_dist': float, 'head_neck_shaft': str}
    synapses = {}

    for filename in os.listdir(graph_dir):
        if not filename.endswith(".pbz2"):
            continue

        name = os.path.join(graph_dir, filename)
        fileid = filename.split("_auto_proof")[0] if "_auto_proof" in filename else filename.split(".")[0]

        try:
            G = su.decompress_pickle(name)
        except Exception:
            continue

        for node in G.nodes:
            if "synapse_data" not in G.nodes[node]:
                continue

            for data in G.nodes[node]["synapse_data"]:
                pos = 0 if data.get("syn_type") == "presyn" else 1
                syn_id = int(data["syn_id"])

                entry = synapses.setdefault(syn_id, {
                    "endpoints":       [-1, -1],
                    "volume":          None,
                    "upstream_dist":   None,
                    "head_neck_shaft": None,
                })

                if entry["endpoints"][pos] == -1:
                    entry["endpoints"][pos] = fileid

                # Attach attrs on first occurrence; both sides carry identical metadata.
                if entry["volume"] is None:
                    entry["volume"] = float(data.get("volume", 0) or 0)
                if entry["upstream_dist"] is None:
                    entry["upstream_dist"] = float(data.get("upstream_dist", 0) or 0)
                if entry["head_neck_shaft"] is None:
                    entry["head_neck_shaft"] = str(data.get("head_neck_shaft", "no_head") or "no_head")

    print("Filtering for complete pairs...")
    complete = {sid: e for sid, e in synapses.items() if -1 not in e["endpoints"]}
    print(f"  {len(complete):,} complete / {len(synapses):,} total")

    print("Converting to PyTorch tensors...")

    unique_strings = set()
    for entry in complete.values():
        unique_strings.add(entry["endpoints"][0])
        unique_strings.add(entry["endpoints"][1])
    node_ids = sorted(list(unique_strings))
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    sources, targets = [], []
    volumes, upstream_dists, head_neck_shafts = [], [], []
    for entry in complete.values():
        pre, post = entry["endpoints"]
        sources.append(id_to_idx[pre])
        targets.append(id_to_idx[post])
        volumes.append(entry["volume"])
        upstream_dists.append(entry["upstream_dist"])
        head_neck_shafts.append(entry["head_neck_shaft"])

    graph_dict = {
        'edge_index':      torch.tensor([sources, targets], dtype=torch.long),
        'node_ids':        node_ids,
        'volume':          torch.tensor(volumes, dtype=torch.float),
        'upstream_dist':   torch.tensor(upstream_dists, dtype=torch.float),
        'head_neck_shaft': head_neck_shafts,
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    torch.save(graph_dict, output_file)

    print(f"Saved -> {output_file}")
    print(f"  edge_index:      {tuple(graph_dict['edge_index'].shape)}")
    print(f"  node_ids:        {len(node_ids):,}")
    print(f"  volume:          {tuple(graph_dict['volume'].shape)}")
    print(f"  upstream_dist:   {tuple(graph_dict['upstream_dist'].shape)}")
    print(f"  head_neck_shaft: list[str] len={len(head_neck_shafts):,}")


def main(config_path=None, output=None):
    if config_path is None or output is None:
        args = parse_args()
        config_path = config_path or args.config
        output = output or args.output

    config = load_config(config_path)
    graph_dir = config["raw_data"]["neurons_directory"]
    extract_synapses_with_features(graph_dir=graph_dir, output_file=output)


if __name__ == "__main__":
    main()
