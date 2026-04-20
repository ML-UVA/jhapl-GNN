"""
Build ``neuron_features.pt`` aligned to ``positions.pt``'s dedupe'd ordering.

Output::

    {
        'features':  FloatTensor[N_canonical, 10],
        'node_ids':  list[int]                 # canonical int neuron IDs
    }

Feature columns:
    0  out_degree
    1  in_degree
    2  total_degree
    3  mean volume
    4  std  volume
    5  mean upstream_dist
    6  std  upstream_dist
    7  fraction of out-synapses labeled 'head'
    8  fraction of out-synapses labeled 'shaft'
    9  fraction of out-synapses labeled 'no_head'
"""

import argparse

import pandas as pd
import torch


def build_features(synapses_path, positions_path, output_path):
    # Canonical ordering from positions.pt: dedupe '_<N>' suffix, first wins.
    pos_blob = torch.load(positions_path, weights_only=False)
    canonical_ids = []
    id_to_row = {}
    for key in pos_blob['node_ids']:
        nid = int(str(key).split('_')[0])
        if nid not in id_to_row:
            id_to_row[nid] = len(canonical_ids)
            canonical_ids.append(nid)
    num_nodes = len(canonical_ids)

    syn_blob = torch.load(synapses_path, weights_only=False)
    edge_index = syn_blob['edge_index']
    node_ids = syn_blob['node_ids']
    id_table = [int(str(k).split('_')[0]) for k in node_ids]
    pre_int = [id_table[i] for i in edge_index[0].tolist()]
    post_int = [id_table[i] for i in edge_index[1].tolist()]

    df = pd.DataFrame({
        'pre_id':          pre_int,
        'post_id':         post_int,
        'volume':          syn_blob['volume'].tolist(),
        'upstream_dist':   syn_blob['upstream_dist'].tolist(),
        'head_neck_shaft': syn_blob['head_neck_shaft'],
    })
    print(f"Loaded {len(df):,} synapses")

    X = torch.zeros((num_nodes, 10), dtype=torch.float)

    out_deg = df['pre_id'].value_counts()
    in_deg = df['post_id'].value_counts()

    for neuron, row in id_to_row.items():
        o = int(out_deg.get(neuron, 0))
        i = int(in_deg.get(neuron, 0))
        X[row, 0] = o
        X[row, 1] = i
        X[row, 2] = o + i

    for neuron, g in df.groupby('pre_id'):
        row = id_to_row.get(neuron)
        if row is None:
            continue

        X[row, 3] = g['volume'].mean()
        X[row, 4] = g['volume'].std() if len(g) > 1 else 0.0
        X[row, 5] = g['upstream_dist'].mean()
        X[row, 6] = g['upstream_dist'].std() if len(g) > 1 else 0.0

        total = len(g)
        X[row, 7] = (g['head_neck_shaft'] == 'head').sum() / total
        X[row, 8] = (g['head_neck_shaft'] == 'shaft').sum() / total
        X[row, 9] = (g['head_neck_shaft'] == 'no_head').sum() / total

    torch.save({'features': X, 'node_ids': canonical_ids}, output_path)
    print(f"Saved features {tuple(X.shape)} -> {output_path}")


if __name__ == '__main__':
    from config import INTERMEDIATE_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument('--synapses_path', type=str,
                        default=str(INTERMEDIATE_DIR / 'synapses_with_features.pt'))
    parser.add_argument('--positions_path', type=str,
                        default=str(INTERMEDIATE_DIR / 'positions.pt'))
    parser.add_argument('--output', type=str,
                        default=str(INTERMEDIATE_DIR / 'neuron_features.pt'))
    args = parser.parse_args()
    build_features(args.synapses_path, args.positions_path, args.output)
