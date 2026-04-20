"""
Trains a Graph Autoencoder (GAE/VGAE) on the connectome graph and computes
first-order In-Run Data Shapley values per node using the ghost dot-product
engine.

USAGE:
    python node_graph_autoencoder.py \
        --csv_path      data/top5_k1.csv \
        --feature_path  data/neuron_features.pt \
        --epochs        200 \
        --latent_dim    16 \
        --output_path   results/graph_node_shapley.value

    # Variational:
    python node_graph_autoencoder.py --variational

    # Linear encoder:
    python node_graph_autoencoder.py --linear
"""

import argparse
import os
import time
import pickle
import networkx as nx
from torch_geometric.utils import from_networkx

import torch
import pandas as pd
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.nn import GAE, VGAE, GCNConv

from ..ghostEngines.graddotprod_engine import GradDotProdEngine



def train_and_compute_shapley(
    G,
    feature_path,
    output_path='results/graph_node_shapley.value',
    epochs=200,
    latent_dim=16,
    lr=0.01,
    save_interval=50,
    variational=False,
    linear=False,
):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"[INFO] Using device: {device}")
    neuron_ids = list(G.nodes())
    neuron_map = {nid: i for i, nid in enumerate(neuron_ids)}

    edge_index = torch.tensor(
        [[neuron_map[u], neuron_map[v]] for u, v in G.edges()],
        dtype=torch.long
    ).t().contiguous()

    feat_blob = torch.load(feature_path, weights_only=False)
    all_features = feat_blob['features']
    id_to_row = {nid: i for i, nid in enumerate(feat_blob['node_ids'])}
    try:
        rows = [id_to_row[nid] for nid in neuron_ids]
    except KeyError as e:
        raise KeyError(
            f"Neuron {e.args[0]} in graph is missing from feature set at {feature_path}"
        ) from e
    x = all_features[rows]
    data = Data(x=x, edge_index=edge_index)
 
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
        T.RandomLinkSplit(
            num_val=0.05,
            num_test=0.1,
            is_undirected=True,
            split_labels=True,
            add_negative_train_samples=False
        ),
    ])
 
    train_data, val_data, test_data = transform(data)
    num_nodes = train_data.num_nodes
    val_nodes = val_data.pos_edge_label_index.unique().numel()
    print(f"[INFO] Nodes: {num_nodes}, Val nodes: {val_nodes}")
    print(f"[INFO] Train edges: {train_data.pos_edge_label_index.size(1)}")
    print(f"[INFO] Val edges:   {val_data.pos_edge_label_index.size(1)}")
    print(f"[INFO] Test edges:  {test_data.pos_edge_label_index.size(1)}")
    class GCNEncoder(torch.nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv1 = GCNConv(in_channels, 2 * out_channels)
            self.conv2 = GCNConv(2 * out_channels, out_channels)
        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index).relu()
            return self.conv2(x, edge_index)
 
    class VariationalGCNEncoder(torch.nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv1       = GCNConv(in_channels, 2 * out_channels)
            self.conv_mu     = GCNConv(2 * out_channels, out_channels)
            self.conv_logstd = GCNConv(2 * out_channels, out_channels)
        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index).relu()
            return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)
 
    class LinearEncoder(torch.nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv = GCNConv(in_channels, out_channels)
        def forward(self, x, edge_index):
            return self.conv(x, edge_index)
 
    class VariationalLinearEncoder(torch.nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv_mu     = GCNConv(in_channels, out_channels)
            self.conv_logstd = GCNConv(in_channels, out_channels)
        def forward(self, x, edge_index):
            return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)
 
    in_channels  = train_data.num_features
    out_channels = latent_dim  # CHANGED: was args.latent_dim
 
    if not variational and not linear:
        model = GAE(GCNEncoder(in_channels, out_channels))
    elif not variational and linear:
        model = GAE(LinearEncoder(in_channels, out_channels))
    elif variational and not linear:
        model = VGAE(VariationalGCNEncoder(in_channels, out_channels))
    else:
        model = VGAE(VariationalLinearEncoder(in_channels, out_channels))
 
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
 
    dot_prod_dir = os.path.join(os.path.dirname(os.path.abspath(output_path)), "grad_dotprods")
    os.makedirs(dot_prod_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
 
    engine = GradDotProdEngine(
        module=model,
        val_batch_size=val_nodes,
        loss_reduction='mean',
        use_dummy_bias=False,
        dot_prod_save_path=dot_prod_dir,
    )
    engine.attach(optimizer)
    engine.attach_and_store_valset(
        X_val=train_data.x,
        val_edge_index=val_data.pos_edge_label_index
    )
 
    node_shapley = torch.zeros(num_nodes, device='cpu')
 
    def train_step(epoch):
        model.train()
        optimizer.zero_grad()
        engine.attach_train_batch(
            X_train=train_data.x,
            iter_num=epoch,
            edge_index=train_data.edge_index,
        )
        z = model.encode(train_data.x, train_data.edge_index)
        loss = model.recon_loss(z, train_data.pos_edge_label_index)
        if variational:
            loss = loss + (1 / train_data.num_nodes) * model.kl_loss()
        loss.backward()
        engine.prepare_gradients()
        optimizer.step()
        engine.aggregate_and_log()
        engine.clear_gradients()
        optimizer.zero_grad(set_to_none=True)
        if engine.dot_product_log:
            latest = engine.dot_product_log[-1]['dot_product']
            node_shapley[:latest.size(0)].add_(latest)
        return float(loss)
 
    @torch.no_grad()
    def test_step(data):
        model.eval()
        z = model.encode(data.x, data.edge_index)
        return model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)
 
    times = []
    for epoch in range(1, epochs + 1):
        start = time.time()
        loss = train_step(epoch)
        auc, ap = test_step(test_data)
        times.append(time.time() - start)
        print(f'Epoch: {epoch:03d} | Loss: {loss:.4f} | AUC: {auc:.4f} | AP: {ap:.4f}')
        if epoch % save_interval == 0:
            try:
                engine.save_dot_product_log(iter_num=epoch)
            except ValueError:
                pass
 
    print(f"\nMedian time per epoch: {torch.tensor(times).median():.4f}s")
 
    engine.detach()
 
    record = {
        "node_index":                      list(range(num_nodes)),
        "neuron_ids":                      [int(x) for x in neuron_ids],
        "First-order In-Run Data Shapley": node_shapley.tolist(),
    }
 
    with open(output_path, "wb") as f:
        pickle.dump(record, f)
 
    print(f"\n[INFO] Saved Shapley values -> {output_path}")
    vals = torch.tensor(record["First-order In-Run Data Shapley"])
    top5 = vals.topk(min(5, len(vals)))
    for idx, val in zip(top5.indices.tolist(), top5.values.tolist()):
        nid = record['neuron_ids'][idx] if idx < len(record['neuron_ids']) else 'unknown'
        print(f"  Node {idx} (neuron {nid}) | Shapley = {val:.6f}")
 
    ckpt_path = output_path.replace(".value", "_model.pt")
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'auc': auc,
        'ap': ap,
    }, ckpt_path)
    print(f"[INFO] Saved model checkpoint -> {ckpt_path}")
 
 
if __name__ == '__main__':
    from config import INTERMEDIATE_DIR, OUTPUT_DIR
    from .filter_graph import build_graph

    parser = argparse.ArgumentParser()
    parser.add_argument('--variational',    action='store_true')
    parser.add_argument('--linear',         action='store_true')
    parser.add_argument('--epochs',         type=int,   default=200)
    parser.add_argument('--latent_dim',     type=int,   default=16)
    parser.add_argument('--lr',             type=float, default=0.01)
    parser.add_argument('--use_existing',   action='store_true')
    parser.add_argument('--existing_csv',   type=str,   default='data/top5_k1.csv')
    parser.add_argument('--synapses_path',  type=str,   default=str(INTERMEDIATE_DIR / 'synapses_with_features.pt'))
    parser.add_argument('--positions_path', type=str,   default=str(INTERMEDIATE_DIR / 'positions.pt'))
    parser.add_argument('--feature_path',   type=str,   default=str(INTERMEDIATE_DIR / 'neuron_features.pt'))
    parser.add_argument('--output_path',    type=str,   default=str(OUTPUT_DIR / 'motifs' / 'graph_node_shapley.value'))
    parser.add_argument('--save_interval',  type=int,   default=50)
    args = parser.parse_args()

    G = build_graph(
        use_existing=args.use_existing,
        existing_csv=args.existing_csv,
        synapses_path=args.synapses_path,
        positions_path=args.positions_path,
    )
    train_and_compute_shapley(
        G=G,
        feature_path=args.feature_path,
        output_path=args.output_path,
        epochs=args.epochs,
        latent_dim=args.latent_dim,
        lr=args.lr,
        save_interval=args.save_interval,
        variational=args.variational,
        linear=args.linear,
    )



"""



# ============================================================
#  CLI
# ============================================================

parser = argparse.ArgumentParser()
parser.add_argument('--variational',   action='store_true')
parser.add_argument('--linear',        action='store_true')
parser.add_argument('--epochs',        type=int,   default=200)
parser.add_argument('--latent_dim',    type=int,   default=16)
parser.add_argument('--lr',            type=float, default=0.01)
parser.add_argument('--csv_path',      type=str,   default='data/top5_k1.csv')
parser.add_argument('--feature_path',  type=str,   default='data/neuron_features.pt')
parser.add_argument('--output_path',   type=str,   default='results/graph_node_shapley.value')
parser.add_argument('--save_interval', type=int,   default=50,
                    help='Save dot-product log every N epochs')
args = parser.parse_args()

# ============================================================
#  DEVICE
# ============================================================

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print(f"[INFO] Using device: {device}")

# ============================================================
#  LOAD GRAPH
# ============================================================

def load_neuron_graph(csv_path, feature_path):
    df = pd.read_csv(csv_path)
    neuron_ids = pd.unique(df[['pre_id', 'post_id']].values.ravel())
    neuron_map = {nid: i for i, nid in enumerate(neuron_ids)}
    edge_index = torch.tensor(
        [[neuron_map[p], neuron_map[q]] for p, q in zip(df.pre_id, df.post_id)],
        dtype=torch.long
    ).t().contiguous()
    x = torch.load(feature_path, weights_only=True)
    return Data(x=x, edge_index=edge_index), neuron_ids

data, neuron_ids = load_neuron_graph(args.csv_path, args.feature_path)

# ============================================================
#  TRAIN / VAL / TEST SPLIT
# ============================================================

transform = T.Compose([
    T.NormalizeFeatures(),
    T.ToDevice(device),
    T.RandomLinkSplit(
        num_val=0.05,
        num_test=0.1,
        is_undirected=True,
        split_labels=True,
        add_negative_train_samples=False
    ),
])

train_data, val_data, test_data = transform(data)

num_nodes    = train_data.num_nodes
val_nodes    = val_data.pos_edge_label_index.unique().numel()

print(f"[INFO] Nodes: {num_nodes}, Val nodes (from held-out edges): {val_nodes}")
print(f"[INFO] Train edges: {train_data.pos_edge_label_index.size(1)}")
print(f"[INFO] Val edges:   {val_data.pos_edge_label_index.size(1)}")
print(f"[INFO] Test edges:  {test_data.pos_edge_label_index.size(1)}")

# ============================================================
#  ENCODER ARCHITECTURES
# ============================================================

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1    = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu  = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


class LinearEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


class VariationalLinearEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_mu     = GCNConv(in_channels, out_channels)
        self.conv_logstd = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


# ============================================================
#  BUILD MODEL
# ============================================================

in_channels  = train_data.num_features
out_channels = args.latent_dim

if not args.variational and not args.linear:
    model = GAE(GCNEncoder(in_channels, out_channels))
elif not args.variational and args.linear:
    model = GAE(LinearEncoder(in_channels, out_channels))
elif args.variational and not args.linear:
    model = VGAE(VariationalGCNEncoder(in_channels, out_channels))
else:
    model = VGAE(VariationalLinearEncoder(in_channels, out_channels))

model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# ============================================================
#  GHOST ENGINE SETUP
# ============================================================

# val_batch_size = number of unique nodes touched by validation edges.
# These are the last `val_nodes` rows when we order nodes by index —
# we use all nodes as the feature matrix but track val nodes separately
# via their edge index.
val_batch_size = val_nodes

dot_prod_dir = os.path.join(os.path.dirname(args.output_path), "grad_dotprods")
os.makedirs(dot_prod_dir, exist_ok=True)
os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)

engine = GradDotProdEngine(
    module=model,
    val_batch_size=val_batch_size,
    loss_reduction='mean',
    use_dummy_bias=False,
    dot_prod_save_path=dot_prod_dir,
)
engine.attach(optimizer)

# Store validation set (node features + held-out edges)
engine.attach_and_store_valset(
    X_val=train_data.x,                          # full node features
    val_edge_index=val_data.pos_edge_label_index  # held-out edges
)

# Accumulator: one Shapley score per node, summed across all epochs
node_shapley = torch.zeros(num_nodes, device='cpu')

# ============================================================
#  TRAIN / EVAL FUNCTIONS
# ============================================================

def train(epoch):
    model.train()
    optimizer.zero_grad()

    # Attach batch info to engine BEFORE forward pass
    engine.attach_train_batch(
        X_train=train_data.x,
        iter_num=epoch,
        edge_index=train_data.edge_index,
    )

    # Forward — concatenate train + val nodes so engine can split them
    # We pass ALL nodes through the encoder (engine hooks split internally)
    z = model.encode(train_data.x, train_data.edge_index)

    # Reconstruction loss on training edges
    edge_loss = model.recon_loss(z, train_data.pos_edge_label_index)

    loss = edge_loss
    if args.variational:
        loss = loss + (1 / train_data.num_nodes) * model.kl_loss()

    loss.backward()

    # Ghost engine: move accumulated train grads to .grad
    engine.prepare_gradients()
    optimizer.step()

    # Aggregate dot products and log them
    engine.aggregate_and_log()
    engine.clear_gradients()
    optimizer.zero_grad(set_to_none=True)

    # Accumulate Shapley values from this epoch's dot products
    if engine.dot_product_log:
        latest = engine.dot_product_log[-1]['dot_product']  # [train_nodes]
        # latest is per training-node; pad to full node count
        node_shapley[:latest.size(0)].add_(latest)

    return float(loss)


@torch.no_grad()
def test(data):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    return model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)


# ============================================================
#  TRAINING LOOP
# ============================================================

times = []

for epoch in range(1, args.epochs + 1):
    start  = time.time()
    loss   = train(epoch)
    auc, ap = test(test_data)
    elapsed = time.time() - start
    times.append(elapsed)

    print(f'Epoch: {epoch:03d} | Loss: {loss:.4f} | AUC: {auc:.4f} | AP: {ap:.4f}')

    # Periodically save dot-product log to disk
    if epoch % args.save_interval == 0:
        try:
            engine.save_dot_product_log(iter_num=epoch)
        except ValueError:
            pass  # nothing to save yet

print(f"\nMedian time per epoch: {torch.tensor(times).median():.4f}s")

# ============================================================
#  SAVE SHAPLEY VALUES
# ============================================================

engine.detach()

record = {
    "node_index":                    list(range(num_nodes)),
    "neuron_ids":                    [int(x) for x in neuron_ids],
    "First-order In-Run Data Shapley": node_shapley.tolist(),
}

with open(args.output_path, "wb") as f:
    pickle.dump(record, f)

print(f"\n[INFO] Saved Shapley values -> {args.output_path}")
print(f"[INFO] Nodes: {num_nodes}")
print(f"[INFO] Top 5 nodes by Shapley value:")
vals = torch.tensor(record["First-order In-Run Data Shapley"])
top5 = vals.topk(5)
for idx, val in zip(top5.indices.tolist(), top5.values.tolist()):
    print(f"  Node {idx} (neuron {record['neuron_ids'][idx]}) | Shapley = {val:.6f}")

# ============================================================
#  SAVE MODEL CHECKPOINT
# ============================================================

ckpt_path = args.output_path.replace(".value", "_model.pt")
torch.save({
    'epoch':      args.epochs,
    'model_state_dict': model.state_dict(),
    'auc':        auc,
    'ap':         ap,
}, ckpt_path)
print(f"[INFO] Saved model checkpoint -> {ckpt_path}")
"""
