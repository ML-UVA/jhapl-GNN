import argparse
import os.path as osp
import time
import pickle

import torch
import pandas as pd
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.nn import GAE, GCNConv

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--latent_dim', type=int, default=16)
parser.add_argument('--csv_path', type=str, default='data/top5_k1.csv')
parser.add_argument('--feature_path', type=str, default='data/neuron_features.pt')
args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

df = pd.read_csv(args.csv_path)
neuron_ids = pd.unique(df[['pre_id', 'post_id']].values.ravel())
neuron_map = {nid: i for i, nid in enumerate(neuron_ids)}

edge_index = torch.tensor(
    [[neuron_map[p], neuron_map[q]] for p, q in zip(df.pre_id, df.post_id)],
    dtype=torch.long
).t().contiguous()

x = torch.load(args.feature_path, weights_only=True)

data = Data(x=x, edge_index=edge_index)

transform = T.Compose([
    T.NormalizeFeatures(),
    T.ToDevice(device),
    T.RandomLinkSplit(
        num_val=0.0,
        num_test=0.0,
        is_undirected=True,
        add_negative_train_samples=False
    ),
])

train_data, _, _ = transform(data)


class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


in_channels = train_data.num_features
out_channels = args.latent_dim

model = GAE(GCNEncoder(in_channels, out_channels)).to(device)

feature_decoder = torch.nn.Sequential(
    torch.nn.Linear(out_channels, out_channels),
    torch.nn.ReLU(),
    torch.nn.Linear(out_channels, in_channels)
).to(device)

optimizer = torch.optim.Adam(
    list(model.parameters()) + list(feature_decoder.parameters()),
    lr=0.01
)

node_shapley = torch.zeros(train_data.num_nodes, device=device)


def train():
    model.train()
    feature_decoder.train()
    optimizer.zero_grad()

    z = model.encode(train_data.x, train_data.edge_index)
    edge_loss = model.recon_loss(z, train_data.edge_index)

    x_recon = feature_decoder(z)
    per_node_loss = ((x_recon - train_data.x) ** 2).mean(dim=1)
    feature_loss = per_node_loss.mean()

    loss = edge_loss + 0.1 * feature_loss
    loss.backward()

    with torch.no_grad():
        grads = []
        for p in model.parameters():
            if p.grad is not None:
                grads.append(p.grad.flatten())
        g = torch.cat(grads)
        node_shapley.add_(per_node_loss * g.norm())

    optimizer.step()
    return float(loss)


times = []

for epoch in range(1, args.epochs + 1):
    start = time.time()
    loss = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    times.append(time.time() - start)

print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")

record = {
    "node_index": list(range(train_data.num_nodes)),
    "First-order In-Run Data Shapley": node_shapley.cpu().tolist()
}

with open("graph_node_shapley.value", "wb") as f:
    pickle.dump(record, f)

print("Saved Shapley values to graph_node_shapley.value")
