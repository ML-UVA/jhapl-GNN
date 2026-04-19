import time
import pickle
import torch
import pandas as pd
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.nn import GAE, GCNConv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

csv_path = "data/top5_k1.csv"
feature_path = "data/neuron_features.pt"
epochs = 200
latent_dim = 16
lr = 0.01
feature_weight = 0.1

# --------------------------------------------------
# Load graph
# --------------------------------------------------
df = pd.read_csv(csv_path)
neuron_ids = pd.unique(df[['pre_id', 'post_id']].values.ravel())
neuron_map = {nid: i for i, nid in enumerate(neuron_ids)}

edge_index = torch.tensor(
    [[neuron_map[p], neuron_map[q]] for p, q in zip(df.pre_id, df.post_id)],
    dtype=torch.long
).t().contiguous()

x = torch.load(feature_path, weights_only=True)

data = Data(x=x, edge_index=edge_index)

transform = T.Compose([
    T.NormalizeFeatures(),
    T.ToDevice(device),
    T.RandomLinkSplit(
        num_val=0.0,
        num_test=0.0,
        is_undirected=True,
        add_negative_train_samples=False
    )
])

train_data, _, _ = transform(data)

# --------------------------------------------------
# Model
# --------------------------------------------------
class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.c1 = GCNConv(in_channels, 2 * out_channels)
        self.c2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.c1(x, edge_index).relu()
        return self.c2(x, edge_index)

model = GAE(Encoder(train_data.num_features, latent_dim)).to(device)

decoder = torch.nn.Sequential(
    torch.nn.Linear(latent_dim, latent_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(latent_dim, train_data.num_features)
).to(device)

optimizer = torch.optim.Adam(
    list(model.parameters()) + list(decoder.parameters()), lr=lr
)

# --------------------------------------------------
# Edge Shapley storage
# --------------------------------------------------
num_edges = train_data.edge_index.size(1)
edge_shapley = torch.zeros(num_edges, device=device)

# --------------------------------------------------
# Training loop
# --------------------------------------------------
for epoch in range(1, epochs + 1):
    t0 = time.time()

    optimizer.zero_grad()

    z = model.encode(train_data.x, train_data.edge_index)

    # ---- edge reconstruction loss (scalar, for training)
    edge_loss = model.recon_loss(z, train_data.edge_index)

    # ---- per-edge loss (for attribution)
    src, dst = train_data.edge_index
    logits = (z[src] * z[dst]).sum(dim=1)

    per_edge_loss = F.binary_cross_entropy_with_logits(
        logits,
        torch.ones_like(logits),
        reduction="none"
    )

    # ---- node feature loss (optional, same as before)
    x_recon = decoder(z)
    feature_loss = ((x_recon - train_data.x) ** 2).mean()

    loss = edge_loss + feature_weight * feature_loss
    loss.backward()

    # ---- first-order In-Run edge Shapley
    with torch.no_grad():
        grads = []
        for p in model.parameters():
            if p.grad is not None:
                grads.append(p.grad.flatten())
        g = torch.cat(grads)
        g_norm = g.norm()

        edge_shapley += per_edge_loss * g_norm

    optimizer.step()

    print(
        f"Epoch {epoch:03d} | "
        f"Loss {loss.item():.4f} | "
        f"Time {time.time() - t0:.2f}s"
    )

# --------------------------------------------------
# Save
# --------------------------------------------------
edge_shapley = edge_shapley.cpu().numpy()

record = {
    "edge_index": train_data.edge_index.cpu().numpy(),
    "First-order In-Run Edge Shapley": edge_shapley.tolist()
}

with open("graph_edge_shapley.value", "wb") as f:
    pickle.dump(record, f)

print("Saved Shapley values to graph_edge_shapley.value")
