# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv


# %%

torch.manual_seed(42)
device = torch.device('cpu')
# %%
#load dataset
dataset = Planetoid(root='./data', name='Cora')
data = dataset[0]

print(data)
# %%

split = T.RandomLinkSplit( num_val=0.1, num_test=0.1, is_undirected=True, add_negative_train_samples=True, neg_sampling_ratio=1.0)
train_data, val_data, test_data = split(data)
train_data, val_data, test_data = train_data.to(device), val_data.to(device), test_data.to(device)
x_all = data.x.to(device)

# %%

class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.2):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# %%   
class MLPDecoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, z, edge_label_index):
        src, dst = edge_label_index
        # Concatenate embeddings for each edge (instead of dot product)
        pair = torch.cat([z[src], z[dst]], dim=-1)
        h = F.relu(self.fc1(pair))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.fc2(h).squeeze(-1)
# %%

model = GCNEncoder( in_channels=dataset.num_node_features, hidden_channels=128, out_channels=64 ).to(device)
decoder = MLPDecoder(embed_dim=64).to(device)
optimizer = torch.optim.Adam( list(model.parameters()) + list(decoder.parameters()), lr=0.01, weight_decay=5e-4 )
criterion = nn.BCEWithLogitsLoss()
# %%   
def train_one_epoch():
    model.train(); decoder.train()
    optimizer.zero_grad()

    z = model(x_all, train_data.edge_index)
    logits = decoder(z, train_data.edge_label_index)
    loss = criterion(logits, train_data.edge_label.float())

    loss.backward()
    optimizer.step()
    return loss.item()
def torch_auc(y_true, y_pred):
    pos = y_pred[y_true == 1]
    neg = y_pred[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return torch.tensor(float('nan'))
    total = len(pos) * len(neg)
    correct = (pos.view(-1, 1) > neg.view(1, -1)).sum().float()
    ties = (pos.view(-1, 1) == neg.view(1, -1)).sum().float() * 0.5
    auc = (correct + ties) / total
    return auc.item()

def evaluate(split):
    model.eval()
    decoder.eval()
    z = model(x_all, train_data.edge_index)
    logits = decoder(z, split.edge_label_index)
    probs = torch.sigmoid(logits)
    y_true = split.edge_label
    auc = torch_auc(y_true, probs)
    return auc
# %%   
best_val, best_test = 0.0, 0.0
for epoch in range(1, 101):
    loss = train_one_epoch()
    val_auc = evaluate(val_data)
    test_auc = evaluate(test_data)

    if val_auc > best_val:
        best_val, best_test = val_auc, test_auc

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d}, loss {loss:.4f}, val AUC {val_auc:.4f}, test AUC {test_auc:.4f}")
# %%   

print(f"Best Val AUC={best_val:.4f}, Test AUC={best_test:.4f}")
