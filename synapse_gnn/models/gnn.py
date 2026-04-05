import torch
from torch_geometric.nn import GraphConv

class SynapsePredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GraphConv(in_channels, hidden_channels, aggr='mean')
        self.conv2 = GraphConv(hidden_channels, hidden_channels, aggr='mean')
        
        # NEW: A true Neural Network Decoder that can accept explicit Edge Features
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * 2 + 1, 64), # src (128) + dst (128) + weight (1)
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1) # Output single logit
        )

    def encode(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight=edge_weight).relu()
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        return x

    def decode(self, z, edge_label_index, explicit_weight=None):
        src_embeddings = z[edge_label_index[0]]
        dst_embeddings = z[edge_label_index[1]]
        
        # If Euclidean, default to 1.0 so the model relies purely on embeddings
        if explicit_weight is None:
            explicit_weight = torch.ones(src_embeddings.size(0), 1, device=z.device)
        else:
            explicit_weight = explicit_weight.view(-1, 1) # Ensure it's a 2D column vector
            
        # Glue the source, destination, and the physical surface area together!
        h = torch.cat([src_embeddings, dst_embeddings, explicit_weight], dim=-1)
        
        # Pass through the linear layers
        return self.decoder(h).squeeze(-1)