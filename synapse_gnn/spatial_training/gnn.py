import torch
from torch_geometric.nn import GraphConv

class SynapsePredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        # Encoder: Swapped SAGEConv for GraphConv!
        # GraphConv natively and explicitly supports continuous edge weights.
        self.conv1 = GraphConv(in_channels, hidden_channels, aggr='mean')
        self.conv2 = GraphConv(hidden_channels, hidden_channels, aggr='mean')

    def encode(self, x, edge_index, edge_weight=None):
        # Forward pass through the graph layers
        # Explicitly passing edge_weight as a keyword argument guarantees no positional crashes
        x = self.conv1(x, edge_index, edge_weight=edge_weight).relu()
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        return x

    def decode(self, z, edge_label_index):
            # Decoder: Dot product similarity
            src_embeddings = z[edge_label_index[0]]
            dst_embeddings = z[edge_label_index[1]]
            
            # Calculate raw dot product
            raw_dot_product = (src_embeddings * dst_embeddings).sum(dim=-1)
            
            # Scale the logits by the square root of the hidden dimension (sqrt of 128)
            # This prevents the sum of 128 dimensions from exploding the final sigmoid!
            scale_factor = z.size(-1) ** 0.5
            
            return raw_dot_product / scale_factor