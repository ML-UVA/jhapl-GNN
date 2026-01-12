import torch
from torch_geometric.nn import SAGEConv

class SynapsePredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        # Encoder: GraphSAGE layers
        # Aggregates information from neighbor neurons to build an embedding
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

    def encode(self, x, edge_index):
        # Forward pass through the graph layers
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        # Decoder: Dot product similarity
        # Predicts the probability of a link based on the similarity of embeddings
        # z: node embeddings [num_nodes, hidden_channels]
        # edge_label_index: pairs of nodes to predict [2, num_pairs]
        
        src_embeddings = z[edge_label_index[0]]
        dst_embeddings = z[edge_label_index[1]]
        
        # Calculate dot product
        return (src_embeddings * dst_embeddings).sum(dim=-1)