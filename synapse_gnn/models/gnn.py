import torch
from torch_geometric.nn import GraphConv

class SynapsePredictor(torch.nn.Module):
    """
    A Graph Neural Network architecture for connectomics edge classification.
    
    This model utilizes two layers of isotropic GraphConv for neighborhood aggregation 
    to learn universal biological morphological rules. It features a custom MLP decoder 
    that conditionally fuses continuous physical touch data (ADP) with the learned 
    node embeddings to differentiate true synapses from structural false positives.

    Args:
        in_channels (int): The number of input biological features (e.g., Axon Length, Soma Vol).
        hidden_channels (int): The dimension of the hidden representation layers.
        use_edge_weights (bool): Toggle to determine if continuous physical weights should 
                                 be ingested during message passing and final decoding.
    """
    def __init__(self, in_channels, hidden_channels, use_edge_weights=True):
        super().__init__()
        self.use_edge_weights = use_edge_weights # The separate parameter
        self.conv1 = GraphConv(in_channels, hidden_channels, aggr='mean')
        self.conv2 = GraphConv(hidden_channels, hidden_channels, aggr='mean')
        
        # We keep the +1 dimension in the decoder so the architecture remains 
        # stable for comparisons; we just feed it "1.0" if weights are disabled.
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * 2 + 1, 64), 
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )

    def encode(self, x, edge_index, edge_weight=None):
        """
        Generates rich contextual embeddings by aggregating morphological features 
        from the local spatial neighborhood.
        """
        # Discard the ingested weights if the toggle is false
        weights = edge_weight if self.use_edge_weights else None
        
        x = self.conv1(x, edge_index, edge_weight=weights).relu()
        x = self.conv2(x, edge_index, edge_weight=weights)
        return x

    def decode(self, z, edge_label_index, explicit_weight=None):
        """
        Evaluates candidate connections using a Multi-Layer Perceptron (MLP).
        
        Bypasses the standard dot-product decoder to explicitly fuse ADP weights 
        with the learned morphological embeddings right before the final prediction step.
        
        If 'use_edge_weights' is False, the model safely defaults to evaluating 
        a purely structural proximity baseline by applying a constant weight of 1.0.
        """
        src_embeddings = z[edge_label_index[0]]
        dst_embeddings = z[edge_label_index[1]]
        
        # Discard physical weights and use '1.0' if the toggle is false
        weights = explicit_weight if self.use_edge_weights else None
        
        if weights is None:
            weights = torch.ones(src_embeddings.size(0), 1, device=z.device)
        else:
            weights = weights.view(-1, 1)
            
        h = torch.cat([src_embeddings, dst_embeddings, weights], dim=-1)
        return self.decoder(h).squeeze(-1)