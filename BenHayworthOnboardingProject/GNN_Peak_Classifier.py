import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.feature_extraction.image import grid_to_graph
from scipy import sparse
import torch
from torch_geometric.utils import from_scipy_sparse_matrix
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

def generate_mountain_peaks(grid_size: int, num_peaks: int) -> tuple[np.ndarray,list[int]]:

    """
    Generate 2D mountain data 
    Args:
        grid_size (int): Number of nodes per dimension
        num_peaks (int): Total number of peaks generated
    Returns:
        tuple[np.ndarray,list[int]]:
        - terrain_map: numpy array of terrain
        - labels: list of labels
    """

    coords = np.linspace(0, 1, grid_size, endpoint=False)
    terrain_map = np.zeros((grid_size, grid_size))
    grid_x, grid_y = np.meshgrid(coords, coords)
    labels = []

    for i in range(num_peaks):
        A = np.random.uniform(10, 50)
        x0, y0 = (np.random.uniform(0, 1), np.random.uniform(0, 1))
        x0_coord = math.floor(x0 / (1 / grid_size))
        y0_coord = math.floor(y0 / (1 / grid_size))
        labels.append(y0_coord * grid_size + x0_coord)
        squared_distances = (grid_x - (x0_coord / grid_size))**2 + (grid_y - (y0_coord / grid_size))**2
        terrain_map += math.sqrt(A) * np.exp(-A * squared_distances)
    #terrain_map = (terrain_map - terrain_map.mean()) / terrain_map.std()
    return terrain_map, labels


def construct_graph_features(grid: np.ndarray, label_coords, num_peaks, num_nodes):
    labels = torch.zeros(num_nodes, dtype=torch.float32, device=device)
    for coord in label_coords:
        labels[coord] = 1.0

    """
    Construct important features regarding graphs:
    - Torch tensor of peak node labels
    - Humidity node grid
    - Edge indices
    - Edge Weights
    """

    h, w = grid.shape
    n = h * w
    undirected_adjacency = grid_to_graph(n_x=w, n_y=h)
    source_indices, destination_indices = undirected_adjacency.nonzero()

    grid_flat = grid.flatten()
    height_diffs = grid_flat[source_indices] - grid_flat[destination_indices]

    edge_weights_np = np.exp(-np.abs(height_diffs))
    sparse_adjacency = sparse.csr_matrix((edge_weights_np, (source_indices, destination_indices)), shape=(n, n))

    noise = np.random.normal(loc=0.0, scale=1.0, size=(h, w))
    humidity_grid = 0.3 * grid + noise
    humidity_grid = (humidity_grid - humidity_grid.mean()) / (humidity_grid.std() + 1e-8)

    torch_adjacency_matrix = from_scipy_sparse_matrix(sparse_adjacency)
    edge_matrix = torch_adjacency_matrix[0].to(device)
    edge_weights = torch_adjacency_matrix[1].to(device).to(torch.float32)

    humidity_grid_flattened = torch.tensor(humidity_grid.flatten(), dtype=torch.float32, device=device)
    return labels, humidity_grid_flattened, edge_matrix, edge_weights


class GCN(torch.nn.Module):

    """
    3-Layer Graph Convolutional Network
    """

    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.leaky_relu(x, 0.1)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = F.leaky_relu(x, 0.1)
        x = self.conv3(x, edge_index, edge_weight=edge_weight)
        return x


def make_batch(num_nodes, batch_size):
    """
    Generates batch of data
    """
    indices = torch.arange(num_nodes)
    indices = indices[torch.randperm(num_nodes)]
    return [indices[i:i + batch_size] for i in range(0, num_nodes, batch_size)]


if __name__ == "__main__":
    epochs = 400
    grid_size = 80 #total nodes will be grid_size x grid_size
    num_peaks = 2000
    learning_rate = 0.001
    weight_d = 5e-3
    threshold = 0.3
    hidden_feature_dim = 5
    
    num_nodes = grid_size ** 2
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    terrain_map, label_coords = generate_mountain_peaks(grid_size, num_peaks)
    labels, humidity_grid_flattened, edge_matrix, edge_weights = construct_graph_features(
        terrain_map, label_coords, num_peaks, num_nodes)

    model = GCN(in_channels=1, hidden_channels=hidden_feature_dim, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_d)
    criterion = torch.nn.BCEWithLogitsLoss()

    losses, accuracies, precision_list, recall_list = [], [], [], []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        feature_nodes = humidity_grid_flattened.reshape(num_nodes, 1)
        outputs = model(feature_nodes, edge_matrix, edge_weights)

        loss = criterion(outputs.reshape(num_nodes), labels)
        loss.backward()
        optimizer.step()

        probs = torch.sigmoid(outputs)
        preds = (probs > 0.3).float()

        with torch.no_grad():
            acc = (preds.view(-1) == labels).float().mean().item()
            tp = ((preds == 1) & (labels == 1)).sum().item()
            tn = ((preds == 0) & (labels == 0)).sum().item()
            fp = ((preds == 1) & (labels == 0)).sum().item()
            fn = ((preds == 0) & (labels == 1)).sum().item()

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)

        losses.append(loss.item())
        accuracies.append(acc)
        precision_list.append(precision)
        recall_list.append(recall)

        print(f"Epoch {epoch+1:03d} | Loss: {loss.item():.4f} | Acc: {acc:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Loss', linewidth=2)
    plt.plot(accuracies, label='Accuracy', linewidth=2)
    plt.plot(precision_list, label='Precision', linestyle='--')
    plt.plot(recall_list, label='Recall', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.title('GCN Training Performance')
    plt.legend()
    plt.grid(True)
    plt.show()

    test_terrain, test_labels = generate_mountain_peaks(grid_size, num_peaks)
    labels, humidity_grid_flattened, edge_matrix, edge_weights = construct_graph_features(
        test_terrain, test_labels, num_peaks, num_nodes)

    feature_nodes = humidity_grid_flattened.reshape(num_nodes, 1)
    outputs = model(feature_nodes, edge_matrix, edge_weights)

    preds = (torch.sigmoid(outputs) > threshold).float()

    with torch.no_grad():
        acc_test = (preds.view(-1) == labels).float().mean().item()
        tp = ((preds == 1) & (labels == 1)).sum().item()
        tn = ((preds == 0) & (labels == 0)).sum().item()
        fp = ((preds == 1) & (labels == 0)).sum().item()
        fn = ((preds == 0) & (labels == 1)).sum().item()

        precision_test = tp / (tp + fp + 1e-8)
        recall_test = tp / (tp + fn + 1e-8)

    print(f"\n✅ Test Precision: {precision_test:.4f}")
    print(f"✅ Test Recall: {recall_test:.4f}")
