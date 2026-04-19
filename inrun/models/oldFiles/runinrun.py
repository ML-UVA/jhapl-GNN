import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.graph_autoencoder import model
from shared.config import TrainingConfig, parse_arguments
from trainer import Trainer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


NUM_NODES = model.num_nodes


def get_node_batch(split, batch_size, return_idx=False):
    idx = torch.randint(0, NUM_NODES, (batch_size,), device=device)
    X = idx
    Y = torch.zeros_like(idx)
    if return_idx:
        return X, Y, idx
    return X, Y


def get_val_node_batch(batch_size, return_idx=False):
    idx = torch.randint(0, NUM_NODES, (batch_size,), device=device)
    X = idx
    Y = torch.zeros_like(idx)
    if return_idx:
        return X, Y, idx
    return X, Y


class GraphWrapper(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, input_ids, labels=None):
        loss = self.base_model.compute_loss(input_ids)
        return type('Out', (), {'loss': loss, 'logits': None})


wrapped_model = GraphWrapper(model).to(device)

optimizer = torch.optim.Adam(wrapped_model.parameters(), lr=3e-4)
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

args = parse_arguments()
config = TrainingConfig(args)

ddp_info = {
    'ddp': False,
    'rank': 0,
    'world_size': 1,
    'device': device,
    'master_process': True
}

ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if torch.cuda.is_available() else torch.no_grad()

trainer = Trainer(
    model=wrapped_model,
    optimizer=optimizer,
    scaler=scaler,
    config=config,
    ddp_info=ddp_info,
    get_batch_fn=get_node_batch,
    get_val_batch_fn=get_val_node_batch,
    ctx=ctx
)

trainer.run_training()
