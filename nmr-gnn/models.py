# mag_eq_nmr/models.py

import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, Embedding
from torch_geometric.nn import GINConv

class NMRGNN(torch.nn.Module):
    def __init__(self,
                 num_node_features=2,  # e.g. atomic_num & degree
                 num_eq_classes=None,   # max equivalence label ID + 1
                 hidden_dim=64,
                 num_layers=3,
                 eq_emb_dim=8):
        super().__init__()

        # Optional embedding for eq_labels
        if num_eq_classes is not None:
            self.eq_embedding = Embedding(num_eq_classes, eq_emb_dim)
        else:
            self.eq_embedding = None

        # GINConv layers
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        # Input dimension includes eq_label embedding if used
        in_dim = num_node_features
        if self.eq_embedding is not None:
            in_dim += eq_emb_dim

        for i in range(num_layers):
            mlp = Sequential(
                Linear(in_dim if i == 0 else hidden_dim, hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim),
                ReLU()
            )
            conv = GINConv(mlp, train_eps=True)
            self.convs.append(conv)
            self.bns.append(BatchNorm1d(hidden_dim))

        self.final_lin = Linear(hidden_dim, 1)  # 1 value per atom (chemical shift)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # If eq_label embedding is used, concatenate to x
        if self.eq_embedding is not None:
            eq_emb = self.eq_embedding(data.eq_labels)
            x = torch.cat([x, eq_emb], dim=-1)

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)

        out = self.final_lin(x).squeeze(-1)  # shape [num_nodes]
        return out