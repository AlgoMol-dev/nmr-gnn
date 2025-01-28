import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool
from torch.nn import Sequential, Linear, ReLU, Embedding

class NMRGNN(torch.nn.Module):
    def __init__(self, 
                 num_node_features,   # size of x (excluding eq_label)
                 num_eq_classes=None, # maximum equivalence label ID + 1
                 hidden_dim=64,
                 num_layers=3):
        super().__init__()
        
        # Optional embedding for equivalence labels
        # If you have up to, say, 200 different eq groups in your entire dataset,
        # set num_eq_classes=200 or higher. This is just a ballpark example.
        if num_eq_classes is not None:
            self.eq_embedding = Embedding(num_eq_classes, embedding_dim=8)
        else:
            self.eq_embedding = None
        
        # GIN layers
        self.convs = torch.nn.ModuleList()
        self.bns   = torch.nn.ModuleList()
        
        # Input dimension includes the eq_label embedding dimension if used
        in_dim = num_node_features
        if self.eq_embedding is not None:
            in_dim += 8  # embedding_dim from eq_embedding
        
        for i in range(num_layers):
            mlp = Sequential(
                Linear(in_dim if i == 0 else hidden_dim, hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim),
                ReLU()
            )
            conv = GINConv(mlp, train_eps=True)
            self.convs.append(conv)
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))
        
        self.final_lin = Linear(hidden_dim, 1)  # predict 1 shift value per node

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # If we have eq_label embeddings, concatenate them to x
        if self.eq_embedding is not None:
            eq_emb = self.eq_embedding(data.eq_labels)
            x = torch.cat([x, eq_emb], dim=-1)
        
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
        
        # We want per-node predictions for chemical shifts
        # So no global pooling needed if you want SHIFT per atom
        out = self.final_lin(x).squeeze(-1)  # shape [num_nodes]

        return out