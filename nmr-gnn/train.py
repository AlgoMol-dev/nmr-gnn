# mag_eq_nmr/train.py

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from .models import NMRGNN
from .utils import equivalence_variance_loss
from .dataset import NMRDataset

def train_model(root, sdf_or_smiles_path, from_sdf=True,
                shift_dict=None, num_epochs=50, batch_size=4,
                lr=1e-3, alpha=0.01):
    """
    Train loop demonstration.
    root: directory for processed data
    sdf_or_smiles_path: file path for SDF or SMILES
    shift_dict: dict for shifts if needed
    alpha: weight for equivalence penalty
    """
    # 1) Create dataset
    dataset = NMRDataset(root, sdf_or_smiles_path, from_sdf=from_sdf, shift_dict=shift_dict)
    # Optionally split dataset into train / val
    train_dataset = dataset[:int(0.8*len(dataset))]
    val_dataset = dataset[int(0.8*len(dataset)):]

    # 2) Determine how many eq classes (optional)
    # e.g., quick hack: find max eq_label among all data
    all_eq_labels = []
    for d in dataset:
        all_eq_labels.extend(d.eq_labels.tolist())
    num_eq_classes = max(all_eq_labels) + 1 if all_eq_labels else None

    # 3) Create model
    model = NMRGNN(num_node_features=2,  # we used 2 features in data_preprocessing
                   num_eq_classes=num_eq_classes,
                   hidden_dim=64,
                   num_layers=3,
                   eq_emb_dim=8)
    print(model)

    # 4) Create loaders, optimizer
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 5) Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch_data in train_loader:
            # A 'batch_data' object merges multiple graphs
            pred_shifts = model(batch_data)
            # standard MSE
            mse_loss = F.mse_loss(pred_shifts, batch_data.y)
            # eq variance penalty
            eq_penalty = equivalence_variance_loss(pred_shifts, batch_data.eq_labels, alpha=alpha)
            loss = mse_loss + eq_penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_data.num_graphs

        avg_train_loss = total_loss / len(train_loader.dataset)

        # Validation
        val_loss = evaluate_model(model, val_loader, alpha=alpha)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Return the trained model
    return model

def evaluate_model(model, data_loader, alpha=0.01):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch_data in data_loader:
            pred_shifts = model(batch_data)
            mse_loss = F.mse_loss(pred_shifts, batch_data.y)
            eq_penalty = equivalence_variance_loss(pred_shifts, batch_data.eq_labels, alpha=alpha)
            loss = mse_loss + eq_penalty
            total_loss += loss.item() * batch_data.num_graphs

    return total_loss / len(data_loader.dataset)

if __name__ == "__main__":
    # Example usage: python -m mag_eq_nmr.train
    model = train_model(root='./data',
                        sdf_or_smiles_path='./data/molecules.sdf',
                        from_sdf=True,
                        shift_dict=None,
                        num_epochs=10,
                        batch_size=2,
                        lr=1e-3,
                        alpha=0.01)
    torch.save(model.state_dict(), 'model_checkpoint.pt')