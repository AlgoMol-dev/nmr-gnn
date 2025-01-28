# mag_eq_nmr/evaluate.py

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from .models import NMRGNN
from .dataset import NMRDataset
from .utils import equivalence_variance_loss

def evaluate_saved_model(checkpoint_path, root, sdf_or_smiles_path, from_sdf=True, alpha=0.01):
    # 1) Load dataset
    dataset = NMRDataset(root, sdf_or_smiles_path, from_sdf=from_sdf)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    # 2) Figure out num_eq_classes
    all_eq_labels = []
    for d in dataset:
        all_eq_labels.extend(d.eq_labels.tolist())
    num_eq_classes = max(all_eq_labels) + 1 if all_eq_labels else None

    # 3) Create model
    model = NMRGNN(num_node_features=2, num_eq_classes=num_eq_classes)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    # 4) Evaluate
    total_mse = 0.0
    total_eq_penalty = 0.0
    num_graphs = len(loader.dataset)

    with torch.no_grad():
        for batch_data in loader:
            pred_shifts = model(batch_data)
            mse = F.mse_loss(pred_shifts, batch_data.y, reduction='sum').item()
            eq_pen = equivalence_variance_loss(pred_shifts, batch_data.eq_labels, alpha=1.0).item()  # alpha=1 to see raw penalty
            total_mse += mse
            total_eq_penalty += eq_pen
    
    avg_mse = total_mse / num_graphs
    avg_eq_penalty = total_eq_penalty / num_graphs
    print(f"Test MSE: {avg_mse:.4f}")
    print(f"Avg Equivalence Penalty: {avg_eq_penalty:.4f} (unweighted)")

if __name__ == "__main__":
    evaluate_saved_model('model_checkpoint.pt', root='./data', sdf_or_smiles_path='./data/molecules.sdf')