# mag_eq_nmr/utils.py

import torch
import torch.nn.functional as F

def equivalence_variance_loss(pred_shifts, eq_labels, alpha=0.01):
    """
    Adds an extra penalty if nodes in the same eq_label group have different predicted shifts.
    alpha: weight factor for this penalty.
    """
    unique_labels = torch.unique(eq_labels)
    penalty = torch.tensor(0.0, device=pred_shifts.device)
    for lbl in unique_labels:
        mask = (eq_labels == lbl)
        group_preds = pred_shifts[mask]
        if len(group_preds) > 1:
            variance = torch.var(group_preds, unbiased=False)
            penalty += variance
    return alpha * penalty