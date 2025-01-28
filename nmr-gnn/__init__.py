# mag_eq_nmr/__init__.py

"""
MagEquivNMR: Magnetic-Equivalence–Aware GNN for NMR Shift Prediction
Developed by AlgoMol
"""

__version__ = "0.1.0"

# Re-export modules so they can be imported directly from mag_eq_nmr
__all__ = [
    "data_preprocessing",
    "dataset",
    "nmr_equivalence",
    "models",
    "train",
    "evaluate",
    "utils",
]

from . import (
    data_preprocessing,
    dataset,
    nmr_equivalence,
    models,
    train,
    evaluate,
    utils,
)