# gnn/data_preprocessing.py

import os
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
from torch_geometric.data import Data
from .nmr_equivalence import assign_equivalence_groups  # your custom function

def load_molecules_from_sdf(sdf_path):
    """
    Loads RDKit molecule objects from an SDF file.
    Returns a list of RDKit Mol objects.
    """
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
    mols = [mol for mol in suppl if mol is not None]
    return mols

def load_molecules_from_smiles(smiles_file):
    """
    Loads RDKit molecule objects from a text file containing one SMILES per line.
    Returns a list of RDKit Mol objects.
    """
    mols = []
    with open(smiles_file, 'r') as f:
        for line in f:
            smi = line.strip()
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                # Generate conformation if needed (for 3D-based equivalences)
                AllChem.EmbedMolecule(mol, AllChem.ETKDG())
                mols.append(mol)
    return mols

def create_pyg_data_from_mol(mol, shifts=None):
    """
    Convert a single RDKit Mol to a PyG Data object.
    'shifts' is a list/array of length num_atoms with the NMR chemical shifts.
    If shifts are not available, set them to 0 or None.
    """
    if shifts is None:
        # Put dummy values if no real shifts available
        shifts = [0.0]*mol.GetNumAtoms()

    # 1) Node features (for demonstration, we use atomic number, degree, etc.)
    atomic_nums = []
    degrees = []
    for atom in mol.GetAtoms():
        atomic_nums.append(atom.GetAtomicNum())
        degrees.append(atom.GetDegree())

    x = torch.tensor([atomic_nums, degrees], dtype=torch.float).t()  # shape [num_atoms, 2]

    # 2) Edges: collect bond indices
    edge_index_list = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index_list.append([i, j])
        edge_index_list.append([j, i])
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()

    # 3) Assign equivalence groups
    eq_labels = assign_equivalence_groups(mol)  # shape [num_atoms], integer IDs
    eq_labels_tensor = torch.tensor(eq_labels, dtype=torch.long)

    # 4) Y targets
    y = torch.tensor(shifts, dtype=torch.float)  # shape [num_atoms]

    data = Data(x=x, edge_index=edge_index, y=y)
    data.eq_labels = eq_labels_tensor

    return data

def preprocess_dataset(sdf_or_smiles_path, from_sdf=True, shift_dict=None):
    """
    - from_sdf: bool, True if input is SDF, False if it's a SMILES file.
    - shift_dict: optional dictionary {mol_id: [list_of_shifts_per_atom]}.
                  If None, we'll just assign zero shifts.
    Returns a list of PyG Data objects.
    """
    if from_sdf:
        mols = load_molecules_from_sdf(sdf_or_smiles_path)
    else:
        mols = load_molecules_from_smiles(sdf_or_smiles_path)

    dataset = []
    for idx, mol in enumerate(mols):
        # If shift_dict is provided, get the shift array for this molecule's ID
        # Adjust logic as needed for how you identify each molecule
        if shift_dict is not None and idx in shift_dict:
            shifts = shift_dict[idx]
        else:
            shifts = None

        data = create_pyg_data_from_mol(mol, shifts)
        dataset.append(data)

    return dataset