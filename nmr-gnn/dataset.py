# gnn/dataset.py

import torch
from torch_geometric.data import InMemoryDataset, Data
from .data_preprocessing import preprocess_dataset

class NMRDataset(InMemoryDataset):
    """
    A simple InMemoryDataset for NMR data using the data_preprocessing pipeline.
    """
    def __init__(self, root, sdf_or_smiles_path, from_sdf=True, shift_dict=None,
                 transform=None, pre_transform=None):
        self.sdf_or_smiles_path = sdf_or_smiles_path
        self.from_sdf = from_sdf
        self.shift_dict = shift_dict
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # We can just treat the input as a single "raw" file
        return [self.sdf_or_smiles_path]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Not implemented because we expect local SDF/SMILES
        pass

    def process(self):
        # Here we do our data preprocessing
        dataset = preprocess_dataset(self.sdf_or_smiles_path,
                                     from_sdf=self.from_sdf,
                                     shift_dict=self.shift_dict)

        if self.pre_transform is not None:
            dataset = [self.pre_transform(d) for d in dataset]

        data, slices = self.collate(dataset)
        torch.save((data, slices), self.processed_paths[0])

    def __len__(self):
        return self.data.y.size(0) if self.data is not None else 0