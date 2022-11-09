import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import h5py


class WDMCEmbDataset(Dataset):

    def __init__(self, file_path, split, cat_embedding_key):
        assert split in ["test", "train", "val"]
        self.cat_embeddings = []
        with h5py.File(file_path, "r") as file_ref:
            split_data = file_ref[split][cat_embedding_key]
            for key in split_data.keys():
                self.cat_embeddings.append(split_data[key][()])
        self.cat_embeddings = np.concatenate(self.cat_embeddings)

    def __len__(self):
        return len(self.cat_embeddings)
    
    def input_size(self):
        return self.cat_embeddings.shape[1]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.cat_embeddings[idx, :]
        return sample