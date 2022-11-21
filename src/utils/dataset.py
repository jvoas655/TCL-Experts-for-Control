import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, GPT2Tokenizer
from transformers import T5Tokenizer
from tqdm import tqdm
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

class WDMCEncDataset(Dataset):

    def __init__(self, file_path, split, target_enc_key, model_name, token_count, batch_size):
        assert split in ["test", "train", "val"]
        self.encodings = []
        self.anit_encodings = []
        c = 0
        with h5py.File(file_path, "r") as file_ref:
            split_data = file_ref[split][target_enc_key]
            anti_split_data = file_ref[split][target_enc_key + "_anti"]
            for key in tqdm(split_data.keys(), total = len(split_data.keys()), desc=f"Loading {split} encodings"):
                self.encodings.append(split_data[key][()].reshape(1, -1))
                self.anit_encodings.append(anti_split_data[key][()].reshape(1, -1))
                if (c > 10000):
                    break
                c+=1
        self.encodings = np.concatenate(self.encodings)
        print(np.mean(self.encodings, axis = 0).tolist())
        self.anit_encodings = np.concatenate(self.anit_encodings)

        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)

        self.summaries = []
        c = 0
        with h5py.File(file_path, "r") as file_ref:
            split_data = file_ref[split]["text"]
            for key in tqdm(split_data.keys(), total = len(split_data.keys()), desc=f"Loading {split} text"):
                self.summaries.append(split_data[key][()].decode("utf-8"))
                if (c > 10000):
                    break
                c+=1
        self.tokens = []
        for i in tqdm(range(0, len(self.summaries), batch_size), desc = f"Tokenizing {split} encodings", total = len(self.summaries) // batch_size):
            #print(self.tokenizer(self.summaries[i:i+batch_size])["input_ids"])
            self.tokens.append(self.tokenizer(self.summaries[i:i+batch_size], return_tensors="pt", truncation=True, max_length=token_count, padding=True))
        input_ids = torch.cat([ind["input_ids"] for ind in self.tokens])
        attention_mask = torch.cat([ind["attention_mask"] for ind in self.tokens])
        self.tokens = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        

    def __len__(self):
        return len(self.encodings)
    
    def output_size(self):
        return self.encodings.shape[1]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        tokens = {
            "input_ids": self.tokens["input_ids"][idx, :],
            "attention_mask": self.tokens["attention_mask"][idx, :],
        }
        sample = {
            "encodings": self.encodings[idx, :],
            "anti_encodings": self.anit_encodings[idx, :],
            "tokens": tokens
        }
        return sample

class WDMCExpLMDataset(Dataset):

    def __init__(self, file_path, split, target_enc_key, max_token_count, batch_size):
        assert split in ["test", "train", "val"]
        self.encodings = []
        lim = None
        c = 0
        with h5py.File(file_path, "r") as file_ref:
            split_data = file_ref[split][target_enc_key]
            for key in tqdm(split_data.keys(), total = len(split_data.keys()), desc=f"Loading {split} encodings"):
                self.encodings.append(split_data[key][()].reshape(1, -1))
                if (lim is not None):
                    c += 1
                    if (c == lim):
                        break
        self.encodings = np.concatenate(self.encodings)

        self.tokenizer = None
        self.max_token_count = max_token_count

        self.summaries = []
        c = 0
        with h5py.File(file_path, "r") as file_ref:
            split_data = file_ref[split]["text"]
            for key in tqdm(split_data.keys(), total = len(split_data.keys()), desc=f"Loading {split} text"):
                self.summaries.append(split_data[key][()].decode("utf-8"))
                if (lim is not None):
                    c += 1
                    if (c == lim):
                        break
        

    def __len__(self):
        return len(self.tokens)
    
    # TODO: probably not needed
    # def output_size(self):
    #     return self.encodings.shape[1]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        elif (type(idx) == int):
            idx = [idx]
        summaries = [self.summaries[i] for i in idx]
        assert self.tokenizer is not None
        tokens = self.tokenizer(summaries, return_tensors="pt", truncation=True, max_length=self.max_token_count, padding="max_length")
        sample = {
            # "encodings": self.encodings[idx, :],
            "tokens": tokens
        }
        return sample