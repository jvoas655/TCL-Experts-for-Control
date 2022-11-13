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
        with h5py.File(file_path, "r") as file_ref:
            split_data = file_ref[split][target_enc_key]
            for key in tqdm(split_data.keys(), total = len(split_data.keys()), desc=f"Loading {split} encodings"):
                self.encodings.append(split_data[key][()].reshape(1, -1))
        self.encodings = np.concatenate(self.encodings)

        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)

        self.summaries = []
        with h5py.File(file_path, "r") as file_ref:
            split_data = file_ref[split]["text"]
            for key in tqdm(split_data.keys(), total = len(split_data.keys()), desc=f"Loading {split} text"):
                self.summaries.append(split_data[key][()].decode("utf-8"))
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
            "tokens": tokens
        }
        return sample

class WDMCGPTEncDataset(Dataset):

    def __init__(self, file_path, split, target_enc_key, model_name, token_count, batch_size):
        assert split in ["test", "train", "val"]
        self.encodings = []
        with h5py.File(file_path, "r") as file_ref:
            split_data = file_ref[split][target_enc_key]
            for key in tqdm(split_data.keys(), total = len(split_data.keys()), desc=f"Loading {split} encodings"):
                self.encodings.append(split_data[key][()].reshape(1, -1))
        self.encodings = np.concatenate(self.encodings)

        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token # TODO: verify this is the desired padding
        self.sentences = []
        with h5py.File(file_path, "r") as file_ref:
            split_data = file_ref[split]["text"]
            for key in tqdm(split_data.keys(), total = len(split_data.keys()), desc=f"Loading {split} text"):
                summary = split_data[key][()].decode("utf-8")
                first_sentence = summary.split(".")[0] + "."
                self.sentences.append(first_sentence)
        
        self.tokens = []
        for i in tqdm(range(0, len(self.sentences), batch_size), desc = f"Tokenizing {split} encodings", total = len(self.sentences) // batch_size):
            self.tokens.append(self.tokenizer(self.sentences[i:i+batch_size], return_tensors="pt", truncation=True, padding=True))
        # TODOL issue, each batch has a different size of 64,x
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

    # TODO: is __getitem__ for GPT as roberta
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        tokens = {
            "input_ids": self.tokens["input_ids"][idx, :],
            "attention_mask": self.tokens["attention_mask"][idx, :],
        }
        # TODO: why sample?
        sample = {
            "encodings": self.encodings[idx, :],
            "tokens": tokens
        }
        return sample