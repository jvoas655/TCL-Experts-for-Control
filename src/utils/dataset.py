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
from scipy.spatial import KDTree
from multiprocessing import Pool


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

    def __init__(self, file_path, split, target_enc_key, max_token_count, batch_size, lim=None):
        assert split in ["test", "train", "val"]
        self.encodings = []
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
        print("Building KDTree of Encodings")
        self.enc_tree = KDTree(self.encodings)

        self.tokenizer = None
        self.anti_encodings = None
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
        self.categories = []
        c = 0
        with h5py.File(file_path, "r") as file_ref:
            split_data = file_ref[split]["categories"]
            for key in tqdm(split_data.keys(), total = len(split_data.keys()), desc=f"Loading {split} text"):
                self.categories.append("; ".join(list(map(lambda c: c.decode("utf-8"), split_data[key][()]))))
                if (lim is not None):
                    c += 1
                    if (c == lim):
                        break
    def get_near_samples(self, sample_encodings):
        _, idxs = self.enc_tree.query(sample_encodings)
        idxs = idxs.flatten().tolist()
        samples = []
        for idx in idxs:
            samples.append(self.__getitem__(idx))
        return samples
    def preprocess_tokens(self, tokenizer):
        self.tokenizer = tokenizer
        self.tokens = self.tokenizer(self.summaries, return_tensors="pt", truncation=True, max_length=self.max_token_count, padding="max_length")
        self.preprompt_tokens = self.tokenizer(self.categories, return_tensors="pt", truncation=True, max_length=16, padding="max_length")
    def batch_sample_som(self, args):
        encodings, som, cluster_map = args
        return som.sample_anti_encoding(encodings, 0.0, 0.0, 1, cluster_map, min_var = True).detach().numpy()
    def preprocess_anit_encodings(self, som, cluster_map, device, num_workers = 24):
        anit_encodings_lists = []
        with tqdm(total = self.encodings.shape[0], desc = "Sampling SOM") as tq:
            for i in range(0, self.encodings.shape[0], 256):
                benc = torch.tensor(self.encodings[i: i + 256, ...]).to(device = device)
                bres = som.sample_anti_encoding(benc, 0.0, 0.0, 1, cluster_map, min_var = True).detach().numpy()
                anit_encodings_lists.append(bres.reshape(bres.shape[0], -1))
                tq.update(256)
        self.anti_encodings = np.concatenate(anit_encodings_lists, axis = 0)
        anti_token_idxs = []
        with tqdm(total = self.anti_encodings.shape[0], desc = "Tokenizing Anti-Encodings") as tq:
            for i in range(0, self.anti_encodings.shape[0], 256):
                benc = self.encodings[i: i + 256, ...]
                bres = self.enc_tree.query(benc)[1]
                anti_token_idxs.append(bres)
                tq.update(256)
        anti_token_idxs = np.concatenate(anti_token_idxs, axis = 0)
        #print(self.anti_encodings.shape, anti_token_idxs.shape)
        self.anti_tokens = {"input_ids": self.tokens["input_ids"][anti_token_idxs, ...], "attention_mask": self.tokens["attention_mask"][anti_token_idxs, ...]}
    def __len__(self):
        return len(self.encodings)
    
    def output_size(self):
        return self.encodings.shape[1]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        elif (type(idx) == int):
            idx = [idx]
        assert self.tokenizer is not None
        assert self.anti_encodings is not None
        sample = {
            "encodings": self.encodings[idx, :],
            "tokens_input_ids": self.tokens["input_ids"][idx, ...],
            "tokens_attention_mask": self.tokens["attention_mask"][idx, ...],
            "anti_encodings": self.anti_encodings[idx, :],
            "anti_tokens_input_ids": self.anti_tokens["input_ids"][idx, ...],
            "anti_tokens_attention_mask": self.anti_tokens["attention_mask"][idx, ...],
            "pretokens_input_ids": self.preprompt_tokens.input_ids[idx, ...],
            "pretokens_attention_mask": self.preprompt_tokens.attention_mask[idx, ...]
        }
        return sample