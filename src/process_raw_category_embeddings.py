from utils.parsers import CatEmbExtractorParser
from tqdm import tqdm
from multiprocessing import Pool
import h5py
from transformers import RobertaTokenizer, RobertaModel
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import numpy as np


def proc_decode(arg):
    return arg.decode("utf-8")

def proc_tokenize(arg):
    text, tokenizer, num_tokens = arg
    return tokenizer(text, return_tensors="pt", truncation=True,max_length=num_tokens, padding=True)

def proc_embedding(arg):
    tokens, model = arg
    print(tokens)
    return model(tokens)

if __name__ == "__main__":
    args = CatEmbExtractorParser.parse_args()
    if ("individual" in args.gen_type):
        data_file = h5py.File(args.path, "r")
        all_cats = []
        for datakey in ["test", "train", "val"]:
            keys = list(data_file[f"{datakey}/categories"].keys())
            for key in keys:
                all_cats += data_file[f"{datakey}/categories/{key}"][()].tolist()
        data_file.close()
        all_cats = list(set(all_cats))
        all_decoded_cats = []
        with tqdm(total=len(all_cats), desc="Decoding Categories") as tq:
            with Pool(args.threads) as pool:
                for res in pool.imap_unordered(proc_decode, all_cats):
                    all_decoded_cats.append(res)
                    tq.update(1)

        device = "cpu"
        if (torch.cuda.is_available() and args.device >= 0):
            device = f"cuda:{args.device}"
        for mi, model_name in enumerate(args.model):
            batch_size = args.batch_size[mi]
            if ("roberta" in model_name):
                tokenizer = RobertaTokenizer.from_pretrained(model_name)
                model = RobertaModel.from_pretrained(model_name).to(device=device)
            elif ("t5" in model_name):
                tokenizer = T5Tokenizer.from_pretrained(model_name)
                model = T5ForConditionalGeneration.from_pretrained(model_name).to(device=device).encoder
            
            proc_args = [(all_decoded_cats[cat_ind:min(len(all_decoded_cats), cat_ind + batch_size)], tokenizer, args.max_tokens_ind) for cat_ind in range(0, len(all_decoded_cats), batch_size)]
            tokenized_cats = []
            with tqdm(total=len(all_decoded_cats), desc=f"Tokenizing Categories: {model_name}") as tq:
                with Pool(args.threads) as pool:
                    for res in pool.imap_unordered(proc_tokenize, proc_args):
                        tokenized_cats.append(res.to(device=device))
                        tq.update(batch_size)
            proc_args = [(token_cat, model) for token_cat in tokenized_cats]
            embedded_cats = []
            with tqdm(total=len(tokenized_cats) * batch_size, desc=f"Extracting Category Embeddings: {model_name}") as tq:
                for token_batch in tokenized_cats:
                    res = model(**token_batch)
                    embedded_cats.append(torch.mean(res.last_hidden_state, dim=1).detach().cpu().numpy())
                    tq.update(batch_size)
            embedded_cats = np.concatenate(embedded_cats)
            
            with h5py.File(args.path, "r+") as data_file:
                grp_name = f"raw_cat_embeddings_ind_{model_name.replace('-', '_')}"
                for datakey in ["test", "train", "val"]:
                    if (grp_name in data_file[f"{datakey}"].keys()):
                        del data_file[f"{datakey}"][grp_name]
                    grp = data_file[f"{datakey}"].create_group(grp_name)
                    keys = list(data_file[f"{datakey}/categories"].keys())
                    for key in tqdm(keys, total = len(keys), desc = f"Writing {datakey} to HDF5: {model_name}"):
                        cats = data_file[f"{datakey}/categories/{key}"][()].tolist()
                        cat_embs = []
                        for cat in cats:
                            cat_ind = all_cats.index(cat)
                            cat_embs.append(embedded_cats[cat_ind, :].reshape(1, -1))
                        cat_embs = np.concatenate(cat_embs)
                        grp.create_dataset(key, data = cat_embs)
    if ("concat" in args.gen_type):
        data_file = h5py.File(args.path, "r")
        all_cats = []
        for datakey in ["test", "train", "val"]:
            keys = list(data_file[f"{datakey}/categories"].keys())
            for key in keys:
                all_cats += ["; ".join(list(map(lambda e: e.decode("utf-8"), data_file[f"{datakey}/categories/{key}"][()].tolist())))]
        data_file.close()
        all_cats = list(set(all_cats))

        device = "cpu"
        if (torch.cuda.is_available() and args.device >= 0):
            device = f"cuda:{args.device}"
        for mi, model_name in enumerate(args.model):
            batch_size = args.batch_size[mi]
            if ("roberta" in model_name):
                tokenizer = RobertaTokenizer.from_pretrained(model_name)
                model = RobertaModel.from_pretrained(model_name).to(device=device)
            elif ("t5" in model_name):
                tokenizer = T5Tokenizer.from_pretrained(model_name)
                model = T5ForConditionalGeneration.from_pretrained(model_name).to(device=device).encoder
            
            proc_args = [(all_cats[cat_ind:min(len(all_cats), cat_ind + batch_size)], tokenizer, args.max_tokens_con) for cat_ind in range(0, len(all_cats), batch_size)]
            tokenized_cats = []
            with tqdm(total=len(all_cats), desc=f"Tokenizing Categories: {model_name}") as tq:
                with Pool(args.threads) as pool:
                    for res in pool.imap_unordered(proc_tokenize, proc_args):
                        tokenized_cats.append(res.to(device=device))
                        tq.update(batch_size)
            proc_args = [(token_cat, model) for token_cat in tokenized_cats]
            embedded_cats = []
            with tqdm(total=len(tokenized_cats) * batch_size, desc=f"Extracting Category Embeddings: {model_name}") as tq:
                for token_batch in tokenized_cats:
                    res = model(**token_batch)
                    embedded_cats.append(torch.mean(res.last_hidden_state, dim=1).detach().cpu().numpy())
                    tq.update(batch_size)
            embedded_cats = np.concatenate(embedded_cats)
            
            with h5py.File(args.path, "r+") as data_file:
                grp_name = f"raw_cat_embeddings_con_{model_name.replace('-', '_')}"
                for datakey in ["test", "train", "val"]:
                    if (grp_name in data_file[f"{datakey}"].keys()):
                        del data_file[f"{datakey}"][grp_name]
                    grp = data_file[f"{datakey}"].create_group(grp_name)
                    keys = list(data_file[f"{datakey}/categories"].keys())
                    for key in tqdm(keys, total = len(keys), desc = f"Writing {datakey} to HDF5: {model_name}"):
                        cats = "; ".join(list(map(lambda e: e.decode("utf-8"), data_file[f"{datakey}/categories/{key}"][()].tolist())))
                        cat_ind = all_cats.index(cats)
                        grp.create_dataset(key, data = embedded_cats[cat_ind, :])





                
        
    
