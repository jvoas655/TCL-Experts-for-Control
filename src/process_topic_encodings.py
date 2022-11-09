import h5py
from utils.parsers import TopicEncodingParser
from models.som import SOM
from tqdm import tqdm
from utils.paths import *
import torch
import numpy as np
import math

if __name__ == "__main__":
    args = TopicEncodingParser.parse_args()
    device = "cpu"
    if (torch.cuda.is_available() and args.device >= 0):
        device = f"cuda:{args.device}"
    
    with h5py.File(args.data_path, "a") as data:
        for cat_ind, category in enumerate(args.target_category):
            for split in data.keys():
                model, _ = SOM.load(LOG_PATH / args.checkpoint_folder / category / f"som_checkpoint_{args.checkpoint_epoch}.npz")
                model = model.to(device = device)
                data_grp = data[split][category]
                encodings = {}
                anti_encodings = {}
                for sum_ind, sum_key in tqdm(enumerate(data_grp.keys()), desc=f"Encoding {category} - {split}", total = len(data_grp.keys())):
                    sub_data = torch.tensor(data_grp[sum_key][()]).to(device = device)
                    encoding = model.encode(sub_data, n=args.num_samples)
                    anit_encoding = model.inv_encode(encoding, method=args.anti_metric)
                    encodings[sum_key] = encoding.detach().cpu().numpy()
                    anti_encodings[sum_key] = anit_encoding.detach().cpu().numpy()
                if (args.save_groups[cat_ind] not in data[split]):
                    new_enc_grp = data[split].create_group(args.save_groups[cat_ind])
                else:
                    new_enc_grp = data[split][args.save_groups[cat_ind]]
                if (args.save_groups[cat_ind] + "_anti" not in data[split]):
                    new_anti_enc_group = data[split].create_group(args.save_groups[cat_ind] + "_anti")
                else:
                    new_anti_enc_group = data[split][args.save_groups[cat_ind] + "_anti"]
                for key in tqdm(encodings, desc=f"Writing {category} - {split}", total = len(data_grp.keys())):
                    new_enc_grp.create_dataset(key, data = encodings[key])
                    new_anti_enc_group.create_dataset(key, data = anti_encodings[key])
