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
                    soft_max = torch.nn.Softmax(dim=0)
                    encoding = model.encode(sub_data, n=args.num_samples, modulate_dist=True)
                    anit_encoding = model.inv_encode(encoding, method=args.anti_metric)
                    encodings[sum_key] = soft_max(encoding).detach().cpu().numpy()#.reshape(1, -1)
                    anti_encodings[sum_key] = soft_max(anit_encoding).detach().cpu().numpy()
                #print(np.sort(np.sum(np.concatenate(list(encodings.values())), axis = 0)))
                #use_args = np.argsort(np.sum(np.concatenate(list(encodings.values())), axis = 0))[:64]
                #continue
                if (args.save_groups[cat_ind] not in data[split]):
                    new_enc_grp = data[split].create_group(args.save_groups[cat_ind])
                else:
                    new_enc_grp = data[split][args.save_groups[cat_ind]]
                if (args.save_groups[cat_ind] + "_anti" not in data[split]):
                    new_anti_enc_group = data[split].create_group(args.save_groups[cat_ind] + "_anti")
                else:
                    new_anti_enc_group = data[split][args.save_groups[cat_ind] + "_anti"]
                for key in tqdm(encodings, desc=f"Writing {category} - {split}", total = len(data_grp.keys())):
                    if (key in new_enc_grp.keys()):
                        del new_enc_grp[key]
                    if (key in new_anti_enc_group.keys()):
                        del new_anti_enc_group[key]
                    new_enc_grp.create_dataset(key, data = encodings[key])
                    new_anti_enc_group.create_dataset(key, data = anti_encodings[key])
