import h5py
from utils.parsers import RedCatEmbExtractorParser
from models.autoencoder import Autoencoder
from tqdm import tqdm
import torch
import numpy as np
import math

if __name__ == "__main__":
    args = RedCatEmbExtractorParser.parse_args()
    device = "cpu"
    if (torch.cuda.is_available() and args.device >= 0):
        device = f"cuda:{args.device}"
    
    with h5py.File(args.data_path, "r+") as data:
        for split in data.keys():
            for i, embedding_key in enumerate(args.target_category):
                spec_data = []
                ind_seps = {}
                ind_count = 0
                for ind in data[split][embedding_key].keys():
                    spec_data.append(data[split][embedding_key][ind][()])
                    num_inds = spec_data[-1].shape[0]
                    ind_seps[ind] = list(range(ind_count, ind_count + num_inds))
                    ind_count += num_inds
                spec_data = torch.tensor(np.concatenate(spec_data)).to(device = device)
                for j, encoder_name in enumerate(args.encoders[i]):
                    print(f"Encoding {split} - {embedding_key} - {encoder_name}")
                    encoder = Autoencoder(args.input_size[i], reduction_steps = args.encoder_depth[i][j]).to(device = device)
                    checkpoint = torch.load(args.log_path / encoder_name / f"{encoder_name}_best.pt")
                    encoder.load_state_dict(checkpoint["model_state_dict"])
                    batch_acum = []
                    for batch in range(math.ceil(spec_data.shape[0] / args.batch_size)):
                        batch_data = spec_data[args.batch_size * batch : min(args.batch_size * (batch + 1), spec_data.shape[0]), :]
                        encoded_data = encoder.encode(batch_data)[0].detach().cpu().numpy()
                        batch_acum.append(encoded_data)
                    reduced_data = np.concatenate(batch_acum)
                    grp = data[split].create_group(encoder_name)
                    for key in ind_seps:
                        sep_data = reduced_data[ind_seps[key], :]
                        grp.create_dataset(key, data = sep_data)


