from models.som import SOM, construct_polyhedra_map
from utils.dataset import WDMCEmbDataset
from utils.parsers import SOMTrainingParser
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    args = SOMTrainingParser.parse_args()
    for exp_ind in range(len(args.target_category)):

        writer = SummaryWriter(args.log_path / args.target_category[exp_ind], comment=args.target_category[exp_ind])
        if (args.checkpoint):
            checkpoint = torch.load(args.checkpoint)
        
        device = "cpu"
        if (torch.cuda.is_available() and args.device >= 0):
            device = f"cuda:{args.device}"

        train_dataset = WDMCEmbDataset(args.path, "train", args.target_category[exp_ind])
        val_dataset = WDMCEmbDataset(args.path, "val", args.target_category[exp_ind])

        train_samples = torch.tensor(train_dataset.cat_embeddings).to(device = device)
        val_samples = torch.tensor(val_dataset.cat_embeddings).to(device = device)

        train_dataloader = DataLoader(train_dataset, batch_size=1,
                            shuffle=True, num_workers=0)
        val_dataloader = DataLoader(val_dataset, batch_size=1,
                            shuffle=True, num_workers=0)

        start_epoch = 0
        if (args.checkpoint):
            model, start_epoch = SOM.load(args.checkpoint)
        else:
            m, n = args.som_params
            locs, values, _ = construct_polyhedra_map(m, n, train_samples.shape[1], train_samples)
            model = SOM(locs, values, args.learning_rate, args.base_sigma, args.lr_decay_rate, args.sigma_decay_rate, args.som_metric)
        model = model.to(device = device)

        for epoch in range(start_epoch, args.train_epochs+1):
            train_dists = []
            for i_batch, sample_batched in tqdm(enumerate(train_dataloader), desc = f"Epoch {epoch}", total=len(train_samples)):
                sample_batched = sample_batched.to(device = device)
                ind, dist = model(sample_batched)
                model.reward(ind, sample_batched, epoch)
                train_dists.append(dist.detach().cpu().numpy().item())
            
            train_dist = np.mean(train_dists)
            inds, dists = model.batch_forward(train_samples)
            fin_train_dist = torch.mean(dists).detach().cpu().numpy().item()
            trn_ratio, trn_max, trn_mean, trn_med, trn_std, trn_min = model.process_cluster_stats(inds)
            print(f"Train - Dist: {train_dist} | Fin Dist: {fin_train_dist} \n\t Ratio: {trn_ratio} | Max: {trn_max} \n\t Mean: {trn_mean} | Med: {trn_med} \n\t Std: {trn_std} | Min: {trn_min}")
            writer.add_scalar(f"trn-dist", train_dist, epoch)
            writer.add_scalar(f"trn-fin-dist", fin_train_dist, epoch)
            writer.add_scalar(f"trn-ratio", trn_ratio, epoch)
            writer.add_scalar(f"trn-max", trn_max, epoch)
            writer.add_scalar(f"trn-mean", trn_mean, epoch)
            writer.add_scalar(f"trn-med", trn_med, epoch)
            writer.add_scalar(f"trn-std", trn_std, epoch)
            writer.add_scalar(f"trn-min", trn_min, epoch)
            if (epoch % args.val_epochs == 0):
                inds, dists = model.batch_forward(val_samples)
                val_dist = torch.mean(dists).detach().cpu().numpy().item()
                val_ratio, val_max, val_mean, val_med, val_std, val_min = model.process_cluster_stats(inds)
                anti_inds, anti_dists = model.batch_forward(val_samples, anti=True)
                val_anti_dist = torch.mean(anti_dists).detach().cpu().numpy().item()
                val_anti_ratio, val_anti_max, val_anti_mean, val_anti_med, val_anti_std, val_anti_min = model.process_cluster_stats(inds)
                print(f"Val - Dist: {val_dist} \n\t Ratio: {val_ratio} | Max: {val_max} \n\t Mean: {val_mean} | Med: {val_med} \n\t Std: {val_std} | Min: {val_min}")
                print(f"Val Anti - Dist: {val_anti_dist} \n\t Ratio: {val_anti_ratio} | Max: {val_anti_max} \n\t Mean: {val_anti_mean} | Med: {val_anti_med} \n\t Std: {val_anti_std} | Min: {val_anti_min}")
                writer.add_scalar(f"val-dist", val_dist, epoch)
                writer.add_scalar(f"val-ratio", val_ratio, epoch)
                writer.add_scalar(f"val-max", val_max, epoch)
                writer.add_scalar(f"val-mean", val_mean, epoch)
                writer.add_scalar(f"val-med", val_med, epoch)
                writer.add_scalar(f"val-std", val_std, epoch)
                writer.add_scalar(f"val-min", val_min, epoch)
                writer.add_scalar(f"val-anti-dist", val_anti_dist, epoch)
                writer.add_scalar(f"val-anti-ratio", val_anti_ratio, epoch)
                writer.add_scalar(f"val-anti-max", val_anti_max, epoch)
                writer.add_scalar(f"val-anti-mean", val_anti_mean, epoch)
                writer.add_scalar(f"val-anti-med", val_anti_med, epoch)
                writer.add_scalar(f"val-anti-std", val_anti_std, epoch)
                writer.add_scalar(f"val-anti-min", val_anti_min, epoch)
                print("-" * 20)
                model.save(args.log_path / args.target_category[exp_ind] / f"som_checkpoint_{epoch}.npz", epoch+1)

