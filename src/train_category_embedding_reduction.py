from models.autoencoder import Autoencoder
from utils.dataset import WDMCAutoEncoderDataset
from utils.parsers import AETrainingParser
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    args = AETrainingParser.parse_args()
    for exp_set_ind in range(len(args.exp_name)):
        for exp_ind in range(len(args.exp_name[exp_set_ind])):

            writer = SummaryWriter(args.log_path / args.exp_name[exp_set_ind][exp_ind], comment=args.exp_name[exp_set_ind][exp_ind])
            use_checkpoint = False
            if (args.checkpoint):
                checkpoint = torch.load(args.checkpoint)
                use_checkpoint = True
            if (exp_ind > 0):
                checkpoint = torch.load(args.log_path / args.exp_name[exp_set_ind][exp_ind-1] / f"{args.exp_name[exp_set_ind][exp_ind-1]}_best.pt")
                use_checkpoint = True
            
            train_dataset = WDMCAutoEncoderDataset(args.path, "train", args.target_category[exp_set_ind])
            val_dataset = WDMCAutoEncoderDataset(args.path, "val", args.target_category[exp_set_ind])

            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                shuffle=True, num_workers=0)
            val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size,
                                shuffle=True, num_workers=0)

            device = "cpu"
            if (torch.cuda.is_available() and args.device >= 0):
                device = f"cuda:{args.device}"
            start_depth = args.start_depth[exp_set_ind][exp_ind]
            if (use_checkpoint):
                advance = max(0, start_depth - checkpoint['cur_depth'])
                start_depth = checkpoint['cur_depth']
            model = Autoencoder(train_dataset.input_size(), reduction_steps = start_depth).to(device = device)

            mse_criterion = torch.nn.MSELoss()
            l1_criterion = torch.nn.L1Loss()
            if (args.loss == "MSE"):
                criterion = mse_criterion
            elif (args.loss == "L1"):
                criterion = l1_criterion
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
            cur_depth = start_depth
            stagnant_val_steps = 0
            best_val_mse = float("inf")
            epoch = 0
            if (use_checkpoint):
                model.load_state_dict(checkpoint['model_state_dict'])
                
                epoch = checkpoint['epoch'] + 1
                if (advance == 0):
                    best_val_mse = checkpoint['best_val_mse']
                    stagnant_val_steps = checkpoint['stagnant_val_steps']
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                else:
                    for i in range(advance):
                        model.increment_depth().to(device = device)
                        cur_depth += 1
                    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
                    epoch = 0
            continue_training = True
            while (continue_training):
                epoch_mses = []
                epoch_losses = []
                for i_batch, sample_batched in tqdm(enumerate(train_dataloader), desc = f"Epoch {epoch}"):
                    sample_batched = sample_batched.to(device = device)
                    optimizer.zero_grad()
                    recon, enc_acts, dec_acts = model.train()(sample_batched)
                    mse_loss = mse_criterion(sample_batched, recon)
                    recon_loss = criterion(sample_batched, recon)
                    if (args.sparsity_target == "parameters"):
                        l1_loss = sum(p.abs().sum() for p in model.parameters())
                        l2_loss = sum(p.pow(2).sum() for p in model.parameters())
                    elif (args.sparsity_target == "activations"):
                        acts = []
                        for act in enc_acts + dec_acts:
                            acts.append(act.flatten())
                        acts = torch.cat(acts)
                        l1_loss = l1_criterion(acts, torch.zeros_like(acts, device = device))
                        l2_loss = mse_criterion(acts, torch.zeros_like(acts, device = device))
                    loss = recon_loss + args.l1_lambda * l1_loss + args.l2_lambda * l2_loss
                    loss.backward()
                    optimizer.step()

                    epoch_losses.append(loss.detach().cpu().numpy())
                    epoch_mses.append(np.sqrt(mse_loss.detach().cpu().numpy()))
                
                train_mse = np.mean(epoch_mses)
                train_loss = np.mean(epoch_losses)
                print(f"Train MSE: {train_mse} | Train Loss: {train_loss}")
                writer.add_scalar(f"trn-mse", train_mse, epoch)
                writer.add_scalar(f"trn-loss", train_loss, epoch)
                if (epoch > 0 and epoch % args.val_epochs == 0):
                    for depth in range(0, cur_depth + 1):
                        val_mses = []
                        val_l1s = []
                        for i_batch, sample_batched in tqdm(enumerate(val_dataloader), desc = f"Val Step {epoch // args.val_epochs}"):
                            sample_batched = sample_batched.to(device = device)
                            optimizer.zero_grad()
                            recon, _, _ = model.eval()(sample_batched, use_depth=depth)
                            mse_loss = mse_criterion(sample_batched, recon)
                            l1_loss = mse_criterion(sample_batched, recon)

                            val_mses.append(np.sqrt(mse_loss.detach().cpu().numpy()))
                            val_l1s.append(l1_loss.detach().cpu().numpy())
                        val_mse = np.mean(val_mses)
                        val_l1 = np.mean(val_l1s)
                        print(f"Val MSE: {val_mse} | Val L1: {val_l1} | Depth: {depth}")
                    writer.add_scalar(f"val-mse", val_mse, epoch)
                    writer.add_scalar(f"val-l1", val_l1, epoch)
                    print("-" * 20)
                    new_best = False
                    if (cur_depth < args.max_depth[exp_set_ind][exp_ind] and val_mse * args.increment_scale < best_val_mse):
                        stagnant_val_steps = 0
                        best_val_mse = val_mse
                        new_best = True
                    elif (cur_depth == args.max_depth[exp_set_ind][exp_ind] and val_mse < best_val_mse):
                        best_val_mse = val_mse
                        stagnant_val_steps = 0
                        new_best = True
                    else:
                        stagnant_val_steps += 1
                    print(stagnant_val_steps, args.increment_guard, best_val_mse)
                    if (stagnant_val_steps > args.increment_guard and args.increment_schedule == "val"):
                        cur_depth += 1
                        model.increment_depth().to(device=device)
                        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
                        stagnant_val_steps = 0
                        best_val_mse = float("inf")
                        epoch = 0
                    elif ((epoch // args.val_epochs) // args.increment_steps == 0 and args.increment_schedule == "step"):
                        cur_depth += 1
                        model.increment_depth().to(device=device)
                        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
                        stagnant_val_steps = 0
                        best_val_mse = float("inf")
                        epoch = 0
                    if (cur_depth > args.max_depth[exp_set_ind][exp_ind]):
                        continue_training = False
                    if (new_best):
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'best_val_mse': best_val_mse,
                            'stagnant_val_steps': stagnant_val_steps,
                            'cur_depth': cur_depth
                        }, args.log_path / args.exp_name[exp_set_ind][exp_ind] / f"checkpoint-{epoch}.pt")
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'best_val_mse': best_val_mse,
                            'stagnant_val_steps': stagnant_val_steps,
                            'cur_depth': cur_depth
                        }, args.log_path / args.exp_name[exp_set_ind][exp_ind] / f"{args.exp_name[exp_set_ind][exp_ind]}_best.pt")
                epoch += 1

