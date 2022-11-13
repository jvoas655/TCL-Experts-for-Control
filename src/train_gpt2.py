import math
from models.adapter_transformer_encoder import TopicPredictorModel
from utils.dataset import WDMCEncDataset, WDMCGPTEncDataset # TODO: delete WDMCEncDataset
from utils.parsers import TopicPredictorTrainingParser
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from models.GPT_2_Fine_Tuning import GPT_With_Adapter_Modules


if __name__ == "__main__":
    args = TopicPredictorTrainingParser.parse_args()
    args.path = "data/text_encoding_pairs.hdf5"
    args.token_count = 256 # TODO: delete
    print(args) # TODO: delete
    for exp_ind in range(len(args.target_category)):

        writer = SummaryWriter(args.log_path / args.target_category[exp_ind], comment=args.target_category[exp_ind])
        use_checkpoint = False
        if (args.checkpoint):
            checkpoint = torch.load(args.checkpoint)
            use_checkpoint = True
        use_checkpoint = False # TODO: delete
        train_dataset = WDMCGPTEncDataset(args.path, "train", args.target_category[exp_ind], args.base_model, args.token_count, args.batch_size)
        val_dataset = WDMCGPTEncDataset(args.path, "val", args.target_category[exp_ind], args.base_model, args.token_count, args.batch_size)

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=0)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=0)

        device = "cpu"
        if (torch.cuda.is_available() and args.device >= 0):
            device = f"cuda:{args.device}"
        
        model = GPT_With_Adapter_Modules() # TODO: pass in args if necessary

        l2_criterion = torch.nn.MSELoss()
        l1_criterion = torch.nn.L1Loss()
        if (args.loss == "L2"):
            criterion = l2_criterion
        elif (args.loss == "L1"):
            criterion = l1_criterion
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters = 20)
        start_epoch = 0
        best_val = float("inf")
        if (use_checkpoint):
            assert True == False # TODO: delete
            start_epoch = checkpoint["epoch"] + 1
            best_val = checkpoint["best_val"]
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        for epoch in range(start_epoch, args.epochs):
            epoch_l2s = []
            epoch_l1s = []
            epoch_losses = []
            for i_batch, sample_batched in tqdm(enumerate(train_dataloader), total = math.ceil(len(train_dataset) / args.batch_size)):
                # TODO: why sampled?
                input_ids = sample_batched["tokens"]["input_ids"].to(device = device)
                attention_mask = sample_batched["tokens"]["attention_mask"].to(device = device)
                encodings = sample_batched["encodings"].to(device = device)
                optimizer.zero_grad()
                pred, z = model.train()({
                        "input_ids": input_ids,
                        "attention_mask": attention_mask
                    })
                l2_loss = l2_criterion(encodings, pred)
                l1_loss = l1_criterion(encodings, pred)
                # TODO: why sparse loss 
                if (args.sparsity_target == "parameters"):
                    assert True == False
                    sparse_l1_loss = sum(p.abs().sum() for p in model.parameters())
                    sparse_l2_loss = sum(p.pow(2).sum() for p in model.parameters())
                elif (args.sparsity_target == "activations"):
                    assert True == False
                    sparse_l1_loss = l1_criterion(z, torch.zeros_like(z, device = device))
                    sparse_l2_loss = l2_criterion(z, torch.zeros_like(z, device = device))
                if (args.loss == "L2"):
                    pred_loss = l2_loss
                elif (args.loss == "L1"):
                    pred_loss = l1_loss
                loss = pred_loss + args.l1_lambda * sparse_l1_loss + args.l2_lambda * sparse_l2_loss
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.detach().cpu().numpy())
                epoch_l2s.append(np.sqrt(l2_loss.detach().cpu().numpy()))
                epoch_l1s.append(np.sqrt(l1_loss.detach().cpu().numpy()))
            scheduler.step()
            
            train_l2 = np.mean(epoch_l2s)
            train_l1 = np.mean(epoch_l1s)
            train_loss = np.mean(epoch_losses)
            print(f"Train L2 (Epoch={epoch}): {train_l2} | Train L1 {train_l1} | Train Loss: {train_loss}")
            writer.add_scalar(f"trn-l2", train_l2, epoch)
            writer.add_scalar(f"trn-l1", train_l1, epoch)
            writer.add_scalar(f"trn-loss", train_loss, epoch)
            if (epoch % args.val_epochs == 0):
                val_l2s = []
                val_l1s = []
                for i_batch, sample_batched in tqdm(enumerate(val_dataloader), total = math.ceil(len(val_dataset) / args.batch_size)):
                    input_ids = sample_batched["tokens"]["input_ids"].to(device = device)
                    attention_mask = sample_batched["tokens"]["attention_mask"].to(device = device)
                    encodings = sample_batched["encodings"].to(device = device)
                    pred, z = model.eval()({
                            "input_ids": input_ids,
                            "attention_mask": attention_mask
                        })
                    l2_loss = l2_criterion(encodings, pred)
                    l1_loss = l1_criterion(encodings, pred)

                    val_l2s.append(np.sqrt(l2_loss.detach().cpu().numpy()))
                    val_l1s.append(np.sqrt(l1_loss.detach().cpu().numpy()))
                
                val_l2 = np.mean(val_l2s)
                val_l1 = np.mean(val_l1s)
                print(f"Val L2 (Epoch={epoch}): {val_l2} | Train L1 {val_l1}")
                writer.add_scalar(f"val-l2", val_l2, epoch)
                writer.add_scalar(f"val-l1", val_l1, epoch)
                new_best = False
                if (args.loss == "L2" and val_l2 < best_val):
                    best_val = val_l2
                    new_best = True
                elif (args.loss == "L1" and val_l1 < best_val):
                    best_val = val_l1
                    new_best = True
                if (new_best):
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': optimizer.state_dict(),
                        'best_val': best_val,
                    }, args.log_path / args.target_category[exp_ind] / f"best_checkpoint.pt")

