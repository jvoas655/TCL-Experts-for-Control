import math
from models.adapter_transformer_encoder import TopicPredictorModel
from utils.dataset import WDMCEncDataset
from utils.parsers import TopicPredictorTrainingParser
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time
import random

if __name__ == "__main__":
    args = TopicPredictorTrainingParser.parse_args()
    start_time = time.time()
    for exp_ind in range(len(args.target_category)):
        writer = SummaryWriter(args.log_path / args.target_category[exp_ind], comment=args.target_category[exp_ind])
        use_checkpoint = False
        if (args.checkpoint):
            checkpoint = torch.load(args.checkpoint)
            use_checkpoint = True
        
        train_dataset = WDMCEncDataset(args.path, "train", args.target_category[exp_ind], args.base_model, args.token_count, args.batch_size)
        val_dataset = WDMCEncDataset(args.path, "val", args.target_category[exp_ind], args.base_model, args.token_count, args.batch_size)

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=0)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=0)

        device = "cpu"
        if (torch.cuda.is_available() and args.device >= 0):
            device = f"cuda:{args.device}"
        
        model = TopicPredictorModel(
            output_dim = train_dataset.output_size(),
            reduction_dim = args.reduction_dim, 
            base_model = args.base_model, 
            learnable_token_count = not args.disable_learn_token_reducer, 
            single_adapater = args.single_adapter, 
            finetune_base = args.finetune_base,
            use_adapters = not args.disable_adapter,
            threshold_value = args.threshold_value
        ).to(device = device)

        l2_criterion = torch.nn.MSELoss()
        l1_criterion = torch.nn.L1Loss()
        cs_criterion = torch.nn.CosineSimilarity()
        if (args.loss == "L2"):
            criterion = l2_criterion
        elif (args.loss == "L1"):
            criterion = l1_criterion
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)
        start_epoch = 0
        best_val = float("inf")
        if (use_checkpoint):
            start_epoch = checkpoint["epoch"] + 1
            best_val = checkpoint["best_val"]
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        for epoch in range(start_epoch, args.epochs):
            epoch_l2s = []
            epoch_l1s = []
            epoch_css = []
            epoch_losses = []
            for i_batch, sample_batched in tqdm(enumerate(train_dataloader), total = math.ceil(len(train_dataset) / args.batch_size)):
                with torch.autocast(device_type='cuda', dtype=torch.float32):
                    length = random.randint(8, args.token_count)
                    start = random.randint(0, args.token_count - length)
                    input_ids = sample_batched["tokens"]["input_ids"][:, start:start + length].to(device = device)
                    attention_mask = sample_batched["tokens"]["attention_mask"][:, start:start + length].to(device = device)
                    encodings = sample_batched["encodings"].to(device = device)
                    anti_encodings = sample_batched["anti_encodings"].to(device = device)
                    optimizer.zero_grad()
                    pred, acts, z = model.train()({
                            "input_ids": input_ids,
                            "attention_mask": attention_mask
                        })
                    l2_loss = l2_criterion(encodings, pred)
                    l1_loss = l1_criterion(encodings, pred)
                    cs_loss = torch.sum(cs_criterion(encodings, pred))
                    if (args.sparsity_target == "parameters"):
                        sparse_l1_loss = sum(p.abs().sum() for p in model.parameters())
                        sparse_l2_loss = sum(p.pow(2).sum() for p in model.parameters())
                    elif (args.sparsity_target == "activations"):
                        acts = torch.cat(acts)
                        sparse_l1_loss = l1_criterion(acts, torch.zeros_like(acts, device = device))
                        sparse_l2_loss = l2_criterion(acts, torch.zeros_like(acts, device = device))
                    elif (args.sparsity_target == "outputs"):
                        acts = torch.cat(acts)
                        sparse_l1_loss = l1_criterion(pred, torch.zeros_like(pred, device = device))
                        sparse_l2_loss = l2_criterion(pred, torch.zeros_like(pred, device = device))
                    if (args.loss == "L2"):
                        pred_loss = l2_loss
                    elif (args.loss == "L1"):
                        pred_loss = l1_loss
                    elif (args.loss == "L1|L2"):
                        pred_loss = l1_loss + l2_loss
                    elif (args.loss == "CS"):
                        pred_loss = cs_loss
                    #cs_loss = torch.mean(cs_criterion(anti_encodings, pred))
                    loss = pred_loss + args.l1_lambda * sparse_l1_loss + args.l2_lambda * sparse_l2_loss
                    assert loss.dtype is torch.float32
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.detach().cpu().numpy())
                epoch_l2s.append(l2_loss.detach().cpu().numpy())
                epoch_l1s.append(l1_loss.detach().cpu().numpy())
                epoch_css.append(cs_loss.detach().cpu().numpy())
            scheduler.step()
            
            train_l2 = np.mean(epoch_l2s)
            train_l1 = np.mean(epoch_l1s)
            train_cs = np.mean(epoch_css)
            train_loss = np.mean(epoch_losses)
            print(f"Train - (Epoch={epoch}): L2 {train_l2} | L1 {train_l1} | CS {train_cs} | Loss: {train_loss}")
            writer.add_scalar(f"trn-l2", train_l2, epoch)
            writer.add_scalar(f"trn-l1", train_l1, epoch)
            writer.add_scalar(f"trn-cs", train_cs, epoch)
            writer.add_scalar(f"trn-loss", train_loss, epoch)
            if (epoch % args.val_epochs == 0):
                val_l2s = []
                val_l1s = []
                val_css = []
                for i_batch, sample_batched in tqdm(enumerate(val_dataloader), total = math.ceil(len(val_dataset) / args.batch_size)):
                    with torch.autocast(device_type='cuda', dtype=torch.float32):
                        input_ids = sample_batched["tokens"]["input_ids"].to(device = device)
                        attention_mask = sample_batched["tokens"]["attention_mask"].to(device = device)
                        encodings = sample_batched["encodings"].to(device = device)
                        anti_encodings = sample_batched["anti_encodings"].to(device = device)
                        pred, _, _ = model.eval()({
                                "input_ids": input_ids,
                                "attention_mask": attention_mask
                            })
                        l2_loss = l2_criterion(encodings, pred)
                        l1_loss = l1_criterion(encodings, pred)
                        cs_loss = torch.sum(cs_criterion(encodings, pred))
                        val_l2s.append(l2_loss.detach().cpu().numpy())
                        val_l1s.append(l1_loss.detach().cpu().numpy())
                        val_css.append(cs_loss.detach().cpu().numpy())
                        if (i_batch == 0):
                            print(encodings[:1, :])
                            #print(anti_encodings[:4, :])
                            print(pred[:1, :])
                
                val_l2 = np.mean(val_l2s)
                val_l1 = np.mean(val_l1s)
                val_cs = np.mean(val_css)
                percent_complete = ((exp_ind + 1) / len(args.target_category)) * (epoch + 1) / (args.epochs - start_epoch)
                hours_passed = ((time.time() - start_time) / 3600)
                remaining_factor = (1 - percent_complete) / max(1e-5, percent_complete)
                print(f"Val - (Epoch={epoch}): L2 {val_l2} | L1 {val_l1} | CS {val_cs} | Comp {'%.2f' % (100 * percent_complete)}% | ETA {'%.2f' % (hours_passed * remaining_factor)} Hours")
                writer.add_scalar(f"val-l2", val_l2, epoch)
                writer.add_scalar(f"val-l1", val_l1, epoch)
                writer.add_scalar(f"val-cs", val_cs, epoch)
                new_best = False
                if (args.loss == "L2" and val_l2 < best_val):
                    best_val = val_l2
                    new_best = True
                elif (args.loss == "L1" and val_l1 < best_val):
                    best_val = val_l1
                    new_best = True
                elif (args.loss == "L1|L2" and val_l1 + val_l2 < best_val):
                    best_val = val_l1 + val_l2
                    new_best = True
                elif (args.loss == "CS" and val_cs < best_val):
                    best_val = val_cs
                    new_best = True
                if (new_best):
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': optimizer.state_dict(),
                        'best_val': best_val,
                    }, args.log_path / args.target_category[exp_ind] / f"best_checkpoint.pt")
            
            

