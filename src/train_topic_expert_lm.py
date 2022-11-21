import math
from models.topic_adapter_lm import TopicExpertLM
from utils.dataset import WDMCExpLMDataset
from utils.parsers import TopicExpertLMTrainingParser
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time
import random

if __name__ == "__main__":
    args = TopicExpertLMTrainingParser.parse_args()
    start_time = time.time()
    for exp_ind in range(len(args.target_category)):
        writer = SummaryWriter(args.log_path / args.target_category[exp_ind], comment=args.target_category[exp_ind])
        use_checkpoint = False
        if (args.checkpoint):
            checkpoint = torch.load(args.checkpoint)
            use_checkpoint = True
        
        train_dataset = WDMCExpLMDataset(args.path, "train", args.target_category[exp_ind], args.max_token_count, args.batch_size)
        val_dataset = WDMCExpLMDataset(args.path, "val", args.target_category[exp_ind], args.max_token_count, args.batch_size)

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=0)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=0)

        device = "cpu"
        if (torch.cuda.is_available() and args.device >= 0):
            device = f"cuda:{args.device}"
        
        model = TopicExpertLM(
            reduction_dim = args.reduction_dim, 
            base_model = args.base_model, 
            single_adapater = args.single_adapter, 
            finetune_base = args.finetune_base,
        ).to(device = device)

        train_dataset.tokenizer = model.tokenizer
        val_dataset.tokenizer = model.tokenizer

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
            epoch_losses = []
            for i_batch, sample_batched in tqdm(enumerate(train_dataloader), total = math.ceil(len(train_dataset) / args.batch_size)):
                with torch.autocast(device_type='cuda', dtype=torch.float32):
                    tokens = sample_batched["tokens"].to(device = device)
                    encodings = sample_batched["encodings"].to(device = device)
                    optimizer.zero_grad()
                    outputs = model.train()(tokens, encodings)
                    print(outputs["logits"].shape)
                    #cs_loss = torch.mean(cs_criterion(anti_encodings, pred))
                    loss = outputs["loss"]
                    assert loss.dtype is torch.float32
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.detach().cpu().numpy())
            scheduler.step()
            
            train_loss = np.mean(epoch_losses)
            print(f"Train - (Epoch={epoch}): Loss: {train_loss}")
            writer.add_scalar(f"trn-loss", train_loss, epoch)
            if (epoch % args.val_epochs == 0):
                val_losses = []
                for i_batch, sample_batched in tqdm(enumerate(val_dataloader), total = math.ceil(len(val_dataset) / args.batch_size)):
                    with torch.autocast(device_type='cuda', dtype=torch.float32):
                        tokens = sample_batched["tokens"].to(device = device)
                        encodings = sample_batched["encodings"].to(device = device)
                        outputs = model.eval()(tokens, encodings)
                        val_losses.append(outputs["loss"].detach().cpu().numpy())
                val_loss = np.mean(val_losses)
                percent_complete = ((exp_ind + 1) / len(args.target_category)) * (epoch + 1) / (args.epochs - start_epoch)
                hours_passed = ((time.time() - start_time) / 3600)
                remaining_factor = (1 - percent_complete) / max(1e-5, percent_complete)
                print(f"Val - (Epoch={epoch}): Loss {val_loss} | Comp {'%.2f' % (100 * percent_complete)}% | ETA {'%.2f' % (hours_passed * remaining_factor)} Hours")
                writer.add_scalar(f"val-loss", val_loss, epoch)
                new_best = False
                if (val_loss < best_val):
                    best_val = val_loss
                    new_best = True
                if (new_best):
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': optimizer.state_dict(),
                        'best_val': best_val,
                    }, args.log_path / args.target_category[exp_ind] / f"best_checkpoint.pt")
            
            

