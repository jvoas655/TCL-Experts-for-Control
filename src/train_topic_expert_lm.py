import math
from models.topic_adapter_lm import TopicExpertLM
from models.som import SOM
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
        writer = SummaryWriter(args.log_path / args.exp_name[exp_ind], comment=args.exp_name[exp_ind])
        use_checkpoint = False
        if (args.checkpoint):
            checkpoint = torch.load(args.checkpoint)
            use_checkpoint = True
        
        train_dataset = WDMCExpLMDataset(args.path, "train", args.target_category[exp_ind], args.max_token_count, args.batch_size, lim=None)
        val_dataset = WDMCExpLMDataset(args.path, "val", args.target_category[exp_ind], args.max_token_count, args.batch_size, lim=None)

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
            topic_dim = args.topic_dim[exp_ind],
            single_adapater = args.single_adapter, 
            finetune_base = args.finetune_base,
        ).to(device = device)
        if (args.train_with_anti_enc or args.som_sample_ref_map):
            som, _ = SOM.load(args.som_checkpoint[exp_ind])
            som = som.to(device = device)
            if (args.som_sample_ref_map):
                som_cluster_map = som.form_cluster_map(torch.tensor(train_dataset.encodings, device = device))

        train_dataset.tokenizer = model.tokenizer
        val_dataset.tokenizer = model.tokenizer

        ce_criterion = torch.nn.CrossEntropyLoss()
        logit_sm = torch.nn.Softmax(dim = 3)

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
            som_cluster_map = checkpoint["cluster_map"]
            som = checkpoint["som"]
        for epoch in range(start_epoch, args.epochs):
            epoch_losses = []
            for i_batch, sample_batched in tqdm(enumerate(train_dataloader), total = math.ceil(len(train_dataset) / args.batch_size)):
                with torch.autocast(device_type='cuda', dtype=torch.float32):
                    tokens = sample_batched["tokens"].to(device = device)
                    encodings = sample_batched["encodings"].to(device = device)
                    
                    optimizer.zero_grad()
                    
                    special_training_check = random.random()
                    if (special_training_check < args.classifier_free_guide_chance): # Classifer free guidance
                        encodings = torch.zeros_like(encodings)
                        outputs = model.train()(tokens, encodings, compute_loss = False)
                        shift_logits = outputs["logits"][..., :-1, :].contiguous()
                        shift_labels = tokens["input_ids"][..., 1:].contiguous()
                        loss = ce_criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    else:
                        som_anti_encodings = som.sample_anti_encoding(encodings, args.som_sample_mean, args.som_sample_std, 1)
                        if (args.train_with_anti_enc):
                            som_anti_encodings = som.sample_anti_encoding(encodings, args.som_sample_mean, args.som_sample_std, 1) # Multi sample not supported
                        elif (args.som_sample_ref_map):
                            som_anti_encodings = som.sample_anti_encoding(encodings, args.som_sample_mean, args.som_sample_std, 1, som_cluster_map)
                        true_tokens = {}
                        for label in tokens.keys():
                            true_tokens[label] = torch.clone(tokens[label])
                        if (special_training_check < args.classifier_free_guide_chance + args.token_masking_chance):
                            mask_type = random.randint(0, 2)
                            if (mask_type == 0):
                                tokens["input_ids"][...] = model.tokenizer.get_vocab()[model.tokenizer.unk_token]
                            elif (mask_type == 1):
                                tokens["input_ids"] = torch.where(torch.rand(tokens["input_ids"].shape, device = device) < 0.1, model.tokenizer.get_vocab()[model.tokenizer.unk_token], tokens["input_ids"])
                            elif (mask_type == 2):
                                chunk_start = random.randint(0, tokens["input_ids"].shape[2] - 2)
                                chunk_end = random.randint(chunk_start + 1, tokens["input_ids"].shape[2])
                                tokens["input_ids"][..., chunk_start:chunk_end] = model.tokenizer.get_vocab()[model.tokenizer.unk_token]

                        outputs = model.train()(tokens, encodings, compute_loss = False)
                        shift_logits = outputs["logits"][..., :-1, :].contiguous()
                        shift_labels = true_tokens["input_ids"][..., 1:].contiguous()
                        ce_loss = ce_criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                        if (args.train_with_anti_enc or args.som_sample_ref_map):
                            anti_outputs = model.train()(tokens, som_anti_encodings, compute_loss = False)
                            anti_shift_logits = anti_outputs["logits"][..., :-1, :].contiguous()
                            anti_ce_loss = ce_criterion(anti_shift_logits.view(-1, anti_shift_logits.size(-1)), shift_labels.view(-1))
                        else:
                            anti_ce_loss = 0

                        loss = ce_loss - args.anit_loss_lambda * anti_ce_loss
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
                anti_val_losses = []
                masked_val_losses = []
                masked_anti_val_losses = []
                for i_batch, sample_batched in tqdm(enumerate(val_dataloader), total = math.ceil(len(val_dataset) / args.batch_size)):
                    with torch.autocast(device_type='cuda', dtype=torch.float32):
                        tokens = sample_batched["tokens"].to(device = device)
                        encodings = sample_batched["encodings"].to(device = device)
                        if (args.train_with_anti_enc):
                            som_anti_encodings = som.sample_anti_encoding(encodings, args.som_sample_mean, args.som_sample_std, 1) # Multi sample not supported
                        elif (args.som_sample_ref_map):
                            som_anti_encodings = som.sample_anti_encoding(encodings, args.som_sample_mean, args.som_sample_std, 1, som_cluster_map)
                        outputs = model.eval()(tokens, encodings, compute_loss = False)
                        shift_logits = outputs["logits"][..., :-1, :].contiguous()
                        shift_labels = tokens["input_ids"][..., 1:].contiguous()
                        ce_loss = ce_criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                        val_losses.append(ce_loss.detach().cpu().numpy())

                        if (args.train_with_anti_enc or args.som_sample_ref_map):
                            anti_outputs = model.train()(tokens, som_anti_encodings, compute_loss = False)
                            anti_shift_logits = anti_outputs["logits"][..., :-1, :].contiguous()
                            anti_ce_loss = ce_criterion(anti_shift_logits.view(-1, anti_shift_logits.size(-1)), shift_labels.view(-1))
                            anti_val_losses.append(anti_ce_loss.detach().cpu().numpy())
                        true_tokens = {}
                        for label in tokens.keys():
                            true_tokens[label] = torch.clone(tokens[label])
                        tokens["input_ids"][...] = model.tokenizer.get_vocab()[model.tokenizer.unk_token]
                        outputs = model.eval()(tokens, encodings, compute_loss = False)
                        shift_logits = outputs["logits"][..., :-1, :].contiguous()
                        shift_labels = true_tokens["input_ids"][..., 1:].contiguous()
                        ce_loss = ce_criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                        masked_val_losses.append(ce_loss.detach().cpu().numpy())
                        if (args.train_with_anti_enc or args.som_sample_ref_map):
                            anti_outputs = model.train()(tokens, som_anti_encodings, compute_loss = False)
                            anti_shift_logits = anti_outputs["logits"][..., :-1, :].contiguous()
                            anti_ce_loss = ce_criterion(anti_shift_logits.view(-1, anti_shift_logits.size(-1)), shift_labels.view(-1))
                            masked_anti_val_losses.append(anti_ce_loss.detach().cpu().numpy())
                val_loss = np.mean(val_losses)
                masked_val_loss = np.mean(masked_val_losses)
                anti_val_loss = np.mean(anti_val_losses)
                masked_anti_val_loss = np.mean(masked_anti_val_losses)
                percent_complete = ((exp_ind + 1) / len(args.target_category)) * (epoch + 1) / (args.epochs - start_epoch)
                hours_passed = ((time.time() - start_time) / 3600)
                remaining_factor = (1 - percent_complete) / max(1e-5, percent_complete)
                if (args.train_with_anti_enc or args.som_sample_ref_map):
                    print(f"Val - (Epoch={epoch}): Loss {val_loss} | Masked Loss {masked_val_loss} | Anti Loss {anti_val_loss} | Anti Masked Loss {masked_anti_val_loss} | Comp {'%.2f' % (100 * percent_complete)}% | ETA {'%.2f' % (hours_passed * remaining_factor)} Hours")
                else:
                    print(f"Val - (Epoch={epoch}): Loss {val_loss} | Masked Loss {masked_val_loss} | Comp {'%.2f' % (100 * percent_complete)}% | ETA {'%.2f' % (hours_passed * remaining_factor)} Hours")
                writer.add_scalar(f"val-loss", val_loss, epoch)
                writer.add_scalar(f"val-msk-loss", masked_val_loss, epoch)
                if (args.train_with_anti_enc or args.som_sample_ref_map):
                    writer.add_scalar(f"val-anti-loss", anti_val_loss, epoch)
                    writer.add_scalar(f"val-anti-msk-loss", masked_anti_val_loss, epoch)
                    writer.add_scalar(f"val-delta-loss", val_loss - args.anit_loss_lambda * anti_val_loss, epoch)
                    writer.add_scalar(f"val-delta-msk-loss", masked_val_loss - args.anit_loss_lambda * masked_anti_val_loss, epoch)
                new_best = False
                if ((args.train_with_anti_enc or args.som_sample_ref_map) and (val_loss - args.anit_loss_lambda * anti_val_loss < best_val)):
                    best_val = val_loss
                    new_best = True
                if (not (args.train_with_anti_enc or args.som_sample_ref_map) and (val_loss < best_val)):
                    best_val = val_loss
                    new_best = True
                if (new_best):
                    pass
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': optimizer.state_dict(),
                    'best_val': best_val,
                    "cluster_map": som_cluster_map,
                    "som": som
                }, args.log_path / args.exp_name[exp_ind] / f"checkpoint_{epoch}.pt")
            
            

