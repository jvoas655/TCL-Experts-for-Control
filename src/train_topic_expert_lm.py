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
    writer = SummaryWriter(args.log_path / args.exp_name, comment=args.exp_name)
    use_checkpoint = False
    if (args.checkpoint):
        checkpoint = torch.load(args.checkpoint)
        use_checkpoint = True
    
    train_dataset = WDMCExpLMDataset(args.path, "train", args.target_category, int(args.max_token_count), int(args.batch_size), lim=None)
    val_dataset = WDMCExpLMDataset(args.path, "val", args.target_category, int(args.max_token_count), int(args.batch_size), lim=None)

    train_dataloader = DataLoader(train_dataset, batch_size=int(args.batch_size),
                        shuffle=True, num_workers=8)
    val_dataloader = DataLoader(val_dataset, batch_size=int(args.batch_size),
                        shuffle=True, num_workers=8)

    device = "cpu"
    if (torch.cuda.is_available() and args.device >= 0):
        device = f"cuda:{args.device}"
    
    model = TopicExpertLM(
        reduction_dim = int(args.reduction_dim), 
        base_model = args.base_model, 
        topic_dim = int(args.topic_dim),
        single_adapater = args.single_adapter, 
        finetune_base = args.finetune_base,
    ).to(device = device)
    
    som, _ = SOM.load(args.som_checkpoint)
    som = som.to(device = device)

    som_cluster_map = som.form_cluster_map(torch.tensor(train_dataset.encodings, device = device))

    print("Preprocessing Train Tokens")
    train_dataset.preprocess_tokens(model.tokenizer)
    print("Preprocessing Train Anti-Encodings")
    train_dataset.preprocess_anit_encodings(som, som_cluster_map, device)
    print("Preprocessing Val Tokens")
    val_dataset.preprocess_tokens(model.tokenizer)
    print("Preprocessing Val Anti-Encodings")
    val_dataset.preprocess_anit_encodings(som, som_cluster_map, device)
    

    ce_criterion = torch.nn.CrossEntropyLoss()
    logit_sm = torch.nn.Softmax(dim = 3)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate))
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)
    start_epoch = 0
    best_val = float("inf")
    if (use_checkpoint):
        if (not args.reset_epoch):
            start_epoch = checkpoint["epoch"] + 1
            best_val = checkpoint["best_val"]
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        model.load_state_dict(checkpoint['model_state_dict'])
        som_cluster_map = checkpoint["cluster_map"]
        som = checkpoint["som"]
        torch.cuda.empty_cache()
    for epoch in range(start_epoch, int(args.epochs)):
        epoch_losses = []
        epoch_anti_losses = []
        epoch_fin_losses = []
        for i_batch, sample_batched in tqdm(enumerate(train_dataloader), total = math.ceil(len(train_dataset) / int(args.batch_size))):
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                input_ids = sample_batched["tokens_input_ids"].to(device = device)
                attention_mask = sample_batched["tokens_attention_mask"].to(device = device)
                encodings = sample_batched["encodings"].to(device = device)
                anti_encodings = sample_batched["anti_encodings"].to(device = device)
                anti_input_ids = sample_batched["anti_tokens_input_ids"].to(device = device)
                anti_attention_mask = sample_batched["anti_tokens_attention_mask"].to(device = device)
                
                optimizer.zero_grad()
                
                special_training_check = random.random()
                if (special_training_check < float(args.classifier_free_guide_chance)): # Classifer free guidance
                    encodings = torch.zeros_like(encodings)
                    outputs = model.train()({"input_ids":input_ids, "attention_mask":attention_mask}, encodings, compute_loss = False)
                    shift_logits = outputs["logits"][..., :-1, :].contiguous()
                    shift_labels = torch.where(attention_mask == 0, -100, input_ids)[..., 1:].contiguous()
                    loss = ce_criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    shift_logits = outputs["logits"][..., :-1, :].contiguous()
                    shift_labels = torch.where(anti_attention_mask == 0, -100, anti_input_ids)[..., 1:].contiguous()
                    anti_loss = ce_criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                else:
                    true_input_ids = torch.clone(input_ids)
                    if (special_training_check < float(args.classifier_free_guide_chance) + float(args.token_masking_chance)):
                        mask_type = random.randint(1, 2)
                        if (mask_type == 1):
                            input_ids = torch.where(torch.rand(input_ids.shape, device = device) < 0.1, model.tokenizer.get_vocab()[model.tokenizer.unk_token], input_ids)
                        elif (mask_type == 2):
                            chunk_start = random.randint(0, input_ids.shape[2] - 2)
                            chunk_end = random.randint(chunk_start + 1, input_ids.shape[2])
                            input_ids[..., chunk_start:chunk_end] = model.tokenizer.get_vocab()[model.tokenizer.unk_token]

                    outputs = model.train()({"input_ids":input_ids, "attention_mask":attention_mask}, encodings, compute_loss = False)
                    shift_logits = outputs["logits"][..., :-1, :].contiguous()
                    shift_labels = torch.where(attention_mask == 0, -100, true_input_ids)[..., 1:].contiguous()
                    loss = ce_criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

                    shift_logits = outputs["logits"][..., :-1, :].contiguous()
                    shift_labels = torch.where(anti_attention_mask == 0, -100, anti_input_ids)[..., 1:].contiguous()
                    anti_loss = ce_criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                final_loss = float(args.anit_loss_lambda) * loss - (1 - float(args.anit_loss_lambda)) * anti_loss
                assert final_loss.dtype is torch.float32
            final_loss.backward()
            optimizer.step()

            epoch_losses.append(loss.detach().cpu().numpy())
            epoch_anti_losses.append(anti_loss.detach().cpu().numpy())
            epoch_fin_losses.append(final_loss.detach().cpu().numpy())
        scheduler.step()
        
        train_loss = np.mean(epoch_losses)
        train_anti_loss = np.mean(epoch_anti_losses)
        train_fin_loss = np.mean(epoch_fin_losses)
        print(f"Train - (Epoch={epoch}): Fin Loss {train_fin_loss} | Loss {train_loss} | Anti-Loss {train_anti_loss}")
        writer.add_scalar(f"trn-loss", train_loss, epoch)
        writer.add_scalar(f"trn-anti-loss", train_anti_loss, epoch)
        writer.add_scalar(f"trn-fin-loss", train_fin_loss, epoch)
        if (epoch % args.val_epochs == 0):
            val_losses = []
            val_anti_losses = []
            masked_val_losses = []
            masked_val_anti_losses = []
            for i_batch, sample_batched in tqdm(enumerate(val_dataloader), total = math.ceil(len(val_dataset) / int(args.batch_size))):
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    with torch.no_grad():
                        input_ids = sample_batched["tokens_input_ids"].to(device = device)
                        attention_mask = sample_batched["tokens_attention_mask"].to(device = device)
                        encodings = sample_batched["encodings"].to(device = device)
                        anti_encodings = sample_batched["anti_encodings"].to(device = device)
                        anti_input_ids = sample_batched["anti_tokens_input_ids"].to(device = device)
                        anti_attention_mask = sample_batched["anti_tokens_attention_mask"].to(device = device)
                        outputs = model.eval()({"input_ids":input_ids, "attention_mask":attention_mask}, encodings, compute_loss = False)
                        shift_logits = outputs["logits"][..., :-1, :].contiguous()
                        shift_labels = torch.where(attention_mask == 0, -100, input_ids)[..., 1:].contiguous()
                        ce_loss = ce_criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                        val_losses.append(ce_loss.detach().cpu().numpy())
                        shift_logits = outputs["logits"][..., :-1, :].contiguous()
                        shift_labels = torch.where(anti_attention_mask == 0, -100, anti_input_ids)[..., 1:].contiguous()
                        ce_loss = ce_criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                        val_anti_losses.append(ce_loss.detach().cpu().numpy())
                        true_input_ids = torch.clone(input_ids)
                        input_ids[...] = model.tokenizer.get_vocab()[model.tokenizer.unk_token]
                        outputs = model.eval()({"input_ids":input_ids, "attention_mask":attention_mask}, encodings, compute_loss = False)
                        shift_logits = outputs["logits"][..., :-1, :].contiguous()
                        shift_labels = torch.where(attention_mask == 0, -100, true_input_ids)[..., 1:].contiguous()
                        ce_loss = ce_criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                        masked_val_losses.append(ce_loss.detach().cpu().numpy())
                        shift_logits = outputs["logits"][..., :-1, :].contiguous()
                        shift_labels = torch.where(anti_attention_mask == 0, -100, anti_input_ids)[..., 1:].contiguous()
                        ce_loss = ce_criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                        masked_val_anti_losses.append(ce_loss.detach().cpu().numpy())
            val_loss = np.mean(val_losses)
            masked_val_loss = np.mean(masked_val_losses)
            val_anti_loss = np.mean(val_anti_losses)
            masked_val_anti_loss = np.mean(masked_val_anti_losses)
            percent_complete = (epoch + 1) / (int(args.epochs) - start_epoch)
            hours_passed = ((time.time() - start_time) / 3600)
            remaining_factor = (1 - percent_complete) / max(1e-5, percent_complete)
            
            print(f"Val - (Epoch={epoch}): Loss {val_loss} | Masked Loss {masked_val_loss} | Anti-Loss {val_anti_loss} | Masked Anti-Loss {masked_val_anti_loss} | Comp {'%.2f' % (100 * percent_complete)}% | ETA {'%.2f' % (hours_passed * remaining_factor)} Hours")
            writer.add_scalar(f"val-loss", val_loss, epoch)
            writer.add_scalar(f"val-msk-loss", masked_val_loss, epoch)
            writer.add_scalar(f"val-anti-loss", val_anti_loss, epoch)
            writer.add_scalar(f"val-msk-anti-loss", masked_val_anti_loss, epoch)
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
                    "cluster_map": som_cluster_map,
                    "som": som
                }, args.log_path / args.exp_name / f"best_checkpoint.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': optimizer.state_dict(),
                'best_val': best_val,
                "cluster_map": som_cluster_map,
                "som": som
            }, args.log_path / args.exp_name / f"checkpoint_{epoch}.pt")
            
            

