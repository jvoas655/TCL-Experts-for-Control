import math
from models.adapter_transformer_encoder import TopicPredictorModel
from utils.dataset import WDMCEncDataset, WDMCGPTokenizedDataset # TODO: delete WDMCEncDataset
from utils.parsers import TopicPredictorTrainingParser
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import random
from models.gpt2_with_adapter_modules import GPT_With_Adapter_Modules
from models.som import SOM

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
        train_dataset = WDMCGPTokenizedDataset(args.path, "train", args.target_category[exp_ind], args.base_model, args.token_count, args.batch_size)
        val_dataset = WDMCGPTokenizedDataset(args.path, "val", args.target_category[exp_ind], args.base_model, args.token_count, args.batch_size)

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=0)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=0)

        device = "cpu"
        if (torch.cuda.is_available() and args.device >= 0):
            device = f"cuda:{args.device}"

        l1_criterion = torch.nn.L1Loss()
        assert args.loss == "L1"

        MAX_TOKENS = 64
        MIN_START = 8
        MAX_START = 48
        MIN_STEP = 1
        MAX_STEP = 4

        # TODO: replace with trained smaller version, and ask jordan for how load or what parameters needed
        topic_encoding_predictor_model = TopicPredictorModel(482, 64, "roberta-large", 64, False, False).to(device = "cuda:0") 
        # freeze topic_encoding_predictor_model
        for param in topic_encoding_predictor_model.features.parameters():
            param.requires_grad = False
        # set topic_encoding_predictor_model to dtype of float16
        # TODO: does this work, or do we rather need to modify the input to pass into the model ???
        topic_encoding_predictor_model.torch_dtype=torch.float16 
        topic_encoding_predictor_model.eval()

        load_path = None # TODO: specify load path, and ask jordan for load path
        som_model, _ = SOM.load(load_path).to(device = device) # or # som_model, _ = SOM.load(LOG_PATH / args.checkpoint_folder / category / f"som_checkpoint_{args.checkpoint_epoch}.npz")model = model.to(device = device)
        
        topic_conditional_lang_model = GPT_With_Adapter_Modules() # TODO: pass in args if necessary
        
        optimizer = torch.optim.AdamW(topic_conditional_lang_model.parameters(), lr=args.learning_rate)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters = 20)
        start_epoch = 0
        best_val = float("inf")
        print("num epochs : {}".format(args.epochs)) # TODO: delete
        for epoch in range(start_epoch, args.epochs):
            # epoch_l2s = []
            epoch_l1s = []
            epoch_losses = []
            for i_batch, sample_batched in tqdm(enumerate(train_dataloader), total = math.ceil(len(train_dataset) / args.batch_size)):

                sample_tokens = sample_batched["tokens"]["input_ids"].to(device = device) # TODO: confirm these are sample tokens
                # attention_mask = sample_batched["tokens"]["attention_mask"].to(device = device) # TODO: do we need attention_mask?
                
                ind = random.randint(MIN_START, MAX_START + 1)
                while (ind < MAX_TOKENS):
                    input_tokens = sample_tokens[:, :ind]
                    # TODO: do I modify dtype before or after passing tensor into model
                    ref_next_token =  sample_tokens[:, ind]
                    pred_topic_encodings = topic_encoding_predictor_model(input_tokens)
                    pred_anti_topic_encodings = som_model.inv_encode(pred_topic_encodings) # TODO:
                    pred_enc_next_token = topic_conditional_lang_model(input_tokens, pred_topic_encodings)
                    pred_anti_next_token = topic_conditional_lang_model(input_tokens, pred_anti_topic_encodings)
                    l1_loss = l1_criterion(ref_next_token, pred_enc_next_token)
                    loss = l1_loss + torch.nn.CosineSimilarity(ref_next_token, pred_enc_next_token)
                    loss.backwards()
                    ind += random.randint(MIN_STEP, MAX_STEP + 1)

                    # TODO: need sqrt? proboably does not matter as long as consistent
                    epoch_l1s.append(np.sqrt(l1_loss.detach().cpu().numpy())) 
                    epoch_losses.append(np.sqrt(loss.detach().cpu().numpy()))

            scheduler.step()
            
            train_l1 = np.mean(epoch_l1s)
            train_loss = np.mean(epoch_losses)
            print(f" Train L1 {train_l1} | Train Loss: {train_loss}")
            writer.add_scalar(f"trn-l1", train_l1, epoch)
            writer.add_scalar(f"trn-loss", train_loss, epoch)
            if (epoch % args.val_epochs == 0):
                # TODO: Put validation code here
                print("Need to implement")
                # val_l2s = []
                # val_l1s = []
                # for i_batch, sample_batched in tqdm(enumerate(val_dataloader), total = math.ceil(len(val_dataset) / args.batch_size)):
                #     input_ids = sample_batched["tokens"]["input_ids"].to(device = device)
                #     attention_mask = sample_batched["tokens"]["attention_mask"].to(device = device)
                #     encodings = sample_batched["encodings"].to(device = device)
                #     pred, z = model.eval()({
                #             "input_ids": input_ids,
                #             "attention_mask": attention_mask
                #         })
                #     l2_loss = l2_criterion(encodings, pred)
                #     l1_loss = l1_criterion(encodings, pred)

                #     val_l2s.append(np.sqrt(l2_loss.detach().cpu().numpy()))
                #     val_l1s.append(np.sqrt(l1_loss.detach().cpu().numpy()))
                
                # val_l2 = np.mean(val_l2s)
                # val_l1 = np.mean(val_l1s)
                # print(f"Val L2 (Epoch={epoch}): {val_l2} | Train L1 {val_l1}")
                # writer.add_scalar(f"val-l2", val_l2, epoch)
                # writer.add_scalar(f"val-l1", val_l1, epoch)
                # new_best = False
                # if (args.loss == "L2" and val_l2 < best_val):
                #     best_val = val_l2
                #     new_best = True
                # elif (args.loss == "L1" and val_l1 < best_val):
                #     best_val = val_l1
                #     new_best = True
                # if (new_best):
                #     torch.save({
                #         'epoch': epoch,
                #         'model_state_dict': model.state_dict(),
                #         'optimizer_state_dict': optimizer.state_dict(),
                #         'scheduler_state_dict': optimizer.state_dict(),
                #         'best_val': best_val,
                #     }, args.log_path / args.target_category[exp_ind] / f"best_checkpoint.pt")

