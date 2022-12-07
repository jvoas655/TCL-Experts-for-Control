import json
import math
from models.topic_adapter_lm import TopicExpertLM
from models.som import SOM
from utils.dataset import WDMCExpLMDataset
from utils.parsers import EvaluationParser
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time
import random
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from rouge import Rouge 
from nltk.translate.bleu_score import sentence_bleu


def compute_perplexity(model, input_ids):
    max_length = primary_model.config.n_positions
    stride = 512

    seq_len = input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over input tokens.
            # Multiply it with trg_len to get the summation instead of average.
            # We will take average over all the tokens to get the true average
            # in the last step of this example.
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc).detach().cpu().numpy()
    return ppl

if __name__ == "__main__":
    args = EvaluationParser.parse_args()
    writer = SummaryWriter(args.log_path, comment=args.exp_name)

    checkpoint = torch.load(args.checkpoint)
    test_dataset = WDMCExpLMDataset(args.path, "test", args.target_category, args.max_token_count, args.batch_size, lim=500)
    
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=0)

    device = "cpu"
    if (torch.cuda.is_available() and args.device >= 0):
        device = f"cuda:{args.device}"
    
    expert_model = TopicExpertLM(
        reduction_dim = args.reduction_dim, 
        base_model = args.expert_base_model, 
        topic_dim = args.topic_dim,
        single_adapater = args.single_adapter, 
        finetune_base = False,
    ).to(device = device).eval()

    print("Preprocessing Train Tokens")
    test_dataset.preprocess_tokens(expert_model.tokenizer)

    primary_model = GPT2LMHeadModel.from_pretrained(args.primary_model).to(device = device).eval()

    expert_model.load_state_dict(checkpoint['model_state_dict'])

    som = checkpoint["som"]
    som = som.to(device = device)

    som_cluster_map = None
    if ("cluster_map" in checkpoint):
        som_cluster_map = checkpoint["cluster_map"]
    print("Preprocessing Train Anti-Encodings")
    test_dataset.preprocess_anit_encodings(som, som_cluster_map, device)
    summary_results = {
        "perplexity": {"preprompt": [], "primary": [], "contrastive": [], "pos": [], "reference":[]}, 
        "rouge-1-r": {"preprompt": [], "primary": [], "contrastive": [], "pos": []}, 
        "rouge-1-p": {"preprompt": [], "primary": [], "contrastive": [], "pos": []}, 
        "rouge-1-f": {"preprompt": [], "primary": [], "contrastive": [], "pos": []}, 
        "rouge-2-r": {"preprompt": [], "primary": [], "contrastive": [], "pos": []}, 
        "rouge-2-p": {"preprompt": [], "primary": [], "contrastive": [], "pos": []}, 
        "rouge-2-f": {"preprompt": [], "primary": [], "contrastive": [], "pos": []},
        "rouge-l-r": {"preprompt": [], "primary": [], "contrastive": [], "pos": []}, 
        "rouge-l-p": {"preprompt": [], "primary": [], "contrastive": [], "pos": []}, 
        "rouge-l-f": {"preprompt": [], "primary": [], "contrastive": [], "pos": []},
        "bleu-1": {"preprompt": [], "primary": [], "contrastive": [], "pos": []}, 
        "bleu-2": {"preprompt": [], "primary": [], "contrastive": [], "pos": []}
    }
    for i_batch, sample_batched in tqdm(enumerate(test_dataloader), total = math.ceil(len(test_dataset) / args.batch_size)):
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            with torch.no_grad():
                input_ids = sample_batched["tokens_input_ids"].to(device = device)
                attention_mask = sample_batched["tokens_attention_mask"].to(device = device)
                encodings = sample_batched["encodings"].to(device = device)
                anti_encodings = sample_batched["anti_encodings"].to(device = device).squeeze(dim=1)
                anti_input_ids = sample_batched["anti_tokens_input_ids"].to(device = device)
                anti_attention_mask = sample_batched["anti_tokens_attention_mask"].to(device = device)
                pre_input_ids = sample_batched["pretokens_input_ids"].to(device = device)
                pre_attention_mask = sample_batched["pretokens_attention_mask"].to(device = device)

                preprompt_generated_tokens = torch.cat((pre_input_ids, input_ids[:, :, :args.eval_start_tokens]), dim=2)
                primary_generated_tokens = input_ids[:, :, :args.eval_start_tokens]
                pos_generated_tokens = input_ids[:, :, :args.eval_start_tokens]
                pos_plus_primary_generated_tokens = input_ids[:, :, :args.eval_start_tokens]
                neg_generated_tokens = input_ids[:, :, :args.eval_start_tokens]
                neg_plus_primary_generated_tokens = input_ids[:, :, :args.eval_start_tokens]
                contrastive_generated_tokens = input_ids[:, :, :args.eval_start_tokens]

                preprompt_generated_probs = torch.ones(1).to(device = input_ids.get_device())
                primary_generated_probs = torch.ones(1).to(device = input_ids.get_device())
                pos_generated_probs = torch.ones(1).to(device = input_ids.get_device())
                pos_plus_primary_generated_probs = torch.ones(1).to(device = input_ids.get_device())
                neg_generated_probs = torch.ones(1).to(device = input_ids.get_device())
                neg_plus_primary_generated_probs = torch.ones(1).to(device = input_ids.get_device())
                contrastive_generated_probs = torch.ones(1).to(device = input_ids.get_device())


                for i in range(args.eval_start_tokens+1, args.max_token_count):
                    # Calculate preprompt output
                    #print(pre_attention_mask.shape, primary_generated_tokens.shape)
                    preprompt_generated_tokens = {"input_ids": preprompt_generated_tokens, "attention_mask": torch.cat((pre_attention_mask.repeat(len(primary_generated_tokens), 1, 1), torch.ones_like(primary_generated_tokens)), dim=2)}
                    #print(preprompt_generated_tokens["input_ids"].shape, preprompt_generated_tokens["attention_mask"].shape)
                    preprompt_logits = primary_model(**preprompt_generated_tokens)["logits"][:, 0, -1, :]
                    preprompt_next_topk = torch.nn.functional.softmax(preprompt_logits, dim = 1).topk(args.beam_size, dim=1, largest = True, sorted = True)
                    preprompt_next_probs = preprompt_next_topk.values
                    preprompt_next_tokens = preprompt_next_topk.indices[..., None]
                    preprompt_candidate_probs = preprompt_next_probs * preprompt_generated_probs[:, None]
                    preprompt_candidate_probs = preprompt_candidate_probs.reshape(args.beam_size ** 2 if i > args.eval_start_tokens+1 else args.beam_size, -1).squeeze(dim=1)
                    preprompt_generated_tokens = preprompt_generated_tokens["input_ids"].repeat(1, args.beam_size, 1)
                    preprompt_candidate_tokens = torch.cat((preprompt_generated_tokens, preprompt_next_tokens), dim=2).reshape(args.beam_size ** 2 if i > args.eval_start_tokens+1 else args.beam_size, -1)
                    preprompt_candidate_topk = preprompt_candidate_probs.topk(args.beam_size, dim = 0, largest = True, sorted = True)
                    preprompt_generated_tokens = preprompt_candidate_tokens[preprompt_candidate_topk.indices, :][:, None, :]
                    preprompt_generated_probs = preprompt_candidate_probs[preprompt_candidate_topk.indices]
                    preprompt_generated_probs /= torch.sum(preprompt_generated_probs)


                    # Calculate primary output
                    primary_generated_tokens = {"input_ids": primary_generated_tokens, "attention_mask": torch.ones_like(primary_generated_tokens)}
                    primary_logits = primary_model(**primary_generated_tokens)["logits"][:, 0, -1, :]
                    primary_next_topk = torch.nn.functional.softmax(primary_logits, dim = 1).topk(args.beam_size, dim=1, largest = True, sorted = True)
                    primary_next_probs = primary_next_topk.values
                    primary_next_tokens = primary_next_topk.indices[..., None]
                    primary_candidate_probs = primary_next_probs * primary_generated_probs[:, None]
                    primary_candidate_probs = primary_candidate_probs.reshape(args.beam_size ** 2 if i > args.eval_start_tokens+1 else args.beam_size, -1).squeeze(dim=1)
                    primary_generated_tokens = primary_generated_tokens["input_ids"].repeat(1, args.beam_size, 1)
                    primary_candidate_tokens = torch.cat((primary_generated_tokens, primary_next_tokens), dim=2).reshape(args.beam_size ** 2 if i > args.eval_start_tokens+1 else args.beam_size, -1)
                    primary_candidate_topk = primary_candidate_probs.topk(args.beam_size, dim = 0, largest = True, sorted = True)
                    primary_generated_tokens = primary_candidate_tokens[primary_candidate_topk.indices, :][:, None, :]
                    primary_generated_probs = primary_candidate_probs[primary_candidate_topk.indices]
                    primary_generated_probs /= torch.sum(primary_generated_probs)

                    '''
                    # Calculate pos output
                    pos_generated_tokens = {"input_ids": pos_generated_tokens, "attention_mask": torch.ones_like(pos_generated_tokens)}
                    pos_expert_logits = expert_model(pos_generated_tokens, encodings, compute_loss = False)["logits"][:, 0, -1, :]
                    pos_next_topk = torch.nn.functional.softmax(pos_expert_logits, dim = 1).topk(args.beam_size, dim=1, largest = True, sorted = True)
                    pos_next_probs = pos_next_topk.values
                    pos_next_tokens = pos_next_topk.indices[..., None]
                    pos_candidate_probs = pos_next_probs * pos_generated_probs[:, None]
                    pos_candidate_probs = pos_candidate_probs.reshape(args.beam_size ** 2 if i > args.eval_start_tokens+1 else args.beam_size, -1).squeeze(dim=1)
                    pos_generated_tokens = pos_generated_tokens["input_ids"].repeat(1, args.beam_size, 1)
                    pos_candidate_tokens = torch.cat((pos_generated_tokens, pos_next_tokens), dim=2).reshape(args.beam_size ** 2 if i > args.eval_start_tokens+1 else args.beam_size, -1)
                    pos_candidate_topk = pos_candidate_probs.topk(args.beam_size, dim = 0, largest = True, sorted = True)
                    pos_generated_tokens = pos_candidate_tokens[pos_candidate_topk.indices, :][:, None, :]
                    pos_generated_probs = pos_candidate_probs[pos_candidate_topk.indices]
                    pos_generated_probs /= torch.sum(pos_generated_probs)
                    '''

                    # Calculate pos plus primary output
                    pos_plus_primary_generated_tokens = {"input_ids": pos_plus_primary_generated_tokens, "attention_mask": torch.ones_like(pos_plus_primary_generated_tokens)}
                    primary_logits = primary_model(**pos_plus_primary_generated_tokens)["logits"][:, 0, -1, :]
                    pos_expert_logits = expert_model(pos_plus_primary_generated_tokens, encodings, compute_loss = False)["logits"][:, 0, -1, :]
                    pos_plus_primary_next_topk = (torch.nn.functional.softmax(primary_logits, dim = 1) + args.expert_guidance * torch.nn.functional.softmax(pos_expert_logits, dim = 1)).topk(args.beam_size, dim=1, largest = True, sorted = True)
                    pos_plus_primary_next_probs = pos_plus_primary_next_topk.values
                    pos_plus_primary_next_tokens = pos_plus_primary_next_topk.indices[..., None]
                    pos_plus_primary_candidate_probs = pos_plus_primary_next_probs * pos_plus_primary_generated_probs[:, None]
                    pos_plus_primary_candidate_probs = pos_plus_primary_candidate_probs.reshape(args.beam_size ** 2 if i > args.eval_start_tokens+1 else args.beam_size, -1).squeeze(dim=1)
                    pos_plus_primary_generated_tokens = pos_plus_primary_generated_tokens["input_ids"].repeat(1, args.beam_size, 1)
                    pos_plus_primary_candidate_tokens = torch.cat((pos_plus_primary_generated_tokens, pos_plus_primary_next_tokens), dim=2).reshape(args.beam_size ** 2 if i > args.eval_start_tokens+1 else args.beam_size, -1)
                    pos_plus_primary_candidate_topk = pos_plus_primary_candidate_probs.topk(args.beam_size, dim = 0, largest = True, sorted = True)
                    pos_plus_primary_generated_tokens = pos_plus_primary_candidate_tokens[pos_plus_primary_candidate_topk.indices, :][:, None, :]
                    pos_plus_primary_generated_probs = pos_plus_primary_candidate_probs[pos_plus_primary_candidate_topk.indices]
                    pos_plus_primary_generated_probs /= torch.sum(pos_plus_primary_generated_probs)

                    # Calculate contrastive output
                    contrastive_generated_tokens = {"input_ids": contrastive_generated_tokens, "attention_mask": torch.ones_like(contrastive_generated_tokens)}
                    primary_logits = primary_model(**contrastive_generated_tokens)["logits"][:, 0, -1, :]
                    pos_expert_logits = expert_model(contrastive_generated_tokens, encodings, compute_loss = False)["logits"][:, 0, -1, :]
                    neg_expert_logits = expert_model(contrastive_generated_tokens, anti_encodings, compute_loss = False)["logits"][:, 0, -1, :]
                    contrastive_next_topk = (torch.nn.functional.softmax(primary_logits, dim = 1) + args.expert_guidance * (torch.nn.functional.softmax(pos_expert_logits, dim = 1) - torch.nn.functional.softmax(neg_expert_logits, dim = 1))).topk(args.beam_size, dim=1, largest = True, sorted = True)
                    contrastive_next_probs = contrastive_next_topk.values
                    contrastive_next_tokens = contrastive_next_topk.indices[..., None]
                    contrastive_candidate_probs = contrastive_next_probs * contrastive_generated_probs[:, None]
                    contrastive_candidate_probs = contrastive_candidate_probs.reshape(args.beam_size ** 2 if i > args.eval_start_tokens+1 else args.beam_size, -1).squeeze(dim=1)
                    contrastive_generated_tokens = contrastive_generated_tokens["input_ids"].repeat(1, args.beam_size, 1)
                    contrastive_candidate_tokens = torch.cat((contrastive_generated_tokens, contrastive_next_tokens), dim=2).reshape(args.beam_size ** 2 if i > args.eval_start_tokens+1 else args.beam_size, -1)
                    contrastive_candidate_topk = contrastive_candidate_probs.topk(args.beam_size, dim = 0, largest = True, sorted = True)
                    contrastive_generated_tokens = contrastive_candidate_tokens[contrastive_candidate_topk.indices, :][:, None, :]
                    contrastive_generated_probs = contrastive_candidate_probs[contrastive_candidate_topk.indices]
                    contrastive_generated_probs /= torch.sum(contrastive_generated_probs)
                
                preprompt_prediction = expert_model.tokenizer.batch_decode(preprompt_generated_tokens[0, :, 16:])[0]
                primary_prediction = expert_model.tokenizer.batch_decode(primary_generated_tokens[0, :, :])[0]
                contrastive_prediction = expert_model.tokenizer.batch_decode(contrastive_generated_tokens[0, :, :])[0]
                pos_prediction = expert_model.tokenizer.batch_decode(pos_plus_primary_generated_tokens[0, :, :])[0]
                reference = expert_model.tokenizer.decode(input_ids[0, 0, :])

                rouge = Rouge()
                preprompt_rouge_scores = rouge.get_scores(preprompt_prediction, reference, avg=True)
                primary_rouge_scores = rouge.get_scores(primary_prediction, reference, avg=True)
                contrastive_rouge_scores = rouge.get_scores(contrastive_prediction, reference, avg=True)
                pos_rouge_scores = rouge.get_scores(pos_prediction, reference, avg=True)

                preprompt_bleu_1_scores = sentence_bleu([reference.split()], preprompt_prediction.split(), weights=(1, 0, 0, 0))
                primary_bleu_1_scores = sentence_bleu([reference.split()], primary_prediction.split(), weights=(1, 0, 0, 0))
                contrastive_bleu_1_scores = sentence_bleu([reference.split()], contrastive_prediction.split(), weights=(1, 0, 0, 0))
                pos_bleu_1_scores = sentence_bleu([reference.split()], pos_prediction.split(), weights=(1, 0, 0, 0))

                preprompt_bleu_2_scores = sentence_bleu([reference.split()], preprompt_prediction.split(), weights=(0, 1, 0, 0))
                primary_bleu_2_scores = sentence_bleu([reference.split()], primary_prediction.split(), weights=(0, 1, 0, 0))
                contrastive_bleu_2_scores = sentence_bleu([reference.split()], contrastive_prediction.split(), weights=(0, 1, 0, 0))
                pos_bleu_2_scores = sentence_bleu([reference.split()], pos_prediction.split(), weights=(0, 1, 0, 0))

                rouge = Rouge()
                preprompt_scores = rouge.get_scores(preprompt_prediction, reference, avg=True)
                primary_scores = rouge.get_scores(primary_prediction, reference, avg=True)
                contrastive_scores = rouge.get_scores(contrastive_prediction, reference, avg=True)
                pos_scores = rouge.get_scores(pos_prediction, reference, avg=True)

                preprompt_perplexity = compute_perplexity(primary_model, preprompt_generated_tokens[0, :, 16:]).item()
                primary_perplexity = compute_perplexity(primary_model, primary_generated_tokens[0, :, :]).item()
                contrastive_perplexity = compute_perplexity(primary_model, contrastive_generated_tokens[0, :, :]).item()
                pos_perplexity = compute_perplexity(primary_model, pos_plus_primary_generated_tokens[0, :, :]).item()
                reference_perplexity = compute_perplexity(primary_model, input_ids[0, :, :]).item()

                summary_results["perplexity"]["preprompt"].append(preprompt_perplexity)
                summary_results["perplexity"]["primary"].append(primary_perplexity)
                summary_results["perplexity"]["contrastive"].append(contrastive_perplexity)
                summary_results["perplexity"]["pos"].append(pos_perplexity)
                summary_results["perplexity"]["reference"].append(reference_perplexity)

                summary_results["bleu-1"]["preprompt"].append(preprompt_bleu_1_scores)
                summary_results["bleu-1"]["primary"].append(primary_bleu_1_scores)
                summary_results["bleu-1"]["contrastive"].append(contrastive_bleu_1_scores)
                summary_results["bleu-1"]["pos"].append(pos_bleu_1_scores)

                summary_results["bleu-2"]["preprompt"].append(preprompt_bleu_2_scores)
                summary_results["bleu-2"]["primary"].append(primary_bleu_2_scores)
                summary_results["bleu-2"]["contrastive"].append(contrastive_bleu_2_scores)
                summary_results["bleu-2"]["pos"].append(pos_bleu_2_scores)

                for key in primary_rouge_scores:
                    for sub_key in primary_rouge_scores[key]:
                        summary_results[key + "-" + sub_key]["preprompt"].append(preprompt_rouge_scores[key][sub_key])
                        summary_results[key + "-" + sub_key]["primary"].append(primary_rouge_scores[key][sub_key])
                        summary_results[key + "-" + sub_key]["contrastive"].append(contrastive_rouge_scores[key][sub_key])
                        summary_results[key + "-" + sub_key]["pos"].append(pos_rouge_scores[key][sub_key])
    #print(summary_results)
    for key in summary_results:
        for sub_key in summary_results[key]:
            summary_results[key][sub_key] = np.mean(summary_results[key][sub_key])
    with open(str(args.log_path / f"{args.exp_name}.json"), "w") as fileref:
        fileref.write(json.dumps(summary_results, indent = 4))
    #print(summary_results)

                

                


                
                
                





            