from utils.paths import *
import argparse



WebScraperParser = argparse.ArgumentParser(description = "Process arguments for data gathering web scraper")
WebScraperParser.add_argument("--output", "-o", default = DATA_PATH / "category_text_pairs_xl", help = "File path to output results")


CatEmbExtractorParser = argparse.ArgumentParser(description = "Process arguments for extration of raw category embeddings")
CatEmbExtractorParser.add_argument("--path", "-p", default = DATA_PATH / "category_text_pairs_xl.hdf5", help = "File path to raw data which will have the embeddings added into")
CatEmbExtractorParser.add_argument("--threads", "-t", default = 1, type = int, help = "Number of threads to utilize")
CatEmbExtractorParser.add_argument("--batch_size", "-b", default = [256, 128, 256, 256, 128], nargs='+', help = "Number of samples per batch")
CatEmbExtractorParser.add_argument("--device", "-d", default = 0, type = int, help = "Device to utilize (-1 = CPU) (> - 1 = GPU)")
CatEmbExtractorParser.add_argument("--max_tokens_ind", default = 5, type = int, help = "Max number of tokens for gen_type individual")
CatEmbExtractorParser.add_argument("--max_tokens_con", default = 20, type = int, help = "Max number of tokens for gen_type concat")
CatEmbExtractorParser.add_argument("--model", "-m", default = ["roberta-base", "roberta-large", "t5-small", "t5-base", "t5-large"], nargs='+', choices = ["roberta-base", "roberta-large", "t5-small", "t5-base", "t5-large", "t5-3b"], help = "Base model to utilize for extraction of embeddings")
CatEmbExtractorParser.add_argument("--gen_type", "-g", default = ["individual", "concat"], help = "Which types of embeddings to produce")


AETrainingParser = argparse.ArgumentParser(description = "Process arguments for training of the category dimensional reduction model")
AETrainingParser.add_argument("--path", "-p", default = DATA_PATH / "category_text_pairs_xl.hdf5", help = "File path to raw data which will have the embeddings added into")
AETrainingParser.add_argument("--batch_size", "-b", default = 512, help = "Number of samples per batch")
AETrainingParser.add_argument("--val_epochs", "-v", default = 10, help = "Number of epochs between each val step")
AETrainingParser.add_argument("--device", "-d", default = 0, type = int, help = "Device to utilize (-1 = CPU) (> - 1 = GPU)")
AETrainingParser.add_argument("--max_depth", "-md", default = 3, help = "Max Depth of the autoencoder (will create a encoder for output size [1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1] respectively")
AETrainingParser.add_argument("--learning_rate", "-lr", default = 1e-5, help = "Model learning rate")
AETrainingParser.add_argument("--increment_scale", default = 1.03, help = "")
AETrainingParser.add_argument("--increment_schedule", default = "val", choices = ["val", "epoch"], help = "Whether the model will increment depth based on val results or fixed epoch steps")
AETrainingParser.add_argument("--increment_steps", default = 20, help = "Increment depth for this number of val stpes, if increment_schedule is ephoch")
AETrainingParser.add_argument("--increment_guard", default = 3, help = "Increment depth if val results has not increased in this number of val steps, if increment_schedule is val")
AETrainingParser.add_argument("--l1_lambda", "-l1s", default = 0.0001, help = "Scaling factor for computing L1 sparsity loss")
AETrainingParser.add_argument("--l2_lambda", "-l2s", default = 0.0001, help = "Scaling factor for computing L2 sparsity loss")
AETrainingParser.add_argument("--sparsity_target", "-st", default = "activations", choices = ["parameters", "activations"], help = "Whether sparsity constraints are applied to parameters or hidden activations")
AETrainingParser.add_argument("--loss", "-l", default = "MSE", choices = ["MSE", "L1", "L2"], help = "Loss to be used on reconstruction results")
AETrainingParser.add_argument("--activation", "-a", default = "ReLU", choices = ["ReLU", "LeakyReLU"], help = "Model activation results")
AETrainingParser.add_argument("--checkpoint", "-ck", default = "", help = "Checkpoint to load")
AETrainingParser.add_argument("--start_depth", "-sd", default = 1, help = "Depth to start at")
AETrainingParser.add_argument("--log_path", "-lp", default = LOG_PATH / "autoencoder", help = "Directory to store checkpoint and log results to")
AETrainingParser.add_argument("--target_category", "-tc", default = 'raw_cat_embeddings_ind_roberta_large', help = "Which set of categories to train for")
AETrainingParser.add_argument("--exp_name", "-en", default = "roberta_large_ae", help = "Experiment name to use for logging and checkpointing")



