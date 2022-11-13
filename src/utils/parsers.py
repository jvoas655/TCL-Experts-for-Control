from utils.paths import *
import argparse



WebScraperParser = argparse.ArgumentParser(description = "Process arguments for data gathering web scraper")
WebScraperParser.add_argument("--output", "-o", default = DATA_PATH / "category_text_pairs_xl", help = "File path to output results")


CatEmbExtractorParser = argparse.ArgumentParser(description = "Process arguments for extration of raw category embeddings")
CatEmbExtractorParser.add_argument("--path", "-p", default = DATA_PATH / "category_text_pairs_xl.hdf5", help = "File path to raw data which will have the embeddings added into")
CatEmbExtractorParser.add_argument("--threads", "-t", default = 1, type = int, help = "Number of threads to utilize")
CatEmbExtractorParser.add_argument("--batch_size", "-b", default = [64, 32], nargs='+', help = "Number of samples per batch")
CatEmbExtractorParser.add_argument("--device", "-d", default = 0, type = int, help = "Device to utilize (-1 = CPU) (> - 1 = GPU)")
CatEmbExtractorParser.add_argument("--max_tokens_ind", default = 12, type = int, help = "Max number of tokens for gen_type individual")
CatEmbExtractorParser.add_argument("--max_tokens_con", default = 40, type = int, help = "Max number of tokens for gen_type concat")
CatEmbExtractorParser.add_argument("--use_last_n_hidden", default = 1, type = int, help = "Number of last hidden layers to use")
CatEmbExtractorParser.add_argument("--con_rand_perms", default = 4, type = int, help = "Randomly sample this number of permutations for concatenation")
CatEmbExtractorParser.add_argument("--model", "-m", default = ["roberta-large", "t5-large"], nargs='+', choices = ["roberta-base", "roberta-large", "t5-small", "t5-base", "t5-large", "t5-3b"], help = "Base model to utilize for extraction of embeddings")
CatEmbExtractorParser.add_argument("--gen_type", "-g", default = ["individual", "concat"], choices = ["individual", "concat"], help = "Which types of embeddings to produce")


AETrainingParser = argparse.ArgumentParser(description = "Process arguments for training of the category dimensional reduction model")
AETrainingParser.add_argument("--path", "-p", default = DATA_PATH / "category_text_pairs_xl.hdf5", help = "File path to raw data which will have the embeddings added into")
AETrainingParser.add_argument("--batch_size", "-b", default = 512, help = "Number of samples per batch")
AETrainingParser.add_argument("--val_epochs", "-v", default = 10, help = "Number of epochs between each val step")
AETrainingParser.add_argument("--device", "-d", default = 0, type = int, help = "Device to utilize (-1 = CPU) (> - 1 = GPU)")
AETrainingParser.add_argument("--max_depth", "-md", default = [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]], help = "Max Depth of the autoencoder (will create a encoder for output size [1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1] respectively")
AETrainingParser.add_argument("--learning_rate", "-lr", default = 1e-5, help = "Model learning rate")
AETrainingParser.add_argument("--increment_scale", default = 1.05, help = "")
AETrainingParser.add_argument("--increment_schedule", default = "val", choices = ["val", "epoch"], help = "Whether the model will increment depth based on val results or fixed epoch steps")
AETrainingParser.add_argument("--increment_steps", default = 20, help = "Increment depth for this number of val stpes, if increment_schedule is ephoch")
AETrainingParser.add_argument("--increment_guard", default = 5, help = "Increment depth if val results has not increased in this number of val steps, if increment_schedule is val")
AETrainingParser.add_argument("--l1_lambda", "-l1s", default = 0.0001, help = "Scaling factor for computing L1 sparsity loss")
AETrainingParser.add_argument("--l2_lambda", "-l2s", default = 0.0001, help = "Scaling factor for computing L2 sparsity loss")
AETrainingParser.add_argument("--sparsity_target", "-st", default = "activations", choices = ["parameters", "activations"], help = "Whether sparsity constraints are applied to parameters or hidden activations")
AETrainingParser.add_argument("--loss", "-l", default = "MSE", choices = ["MSE", "L1"], help = "Loss to be used on reconstruction results")
AETrainingParser.add_argument("--activation", "-a", default = "ReLU", choices = ["ReLU", "LeakyReLU"], help = "Model activation results")
AETrainingParser.add_argument("--checkpoint", "-ck", default = "", help = "Checkpoint to load")
AETrainingParser.add_argument("--start_depth", "-sd", default = [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]], help = "Depth to start at")
AETrainingParser.add_argument("--log_path", "-lp", default = LOG_PATH / "autoencoder", help = "Directory to store checkpoint and log results to")
AETrainingParser.add_argument("--target_category", "-tc", default = ['raw_cat_embeddings_ind_roberta_large', 'raw_cat_embeddings_ind_t5_large', 'raw_cat_embeddings_con_roberta_large', 'raw_cat_embeddings_con_t5_large'], help = "Which set of categories to train for")
AETrainingParser.add_argument("--exp_name", "-en", default = [
                                                                ["ind_roberta_large_512_ae", "ind_roberta_large_256_ae", "ind_roberta_large_128_ae"], 
                                                                ["ind_t5_large_512_ae", "ind_t5_large_256_ae", "ind_t5_large_128_ae"],
                                                                ["con_roberta_large_512_ae", "con_roberta_large_256_ae", "con_roberta_large_128_ae"], 
                                                                ["con_t5_large_512_ae", "con_t5_large_256_ae", "con_t5_large_128_ae"]
                                                            ], help = "Experiment name to use for logging and checkpointing")



RedCatEmbExtractorParser = argparse.ArgumentParser(description = "Process arguments for extration of reduced category embeddings")
RedCatEmbExtractorParser.add_argument("--data_path", "-p", default = DATA_PATH / "category_text_pairs_xl.hdf5", help = "File path to raw data which will have the embeddings added into")
RedCatEmbExtractorParser.add_argument("--log_path", "-lp", default = LOG_PATH / "autoencoder", help = "Directory to store checkpoint and log results to")
RedCatEmbExtractorParser.add_argument("--target_category", "-tc", default = ['raw_cat_embeddings_ind_roberta_large', 'raw_cat_embeddings_ind_t5_large', 'raw_cat_embeddings_con_roberta_large', 'raw_cat_embeddings_con_t5_large'], help = "Which set of categories to reduce")
RedCatEmbExtractorParser.add_argument("--encoders", "-en", default = [
                                                                ["ind_roberta_large_512_ae", "ind_roberta_large_256_ae", "ind_roberta_large_128_ae"], 
                                                                ["ind_t5_large_512_ae", "ind_t5_large_256_ae", "ind_t5_large_128_ae"],
                                                                ["con_roberta_large_512_ae", "con_roberta_large_256_ae", "con_roberta_large_128_ae"], 
                                                                ["con_t5_large_512_ae", "con_t5_large_256_ae", "con_t5_large_128_ae"]
                                                            ], help = "Encoders to process each embedding with")
RedCatEmbExtractorParser.add_argument("--batch_size", "-b", default = 512, help = "Number of samples per batch")
RedCatEmbExtractorParser.add_argument("--encoder_depth", default = [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]], help = "Depth of encoder")
RedCatEmbExtractorParser.add_argument("--input_size", default = [1024, 1024, 1024, 1024], help = "Input size of encoder")
RedCatEmbExtractorParser.add_argument("--device", "-d", default = 0, type = int, help = "Device to utilize (-1 = CPU) (> - 1 = GPU)")


SOMTrainingParser = argparse.ArgumentParser(description = "Process arguments for training of the topic SOM")
SOMTrainingParser.add_argument("--path", "-p", default = DATA_PATH / "category_text_pairs_xl.hdf5", help = "File path to raw data which will have the embeddings added into")
SOMTrainingParser.add_argument("--val_epochs", "-v", default = 1, help = "Number of epochs between each val step")
SOMTrainingParser.add_argument("--train_epochs", "-e", default = 40, help = "Number of epochs (cycles through training data) to train for")
SOMTrainingParser.add_argument("--device", "-d", default = 0, type = int, help = "Device to utilize (-1 = CPU) (> - 1 = GPU)")
SOMTrainingParser.add_argument("--som_params", default = (4, 4), help = "Parameters to initialize SOM with")
SOMTrainingParser.add_argument("--learning_rate", "-lr", default = 1e-4, help = "Model learning rate")
SOMTrainingParser.add_argument("--base_sigma", default = 2, help = "Model base sigma to for distance scaling")
SOMTrainingParser.add_argument("--lr_decay_rate", default = 8, help = "Number of epoch for which the LR will decay")
SOMTrainingParser.add_argument("--sigma_decay_rate", default = 2, help = "Number of epoch for which the kernel standard deviation will decay")
SOMTrainingParser.add_argument("--som_metric", default = "dist", choices = ["dist", "ang"], help = "Metric to scale based on")
SOMTrainingParser.add_argument("--checkpoint", "-ck", default = "", help = "Checkpoint to load")
SOMTrainingParser.add_argument("--log_path", "-lp", default = LOG_PATH / "som", help = "Directory to store checkpoint and log results to")
SOMTrainingParser.add_argument("--target_category", "-tc", default = [
                                                                'raw_cat_embeddings_ind_roberta_large',
                                                                "ind_roberta_large_512_ae", 
                                                                "ind_roberta_large_256_ae", 
                                                                "ind_roberta_large_128_ae",
                                                                'raw_cat_embeddings_ind_t5_large', 
                                                                "ind_t5_large_512_ae", 
                                                                "ind_t5_large_256_ae", 
                                                                "ind_t5_large_128_ae",
                                                                'raw_cat_embeddings_con_roberta_large', 
                                                                "con_roberta_large_512_ae", 
                                                                "con_roberta_large_256_ae", 
                                                                "con_roberta_large_128_ae",
                                                                'raw_cat_embeddings_con_t5_large',
                                                                "con_t5_large_512_ae", 
                                                                "con_t5_large_256_ae", 
                                                                "con_t5_large_128_ae"
                                                                ], help = "Which set of categories to train for")
                                                            
TopicEncodingParser = argparse.ArgumentParser(description = "Process arguments for training of the topic SOM")
TopicEncodingParser.add_argument("--data_path", "-p", default = DATA_PATH / "category_text_pairs_xl.hdf5", help = "File path to raw data which will have the embeddings added into")
TopicEncodingParser.add_argument("--device", "-d", default = 0, type = int, help = "Device to utilize (-1 = CPU) (> - 1 = GPU)")
TopicEncodingParser.add_argument("--num_samples", "-n", default = 3, type = int, help = "Number of encoded values to use per category")
TopicEncodingParser.add_argument("--checkpoint_folder", default = "som", help = "Checkpoint parent folder")
TopicEncodingParser.add_argument("--checkpoint_epoch", default = 40, help = "Checkpoint epoch to use")
TopicEncodingParser.add_argument("--anti_metric", default = "som", choices = ["som", "vals"], help = "Metric to invert encoding based on")
TopicEncodingParser.add_argument("--target_category", "-tc", default = [
                                                                'raw_cat_embeddings_ind_roberta_large',
                                                                "ind_roberta_large_512_ae", 
                                                                "ind_roberta_large_256_ae", 
                                                                "ind_roberta_large_128_ae",
                                                                'raw_cat_embeddings_ind_t5_large', 
                                                                "ind_t5_large_512_ae", 
                                                                "ind_t5_large_256_ae", 
                                                                "ind_t5_large_128_ae",
                                                                'raw_cat_embeddings_con_roberta_large', 
                                                                "con_roberta_large_512_ae", 
                                                                "con_roberta_large_256_ae", 
                                                                "con_roberta_large_128_ae",
                                                                'raw_cat_embeddings_con_t5_large',
                                                                "con_t5_large_512_ae", 
                                                                "con_t5_large_256_ae", 
                                                                "con_t5_large_128_ae"
                                                                ], help = "Which set of categories to train for")
TopicEncodingParser.add_argument("--save_groups", default = [
                                                                'ind_roberta_large_1024_encoding',
                                                                'ind_roberta_large_512_encoding', 
                                                                'ind_roberta_large_256_encoding', 
                                                                'ind_roberta_large_128_encoding',
                                                                'ind_t5_large_1024_encoding',
                                                                'ind_t5_large_512_encoding', 
                                                                'ind_t5_large_256_encoding', 
                                                                'ind_t5_large_128_encoding',
                                                                'con_roberta_large_1024_encoding',
                                                                'con_roberta_large_512_encoding', 
                                                                'con_roberta_large_256_encoding', 
                                                                'con_roberta_large_128_encoding',
                                                                'con_t5_large_1024_encoding',
                                                                'con_t5_large_512_encoding', 
                                                                'con_t5_large_256_encoding', 
                                                                'con_t5_large_128_encoding',
                                                                ], help = "Names to store each encoding under")


DataCleaner = argparse.ArgumentParser(description = "Process arguments for training of the topic SOM")
DataCleaner.add_argument("--source_path", default = DATA_PATH / "category_text_pairs_xl.hdf5", help = "File path for data to duplicate with a clean version")
DataCleaner.add_argument("--target_path", default = DATA_PATH / "text_encoding_pairs.hdf5", help = "File path of duplicated and cleaned data")
DataCleaner.add_argument("--keep_cats", default = [
                                                                "text",
                                                                'ind_roberta_large_1024_encoding',
                                                                'ind_roberta_large_512_encoding', 
                                                                'ind_roberta_large_256_encoding', 
                                                                'ind_roberta_large_128_encoding',
                                                                'ind_t5_large_1024_encoding',
                                                                'ind_t5_large_512_encoding', 
                                                                'ind_t5_large_256_encoding', 
                                                                'ind_t5_large_128_encoding',
                                                                'con_roberta_large_1024_encoding',
                                                                'con_roberta_large_512_encoding', 
                                                                'con_roberta_large_256_encoding', 
                                                                'con_roberta_large_128_encoding',
                                                                'con_t5_large_1024_encoding',
                                                                'con_t5_large_512_encoding', 
                                                                'con_t5_large_256_encoding', 
                                                                'con_t5_large_128_encoding',
                                                                'ind_roberta_large_1024_encoding_anti',
                                                                'ind_roberta_large_512_encoding_anti', 
                                                                'ind_roberta_large_256_encoding_anti', 
                                                                'ind_roberta_large_128_encoding_anti',
                                                                'ind_t5_large_1024_encoding_anti',
                                                                'ind_t5_large_512_encoding_anti', 
                                                                'ind_t5_large_256_encoding_anti', 
                                                                'ind_t5_large_128_encoding_anti',
                                                                'con_roberta_large_1024_encoding_anti',
                                                                'con_roberta_large_512_encoding_anti', 
                                                                'con_roberta_large_256_encoding_anti', 
                                                                'con_roberta_large_128_encoding_anti',
                                                                'con_t5_large_1024_encoding_anti',
                                                                'con_t5_large_512_encoding_anti', 
                                                                'con_t5_large_256_encoding_anti', 
                                                                'con_t5_large_128_encoding_anti',
                                                                ], help = "Names to store each encoding under")


TopicPredictorTrainingParser = argparse.ArgumentParser(description = "Process arguments for training of the text based topic predictor")
TopicPredictorTrainingParser.add_argument("--path", "-p", default = DATA_PATH / "category_text_pairs_xl.hdf5", help = "File path to raw data which will have the embeddings added into")
TopicPredictorTrainingParser.add_argument("--batch_size", "-b", default = 16, help = "Number of samples per batch")
TopicPredictorTrainingParser.add_argument("--epochs", "-e", default = 20, help = "Number of epochs to train for")
TopicPredictorTrainingParser.add_argument("--val_epochs", "-ve", default = 1, help = "Number of epochs between each val step")
TopicPredictorTrainingParser.add_argument("--device", "-d", default = 0, type = int, help = "Device to utilize (-1 = CPU) (> - 1 = GPU)")
TopicPredictorTrainingParser.add_argument("--reduction_dim", default = 256, help = "")
TopicPredictorTrainingParser.add_argument("--base_model", default = "roberta-large", help = "")
TopicPredictorTrainingParser.add_argument("--token_count", default=64, help = "")
TopicPredictorTrainingParser.add_argument("--disable_learn_token_reducer", action="store_true", help = "")
TopicPredictorTrainingParser.add_argument("--single_adapter", action="store_true", help = "")
TopicPredictorTrainingParser.add_argument("--disable_finetune_base", action="store_true", help = "")
TopicPredictorTrainingParser.add_argument("--disable_use_adapter", action="store_true", help = "")
TopicPredictorTrainingParser.add_argument("--threshold_value", default=0.03, help = "")
TopicPredictorTrainingParser.add_argument("--learning_rate", "-lr", default = 1e-5, help = "Model learning rate")
TopicPredictorTrainingParser.add_argument("--l1_lambda", "-l1s", default = 1e-6, help = "Scaling factor for computing L1 sparsity loss")
TopicPredictorTrainingParser.add_argument("--l2_lambda", "-l2s", default = 1e-6, help = "Scaling factor for computing L2 sparsity loss")
TopicPredictorTrainingParser.add_argument("--sparsity_target", "-st", default = "activations", choices = ["parameters", "activations"], help = "Whether sparsity constraints are applied to parameters or hidden activations")
TopicPredictorTrainingParser.add_argument("--loss", "-l", default = "L1|L2", choices = ["L2", "L1", "L1|L2"], help = "Loss to be used on reconstruction results")
TopicPredictorTrainingParser.add_argument("--checkpoint", "-ck", default = "", help = "Checkpoint to load")
TopicPredictorTrainingParser.add_argument("--log_path", "-lp", default = LOG_PATH / "topic_predictor", help = "Directory to store checkpoint and log results to")
TopicPredictorTrainingParser.add_argument("--target_category", "-tc", default = [
                                                                #'ind_roberta_large_1024_encoding',
                                                                #'ind_roberta_large_512_encoding', 
                                                                #'ind_roberta_large_256_encoding', 
                                                                #'ind_roberta_large_128_encoding',
                                                                #'ind_t5_large_1024_encoding',
                                                                'ind_t5_large_512_encoding', 
                                                                #'ind_t5_large_256_encoding', 
                                                                #'ind_t5_large_128_encoding',
                                                                #'con_roberta_large_1024_encoding',
                                                                #'con_roberta_large_512_encoding', 
                                                                #'con_roberta_large_256_encoding', 
                                                                #'con_roberta_large_128_encoding',
                                                                #'con_t5_large_1024_encoding',
                                                                #'con_t5_large_512_encoding', 
                                                                #'con_t5_large_256_encoding', 
                                                                #'con_t5_large_128_encoding',
                                                                ], help = "Which set of categories to train for")
                            