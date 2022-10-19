from utils.paths import *
import argparse



WebScraperParser = argparse.ArgumentParser(description = "Process arguments for data gathering web scraper")
WebScraperParser.add_argument("--output", "-o", default = DATA_PATH / "category_text_pairs_xl", help = "File path to output results")


CatEmbExtractorParser = argparse.ArgumentParser(description = "Process arguments for data gathering web scraper")
CatEmbExtractorParser.add_argument("--path", "-p", default = DATA_PATH / "category_text_pairs_xl.hdf5", help = "File path to raw data which will have the embeddings added into")
CatEmbExtractorParser.add_argument("--threads", "-t", default = 1, type = int, help = "Number of threads to utilize")
CatEmbExtractorParser.add_argument("--batch_size", "-b", default = [256, 128, 256, 256, 128], nargs='+', help = "Number of samples per batch")
CatEmbExtractorParser.add_argument("--device", "-d", default = 0, type = int, help = "Device to utilize (-1 = CPU) (> - 1 = GPU)")
CatEmbExtractorParser.add_argument("--max_tokens_ind", default = 5, type = int, help = "Max number of tokens for gen_type individual")
CatEmbExtractorParser.add_argument("--max_tokens_con", default = 20, type = int, help = "Max number of tokens for gen_type concat")
CatEmbExtractorParser.add_argument("--model", "-m", default = ["roberta-base", "roberta-large", "t5-small", "t5-base", "t5-large"], nargs='+', choices = ["roberta-base", "roberta-large", "t5-small", "t5-base", "t5-large", "t5-3b"], help = "Base model to utilize for extraction of embeddings")
CatEmbExtractorParser.add_argument("--gen_type", "-g", default = ["concat"], help = "Which types of embeddings to produce")