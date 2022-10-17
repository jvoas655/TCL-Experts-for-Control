from utils.paths import *
import argparse



WebScraperParser = argparse.ArgumentParser(description = "Process arguments for data gathering web scraper")
WebScraperParser.add_argument("--output", "-o", default = DATA_PATH / "category_text_pairs_xl", help = "File path to output results")
