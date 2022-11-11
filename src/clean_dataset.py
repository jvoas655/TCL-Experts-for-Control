import h5py
from utils.parsers import DataCleaner
from tqdm import tqdm
import shutil



if __name__ == "__main__":
    args = DataCleaner.parse_args()
    source_data = h5py.File(args.source_path, "r")
    with h5py.File(args.target_path, "w") as target_data:
        for split in source_data.keys():
            print(split)
            split_grp = target_data.create_group(split)
            for cat in source_data[split].keys():
                if (cat not in args.keep_cats):
                    continue
                else:
                    print(" " * 3, cat)
                    cat_grp = split_grp.create_group(cat)
                    for key in tqdm(source_data[split][cat].keys(), total = len(source_data[split][cat].keys())):
                        cat_grp.create_dataset(key, data=source_data[split][cat][key][()])
