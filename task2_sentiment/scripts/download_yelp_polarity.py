"""Download the Yelp Polarity dataset from Hugging Face into task2_sentiment/data/yelp_polarity/.

Official splits: train 560,000 reviews, test 38,000 reviews (both perfectly balanced).
Labels: 0 = negative (1-2 stars), 1 = positive (3-4 stars), as defined by the dataset authors.

Usage (from the repo root):
    python task2_sentiment/scripts/download_yelp_polarity.py
"""
from pathlib import Path

from datasets import load_dataset

# Pinned dataset version, so every team member gets exactly the same data.
REVISION = "bbf1c97a1f0cf005e5aded43839fd814654a1557"

# find task2_sentiment folder
current_file = Path(__file__).resolve()
scripts_folder = current_file.parent
task2 = scripts_folder.parent
# find data folder in task2_sentiment
data_folder = task2 / "data"
# find yelp_polarity folder in task2_sentiment/data
dataset_folder = data_folder / "yelp_polarity"


def download_yelp_polarity():
    """Download and save the dataset if it is not there yet; return its folder."""
    data_folder.mkdir(parents=True, exist_ok=True)
    if dataset_folder.is_dir():
        print("Dataset already exists.")
    else:
        dataset = load_dataset("fancyzhx/yelp_polarity", revision=REVISION)
        dataset.save_to_disk(dataset_folder)
        print("Download complete")
        for split, part in dataset.items():
            print(f"  {split}: {len(part):,} reviews")
    return dataset_folder


if __name__ == "__main__":
    download_yelp_polarity()
