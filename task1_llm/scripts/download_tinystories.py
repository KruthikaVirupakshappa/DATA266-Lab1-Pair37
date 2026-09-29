"""Download the TinyStories dataset from Hugging Face into task1_llm/data/tinystories/.

It keeps Hugging Face's two official splits:
    train       ~2.1M stories -> we sample our 100K training stories from here
    validation  ~22K stories  -> we sample our 10K validation stories from here
Because they are separate splits, validation stories are never used for training.

Usage (from the repo root):
    python task1_llm/scripts/download_tinystories.py
"""
from pathlib import Path

from datasets import load_dataset

# Pinned dataset version, so every team member gets exactly the same data.
REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"

# find task1_llm folder
current_file = Path(__file__).resolve()
scripts_folder = current_file.parent
task1_llm = scripts_folder.parent
# find data folder in task1_llm
data_folder = task1_llm / "data"
# find tinystories folder in task1_llm/data
dataset_folder = data_folder / "tinystories"


def download_tinystories():
    """Download and save the dataset if it is not there yet; return its folder."""
    # create data folder if it does not exist
    data_folder.mkdir(parents=True, exist_ok=True)

    # check if tinystories exist, if not: download the dataset from hf and save it to the folder.
    if dataset_folder.is_dir():
        print("Dataset already exists.")
    else:
        dataset = load_dataset("roneneldan/TinyStories", revision=REVISION)
        dataset.save_to_disk(dataset_folder)
        print("Download complete")
        for split, part in dataset.items():
            print(f"  {split}: {len(part):,} stories")
    return dataset_folder


if __name__ == "__main__":
    download_tinystories()
