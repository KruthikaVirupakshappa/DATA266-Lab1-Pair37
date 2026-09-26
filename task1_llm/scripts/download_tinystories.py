from pathlib import Path
from datasets import load_dataset

# find task1_llm folder
current_file = Path(__file__).resolve()
scripts_folder = current_file.parent
task1_llm = scripts_folder.parent
# find data folder in task1_llm
data_folder = task1_llm / "data"
# find tinystories folder in task1_llm/data
dataset_folder = data_folder / "tinystories"

# create data folder if it does not exist
data_folder.mkdir(parents=True, exist_ok=True)

# check if tinystories exist, if not: download the dataset from hf and save it to the folder.
if dataset_folder.is_dir():
    print("Dataset already exists.")
else:
    dataset = load_dataset("roneneldan/TinyStories")
    dataset.save_to_disk(dataset_folder)
    print("Download complete")
