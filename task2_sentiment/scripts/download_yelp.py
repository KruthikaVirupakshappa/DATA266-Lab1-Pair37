from pathlib import Path
from datasets import load_dataset

# find task2_sentiment folder
current_file = Path(__file__).resolve()
scripts_folder = current_file.parent
task2_sentiment = scripts_folder.parent

# find data folder
data_folder = task2_sentiment / "data"

# dataset will be saved here
dataset_folder = data_folder / "yelp_polarity"

# create data folder if it does not exist
data_folder.mkdir(parents=True, exist_ok=True)

# download only if dataset is not already saved
if dataset_folder.is_dir():
    print("Dataset already exists.")
else:
    dataset = load_dataset("fancyzhx/yelp_polarity")
    dataset.save_to_disk(dataset_folder)
    print("Download complete.")
