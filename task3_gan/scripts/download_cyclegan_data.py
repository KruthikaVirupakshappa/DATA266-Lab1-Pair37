from pathlib import Path
import shutil
import sys

import kagglehub
from kagglehub.exceptions import UnauthenticatedError


COMPETITION = "data-266-fall-2026-gan-image-style-transfer"

# task3_gan/
TASK3_DIR = Path(__file__).resolve().parents[1]

# task3_gan/data/
DATA_DIR = TASK3_DIR / "data"
MONET_DIR = DATA_DIR / "monet_jpg"
PHOTO_DIR = DATA_DIR / "photo_jpg"

DATA_DIR.mkdir(parents=True, exist_ok=True)


# Skip download if dataset is already prepared
if MONET_DIR.is_dir() and PHOTO_DIR.is_dir():
    print("Task 3 dataset already exists.")
    print("Monet:", MONET_DIR)
    print("Photo:", PHOTO_DIR)
    sys.exit(0)


# Download competition data using the existing Kaggle authentication
try:
    download_path = Path(
        kagglehub.competition_download(COMPETITION)
    )
except UnauthenticatedError:
    print(
        "Kaggle authentication is required.\n"
        "Please authenticate with a Kaggle account that has access to "
        "the DATA 266 class competition, then run this script again."
    )
    sys.exit(1)


# Actual directory structure inside the Kaggle competition download
SOURCE_DIR = download_path / "dataset" / "dataset"

MONET_SOURCE = SOURCE_DIR / "monet_jpg"
PHOTO_SOURCE = SOURCE_DIR / "photo_jpg"


# Check that expected folders exist
if not MONET_SOURCE.is_dir() or not PHOTO_SOURCE.is_dir():
    print("Could not find the expected Monet/Photo folders.")
    print("Downloaded data:", download_path)
    sys.exit(1)

shutil.copytree(MONET_SOURCE, MONET_DIR, dirs_exist_ok=True)
shutil.copytree(PHOTO_SOURCE, PHOTO_DIR, dirs_exist_ok=True)

print("Task 3 dataset prepared successfully.")
print("Monet:", MONET_DIR)
print("Photo:", PHOTO_DIR)
