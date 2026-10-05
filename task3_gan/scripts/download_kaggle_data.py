"""Download the class Kaggle competition data into task3_gan/data/.

Result:
    task3_gan/data/monet_jpg/    300 Monet paintings (256x256 RGB JPG)
    task3_gan/data/photo_jpg/    7,038 photos (256x256 RGB JPG)
    task3_gan/data/real_stats.npz  Inception statistics of the 300 Monets shipped with the competition
                                   (not used by the evaluation script, which recomputes features)

Needs a Kaggle API token (Kaggle -> Settings -> API -> Create New Token):
either the KAGGLE_API_TOKEN environment variable (e.g. from Colab Secrets) or ~/.kaggle/access_token.
You must have joined the competition on its Kaggle page.

Usage (from the repo root):
    python task3_gan/scripts/download_kaggle_data.py
"""
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

COMPETITION = "data-266-fall-2026-gan-image-style-transfer"

current_file = Path(__file__).resolve()
task3 = current_file.parent.parent
data_folder = task3 / "data"


def download_kaggle_data():
    """Download and arrange the data if it is not there yet; return the data folder."""
    if (data_folder / "monet_jpg").is_dir() and (data_folder / "photo_jpg").is_dir() \
            and (data_folder / "real_stats.npz").exists():
        print("Dataset already exists.")
        return data_folder
    data_folder.mkdir(parents=True, exist_ok=True)
    kaggle = shutil.which("kaggle") or str(Path(sys.executable).parent / "kaggle")
    subprocess.run([kaggle, "competitions", "download", "-c", COMPETITION, "-p", str(data_folder)], check=True)
    zip_path = data_folder / f"{COMPETITION}.zip"
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(data_folder)
    zip_path.unlink()
    # The zip nests the images under dataset/dataset/; move them to the layout the brief uses.
    nested = data_folder / "dataset" / "dataset"
    for name in ("monet_jpg", "photo_jpg"):
        if (nested / name).is_dir():
            shutil.move(str(nested / name), str(data_folder / name))
    shutil.rmtree(data_folder / "dataset", ignore_errors=True)
    for name in ("monet_jpg", "photo_jpg"):
        print(f"  {name}: {len(list((data_folder / name).glob('*.jpg'))):,} images")
    return data_folder


if __name__ == "__main__":
    download_kaggle_data()
