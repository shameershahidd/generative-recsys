"""
Downloads and extracts MovieLens-1M into data/ml-1m/.
Run once before anything else: python scripts/download_data.py
"""

import urllib.request
import zipfile
import shutil
from pathlib import Path

URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
DATA_DIR = Path(__file__).parent.parent / "data"


def download():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / "ml-1m.zip"

    if (DATA_DIR / "ml-1m" / "ratings.dat").exists():
        print("Data already downloaded.")
        return

    print(f"Downloading MovieLens-1M from {URL} ...")
    import subprocess
    subprocess.run(["curl", "-L", "--insecure", "-o", str(zip_path), URL], check=True)
    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATA_DIR)
    zip_path.unlink()
    print(f"Done. Data in {DATA_DIR / 'ml-1m'}")


def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    pct = min(downloaded / total_size * 100, 100)
    print(f"\r  {pct:.1f}%", end="", flush=True)


if __name__ == "__main__":
    download()
