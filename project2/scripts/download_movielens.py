"""
Download and extract MovieLens 100k dataset from GroupLens.
"""
import os
import requests
import zipfile
from pathlib import Path

URL = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"


def download(url: str, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} to {target}")
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    with open(target, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)


def extract(zip_path: Path, out_dir: Path):
    print(f"Extracting {zip_path} to {out_dir}")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)


def main():
    workspace = Path(__file__).resolve().parents[2]
    data_dir = workspace / "data" / "raw"
    zip_path = data_dir / "ml-100k.zip"
    extract_dir = data_dir

    if not (data_dir / "ml-100k").exists():
        download(URL, zip_path)
        extract(zip_path, extract_dir)
        print("Download and extraction complete.")
    else:
        print("Dataset already exists at", data_dir / "ml-100k")


if __name__ == "__main__":
    main()
