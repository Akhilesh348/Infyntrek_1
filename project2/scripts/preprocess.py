"""
Preprocess MovieLens 100k raw files into CSVs for modeling.
"""
import os
from pathlib import Path
import pandas as pd


def load_raw_ml100k(raw_dir: Path):
    data_file = raw_dir / "ml-100k" / "u.data"
    item_file = raw_dir / "ml-100k" / "u.item"
    if not data_file.exists() or not item_file.exists():
        raise FileNotFoundError("Raw MovieLens files not found. Run download script first.")

    # u.data: user id | item id | rating | timestamp (tab-separated)
    ratings = pd.read_csv(data_file, sep="\t", header=None, names=["userId", "movieId", "rating", "timestamp"])

    # u.item: movie id | movie title | release date | ... (pipe-separated)
    items = pd.read_csv(item_file, sep="|", header=None, encoding="latin-1", usecols=[0, 1, 2])
    items.columns = ["movieId", "title", "release_date"]

    return ratings, items


def clean_and_save(ratings: pd.DataFrame, items: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Drop timestamp (not needed for basic EDA)
    ratings = ratings.drop(columns=["timestamp"]) 

    # Ensure proper dtypes
    ratings["userId"] = ratings["userId"].astype(int)
    ratings["movieId"] = ratings["movieId"].astype(int)
    ratings["rating"] = ratings["rating"].astype(float)

    # Normalize ratings to 0-1 (original ratings 1-5)
    ratings["rating_norm"] = (ratings["rating"] - 1.0) / 4.0

    # Save processed files
    ratings.to_csv(out_dir / "ratings.csv", index=False)
    items.to_csv(out_dir / "movies.csv", index=False)
    print(f"Saved processed ratings to {out_dir / 'ratings.csv'} and movies to {out_dir / 'movies.csv'}")


def main():
    workspace = Path(__file__).resolve().parents[2]
    raw_dir = workspace / "data" / "raw"
    out_dir = workspace / "data" / "processed"

    ratings, items = load_raw_ml100k(raw_dir)
    clean_and_save(ratings, items, out_dir)


if __name__ == "__main__":
    main()
