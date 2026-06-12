"""
Data loading utilities for MovieLens dataset.
Downloads the small dataset automatically if not present.
"""

import os
import zipfile
import urllib.request
import pandas as pd

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
ZIP_PATH = os.path.join(DATA_DIR, "ml-latest-small.zip")
EXTRACT_DIR = os.path.join(DATA_DIR, "ml-latest-small")


def download_movielens():
    """Download MovieLens small dataset if not already present."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(EXTRACT_DIR):
        print("Downloading MovieLens dataset...")
        urllib.request.urlretrieve(MOVIELENS_URL, ZIP_PATH)
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(DATA_DIR)
        print("Dataset downloaded and extracted.")
    else:
        print("Dataset already present.")


def load_data():
    """
    Load and return (movies, ratings, tags) DataFrames.
    Auto-downloads if not present.
    """
    download_movielens()

    movies = pd.read_csv(os.path.join(EXTRACT_DIR, "movies.csv"))
    ratings = pd.read_csv(os.path.join(EXTRACT_DIR, "ratings.csv"))
    tags = pd.read_csv(os.path.join(EXTRACT_DIR, "tags.csv"))

    # Clean up
    movies["genres"] = movies["genres"].str.replace("|", ", ", regex=False)
    movies["genres"] = movies["genres"].replace("(no genres listed)", "")

    # Extract year from title
    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)$").astype("float")
    movies["title_clean"] = movies["title"].str.replace(r"\s*\(\d{4}\)$", "", regex=True)

    print(f"Loaded: {len(movies)} movies, {len(ratings)} ratings, {len(tags)} tags")
    return movies, ratings, tags


def get_movie_stats(movies: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    """Merge movies with aggregated rating stats."""
    stats = ratings.groupby("movieId").agg(
        rating_count=("rating", "count"),
        rating_mean=("rating", "mean")
    ).reset_index()
    return movies.merge(stats, on="movieId", how="left")
