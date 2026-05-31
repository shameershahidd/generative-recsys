"""
MovieLens-1M data pipeline.
Loads ratings + movies, splits train/test via leave-last-out,
and builds item metadata and interaction history per user.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data" / "ml-1m"


def load_ratings(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    ratings_path = data_dir / "ratings.dat"
    df = pd.read_csv(
        ratings_path,
        sep="::",
        engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
        encoding="latin-1",
    )
    return df


def load_movies(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    movies_path = data_dir / "movies.dat"
    df = pd.read_csv(
        movies_path,
        sep="::",
        engine="python",
        names=["movie_id", "title", "genres"],
        encoding="latin-1",
    )
    return df


def build_item_metadata(movies_df: pd.DataFrame) -> dict:
    """Returns {movie_id: {title, genres, description}} dict."""
    metadata = {}
    for _, row in movies_df.iterrows():
        genres = row["genres"].replace("|", ", ")
        metadata[row["movie_id"]] = {
            "title": row["title"],
            "genres": genres,
            "description": f"{row['title']} — Genres: {genres}",
        }
    return metadata


def split_train_test(
    ratings_df: pd.DataFrame,
    min_interactions: int = 5,
    rating_threshold: float = 3.5,
) -> tuple[dict, dict]:
    """
    Leave-last-out split by timestamp.
    Only considers ratings >= rating_threshold as positive interactions.
    Returns (train_history, test_items) where each is {user_id: [movie_ids]}.
    """
    # Keep only positive interactions
    pos = ratings_df[ratings_df["rating"] >= rating_threshold].copy()
    pos = pos.sort_values("timestamp")

    # Filter users with enough history
    counts = pos.groupby("user_id").size()
    valid_users = counts[counts >= min_interactions].index
    pos = pos[pos["user_id"].isin(valid_users)]

    train_history: dict[int, list[int]] = {}
    test_items: dict[int, list[int]] = {}

    for user_id, group in pos.groupby("user_id"):
        movie_ids = group["movie_id"].tolist()
        # Last item is the test item; rest is training history
        train_history[user_id] = movie_ids[:-1]
        test_items[user_id] = [movie_ids[-1]]

    return train_history, test_items


def load_dataset(
    data_dir: Path = DATA_DIR,
    min_interactions: int = 5,
    rating_threshold: float = 3.5,
) -> tuple[dict, dict, dict, list[int]]:
    """
    Returns:
        train_history: {user_id: [movie_ids]}
        test_items:    {user_id: [movie_ids]}
        item_metadata: {movie_id: {title, genres, description}}
        all_item_ids:  sorted list of all movie_ids
    """
    ratings_df = load_ratings(data_dir)
    movies_df = load_movies(data_dir)

    item_metadata = build_item_metadata(movies_df)
    train_history, test_items = split_train_test(
        ratings_df, min_interactions, rating_threshold
    )

    all_item_ids = sorted(movies_df["movie_id"].tolist())
    return train_history, test_items, item_metadata, all_item_ids


if __name__ == "__main__":
    train, test, metadata, items = load_dataset()
    print(f"Users: {len(train)}")
    print(f"Items: {len(items)}")
    sample_user = next(iter(train))
    print(f"Sample user {sample_user} history length: {len(train[sample_user])}")
    print(f"Sample test item: {test[sample_user]}")
    print(f"Sample metadata: {metadata[train[sample_user][0]]}")
