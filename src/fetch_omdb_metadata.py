import os
import time
import random

import pandas as pd
import requests

# Number of movies to select for the movie-based dataset
N_MOVIES = 20


def fetch_single_movie(movie_id: str, api_key: str):
    """Fetch metadata for a single IMDb ID from OMDb.

    Returns a dict with movie_id, movie_title, type, and imdb_rating, or None if the API call fails.
    """
    url = f"http://www.omdbapi.com/?i={movie_id}&apikey={api_key}"

    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()

        if data.get("Response") == "False":
            print(f"OMDb error for {movie_id}: {data.get('Error')}")
            return None

        return {
            "movie_id": movie_id,
            "movie_title": data.get("Title"),
            "type": data.get("Type"),      # movie / series / episode
            "imdb_rating": data.get("imdbRating"),  # string like "7.3"
        }
    except Exception as e:
        print(f"Error fetching {movie_id}: {e}")
        return None


def main():
    api_key = os.getenv("OMDB_API_KEY")
    if not api_key:
        raise ValueError("OMDB_API_KEY environment variable not set.")

    # 1. Load all reviews (built by build_full_reviews.py)
    reviews_path = "data/processed/all_reviews.csv"
    df = pd.read_csv(reviews_path)

    # Unique IMDb IDs present in the reviews
    unique_ids = sorted(df["movie_id"].unique().tolist())
    print(f"Found {len(unique_ids)} unique IMDb IDs.")

    # 2. Randomly select N_MOVIES from the available IDs
    random.seed(91324)
    n_to_pick = min(N_MOVIES, len(unique_ids))
    selected_ids = random.sample(unique_ids, n_to_pick)
    print(f"Randomly selected {len(selected_ids)} movie IDs for metadata fetch.")

    # 3. Fetch OMDb metadata for the selected movies only
    rows = []
    for i, imdb_id in enumerate(selected_ids, start=1):
        print(f"[{i}/{len(selected_ids)}] Fetching {imdb_id} ...")
        info = fetch_single_movie(imdb_id, api_key)
        if info:
            rows.append(info)
        time.sleep(0.4)  # be polite with the free API

    meta_df = pd.DataFrame(rows)

    # 4. Merge metadata with all reviews, but only for selected movie_ids
    selected_reviews = df[df["movie_id"].isin(selected_ids)].copy()
    merged = selected_reviews.merge(meta_df, on="movie_id", how="left")

    # 5. Keep only actual movies (type == "movie")
    movies_only = merged[merged["type"] == "movie"].copy()
    # `type` is only used to filter to movies; drop it from the exported dataset
    if "type" in movies_only.columns:
        movies_only = movies_only.drop(columns=["type"])
    print("Movie-only shape:", movies_only.shape)

    # 6. Save the final enriched dataset for the selected movies
    out_path = "data/processed/selected_movies_reviews_enriched.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    movies_only.to_csv(out_path, index=False)
    print("Saved reviews for selected movies:", out_path)


if __name__ == "__main__":
    main()