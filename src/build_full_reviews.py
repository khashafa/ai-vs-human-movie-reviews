import json
import os
from pathlib import Path

import pandas as pd


def load_all_reviews():
    root_dir = Path("data/raw/IMDB-Reviews")

    rows = []

    for path in root_dir.glob("*_reviews.json"):
        # movie_id is either in the JSON or we take it from the filename
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        movie_id = data.get("movie_id")
        if not movie_id:
            # fall back to filename like "tt0816692_reviews.json"
            movie_id = path.stem.split("_")[0]

        reviews = data.get("reviews", [])

        for r in reviews:
            rating = r.get("rating")
            text = (r.get("review") or "").strip()
            title = r.get("title") or ""

            # skip things that aren't proper numeric ratings
            if not rating or rating == "[No Rating]":
                continue

            try:
                rating_int = int(rating)
            except ValueError:
                continue

            word_count = len(text.split())

            rows.append(
                {
                    "movie_id": movie_id,
                    "review_title": title,
                    "review": text,
                    "human_rating": rating_int,
                    "review_word_count": word_count,
                }
            )

    df = pd.DataFrame(rows)
    return df


def main():
    df = load_all_reviews()
    print("Full reviews shape:", df.shape)

    out_path = "data/processed/all_reviews.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)

    print("Saved:", out_path)


if __name__ == "__main__":
    main()