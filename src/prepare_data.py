import os
from pathlib import Path

import pandas as pd

from textblob import TextBlob
import textstat
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def main():
    """Prepare analysis-ready CSV with combined title+review text features.

    Dataset: selected_movies_reviews_enriched.csv (movie-based design)

    Features created (based on "review_title: review"):
      - sentiment_polarity   (VADER compound)
      - sentiment_conflict   (min(pos, neg) from VADER)
      - subjectivity         (TextBlob subjectivity)
      - readability          (textstat Flesch–Kincaid grade)
    """

    # 1) Load the enriched movie-based dataset
    input_path = Path("data/processed/selected_movies_reviews_enriched.csv")
    df = pd.read_csv(input_path)

    # 3) Drop ID columns we don't need in the final analysis CSV
    for col in ["movie_id"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Ensure imdb_rating is numeric if it exists
    if "imdb_rating" in df.columns:
        df["imdb_rating"] = pd.to_numeric(df["imdb_rating"], errors="coerce")

    # 5) Build combined text "review_title: review" and compute features
    analyzer = SentimentIntensityAnalyzer()

    def make_combined_text(row):
        title = "" if pd.isna(row.get("review_title")) else str(row["review_title"])
        review = "" if pd.isna(row.get("review")) else str(row["review"])
        if title and review:
            return f"{title}: {review}"
        else:
            return title or review

    df["combined_text"] = df.apply(make_combined_text, axis=1)

    def compute_all_features(text: str):
        t = "" if pd.isna(text) else str(text).strip()
        # VADER scores
        vs = analyzer.polarity_scores(t)
        compound = vs["compound"]
        conflict = min(vs["pos"], vs["neg"])
        # TextBlob subjectivity
        tb = TextBlob(t)
        subj = tb.sentiment.subjectivity
        # Readability (Flesch–Kincaid grade)
        if t:
            try:
                read = textstat.flesch_kincaid_grade(t)
            except Exception:
                read = None
        else:
            read = None
        return compound, conflict, subj, read

    (
        df["sentiment_polarity"],
        df["sentiment_conflict"],
        df["subjectivity"],
        df["readability"],
    ) = zip(*df["combined_text"].apply(compute_all_features))

    # We don't need combined_text in the final CSV
    df = df.drop(columns=["combined_text"])

    # 6) Reorder columns: movie_title first, then key fields, then features
    base_cols = [
        "movie_title",
        "imdb_rating",
        "review_title",
        "review",
        "human_rating",
        "review_word_count",
        "sentiment_polarity",
        "sentiment_conflict",
        "subjectivity",
        "readability",
    ]

    other_cols = [c for c in df.columns if c not in base_cols]
    ordered_cols = base_cols + other_cols
    df = df[ordered_cols]

    # 7) Save the final analysis-ready CSV
    output_path = Path("data/processed/selected_movies_reviews_ready.csv")
    os.makedirs(output_path.parent, exist_ok=True)
    df.to_csv(output_path, index=False)

    print("Saved:", output_path)
    print("Final shape:", df.shape)


if __name__ == "__main__":
    main()