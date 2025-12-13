import os
import time
import re
from pathlib import Path

import pandas as pd
from openai import OpenAI
import anthropic


# ====== MODEL NAMES ======
GPT_MODEL = "gpt-5.1"  # same as your current GPT script
CLAUDE_MODEL = "claude-opus-4-1"  # or whatever you used in add_claude_ratings.py


# ====== SHARED PROMPT BUILDER (GPT + CLAUDE USE THIS) ======
def build_prompt(review_title: str, review_text: str) -> str:
    """
    Build the prompt for rating a review.
    The model must output ONLY a single integer 1–10.
    This is shared between GPT and Claude for a fair comparison.
    """
    # Clean text fields / fallbacks
    title = (review_title or "").strip() or "(no review title)"

    raw_text = review_text if review_text is not None else ""
    if not isinstance(raw_text, str):
        raw_text = str(raw_text)
    text_clean = raw_text.strip()

    # Handle missing/no-body reviews exactly once, same for both models
    if text_clean == "" or text_clean == "[No Review]":
        review_block = (
            "There is no body text for this review. You must base your rating only "
            "on the review title above.\n"
        )
        review_text_for_prompt = "(no additional review text)"
    else:
        review_block = (
            "Use both the review title and the full review text below when deciding "
            "the rating.\n"
        )
        review_text_for_prompt = text_clean

    return f"""
You are evaluating this user review from 1 to 10:

- 1 = extremely negative
- 5 = mixed/neutral
- 10 = extremely positive

You must decide what rating *you* would give this review, based only on its content.
You are not given any existing numeric rating, and you should not guess or imagine one.

Review title:
\"\"\"{title}\"\"\"

{review_block}Review text:
\"\"\"{review_text_for_prompt}\"\"\"

Return ONLY a single integer between 1 and 10, with no words or explanation.
You are not allowed to refuse; you must always choose a number from 1 to 10.
""".strip()


# ====== RATING PARSER (SHARED) ======
def extract_rating(raw: str) -> int | None:
    """
    Parse a valid integer 1–10 from the model output.
    Works for both GPT and Claude responses.
    """
    if raw is None:
        return None

    match = re.search(r"\b(10|[1-9])\b", raw)
    if match:
        try:
            return int(match.group(1))
        except Exception:
            return None

    return None


# ====== GPT CALL ======
def call_gpt(client: OpenAI, prompt: str) -> str:
    """
    Call the OpenAI Responses API and return the raw string response.
    """
    resp = client.responses.create(
        model=GPT_MODEL,
        temperature=0.0,
        max_output_tokens=16,
        input=[
            {
                "role": "system",
                "content": "You output ONLY a single integer between 1 and 10. No words.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    return resp.output_text.strip()


# ====== CLAUDE CALL ======
def call_claude(client: anthropic.Anthropic, prompt: str) -> str:
    """
    Call the Claude Messages API and return the raw string response.
    """
    resp = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=16,
        temperature=0.0,
        system="You output ONLY a single integer between 1 and 10. No words.",
        messages=[{"role": "user", "content": prompt}],
    )
    # Claude's content is a list of blocks; assume first block is text
    if resp.content and hasattr(resp.content[0], "text"):
        return resp.content[0].text.strip()
    return ""


# ====== MAIN PIPELINE ======
def main() -> None:
    # --- Keys ---
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Export it before running.")

    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set. Export it before running.")

    gpt_client = OpenAI(api_key=openai_key)
    claude_client = anthropic.Anthropic(api_key=anthropic_key)

    # --- Load data ---
    input_path = Path("data/processed/selected_movies_with_ai.csv")
    df = pd.read_csv(input_path)

    print("Loaded:", input_path)
    print("Rows:", len(df))

    gpt5_ratings: list[int | None] = []
    claude4_ratings: list[int | None] = []

    # --- Loop rows once, call BOTH models on SAME prompt ---
    for idx, row in df.iterrows():
        review_title = row.get("review_title", "")
        review_text = row.get("review", "")

        prompt = build_prompt(review_title, review_text)

        # GPT call
        try:
            gpt_raw = call_gpt(gpt_client, prompt)
            gpt_score = extract_rating(gpt_raw)
            if gpt_score is None:
                print(f"[WARN] Row {idx}: GPT failed to parse rating. Raw: {gpt_raw!r}")
        except Exception as e:
            print(f"[ERROR] Row {idx}: GPT call failed: {e!r}")
            gpt_score = None

        # Short pause between providers to avoid rate spikes
        time.sleep(0.2)

        # Claude call
        try:
            claude_raw = call_claude(claude_client, prompt)
            claude_score = extract_rating(claude_raw)
            if claude_score is None:
                print(
                    f"[WARN] Row {idx}: Claude failed to parse rating. Raw: {claude_raw!r}"
                )
        except Exception as e:
            print(f"[ERROR] Row {idx}: Claude call failed: {e!r}")
            claude_score = None

        gpt5_ratings.append(gpt_score)
        claude4_ratings.append(claude_score)

        # Optional: small sleep each row overall
        time.sleep(0.1)

        if (idx + 1) % 25 == 0:
            print(f"Processed {idx + 1}/{len(df)}")

    # --- Attach big‑model ratings ---
    df["gpt5_rating"] = gpt5_ratings
    df["claude4_rating"] = claude4_ratings

    # --- Column order for readability & consistency ---
    base_cols = [
        "movie_title",
        "imdb_rating",
        "review_title",
        "review",
        "rating",         # human rating
        "gpt4_rating",    # small OpenAI model (from add_ai_rating_old.py)
        "claude3_rating", # small Anthropic model (from add_ai_rating_old.py)
        "gpt5_rating",    # larger OpenAI model (this script)
        "claude4_rating", # larger Anthropic model (this script)
        "review_word_count",
        "year",
        "genres",
        "sentiment_polarity",
        "subjectivity",
        "readability",
        "sentiment_conflict",
    ]

    other_cols = [c for c in df.columns if c not in base_cols]
    df = df[base_cols + other_cols]

    # --- Save ---
    output_path = Path("data/processed/selected_movies_with_ai_4models.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print("\nSaved:", output_path)
    print("Final shape:", df.shape)


if __name__ == "__main__":
    main()