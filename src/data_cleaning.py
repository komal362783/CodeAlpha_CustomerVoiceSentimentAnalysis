"""
data_cleaning.py
-----------------
Data cleaning and NLP text-preprocessing utilities for the
Customer Voice Analytics project.

Two separate concerns are kept apart on purpose:

1. DATA CLEANING  -> fixes structural / data-quality issues (missing values,
   duplicates, empty strings, invalid ratings, stray HTML/URLs, whitespace).
   The output is still a human-readable review (e.g. "Great product, works
   well!") because we still want to *display* this text and feed it to
   VADER, which is tuned on natural (not aggressively stripped) English,
   including punctuation and casing cues for sentiment.

2. NLP PREPROCESSING -> produces a second, heavily-normalised column
   (tokenized, lower-cased, stopword-free, lemmatized) used only for
   word-frequency analysis / word clouds / keyword extraction, where
   stripping stopwords and punctuation actually helps.

We deliberately do NOT run the aggressive NLP preprocessing before VADER
sentiment scoring, because VADER is a lexicon+rule-based tool that uses
capitalization, punctuation ("!!!") and negation words ("not good") as
sentiment signals. Removing those would *hurt* sentiment accuracy, not help.
"""

import re
import string
import pandas as pd
import nltk

for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(pkg)
    except LookupError:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

STOPWORDS = set(stopwords.words("english"))
# Keep negation words -- critical for sentiment meaning ("not good" != "good")
NEGATIONS = {"no", "not", "nor", "never", "none", "n't"}
STOPWORDS = STOPWORDS - NEGATIONS
LEMMATIZER = WordNetLemmatizer()

URL_PATTERN = re.compile(r"http\S+|www\.\S+")
HTML_PATTERN = re.compile(r"<.*?>|&\w+;")
MULTISPACE_PATTERN = re.compile(r"\s+")
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002700-\U000027BF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002600-\U000026FF"
    "]+",
    flags=re.UNICODE,
)


# --------------------------------------------------------------------------
# STEP 1: STRUCTURAL DATA CLEANING
# --------------------------------------------------------------------------
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the raw review dataframe:
      - drops rows with missing/empty review text (can't analyse sentiment
        of text that doesn't exist)
      - drops exact duplicate reviews (same product + same text = duplicate
        submission/scrape artifact)
      - drops rows with missing/invalid rating (1-5 only)
      - normalises whitespace, strips stray HTML tags and URLs from text
      - parses review_date into a real datetime column
      - resets a clean, contiguous index
    """
    original_len = len(df)
    df = df.copy()

    # --- Missing / empty text -------------------------------------------------
    df["review_text"] = df["review_text"].astype(str)
    df.loc[df["review_text"].str.strip().isin(["nan", "None", ""]), "review_text"] = None
    df = df.dropna(subset=["review_text"])
    df = df[df["review_text"].str.strip() != ""]

    # --- Invalid / missing ratings ---------------------------------------
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"])
    df = df[df["rating"].between(1, 5)]
    df["rating"] = df["rating"].astype(int)

    # --- Strip HTML tags & URLs from the display text ---------------------
    df["review_text"] = df["review_text"].apply(lambda t: HTML_PATTERN.sub(" ", t))
    df["review_text"] = df["review_text"].apply(lambda t: URL_PATTERN.sub(" ", t))

    # --- Normalise whitespace ------------------------------------------------
    df["review_text"] = df["review_text"].apply(lambda t: MULTISPACE_PATTERN.sub(" ", t).strip())

    # Drop anything that became empty after stripping HTML/URLs
    df = df[df["review_text"].str.len() > 0]

    # --- Remove exact duplicate reviews (same product + same text) --------
    df = df.drop_duplicates(subset=["product_name", "review_text"], keep="first")

    # --- Parse dates -----------------------------------------------------
    df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
    df = df.dropna(subset=["review_date"])

    df = df.reset_index(drop=True)

    removed = original_len - len(df)
    print(f"[data_cleaning] Removed {removed} invalid/duplicate/empty rows "
          f"({removed/original_len:.1%} of raw data). {len(df)} rows remain.")
    return df


# --------------------------------------------------------------------------
# STEP 2: NLP TEXT PREPROCESSING (for word-frequency / word-cloud use only)
# --------------------------------------------------------------------------
def preprocess_text(text: str) -> str:
    """
    Heavy NLP normalisation for keyword/word-cloud analysis:
    lowercase -> remove emojis -> remove punctuation/digits ->
    tokenize -> remove stopwords (keeping negations) -> lemmatize.

    NOTE: This is intentionally NOT used before VADER sentiment scoring
    (see module docstring) -- only for word-frequency style analysis.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = EMOJI_PATTERN.sub(" ", text)
    text = URL_PATTERN.sub(" ", text)
    text = HTML_PATTERN.sub(" ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    text = MULTISPACE_PATTERN.sub(" ", text).strip()

    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]

    return " ".join(tokens)


def add_preprocessed_column(df: pd.DataFrame, source_col: str = "review_text",
                             target_col: str = "clean_tokens") -> pd.DataFrame:
    df = df.copy()
    df[target_col] = df[source_col].apply(preprocess_text)
    return df
