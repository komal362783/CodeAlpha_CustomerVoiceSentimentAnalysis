"""
sentiment_analysis.py
----------------------
Lexicon-based sentiment classification (VADER) and emotion analysis
(NRC Emotion Lexicon via NRCLex) for the Customer Voice Analytics project.

METHODOLOGY & LIMITATIONS (documented honestly, no overclaiming):

SENTIMENT (VADER):
  VADER (Valence Aware Dictionary and sEntiment Reasoner) is a rule-based /
  lexicon sentiment tool tuned for short, informal text such as reviews and
  social media posts. It returns a compound score in [-1, 1]. We bucket this
  score into Positive / Negative / Neutral using VADER's own recommended
  thresholds (compound >= 0.05 -> positive, <= -0.05 -> negative, else
  neutral). This is a lexicon-based method, NOT a trained supervised
  classifier -- we do not claim a classification "accuracy" figure because
  no labelled ground truth is being validated against. Where the dataset
  also has star ratings, we separately compare VADER sentiment to rating
  bands purely as a cross-check / insight, not as an accuracy metric.

EMOTION (NRC Emotion Lexicon):
  The NRC lexicon maps individual English words to one or more of 8 basic
  emotions (anger, anticipation, disgust, fear, joy, sadness, surprise,
  trust) plus 2 sentiment polarities. It is a word-level, non-contextual
  lexicon -- it does not understand negation, sarcasm, or sentence
  structure, so results are indicative of emotional *tone* at the word
  level rather than a definitive emotional diagnosis of the reviewer.
  Reviews with no emotion-bearing words are labelled "None Detected".
"""

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nrclex import NRCLex

_vader = SentimentIntensityAnalyzer()
_nrc = NRCLex()  # reused across calls; load_raw_text() resets its internal state

EMOTION_LABELS = ["anger", "anticipation", "disgust", "fear",
                   "joy", "sadness", "surprise", "trust"]


# --------------------------------------------------------------------------
# SENTIMENT
# --------------------------------------------------------------------------
def classify_sentiment(text: str) -> dict:
    """Returns VADER compound/pos/neu/neg scores + a Positive/Negative/Neutral label."""
    if not isinstance(text, str) or not text.strip():
        return {"vader_compound": 0.0, "vader_pos": 0.0, "vader_neu": 1.0,
                "vader_neg": 0.0, "sentiment_label": "Neutral"}

    scores = _vader.polarity_scores(text)
    compound = scores["compound"]

    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    return {
        "vader_compound": compound,
        "vader_pos": scores["pos"],
        "vader_neu": scores["neu"],
        "vader_neg": scores["neg"],
        "sentiment_label": label,
    }


def add_sentiment_columns(df: pd.DataFrame, text_col: str = "review_text") -> pd.DataFrame:
    df = df.copy()
    results = df[text_col].apply(classify_sentiment).apply(pd.Series)
    return pd.concat([df, results], axis=1)


# --------------------------------------------------------------------------
# EMOTION
# --------------------------------------------------------------------------
def classify_emotion(text: str) -> dict:
    """Returns the dominant NRC emotion for a piece of text (word-level lexicon lookup)."""
    if not isinstance(text, str) or not text.strip():
        return {"dominant_emotion": "None Detected"}

    _nrc.load_raw_text(text)
    scores = {k: v for k, v in _nrc.raw_emotion_scores.items() if k in EMOTION_LABELS}

    if not scores or max(scores.values()) == 0:
        return {"dominant_emotion": "None Detected"}

    dominant = max(scores, key=scores.get)
    return {"dominant_emotion": dominant.capitalize()}


def add_emotion_column(df: pd.DataFrame, text_col: str = "review_text") -> pd.DataFrame:
    df = df.copy()
    results = df[text_col].apply(classify_emotion).apply(pd.Series)
    return pd.concat([df, results], axis=1)


# --------------------------------------------------------------------------
# RATING vs SENTIMENT CROSS-CHECK (insight helper, not an accuracy metric)
# --------------------------------------------------------------------------
def rating_sentiment_agreement(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-tabulates star rating against VADER sentiment label.
    Used purely as a descriptive cross-check / insight, e.g. to spot
    reviews where the written text sentiment disagrees with the numeric
    star rating (useful for flagging sarcasm or mismatched reviews).
    """
    return pd.crosstab(df["rating"], df["sentiment_label"], normalize="index").round(3)
