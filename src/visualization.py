"""
visualization.py
-----------------
Generates all professional matplotlib/seaborn/plotly/wordcloud charts for
the Customer Voice Analytics project, saved to outputs/charts/ (and a
subset re-saved to /screenshots/ for the README gallery).
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS as WC_STOPWORDS

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
})

SENTIMENT_COLORS = {"Positive": "#2ecc71", "Neutral": "#95a5a6", "Negative": "#e74c3c"}
EMOTION_PALETTE = "Set2"


def _save(fig, path, tight=True):
    if tight:
        fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {path}")


def chart_sentiment_distribution(df, out_dir):
    fig, ax = plt.subplots(figsize=(6, 5))
    order = ["Positive", "Neutral", "Negative"]
    counts = df["sentiment_label"].value_counts().reindex(order)
    bars = ax.bar(counts.index, counts.values, color=[SENTIMENT_COLORS[l] for l in order])
    ax.set_ylim(0, max(counts.values) * 1.18)
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + max(counts.values) * 0.03,
                f"{int(h)}\n({h/counts.sum():.1%})", ha="center", va="bottom", fontsize=10)
    ax.set_title("Overall Sentiment Distribution of Customer Reviews", pad=14)
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Number of Reviews")
    _save(fig, os.path.join(out_dir, "sentiment_distribution.png"))


def chart_rating_distribution(df, out_dir):
    fig, ax = plt.subplots(figsize=(6, 5))
    counts = df["rating"].value_counts().sort_index()
    ax.bar(counts.index.astype(str), counts.values, color="#3498db")
    ax.set_title("Star Rating Distribution")
    ax.set_xlabel("Star Rating")
    ax.set_ylabel("Number of Reviews")
    _save(fig, os.path.join(out_dir, "rating_distribution.png"))


def chart_sentiment_vs_rating(df, out_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    ct = pd.crosstab(df["rating"], df["sentiment_label"], normalize="index") * 100
    ct = ct[["Positive", "Neutral", "Negative"]]
    ct.plot(kind="bar", stacked=True, ax=ax,
            color=[SENTIMENT_COLORS[c] for c in ct.columns])
    ax.set_title("Sentiment Composition Within Each Star Rating")
    ax.set_xlabel("Star Rating")
    ax.set_ylabel("Percentage of Reviews (%)")
    ax.legend(title="Sentiment", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.tick_params(axis="x", rotation=0)
    _save(fig, os.path.join(out_dir, "sentiment_vs_rating.png"))


def chart_emotion_distribution(df, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df["dominant_emotion"].value_counts()
    counts = counts.drop("None Detected", errors="ignore")
    sns.barplot(x=counts.values, y=counts.index, ax=ax, palette=EMOTION_PALETTE, hue=counts.index, legend=False)
    ax.set_title("Dominant Emotion Distribution in Customer Reviews")
    ax.set_xlabel("Number of Reviews")
    ax.set_ylabel("Emotion")
    _save(fig, os.path.join(out_dir, "emotion_distribution.png"))


def chart_sentiment_by_category(df, out_dir):
    fig, ax = plt.subplots(figsize=(9, 6))
    ct = pd.crosstab(df["product_category"], df["sentiment_label"], normalize="index") * 100
    ct = ct[["Positive", "Neutral", "Negative"]].sort_values("Positive")
    ct.plot(kind="barh", stacked=True, ax=ax, color=[SENTIMENT_COLORS[c] for c in ct.columns])
    ax.set_title("Sentiment Composition by Product Category")
    ax.set_xlabel("Percentage of Reviews (%)")
    ax.set_ylabel("Product Category")
    ax.legend(title="Sentiment", bbox_to_anchor=(1.02, 1), loc="upper left")
    _save(fig, os.path.join(out_dir, "sentiment_by_category.png"))


def chart_negative_by_category(df, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    neg = df[df["sentiment_label"] == "Negative"]
    counts = neg["product_category"].value_counts().sort_values()
    ax.barh(counts.index, counts.values, color="#e74c3c")
    ax.set_title("Negative Reviews by Product Category")
    ax.set_xlabel("Number of Negative Reviews")
    _save(fig, os.path.join(out_dir, "negative_by_category.png"))


def chart_positive_by_category(df, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    pos = df[df["sentiment_label"] == "Positive"]
    counts = pos["product_category"].value_counts().sort_values()
    ax.barh(counts.index, counts.values, color="#2ecc71")
    ax.set_title("Positive Reviews by Product Category")
    ax.set_xlabel("Number of Positive Reviews")
    _save(fig, os.path.join(out_dir, "positive_by_category.png"))


def chart_sentiment_trend(df, out_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    tmp = df.copy()
    tmp["month"] = pd.to_datetime(tmp["review_date"]).dt.to_period("M").dt.to_timestamp()
    monthly = tmp.groupby(["month", "sentiment_label"]).size().unstack(fill_value=0)
    monthly = monthly.reindex(columns=["Positive", "Neutral", "Negative"], fill_value=0)
    monthly_pct = monthly.div(monthly.sum(axis=1), axis=0) * 100
    for col in monthly_pct.columns:
        ax.plot(monthly_pct.index, monthly_pct[col], marker="o", label=col, color=SENTIMENT_COLORS[col])
    ax.set_title("Monthly Sentiment Trend Over Time")
    ax.set_xlabel("Month")
    ax.set_ylabel("Share of Reviews (%)")
    ax.legend(title="Sentiment")
    fig.autofmt_xdate(rotation=45)
    _save(fig, os.path.join(out_dir, "sentiment_trend.png"))


def _wordcloud_from_series(text_series, title, color, out_path):
    text = " ".join(text_series.dropna().tolist())
    if not text.strip():
        text = "no_data_available"
    wc = WordCloud(width=1000, height=550, background_color="white",
                    colormap=color, stopwords=WC_STOPWORDS, max_words=100,
                    collocations=False).generate(text)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold")
    _save(fig, out_path, tight=False)


def chart_wordcloud_positive(df, out_dir):
    pos_text = df.loc[df["sentiment_label"] == "Positive", "clean_tokens"]
    _wordcloud_from_series(pos_text, "Most Frequent Words in Positive Reviews",
                            "Greens", os.path.join(out_dir, "wordcloud_positive.png"))


def chart_wordcloud_negative(df, out_dir):
    neg_text = df.loc[df["sentiment_label"] == "Negative", "clean_tokens"]
    _wordcloud_from_series(neg_text, "Most Frequent Words in Negative Reviews",
                            "Reds", os.path.join(out_dir, "wordcloud_negative.png"))


def chart_top_keywords(df, out_dir, top_n=20):
    from collections import Counter
    all_tokens = " ".join(df["clean_tokens"].dropna()).split()
    counts = Counter(all_tokens).most_common(top_n)
    words, freqs = zip(*counts)
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.barplot(x=list(freqs), y=list(words), ax=ax, palette="viridis", hue=list(words), legend=False)
    ax.set_title(f"Top {top_n} Keywords in Customer Feedback (All Reviews)")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Keyword")
    _save(fig, os.path.join(out_dir, "top_keywords.png"))


def generate_all_charts(df, out_dir="outputs/charts"):
    os.makedirs(out_dir, exist_ok=True)
    chart_sentiment_distribution(df, out_dir)
    chart_rating_distribution(df, out_dir)
    chart_sentiment_vs_rating(df, out_dir)
    chart_emotion_distribution(df, out_dir)
    chart_sentiment_by_category(df, out_dir)
    chart_negative_by_category(df, out_dir)
    chart_positive_by_category(df, out_dir)
    chart_sentiment_trend(df, out_dir)
    chart_wordcloud_positive(df, out_dir)
    chart_wordcloud_negative(df, out_dir)
    chart_top_keywords(df, out_dir)
    print("All charts generated.")
