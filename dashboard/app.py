"""
Customer Voice Analytics — Interactive Streamlit Dashboard
------------------------------------------------------------
Run with:  streamlit run dashboard/app.py   (from the project root)

Loads the raw dataset, runs the same cleaning / sentiment / emotion
pipeline used in the notebook (src/data_cleaning.py, src/sentiment_analysis.py),
and renders an interactive, filterable dashboard.
"""

import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st
from wordcloud import WordCloud, STOPWORDS as WC_STOPWORDS
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from data_cleaning import clean_dataframe, add_preprocessed_column          # noqa: E402
from sentiment_analysis import add_sentiment_columns, add_emotion_column    # noqa: E402

st.set_page_config(page_title="Customer Voice Analytics", page_icon="📊", layout="wide")

SENTIMENT_COLORS = {"Positive": "#2ecc71", "Neutral": "#95a5a6", "Negative": "#e74c3c"}
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "dataset.csv")


@st.cache_data(show_spinner="Running cleaning + sentiment + emotion pipeline...")
def load_analyzed_data():
    df = pd.read_csv(DATA_PATH)
    df = clean_dataframe(df)
    df = add_preprocessed_column(df)
    df = add_sentiment_columns(df)
    df = add_emotion_column(df)
    df["review_date"] = pd.to_datetime(df["review_date"])
    return df


df = load_analyzed_data()

# ----------------------------- Sidebar filters -----------------------------
st.sidebar.header("🔎 Filters")
categories = sorted(df["product_category"].unique())
selected_categories = st.sidebar.multiselect("Product Category", categories, default=categories)

ratings = sorted(df["rating"].unique())
selected_ratings = st.sidebar.multiselect("Star Rating", ratings, default=ratings)

sentiments = ["Positive", "Neutral", "Negative"]
selected_sentiments = st.sidebar.multiselect("Sentiment", sentiments, default=sentiments)

date_min, date_max = df["review_date"].min(), df["review_date"].max()
date_range = st.sidebar.date_input("Review Date Range", value=(date_min, date_max),
                                    min_value=date_min, max_value=date_max)

filtered = df[
    df["product_category"].isin(selected_categories)
    & df["rating"].isin(selected_ratings)
    & df["sentiment_label"].isin(selected_sentiments)
]
if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    filtered = filtered[(filtered["review_date"] >= start) & (filtered["review_date"] <= end)]

st.title("📊 Customer Voice Analytics")
st.caption("NLP-powered sentiment & emotion intelligence for customer product reviews")

if filtered.empty:
    st.warning("No reviews match the current filters. Try widening your selection.")
    st.stop()

# ----------------------------- Top metric row -------------------------------
total = len(filtered)
pos_pct = (filtered["sentiment_label"] == "Positive").mean() * 100
neg_pct = (filtered["sentiment_label"] == "Negative").mean() * 100
neu_pct = (filtered["sentiment_label"] == "Neutral").mean() * 100
avg_rating = filtered["rating"].mean()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Reviews", f"{total:,}")
c2.metric("Positive %", f"{pos_pct:.1f}%")
c3.metric("Neutral %", f"{neu_pct:.1f}%")
c4.metric("Negative %", f"{neg_pct:.1f}%")
c5.metric("Average Rating", f"{avg_rating:.2f} / 5")

st.divider()

# ----------------------------- Row 1: sentiment + emotion -------------------
col1, col2 = st.columns(2)
with col1:
    st.subheader("Sentiment Distribution")
    counts = filtered["sentiment_label"].value_counts().reindex(sentiments).fillna(0)
    fig = px.bar(x=counts.index, y=counts.values,
                 color=counts.index, color_discrete_map=SENTIMENT_COLORS,
                 labels={"x": "Sentiment", "y": "Number of Reviews"})
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Emotion Distribution")
    emo_counts = filtered["dominant_emotion"].value_counts()
    emo_counts = emo_counts.drop("None Detected", errors="ignore")
    fig = px.bar(x=emo_counts.values, y=emo_counts.index, orientation="h",
                 labels={"x": "Number of Reviews", "y": "Emotion"},
                 color=emo_counts.index)
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------- Row 2: rating vs sentiment / category --------
col3, col4 = st.columns(2)
with col3:
    st.subheader("Rating vs Sentiment")
    ct = pd.crosstab(filtered["rating"], filtered["sentiment_label"], normalize="index") * 100
    ct = ct.reindex(columns=sentiments, fill_value=0).reset_index().melt(
        id_vars="rating", var_name="Sentiment", value_name="Percentage")
    fig = px.bar(ct, x="rating", y="Percentage", color="Sentiment",
                 color_discrete_map=SENTIMENT_COLORS, barmode="stack",
                 labels={"rating": "Star Rating", "Percentage": "% of Reviews"})
    st.plotly_chart(fig, use_container_width=True)

with col4:
    st.subheader("Sentiment by Product Category")
    ct2 = pd.crosstab(filtered["product_category"], filtered["sentiment_label"], normalize="index") * 100
    ct2 = ct2.reindex(columns=sentiments, fill_value=0).reset_index().melt(
        id_vars="product_category", var_name="Sentiment", value_name="Percentage")
    fig = px.bar(ct2, x="Percentage", y="product_category", color="Sentiment",
                 color_discrete_map=SENTIMENT_COLORS, orientation="h", barmode="stack",
                 labels={"product_category": "Category", "Percentage": "% of Reviews"})
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------- Row 3: trend + keyword cloud -----------------
st.subheader("Monthly Sentiment Trend")
tmp = filtered.copy()
tmp["month"] = tmp["review_date"].dt.to_period("M").dt.to_timestamp()
monthly = tmp.groupby(["month", "sentiment_label"]).size().unstack(fill_value=0)
monthly = monthly.reindex(columns=sentiments, fill_value=0)
monthly_pct = monthly.div(monthly.sum(axis=1), axis=0).fillna(0) * 100
monthly_pct = monthly_pct.reset_index().melt(id_vars="month", var_name="Sentiment", value_name="Percentage")
fig = px.line(monthly_pct, x="month", y="Percentage", color="Sentiment",
              color_discrete_map=SENTIMENT_COLORS, markers=True,
              labels={"month": "Month", "Percentage": "% of Reviews"})
st.plotly_chart(fig, use_container_width=True)

col5, col6 = st.columns(2)
with col5:
    st.subheader("☁️ Positive Review Word Cloud")
    pos_text = " ".join(filtered.loc[filtered["sentiment_label"] == "Positive", "clean_tokens"].dropna())
    if pos_text.strip():
        wc = WordCloud(width=800, height=400, background_color="white",
                        colormap="Greens", stopwords=WC_STOPWORDS, collocations=False).generate(pos_text)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.info("No positive reviews in current filter selection.")

with col6:
    st.subheader("☁️ Negative Review Word Cloud")
    neg_text = " ".join(filtered.loc[filtered["sentiment_label"] == "Negative", "clean_tokens"].dropna())
    if neg_text.strip():
        wc = WordCloud(width=800, height=400, background_color="white",
                        colormap="Reds", stopwords=WC_STOPWORDS, collocations=False).generate(neg_text)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.info("No negative reviews in current filter selection.")

st.divider()
st.subheader("📋 Review Explorer")
st.dataframe(
    filtered[["review_id", "product_category", "product_name", "review_text",
              "rating", "sentiment_label", "dominant_emotion", "review_date"]]
    .sort_values("review_date", ascending=False),
    use_container_width=True, height=350,
)

st.caption("Sentiment: VADER lexicon-based classification · Emotion: NRC Emotion Lexicon · "
           "Dataset: synthetic, documented review corpus (see README)")
