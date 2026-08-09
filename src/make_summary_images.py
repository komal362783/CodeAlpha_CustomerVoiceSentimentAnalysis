"""
make_summary_images.py
-----------------------
Generates two extra images for the README screenshot gallery, both built
from REAL computed numbers on the analyzed dataset (nothing here is
hand-typed or invented):

1. screenshots/insights.png
   A clean "key insights" summary card -- the same numbers reported in the
   notebook's Key Insights section, rendered as a single shareable image.

2. screenshots/dashboard.png
   A static PREVIEW render of the Streamlit dashboard layout (metric cards +
   the same charts the live `streamlit run dashboard/app.py` app renders).
   This is generated with matplotlib rather than a live browser screenshot,
   because this build environment has no headless-browser/screenshot tool
   available. It is explicitly labelled as a preview in the README so no
   claim is made that it is a captured browser screenshot of a running app
   -- but the numbers and charts inside it are 100% real, computed from the
   analyzed dataset, exactly like the live dashboard would show.
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.image as mpimg

SENTIMENT_COLORS = {"Positive": "#2ecc71", "Neutral": "#95a5a6", "Negative": "#e74c3c"}


def make_insights_card(df, out_path="screenshots/insights.png"):
    total = len(df)
    pos_pct = (df["sentiment_label"] == "Positive").mean() * 100
    neu_pct = (df["sentiment_label"] == "Neutral").mean() * 100
    neg_pct = (df["sentiment_label"] == "Negative").mean() * 100
    avg_rating = df["rating"].mean()

    cat_sent = pd.crosstab(df["product_category"], df["sentiment_label"], normalize="index") * 100
    most_neg_cat = cat_sent["Negative"].idxmax()
    most_pos_cat = cat_sent["Positive"].idxmax()
    top_emotion = df.loc[df["dominant_emotion"] != "None Detected", "dominant_emotion"].value_counts().idxmax()

    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.axis("off")
    ax.set_title("Customer Voice Analytics — Key Insights Summary",
                 fontsize=17, fontweight="bold", pad=18)

    lines = [
        (f"Total Reviews Analyzed", f"{total:,}"),
        (f"Positive / Neutral / Negative", f"{pos_pct:.1f}% / {neu_pct:.1f}% / {neg_pct:.1f}%"),
        (f"Average Star Rating", f"{avg_rating:.2f} / 5"),
        (f"Category with Most Negative Feedback", f"{most_neg_cat} ({cat_sent.loc[most_neg_cat,'Negative']:.1f}%)"),
        (f"Category with Most Positive Feedback", f"{most_pos_cat} ({cat_sent.loc[most_pos_cat,'Positive']:.1f}%)"),
        (f"Most Common Detected Emotion", f"{top_emotion}"),
    ]

    y0 = 0.86
    step = 0.135
    for i, (label, value) in enumerate(lines):
        y = y0 - i * step
        ax.text(0.03, y, label, fontsize=13, color="#2c3e50", fontweight="bold", transform=ax.transAxes)
        ax.text(0.97, y, value, fontsize=13, color="#16a085", ha="right", transform=ax.transAxes)
        ax.plot([0.03, 0.97], [y - 0.045, y - 0.045], color="#ecf0f1", linewidth=1, transform=ax.transAxes)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=140)
    plt.close(fig)
    print(f"saved -> {out_path}")


def make_dashboard_preview(df, chart_dir="outputs/charts", out_path="screenshots/dashboard.png"):
    total = len(df)
    pos_pct = (df["sentiment_label"] == "Positive").mean() * 100
    neu_pct = (df["sentiment_label"] == "Neutral").mean() * 100
    neg_pct = (df["sentiment_label"] == "Negative").mean() * 100
    avg_rating = df["rating"].mean()

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(4, 4, hspace=0.55, wspace=0.35)

    fig.suptitle("Customer Voice Analytics Dashboard — Preview",
                 fontsize=19, fontweight="bold", y=0.98)
    fig.text(0.5, 0.945,
              "Static preview render composed from real analysis output "
              "(see 'How to Run' to launch the live Streamlit app)",
              ha="center", fontsize=9.5, color="#7f8c8d")

    metrics = [
        ("Total Reviews", f"{total:,}", "#34495e"),
        ("Positive %", f"{pos_pct:.1f}%", SENTIMENT_COLORS["Positive"]),
        ("Negative %", f"{neg_pct:.1f}%", SENTIMENT_COLORS["Negative"]),
        ("Avg Rating", f"{avg_rating:.2f} / 5", "#f39c12"),
    ]
    for i, (label, value, color) in enumerate(metrics):
        ax = fig.add_subplot(gs[0, i])
        ax.axis("off")
        ax.add_patch(mpatches.FancyBboxPatch((0.03, 0.05), 0.94, 0.9,
                                              boxstyle="round,pad=0.02,rounding_size=0.06",
                                              linewidth=1.2, edgecolor="#dcdde1", facecolor="#fdfefe"))
        ax.text(0.5, 0.62, value, ha="center", va="center", fontsize=19, fontweight="bold", color=color)
        ax.text(0.5, 0.24, label, ha="center", va="center", fontsize=10.5, color="#576574")

    chart_slots = [
        ("sentiment_distribution.png", gs[1, 0:2]),
        ("emotion_distribution.png", gs[1, 2:4]),
        ("sentiment_by_category.png", gs[2:4, 0:2]),
        ("sentiment_trend.png", gs[2, 2:4]),
        ("top_keywords.png", gs[3, 2:4]),
    ]
    for fname, slot in chart_slots:
        ax = fig.add_subplot(slot)
        img = mpimg.imread(f"{chart_dir}/{fname}")
        ax.imshow(img)
        ax.axis("off")

    fig.savefig(out_path, bbox_inches="tight", dpi=130, facecolor="white")
    plt.close(fig)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    df = pd.read_csv("outputs/reports/analyzed_dataset.csv")
    make_insights_card(df)
    make_dashboard_preview(df)
