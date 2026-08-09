"""
generate_dataset.py
--------------------
Generates a realistic, synthetic e-commerce customer review dataset for the
Customer Voice Analytics project.

WHY A SYNTHETIC DATASET?
Publicly hosted review datasets (Amazon Reviews, Flipkart Reviews, Yelp, etc.)
are typically distributed as multi-GB archives on Kaggle/UCI, which cannot be
reliably fetched inside this offline build environment. To keep the project
100% reproducible for anyone cloning the repository (no manual download,
no broken links, no API keys), we generate a large, realistic review corpus
programmatically.

The generation process is NOT random noise. Reviews are built by combining:
  - a product category (Electronics, Fashion, Home & Kitchen, Beauty,
    Sports & Outdoors, Books)
  - a star rating (1-5), which drives the sentiment "intent" of the text
  - a bank of realistic sentence fragments that real shoppers commonly write
    (praise, complaints, shipping/quality/service remarks, mixed opinions)
  - randomised noise: typos, casing, punctuation, emojis, extra spaces,
    occasional HTML fragments and stray URLs -- so the cleaning pipeline
    downstream has real work to do, just like production review data.

This script is fully deterministic (fixed random seed) so re-running it
reproduces the exact same dataset.csv shipped in /data.
"""

import random
import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

CATEGORIES = [
    "Electronics", "Fashion", "Home & Kitchen",
    "Beauty & Personal Care", "Sports & Outdoors", "Books & Media"
]

PRODUCTS = {
    "Electronics": ["Wireless Earbuds", "Smartwatch", "Bluetooth Speaker", "Power Bank", "Laptop Stand", "USB-C Hub"],
    "Fashion": ["Running Shoes", "Denim Jacket", "Cotton T-Shirt", "Leather Wallet", "Sunglasses", "Backpack"],
    "Home & Kitchen": ["Air Fryer", "Non-Stick Pan Set", "Electric Kettle", "Vacuum Cleaner", "Blender", "Bedsheet Set"],
    "Beauty & Personal Care": ["Face Serum", "Hair Dryer", "Electric Trimmer", "Moisturizer Cream", "Sunscreen Lotion", "Lip Balm Set"],
    "Sports & Outdoors": ["Yoga Mat", "Dumbbell Set", "Camping Tent", "Cycling Helmet", "Water Bottle", "Resistance Bands"],
    "Books & Media": ["Self-Help Book", "Fiction Novel", "Cookbook", "Biography", "Kids Story Set", "Comic Collection"],
}

# ---- Sentence banks keyed by rating "intent" ------------------------------
POSITIVE_OPENERS = [
    "Absolutely loved this product!", "Exceeded my expectations.", "Best purchase I've made this year.",
    "I'm so impressed with the quality.", "Highly recommend this to everyone.", "Five stars, no doubt.",
    "This is exactly what I needed.", "Fantastic value for the price.", "Works like a charm.",
    "Couldn't be happier with this purchase."
]
POSITIVE_DETAILS = [
    "The build quality feels premium and sturdy.", "Delivery was super fast, arrived a day early.",
    "Customer service was quick to respond when I had a question.", "It looks even better than the pictures online.",
    "Setup was simple and took less than five minutes.", "Battery life lasts way longer than advertised.",
    "The packaging was neat and nothing was damaged.", "My family loves it just as much as I do.",
    "It's comfortable and fits perfectly.", "The color and finish look great in person."
]
NEGATIVE_OPENERS = [
    "Really disappointed with this purchase.", "Not worth the money at all.", "I regret buying this.",
    "Would not recommend this to anyone.", "This was a complete waste of time.", "Extremely poor quality.",
    "One of the worst products I've bought online.", "I want a refund immediately.",
    "This broke within a few days of use.", "Very frustrating experience overall."
]
NEGATIVE_DETAILS = [
    "The material feels cheap and flimsy.", "It arrived late and the box was damaged.",
    "Customer support never replied to my emails.", "It stopped working after just one week.",
    "The size was completely different from what was described.", "There was a strange smell straight out of the box.",
    "The instructions were confusing and unclear.", "It doesn't match the pictures shown online at all.",
    "I found scratches on it right out of the packaging.", "The stitching came apart almost immediately."
]
NEUTRAL_OPENERS = [
    "It's an okay product, nothing special.", "Does the job but nothing extraordinary.",
    "Average quality for the price.", "It's fine, meets basic expectations.",
    "Not bad, but not great either.", "A decent option if you're on a budget.",
    "It works as described, no complaints, no excitement.", "Reasonable purchase overall.",
    "It's exactly what was advertised, nothing more.", "Middle of the road experience."
]
NEUTRAL_DETAILS = [
    "Some features could be improved in future versions.", "Shipping took the standard amount of time.",
    "The design is simple but functional.", "It's a bit bulkier than I expected.",
    "Packaging was standard, nothing memorable.", "It's comparable to other products in this price range.",
    "There are better alternatives but this works fine too.", "I might consider other brands next time.",
    "It does what it says, just don't expect much extra.", "Good enough for occasional use."
]
MIXED_DETAILS = [
    "although the price is a bit high for what you get.",
    "but the delivery experience could definitely be better.",
    "even though customer service was slow to respond.",
    "but I wish it came in more color options.",
    "although it took longer to arrive than promised.",
]

EMOJIS_POS = [" 😊", " 👍", " ❤️", " 🔥", " ✨", ""]
EMOJIS_NEG = [" 😡", " 👎", " 😢", " 💔", ""]
EMOJIS_NEU = [" 🙂", " 🤷", ""]

HTML_NOISE = ["<br>", "<p>", "</p>", "&amp;"]
URL_NOISE = ["http://example-shop.com/review", "www.shopreview-site.com/r/12345", "https://bit.ly/3xample"]


def build_review_text(rating: int) -> str:
    """Compose a realistic review sentence based on the star rating, then
    inject realistic 'messiness' (casing, punctuation, emoji, occasional
    HTML/URL fragments, extra whitespace) so cleaning has genuine work to do."""
    if rating >= 4:
        opener = random.choice(POSITIVE_OPENERS)
        detail = random.choice(POSITIVE_DETAILS)
        emoji = random.choice(EMOJIS_POS)
        text = f"{opener} {detail}{emoji}"
        if random.random() < 0.15:
            text += " " + random.choice(MIXED_DETAILS)
    elif rating == 3:
        opener = random.choice(NEUTRAL_OPENERS)
        detail = random.choice(NEUTRAL_DETAILS)
        emoji = random.choice(EMOJIS_NEU)
        text = f"{opener} {detail}{emoji}"
    else:
        opener = random.choice(NEGATIVE_OPENERS)
        detail = random.choice(NEGATIVE_DETAILS)
        emoji = random.choice(EMOJIS_NEG)
        text = f"{opener} {detail}{emoji}"
        if random.random() < 0.1:
            text += " On the plus side, it did arrive quickly."

    # Inject noise a fraction of the time (mirrors real-world scraped data)
    if random.random() < 0.08:
        text = random.choice(HTML_NOISE) + text
    if random.random() < 0.05:
        text += " " + random.choice(URL_NOISE)
    if random.random() < 0.10:
        text = text.upper() if random.random() < 0.5 else text.lower()
    if random.random() < 0.12:
        text = re.sub(" ", "  ", text, count=1)  # double space
    if random.random() < 0.04:
        text = text + "   "  # trailing whitespace

    return text


def random_date(start: datetime, end: datetime) -> datetime:
    delta = end - start
    rand_days = random.randint(0, delta.days)
    return start + timedelta(days=rand_days)


def generate_dataset(n_rows: int = 1800) -> pd.DataFrame:
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2025, 12, 31)

    # Rating distribution skewed slightly positive, like real e-commerce data
    rating_choices = [1, 2, 3, 4, 5]
    rating_weights = [0.09, 0.08, 0.14, 0.32, 0.37]

    rows = []
    for i in range(n_rows):
        category = random.choice(CATEGORIES)
        product = random.choice(PRODUCTS[category])
        rating = np.random.choice(rating_choices, p=rating_weights)
        text = build_review_text(rating)
        date = random_date(start_date, end_date)
        verified = random.random() < 0.8
        helpful_votes = max(0, int(np.random.exponential(scale=3)))

        rows.append({
            "review_id": f"REV{i+1:05d}",
            "product_category": category,
            "product_name": product,
            "review_text": text,
            "rating": int(rating),
            "review_date": date.strftime("%Y-%m-%d"),
            "verified_purchase": verified,
            "helpful_votes": helpful_votes,
        })

    df = pd.DataFrame(rows)

    # --- Deliberately introduce realistic data-quality issues -------------
    # 1. Missing review text in a few rows
    missing_idx = df.sample(frac=0.015, random_state=1).index
    df.loc[missing_idx, "review_text"] = np.nan

    # 2. Empty-string reviews (different from NaN, common in scraped data)
    empty_idx = df.sample(frac=0.01, random_state=2).index
    df.loc[empty_idx, "review_text"] = "   "

    # 3. Duplicate reviews (users double-submitting / scraping duplicates)
    dup_rows = df.sample(frac=0.03, random_state=3).copy()
    df = pd.concat([df, dup_rows], ignore_index=True)

    # 4. A few missing ratings
    missing_rating_idx = df.sample(frac=0.01, random_state=4).index
    df.loc[missing_rating_idx, "rating"] = np.nan

    df = df.sample(frac=1, random_state=5).reset_index(drop=True)  # shuffle
    return df


if __name__ == "__main__":
    df = generate_dataset(1800)
    out_path = "/home/claude/customer-voice-sentiment-analysis/data/dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} rows -> {out_path}")
    print(df.head())
    print(df.isna().sum())
