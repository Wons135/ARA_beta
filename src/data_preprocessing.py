import pandas as pd
import numpy as np
import json
import re
from sklearn.model_selection import train_test_split
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import download
from datetime import datetime
from tqdm import tqdm

# Download VADER lexicon for sentiment analysis
download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

# ---------- Load Reviews ----------

reviews_path = '../datasets/Movies_and_TV.jsonl'

reviews_list = []
with open(reviews_path, 'r') as fp:
    for line in tqdm(fp, desc="Loading reviews"):
        data = json.loads(line.strip())
        reviews_list.append(data)

reviews_df = pd.DataFrame(reviews_list)

# ---------- Preprocess Reviews ----------

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = text.lower()
    return text.strip()

reviews_df["clean_text"] = reviews_df["text"].apply(clean_text)
reviews_df["clean_title"] = reviews_df["title"].apply(clean_text)
reviews_df["full_text"] = reviews_df["clean_title"].fillna('') + ". " + reviews_df["clean_text"].fillna('')

# Type conversion for consistency
reviews_df['helpful_vote'] = reviews_df['helpful_vote'].astype(int)
reviews_df['rating'] = reviews_df['rating'].astype(int)

# Compute helpfulness ratio
reviews_df["helpfulness_ratio"] = reviews_df.apply(
    lambda row: row["helpful_vote"] / row["rating"] if row["rating"] > 0 else np.nan,
    axis=1
)

# Filter reviews
reviews_df = reviews_df[reviews_df["helpful_vote"] >= 5].copy()
reviews_df = reviews_df.dropna(subset=["helpfulness_ratio"])

# Log-transform helpfulness ratio
reviews_df["helpfulness_ratio_log"] = np.log1p(reviews_df["helpfulness_ratio"])

# Additional features
reviews_df["review_length"] = reviews_df["clean_text"].apply(lambda x: len(x.split()))
reviews_df["sentiment_score"] = reviews_df["clean_text"].apply(lambda x: sia.polarity_scores(x)["compound"])
reviews_df["is_verified"] = reviews_df["verified_purchase"].astype(int)
reviews_df["timestamp"] = reviews_df["timestamp"]

# ---------- Filter Metadata ---------- # commented out as not explicitly needed in research

# meta_df = meta_df[meta_df["main_category"].isin(["Movies & TV", "Prime Video"])].copy()

# ---------- Preprocess Metadata ----------

# meta_df["price"] = pd.to_numeric(meta_df["price"], errors='coerce')
# meta_df["has_description"] = meta_df["description"].notnull().astype(int)
# meta_df["store_known"] = meta_df["store"].notnull().astype(int)
# meta_df["main_category"] = meta_df["main_category"].fillna("Unknown")

# ---------- Merge ----------

# merged_df = reviews_df.merge(meta_df[["parent_asin", "price", "has_description", "store_known", "main_category"]],
#                              left_on="parent_asin", right_on="parent_asin", how="left")
#
# merged_df = pd.get_dummies(merged_df, columns=["main_category"], drop_first=False)

# ---------- Select Features ----------

feature_columns = [
    "review_length",
    "sentiment_score",
    "is_verified",
    "timestamp",
]

X = reviews_df[feature_columns]
y = reviews_df["helpfulness_ratio_log"]

# Drop rows with missing values
mask = X.notnull().all(axis=1) & y.notnull()
X = X[mask]
y = y[mask]
reviews_df = reviews_df.loc[mask]

# ---------- Split ----------

X_train, X_temp, y_train, y_temp, text_train, text_temp, rating_train, rating_temp, hv_train, hv_temp = train_test_split(
    X, y, reviews_df["full_text"], reviews_df["rating"], reviews_df["helpful_vote"],
    test_size=0.3,
    random_state=42,
    stratify=pd.qcut(y, q=10, duplicates="drop")
)

X_val, X_test, y_val, y_test, text_val, text_test, rating_val, rating_test, hv_val, hv_test = train_test_split(
    X_temp, y_temp, text_temp, rating_temp, hv_temp,
    test_size=0.5,
    random_state=42,
    stratify=pd.qcut(y_temp, q=5, duplicates="drop")
)

# ---------- Save ----------

train_df = pd.concat([X_train, y_train, text_train, rating_train, hv_train], axis=1)
val_df = pd.concat([X_val, y_val, text_val, rating_val, hv_val], axis=1)
test_df = pd.concat([X_test, y_test, text_test, rating_test, hv_test], axis=1)

train_df.to_csv("../datasets/preprocessed/train.csv", index=False)
val_df.to_csv("../datasets/preprocessed/val.csv", index=False)
test_df.to_csv("../datasets/preprocessed/test.csv", index=False)
