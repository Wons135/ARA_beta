# preprocess_and_split.py (updated with metadata integration)

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

reviews_path = 'datasets/Movies_and_TV.jsonl'
meta_path = 'datasets/meta_Movies_and_TV.jsonl'

reviews_list = []
with open(reviews_path, 'r') as fp:
    for line in tqdm(fp, desc="Loading reviews"):
        data = json.loads(line.strip())
        reviews_list.append(data)

reviews_df = pd.DataFrame(reviews_list)

# ---------- Load Metadata ----------

meta_list = []
with open(meta_path, 'r') as fp:
    for line in tqdm(fp, desc="Loading metadata"):
        data = json.loads(line.strip())
        meta_list.append(data)

meta_df = pd.DataFrame(meta_list)

# ---------- Preprocess Reviews ----------

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = text.lower()
    return text.strip()

def compute_helpfulness(row):
    hv = row.get("helpfulVotes", 0)
    tv = row.get("totalVotes", 0)
    if tv == 0:
        return np.nan
    return hv / tv

reviews_df["clean_review"] = reviews_df["reviewText"].apply(clean_text)
reviews_df["clean_summary"] = reviews_df["summary"].apply(clean_text)
reviews_df["full_text"] = reviews_df["clean_summary"].fillna('') + ". " + reviews_df["clean_review"].fillna('')
reviews_df["helpfulness_ratio"] = reviews_df.apply(compute_helpfulness, axis=1)

reviews_df = reviews_df[reviews_df["totalVotes"] >= 5].copy()
reviews_df = reviews_df.dropna(subset=["helpfulness_ratio"])

reviews_df["review_length"] = reviews_df["clean_review"].apply(lambda x: len(x.split()))
reviews_df["sentiment_score"] = reviews_df["clean_review"].apply(lambda x: sia.polarity_scores(x)["compound"])
reviews_df["is_verified"] = reviews_df["verified"].astype(int)

def compute_review_age(date_str):
    try:
        return (datetime.now() - datetime.strptime(date_str, "%m %d, %Y")).days
    except:
        return np.nan

reviews_df["review_age"] = reviews_df["reviewTime"].apply(compute_review_age)

# ---------- Preprocess Metadata ----------

meta_df["price"] = pd.to_numeric(meta_df.get("price", np.nan), errors='coerce')
meta_df["has_brand"] = meta_df["brand"].notnull().astype(int)
meta_df["has_description"] = meta_df["description"].notnull().astype(int)
meta_df["main_category"] = meta_df["category"].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else "Unknown")

# ---------- Merge ----------

merged_df = reviews_df.merge(meta_df[["asin", "price", "has_brand", "has_description", "main_category"]], on="asin", how="left")

# One-hot encode main_category
merged_df = pd.get_dummies(merged_df, columns=["main_category"], drop_first=True)

# ---------- Select Features ----------

feature_columns = [
    "review_length",
    "sentiment_score",
    "is_verified",
    "review_age",
    "price",
    "has_brand",
    "has_description"
] + [col for col in merged_df.columns if col.startswith("main_category_")]

X = merged_df[feature_columns]
y = merged_df["helpfulness_ratio"]

# Drop rows with missing values
mask = X.notnull().all(axis=1) & y.notnull()
X = X[mask]
y = y[mask]
merged_df = merged_df.loc[mask]

# ---------- Split ----------

X_train, X_temp, y_train, y_temp, text_train, text_temp = train_test_split(
    X, y, merged_df["full_text"],
    test_size=0.3,
    random_state=42,
    stratify=pd.qcut(y, q=10, duplicates="drop")
)

X_val, X_test, y_val, y_test, text_val, text_test = train_test_split(
    X_temp, y_temp, text_temp,
    test_size=0.5,
    random_state=42,
    stratify=pd.qcut(y_temp, q=5, duplicates="drop")
)

# ---------- Save ----------

train_df = pd.concat([X_train, y_train, text_train], axis=1)
val_df = pd.concat([X_val, y_val, text_val], axis=1)
test_df = pd.concat([X_test, y_test, text_test], axis=1)

train_df.to_csv("datasets/preprocessed/train.csv", index=False)
val_df.to_csv("datasets/preprocessed/val.csv", index=False)
test_df.to_csv("datasets/preprocessed/test.csv", index=False)
