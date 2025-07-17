import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob

# Load pre-split datasets
train_df = pd.read_csv("../datasets/preprocessed/train.csv")
test_df = pd.read_csv("../datasets/preprocessed/test.csv")
val_df = pd.read_csv("../datasets/preprocessed/val.csv")

# ---------- Add smoothed helpfulness score ----------

def smoothed_helpfulness(hv, tv, alpha=1, beta=1):
    return (hv + alpha) / (tv + alpha + beta)

# Apply to all splits
for df in [train_df, val_df, test_df]:
    # Assuming 'helpful_vote' and 'rating' columns exist in your CSV
    df["smoothed_helpfulness_score"] = df.apply(
        lambda row: smoothed_helpfulness(row["helpful_vote"], row["rating"]) if row["rating"] > 0 else np.nan,
        axis=1
    )
    # Replace original target column
    df["helpfulness_score"] = df["smoothed_helpfulness_score"]
    df.drop(columns=["smoothed_helpfulness_score"], inplace=True)

# ---------- Check target distribution ----------

def plot_target_distribution(df, target_column, set_name):
    data = df[target_column]
    upper_limit = np.percentile(data, 99)
    plt.hist(data[data <= upper_limit], bins=30, edgecolor='black')
    plt.title(f"{set_name} Target Distribution")
    plt.xlabel("Target Value (Helpfulness score)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

plot_target_distribution(train_df, "helpfulness_score", "Train")
plot_target_distribution(val_df, "helpfulness_score", "Validation")
plot_target_distribution(test_df, "helpfulness_score", "Test")

# ---------- Feature extraction ----------

def extract_features(df, text_column):
    features = pd.DataFrame()

    tfidf = TfidfVectorizer(max_features=5000)
    if 'fit_tfidf' not in globals():
        global fit_tfidf
        fit_tfidf = tfidf.fit(train_df[text_column])
    tfidf_features = fit_tfidf.transform(df[text_column]).toarray()
    tfidf_df = pd.DataFrame(tfidf_features, columns=[f"tfidf_{i}" for i in range(tfidf_features.shape[1])])

    sentiments = df[text_column].apply(lambda x: TextBlob(x).sentiment.polarity)
    features["sentiment_score"] = sentiments

    features["review_length"] = df[text_column].apply(lambda x: len(x.split()))

    final_features = pd.concat([features, tfidf_df], axis=1)

    scaler = StandardScaler()
    features_to_scale = ["sentiment_score", "review_length"]
    final_features[features_to_scale] = scaler.fit_transform(final_features[features_to_scale])

    return final_features

X_train = extract_features(train_df, "full_text")
X_val = extract_features(val_df, "full_text")
X_test = extract_features(test_df, "full_text")

# ---------- Save extracted features and updated targets ----------

X_train.to_csv("../datasets/preprocessed/X_train_features.csv", index=False)
X_val.to_csv("../datasets/preprocessed/X_val_features.csv", index=False)
X_test.to_csv("../datasets/preprocessed/X_test_features.csv", index=False)

train_df[["helpfulness_score"]].to_csv("../datasets/preprocessed/y_train.csv", index=False)
val_df[["helpfulness_score"]].to_csv("../datasets/preprocessed/y_val.csv", index=False)
test_df[["helpfulness_score"]].to_csv("../datasets/preprocessed/y_test.csv", index=False)
