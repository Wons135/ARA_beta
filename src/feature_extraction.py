# feature_extraction.py (Windows-friendly, CPU-only, tuned TF-IDF + SVD)
import os, json, time, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import sparse as sp
from sklearn.feature_extraction.text import (
    TfidfVectorizer as SK_TfidfVectorizer,
    HashingVectorizer as SK_HashingVectorizer,
    TfidfTransformer as SK_TfidfTransformer,
)
from sklearn.preprocessing import StandardScaler as SK_StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, SelectFromModel
from sklearn.decomposition import TruncatedSVD as SK_TruncatedSVD
from textblob import TextBlob
from tqdm import tqdm

# =============================
# CONFIG
# =============================
TASK = "regression"  # "regression" | "binary"

# ---- Speed knobs ----
USE_TEXTBLOB = False           # OFF for big runs (major speedup)
SENT_BATCH   = 50_000

# ---- Text config (tuned) ----
TFIDF_BACKEND = "tfidf"        # "tfidf" | "hashing"
TFIDF_MAX_FEATURES = 20_000    # ↑ features
TFIDF_NGRAM_RANGE = (1, 2)     # add bigrams
TFIDF_MIN_DF = 2               # prune rare terms
TFIDF_SUBLINEAR_TF = True

# Hashing backend settings (only if TFIDF_BACKEND="hashing")
HASHING_N_FEATURES = 2**20
HASHING_ALT_SIGN   = False

# ---- Dimensionality reduction (tuned) ----
APPLY_FEATURE_SELECTION   = False        # OFF by default (SVD is enough)
FEATURE_SELECTION_METHOD  = "kbest"      # kept for optional use
NUM_FEATURES_SELECTED     = 2000
APPLY_DIM_REDUCTION = True
NUM_SVD_COMPONENTS  = 300                # ↑ components

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

task_to_label_column = {"regression": "regression_score", "binary": "binary_label"}
label_column = task_to_label_column[TASK]
task_suffix = "regression" if TASK == "regression" else "binary"

# Paths
base_path  = "../datasets/preprocessed"
plots_dir  = "../outputs/plots"
model_path = "../outputs/feature_models"
os.makedirs(model_path, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

train_path = os.path.join(base_path, f"{label_column}_train.csv")
val_path   = os.path.join(base_path, f"{label_column}_val.csv")
test_path  = os.path.join(base_path, f"{label_column}_test.csv")

AUX_COLS = ["sentiment_score", "review_length", "is_verified", "timestamp_recency_days"]

# =============================
# Load & clean
# =============================
def load_and_clean(path: str) -> pd.DataFrame:
    t0 = time.time()
    df = pd.read_csv(path)
    df = df[df["full_text"].notna()]
    df = df[df["full_text"].astype(str).str.split().str.len() >= 10]
    df = df.reset_index(drop=True)
    print(f"Loaded {path} -> rows={len(df)} in {time.time()-t0:.1f}s")
    return df

train_df = load_and_clean(train_path)
val_df   = load_and_clean(val_path)
test_df  = load_and_clean(test_path)

# =============================
# Helpers
# =============================
def to_days(series): return series.astype(float) / 86400.0

def plot_target_distribution(df, target_column, set_name):
    data = df[target_column].copy()
    if TASK == "regression":
        upper = np.percentile(data, 99.0); data = data[data <= upper]
    plt.figure(figsize=(6,4))
    plt.hist(data, bins=30, edgecolor='black')
    plt.title(f"{set_name} Target Distribution ({TASK})")
    plt.xlabel("Target"); plt.ylabel("Freq"); plt.grid(True, alpha=0.3)
    out = os.path.join(plots_dir, f"target_hist_{set_name.lower()}_{task_suffix}.png")
    plt.tight_layout(); plt.savefig(out); plt.close()

print("Plotting target histograms…")
plot_target_distribution(train_df, label_column, "Train")
plot_target_distribution(val_df,   label_column, "Validation")
plot_target_distribution(test_df,  label_column, "Test")

t0_days = to_days(train_df["timestamp"]).min() if "timestamp" in train_df.columns else 0.0

def compute_sentiment_series(text_series: pd.Series) -> pd.Series:
    if not USE_TEXTBLOB:
        return pd.Series(np.zeros(len(text_series), dtype=np.float32), index=text_series.index)
    arr = text_series.astype(str).values
    scores = np.empty(len(arr), dtype=np.float32)
    for start in tqdm(range(0, len(arr), SENT_BATCH), desc="Sentiment (TextBlob)", unit="batch"):
        end = min(start + SENT_BATCH, len(arr))
        batch = arr[start:end]
        scores[start:end] = [float(TextBlob(x).sentiment.polarity) for x in batch]
    return pd.Series(scores, index=text_series.index)

# =============================
# Feature extraction
# =============================
def extract_features(df: pd.DataFrame, text_column: str, is_train: bool = False) -> sp.csr_matrix:
    stage_t0 = time.time()
    n = len(df)
    print(f"\n=== Extracting features ({'TRAIN' if is_train else 'VAL/TEST'}) on {n:,} rows ===")
    txt = df[text_column].fillna("")

    # ---------- Text → vectors ----------
    t0 = time.time()
    if TFIDF_BACKEND == "hashing":
        print(f"Vectorizer: HashingVectorizer(n_features={HASHING_N_FEATURES}, alt_sign={HASHING_ALT_SIGN}) + IDF")
        hv = SK_HashingVectorizer(n_features=HASHING_N_FEATURES,
                                  alternate_sign=HASHING_ALT_SIGN,
                                  norm=None, dtype=np.float32)
        X_counts = hv.transform(txt)
        if is_train:
            idf = SK_TfidfTransformer(use_idf=True, norm="l2")
            X_tfidf = idf.fit_transform(X_counts)
            joblib.dump(idf, os.path.join(model_path, f"idf_{task_suffix}.pkl"))
            joblib.dump({"n_features": HASHING_N_FEATURES, "alternate_sign": HASHING_ALT_SIGN},
                        os.path.join(model_path, f"hashing_meta_{task_suffix}.pkl"))
        else:
            idf = joblib.load(os.path.join(model_path, f"idf_{task_suffix}.pkl"))
            X_tfidf = idf.transform(X_counts)
        tfidf_matrix = X_tfidf
    else:
        print(f"Vectorizer: TfidfVectorizer(max_features={TFIDF_MAX_FEATURES}, ngram_range={TFIDF_NGRAM_RANGE})")
        if is_train:
            tfidf = SK_TfidfVectorizer(
                max_features=TFIDF_MAX_FEATURES,
                ngram_range=TFIDF_NGRAM_RANGE,
                min_df=TFIDF_MIN_DF,
                sublinear_tf=TFIDF_SUBLINEAR_TF,
                dtype=np.float32,
            )
            tfidf_matrix = tfidf.fit_transform(txt)
            joblib.dump(tfidf, os.path.join(model_path, f"tfidf_{task_suffix}.pkl"))
        else:
            tfidf = joblib.load(os.path.join(model_path, f"tfidf_{task_suffix}.pkl"))
            tfidf_matrix = tfidf.transform(txt)
    print(f"  Text vectorization done in {time.time()-t0:.1f}s -> shape={tfidf_matrix.shape}")

    # ---------- AUX features ----------
    print("AUX features: sentiment/length/flags/recency…")
    t0 = time.time()
    feats = pd.DataFrame(index=df.index)
    feats["sentiment_score"] = compute_sentiment_series(txt)
    feats["review_length"] = 0.0
    for i in tqdm(range(0, n, SENT_BATCH), desc="Review length", unit="batch"):
        j = min(i+SENT_BATCH, n)
        feats.iloc[i:j, feats.columns.get_loc("review_length")] = [
            float(len(s.split())) for s in txt.iloc[i:j].astype(str).values
        ]
    feats["is_verified"] = (df["is_verified"].astype(float) if "is_verified" in df.columns else 0.0)
    if "timestamp" in df.columns:
        feats["timestamp_recency_days"] = to_days(df["timestamp"]) - t0_days
    else:
        feats["timestamp_recency_days"] = 0.0
    feats["review_length"] = np.log1p(feats["review_length"].astype(float))
    feats = feats[AUX_COLS].astype(np.float32)
    print(f"  AUX compute done in {time.time()-t0:.1f}s")

    # ---------- Scale AUX ----------
    print("Scaling AUX (StandardScaler)…")
    t0 = time.time()
    scaler = SK_StandardScaler(with_mean=True, with_std=True)
    if is_train:
        scaled_aux = scaler.fit_transform(feats.values)
        joblib.dump(scaler, os.path.join(model_path, f"scaler_{task_suffix}.pkl"))
        joblib.dump({"aux_cols": AUX_COLS, "t0_days": float(t0_days)},
                    os.path.join(model_path, f"aux_meta_{task_suffix}.pkl"))
    else:
        scaler = joblib.load(os.path.join(model_path, f"scaler_{task_suffix}.pkl"))
        scaled_aux = scaler.transform(feats.values)
    aux_sparse = sp.csr_matrix(scaled_aux.astype(np.float32))
    print(f"  Scaling done in {time.time()-t0:.1f}s -> aux_shape={aux_sparse.shape}")

    # ---------- Combine AUX + TF-IDF ----------
    print("Combining AUX + text features…")
    t0 = time.time()
    combined = sp.hstack([aux_sparse, tfidf_matrix], format="csr", dtype=np.float32)
    print(f"  Combine done in {time.time()-t0:.1f}s -> shape={combined.shape}")

    # ---------- (Optional) Feature selection ----------
    if APPLY_FEATURE_SELECTION:
        print(f"Feature selection: {FEATURE_SELECTION_METHOD} ({'fit→transform' if is_train else 'transform'})")
        tfs = time.time()
        if is_train:
            if FEATURE_SELECTION_METHOD == "kbest":
                score_func = f_regression if TASK == "regression" else f_classif
                k = min(NUM_FEATURES_SELECTED, combined.shape[1])
                selector = SelectKBest(score_func=score_func, k=k)
            else:
                # Model-based FS is expensive on 7.7M rows; keep off unless you really need it.
                raise RuntimeError("Model-based feature selection is disabled for scale.")
            selector.fit(combined, df[label_column].values)
            joblib.dump(selector, os.path.join(model_path, f"selector_{task_suffix}.pkl"))
        else:
            selector = joblib.load(os.path.join(model_path, f"selector_{task_suffix}.pkl"))
        combined = selector.transform(combined)
        if not sp.issparse(combined):
            combined = sp.csr_matrix(combined, dtype=np.float32)
        print(f"  FS done in {time.time()-tfs:.1f}s -> shape={combined.shape}")

    # ---------- Dimensionality reduction ----------
    if APPLY_DIM_REDUCTION:
        n_comp = min(NUM_SVD_COMPONENTS, max(1, combined.shape[1] - 1))
        print(f"SVD (TruncatedSVD, n_components={n_comp})…")
        tsvd = time.time()
        svd = SK_TruncatedSVD(n_components=n_comp, random_state=RANDOM_STATE)
        if is_train:
            reduced = svd.fit_transform(combined)
            joblib.dump(svd, os.path.join(model_path, f"svd_{task_suffix}.pkl"))
        else:
            svd = joblib.load(os.path.join(model_path, f"svd_{task_suffix}.pkl"))
            reduced = svd.transform(combined)
        combined = sp.csr_matrix(reduced.astype(np.float32))
        print(f"  SVD done in {time.time()-tsvd:.1f}s -> shape={combined.shape}")

    print(f"=== Done ({'TRAIN' if is_train else 'VAL/TEST'}) in {time.time()-stage_t0:.1f}s ===")
    return combined

# Build & save
X_train = extract_features(train_df, "full_text", is_train=True)
X_val   = extract_features(val_df,   "full_text", is_train=False)
X_test  = extract_features(test_df,  "full_text", is_train=False)

print("Saving sparse matrices and labels…")
t0 = time.time()
sp.save_npz(os.path.join(base_path, f"X_train_features_{task_suffix}.npz"), X_train)
sp.save_npz(os.path.join(base_path, f"X_val_features_{task_suffix}.npz"),   X_val)
sp.save_npz(os.path.join(base_path, f"X_test_features_{task_suffix}.npz"),  X_test)

train_df[[label_column]].to_csv(os.path.join(base_path, f"y_train_{task_suffix}.csv"), index=False)
val_df[[label_column]].to_csv(  os.path.join(base_path, f"y_val_{task_suffix}.csv"),   index=False)
test_df[[label_column]].to_csv( os.path.join(base_path, f"y_test_{task_suffix}.csv"),  index=False)
print(f"Saved in {time.time()-t0:.1f}s")

meta = {
    "task": TASK,
    "label_column": label_column,
    "rows": {"train": int(X_train.shape[0]), "val": int(X_val.shape[0]), "test": int(X_test.shape[0])},
    "tfidf_backend": TFIDF_BACKEND,
    "tfidf_max_features": TFIDF_MAX_FEATURES,
    "tfidf_ngram_range": TFIDF_NGRAM_RANGE,
    "apply_feature_selection": APPLY_FEATURE_SELECTION,
    "num_svd_components": int(NUM_SVD_COMPONENTS) if APPLY_DIM_REDUCTION else None,
    "use_textblob": USE_TEXTBLOB,
    "sent_batch": SENT_BATCH,
}
with open(os.path.join(base_path, f"labels_meta_{task_suffix}.json"), "w") as f:
    json.dump(meta, f, indent=2)

print(f"✅ Saved sparse features and labels for TASK='{TASK}' ({task_suffix}).")
print(f"   X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
