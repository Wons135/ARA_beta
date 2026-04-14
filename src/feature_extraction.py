from __future__ import annotations

import os, json, time, argparse, logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from numpy.lib.format import open_memmap  # ensures .npy header
from scipy import sparse as sp
from sklearn.feature_extraction.text import (
    TfidfVectorizer as SK_TfidfVectorizer,
    HashingVectorizer as SK_HashingVectorizer,
    TfidfTransformer as SK_TfidfTransformer,
)
from sklearn.preprocessing import StandardScaler as SK_StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.decomposition import TruncatedSVD as SK_TruncatedSVD
from textblob import TextBlob
from tqdm import tqdm
import joblib

# =============================
# CONFIG (defaults; can be overridden via CLI)
# =============================
TASK = "regression"  # "regression" | "binary"

# ---- Speed knobs ----
USE_TEXTBLOB = False
SENT_BATCH   = 50_000

# ---- Text config (tuned) ----
TFIDF_BACKEND = "tfidf"        # "tfidf" | "hashing"
TFIDF_MAX_FEATURES = 20_000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 2
TFIDF_SUBLINEAR_TF = True

# Hashing backend settings
HASHING_N_FEATURES = 2**20
HASHING_ALT_SIGN   = False

# ---- Dimensionality reduction (tuned) ----
APPLY_FEATURE_SELECTION   = False
FEATURE_SELECTION_METHOD  = "kbest"
NUM_FEATURES_SELECTED     = 2000

APPLY_DIM_REDUCTION = True
NUM_SVD_COMPONENTS  = 100
SVD_BATCH_ROWS      = 200_000
SAVE_DENSE_NPY      = True

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Paths (defaults; canonicalized later)
BASE_PATH  = "../datasets/preprocessed"
PLOTS_DIR  = "../outputs/plots"
MODEL_PATH = "../outputs/feature_models"

task_to_label_column = {"regression": "regression_score", "binary": "binary_label"}

AUX_COLS = ["sentiment_score", "review_length", "is_verified", "timestamp_recency_days"]

# =============================
# Logging & safety
# =============================
def setup_logger(level: str = "INFO") -> logging.LoggerAdapter:
    logger = logging.getLogger("features")
    if not logger.handlers:
        h = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s [%(levelname)s) trace_id=%(trace_id)s %(message)s",
                                datefmt="%Y-%m-%d %H:%M:%S")
        h.setFormatter(fmt)
        logger.addHandler(h)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logging.LoggerAdapter(logger, {"trace_id": os.getpid()})

def canonicalize(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()

def ensure_under(base: Path, *paths: Path) -> None:
    """
    Assert that each path is contained under 'base', using proper path semantics.
    Avoids fragile string-prefix checks (see pathlib docs).
    """
    base = base.resolve()
    for p in paths:
        p = p.resolve()
        # Path.is_relative_to is available in Python 3.9+ and is the safest check
        if hasattr(p, "is_relative_to"):
            if not p.is_relative_to(base):
                raise ValueError(f"Unsafe path outside base: {p} (base: {base})")
        else:
            # Fallback: attempt relative_to, which raises ValueError if not under base
            try:
                p.relative_to(base)
            except ValueError:
                raise ValueError(f"Unsafe path outside base: {p} (base: {base})")

def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)

def ensure_headless(headless: bool, logger: logging.LoggerAdapter) -> None:
    if headless:
        matplotlib.use("Agg")
        logger.info("Using headless matplotlib backend 'Agg'.")

# =============================
# Load & schema checks
# =============================
def load_and_clean(path: Path, logger: logging.LoggerAdapter) -> pd.DataFrame:
    t0 = time.time()
    df = pd.read_csv(path)
    # Basic schema presence
    required = {"full_text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df[df["full_text"].notna()]
    df = df[df["full_text"].astype(str).str.split().str.len() >= 10]
    df = df.reset_index(drop=True)
    logger.info(f"Loaded {path.name}: rows={len(df)} in {time.time()-t0:.1f}s")
    return df

def to_days(series: pd.Series) -> pd.Series:
    # If already in days, keep; else treat as seconds and convert.
    s = pd.to_numeric(series, errors="coerce")
    return (s / 86400.0).astype(float)

def plot_target_distribution(df: pd.DataFrame, target_column: str, set_name: str,
                             out_dir: Path, task_suffix: str, logger: logging.LoggerAdapter) -> None:
    data = df[target_column].copy()
    if TASK == "regression":
        upper = np.percentile(data, 99.0)
        data = data[data <= upper]
    plt.figure(figsize=(6,4))
    plt.hist(data, bins=30, edgecolor='black')
    plt.title(f"{set_name} Target Distribution ({TASK})")
    plt.xlabel("Target"); plt.ylabel("Freq"); plt.grid(True, alpha=0.3)
    out = out_dir / f"target_hist_{set_name.lower()}_{task_suffix}.png"
    plt.tight_layout(); plt.savefig(out); plt.close()
    logger.info(f"Saved plot: {out}")

def compute_sentiment_series(text_series: pd.Series, logger: logging.LoggerAdapter) -> pd.Series:
    if not USE_TEXTBLOB:
        return pd.Series(np.zeros(len(text_series), dtype=np.float32), index=text_series.index)
    arr = text_series.astype(str).values
    scores = np.empty(len(arr), dtype=np.float32)
    for start in tqdm(range(0, len(arr), SENT_BATCH), desc="Sentiment (TextBlob)", unit="batch"):
        end = min(start + SENT_BATCH, len(arr))
        batch = arr[start:end]
        scores[start:end] = [float(TextBlob(x).sentiment.polarity) for x in batch]
    logger.info("Computed TextBlob sentiment.")
    return pd.Series(scores, index=text_series.index)

# =============================
# Streamed SVD helper (.npy with header)
# =============================
def svd_fit_and_transform_to_memmap(
    X_sparse, n_components: int, out_path: Path, is_train: bool,
    model_dir: Path, task_suffix: str, allow_untrusted: bool, logger: logging.LoggerAdapter
) -> Path:
    """Fit (if train) TruncatedSVD and stream-transform X into a disk-backed .npy memmap."""
    svd_path = model_dir / f"svd_{task_suffix}.pkl"
    if is_train:
        _t = time.time()
        svd = SK_TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
        svd.fit(X_sparse)
        logger.info(f"SVD fit in {time.time()-_t:.1f}s")
        joblib.dump(svd, svd_path)
    else:
        # Guard artifact loading: only canonical path under model_dir unless explicitly allowed.
        if not allow_untrusted:
            ensure_under(model_dir, svd_path)
        svd = joblib.load(svd_path)

    n_rows = X_sparse.shape[0]
    n_comp = svd.n_components
    mm = open_memmap(out_path, mode="w+", dtype=np.float32, shape=(n_rows, n_comp))

    for start in range(0, n_rows, SVD_BATCH_ROWS):
        end = min(start + SVD_BATCH_ROWS, n_rows)
        _tb = time.time()
        chunk = X_sparse[start:end]               # CSR slice
        reduced = svd.transform(chunk)            # dense
        mm[start:end, :] = reduced.astype(np.float32, copy=False)
        logger.info(f"SVD transform {start:,}:{end:,} in {time.time()-_tb:.1f}s")

    del mm  # flush
    return out_path

# =============================
# Feature extraction
# =============================
def extract_features(
    df: pd.DataFrame, text_column: str, split: str, is_train: bool,
    base_path: Path, plots_dir: Path, model_dir: Path, task: str,
    allow_untrusted: bool, logger: logging.LoggerAdapter
):
    stage_t0 = time.time()
    n = len(df)
    label_column = task_to_label_column[task]
    task_suffix = "regression" if task == "regression" else "binary"
    logger.info(f"=== Extracting features ({split.upper()}) on {n:,} rows ===")

    txt = df[text_column].fillna("")

    # ---------- Text → vectors ----------
    t0 = time.time()
    if TFIDF_BACKEND == "hashing":
        logger.info(f"Vectorizer: HashingVectorizer(n_features={HASHING_N_FEATURES}, alt_sign={HASHING_ALT_SIGN}) + IDF")
        hv = SK_HashingVectorizer(n_features=HASHING_N_FEATURES,
                                  alternate_sign=HASHING_ALT_SIGN,
                                  norm=None, dtype=np.float32)
        X_counts = hv.transform(txt)
        idf_path = model_dir / f"idf_{task_suffix}.pkl"
        if is_train:
            idf = SK_TfidfTransformer(use_idf=True, norm="l2")
            X_tfidf = idf.fit_transform(X_counts)
            joblib.dump(idf, idf_path)
            meta_path = model_dir / f"hashing_meta_{task_suffix}.pkl"
            joblib.dump({"n_features": HASHING_N_FEATURES, "alternate_sign": HASHING_ALT_SIGN}, meta_path)
        else:
            if not allow_untrusted:
                ensure_under(model_dir, idf_path)
            idf = joblib.load(idf_path)
            X_tfidf = idf.transform(X_counts)
        tfidf_matrix = X_tfidf
    else:
        logger.info(f"Vectorizer: TfidfVectorizer(max_features={TFIDF_MAX_FEATURES}, ngram_range={TFIDF_NGRAM_RANGE})")
        tfidf_path = model_dir / f"tfidf_{task_suffix}.pkl"
        if is_train:
            tfidf = SK_TfidfVectorizer(
                max_features=TFIDF_MAX_FEATURES,
                ngram_range=TFIDF_NGRAM_RANGE,
                min_df=TFIDF_MIN_DF,
                sublinear_tf=TFIDF_SUBLINEAR_TF,
                dtype=np.float32,
            )
            tfidf_matrix = tfidf.fit_transform(txt)
            joblib.dump(tfidf, tfidf_path)
        else:
            if not allow_untrusted:
                ensure_under(model_dir, tfidf_path)
            tfidf = joblib.load(tfidf_path)
            tfidf_matrix = tfidf.transform(txt)
    logger.info(f"Text vectorization in {time.time()-t0:.1f}s -> shape={tfidf_matrix.shape}")

    # ---------- AUX features ----------
    logger.info("AUX features: sentiment/length/flags/recency…")
    t0 = time.time()
    feats = pd.DataFrame(index=df.index)
    feats["sentiment_score"] = compute_sentiment_series(txt, logger)
    # batched review length
    feats["review_length"] = 0.0
    for i in tqdm(range(0, n, SENT_BATCH), desc="Review length", unit="batch"):
        j = min(i+SENT_BATCH, n)
        feats.iloc[i:j, feats.columns.get_loc("review_length")] = [
            float(len(s.split())) for s in txt.iloc[i:j].astype(str).values
        ]
    feats["is_verified"] = (df["is_verified"].astype(float) if "is_verified" in df.columns else 0.0)
    if "timestamp" in df.columns:
        t0_days = to_days(df["timestamp"]).min()
        feats["timestamp_recency_days"] = to_days(df["timestamp"]) - t0_days
    else:
        feats["timestamp_recency_days"] = 0.0
    feats["review_length"] = np.log1p(feats["review_length"].astype(float))
    feats = feats[AUX_COLS].astype(np.float32)
    logger.info(f"AUX compute in {time.time()-t0:.1f}s")

    # ---------- Scale AUX ----------
    logger.info("Scaling AUX (StandardScaler)…")
    t0 = time.time()
    scaler = SK_StandardScaler(with_mean=True, with_std=True)
    scaler_path = model_dir / f"scaler_{task_suffix}.pkl"
    aux_meta_path = model_dir / f"aux_meta_{task_suffix}.pkl"
    if is_train:
        scaled_aux = scaler.fit_transform(feats.values)
        joblib.dump(scaler, scaler_path)
        joblib.dump({"aux_cols": AUX_COLS}, aux_meta_path)
    else:
        if not allow_untrusted:
            ensure_under(model_dir, scaler_path)
        scaler = joblib.load(scaler_path)
        scaled_aux = scaler.transform(feats.values)
    aux_sparse = sp.csr_matrix(scaled_aux.astype(np.float32))
    logger.info(f"Scaling in {time.time()-t0:.1f}s -> aux_shape={aux_sparse.shape}")

    # ---------- Combine AUX + TF-IDF ----------
    logger.info("Combining AUX + text features…")
    t0 = time.time()
    combined = sp.hstack([aux_sparse, tfidf_matrix], format="csr", dtype=np.float32)
    logger.info(f"Combine in {time.time()-t0:.1f}s -> shape={combined.shape}")

    # ---------- (Optional) Feature selection ----------
    if APPLY_FEATURE_SELECTION:
        logger.info(f"Feature selection: {FEATURE_SELECTION_METHOD} ({'fit→transform' if is_train else 'transform'})")
        tfs = time.time()
        if is_train:
            score_func = f_regression if task == "regression" else f_classif
            k = min(NUM_FEATURES_SELECTED, combined.shape[1])
            selector = SelectKBest(score_func=score_func, k=k)
            selector.fit(combined, df[task_to_label_column[task]].values)
            joblib.dump(selector, model_dir / f"selector_{task_suffix}.pkl")
        else:
            sel_path = model_dir / f"selector_{task_suffix}.pkl"
            if not allow_untrusted:
                ensure_under(model_dir, sel_path)
            selector = joblib.load(sel_path)
        combined = selector.transform(combined)
        if not sp.issparse(combined):
            combined = sp.csr_matrix(combined, dtype=np.float32)
        logger.info(f"FS in {time.time()-tfs:.1f}s -> shape={combined.shape}")

    # ---------- Dimensionality reduction (streamed to .npy) ----------
    if APPLY_DIM_REDUCTION:
        n_comp = min(NUM_SVD_COMPONENTS, max(1, combined.shape[1] - 1))
        logger.info(f"SVD (TruncatedSVD → .npy memmap, n_components={n_comp})…")
        tsvd = time.time()
        out_path = base_path / f"X_{split}_features_{task_suffix}_svd.npy"
        svd_fit_and_transform_to_memmap(
            combined, n_comp, out_path, is_train=is_train,
            model_dir=model_dir, task_suffix=task_suffix,
            allow_untrusted=allow_untrusted, logger=logger
        )
        logger.info(f"SVD stream in {time.time()-tsvd:.1f}s -> saved: {out_path}")
        logger.info(f"=== Done ({split.upper()}) in {time.time()-stage_t0:.1f}s ===")
        return str(out_path)
    else:
        logger.info(f"=== Done ({split.upper()}) in {time.time()-stage_t0:.1f}s ===")
        return combined  # sparse CSR

# =============================
# CLI entry & main
# =============================
def main():
    parser = argparse.ArgumentParser(description="Feature extraction with targeted hardening.")
    parser.add_argument("--task", choices=["regression", "binary"], default=TASK)
    parser.add_argument("--base-path", default=BASE_PATH)
    parser.add_argument("--plots-dir", default=PLOTS_DIR)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--headless-plots", action="store_true", help="Force Agg backend for headless runs.")
    parser.add_argument("--allow-untrusted-artifacts", action="store_true",
                        help="Allow loading joblib artifacts outside the trusted model directory (NOT RECOMMENDED).")
    args = parser.parse_args()

    logger = setup_logger(args.log_level)
    ensure_headless(args.headless_plots, logger)

    # Canonicalize and restrict directories
    base_path  = canonicalize(args.base_path)
    plots_dir  = canonicalize(args.plots_dir)
    model_path = canonicalize(args.model_path)
    ensure_dirs(base_path, plots_dir, model_path)

    # Compute a trusted root that actually contains *all* of our paths.
    # Using commonpath is correct for this purpose (official docs).
    # If paths are on different drives on Windows, commonpath will raise; surface a clear error.
    try:
        common_root_str = os.path.commonpath([str(base_path), str(plots_dir), str(model_path)])
    except ValueError as e:
        raise ValueError(
            "Configured paths are on different drives or have no common ancestor; "
            "cannot establish a trusted root for artifact safety. "
            f"base_path={base_path}, plots_dir={plots_dir}, model_path={model_path}"
        ) from e
    trusted_root = Path(common_root_str).resolve()

    # Guard that all paths are under the computed trusted root.
    ensure_under(trusted_root, base_path, plots_dir, model_path)

    task = args.task
    task_suffix = "regression" if task == "regression" else "binary"
    label_column = task_to_label_column[task]

    # Inputs
    train_csv = base_path / f"{label_column}_train.csv"
    val_csv   = base_path / f"{label_column}_val.csv"
    test_csv  = base_path / f"{label_column}_test.csv"

    # Load
    train_df = load_and_clean(train_csv, logger)
    val_df   = load_and_clean(val_csv, logger)
    test_df  = load_and_clean(test_csv, logger)

    # Plots (optional but on by default)
    logger.info("Plotting target histograms…")
    plot_target_distribution(train_df, label_column, "Train", plots_dir, task_suffix, logger)
    plot_target_distribution(val_df,   label_column, "Validation", plots_dir, task_suffix, logger)
    plot_target_distribution(test_df,  label_column, "Test", plots_dir, task_suffix, logger)

    # Extract
    X_train = extract_features(train_df, "full_text", "train", True,
                               base_path, plots_dir, model_path, task, args.allow_untrusted_artifacts, logger)
    X_val   = extract_features(val_df,   "full_text", "val",   False,
                               base_path, plots_dir, model_path, task, args.allow_untrusted_artifacts, logger)
    X_test  = extract_features(test_df,  "full_text", "test",  False,
                               base_path, plots_dir, model_path, task, args.allow_untrusted_artifacts, logger)

    # Save features paths + labels
    logger.info("Saving features and labels…")
    t0 = time.time()

    def _maybe_save_sparse_or_npy(obj_or_path: str | sp.csr_matrix, split: str) -> str:
        if isinstance(obj_or_path, str) and obj_or_path.endswith(".npy"):
            return obj_or_path
        out = base_path / f"X_{split}_features_{task_suffix}.npz"
        sp.save_npz(out, obj_or_path)  # type: ignore[arg-type]
        return str(out)

    train_path_out = _maybe_save_sparse_or_npy(X_train, "train")
    val_path_out   = _maybe_save_sparse_or_npy(X_val,   "val")
    test_path_out  = _maybe_save_sparse_or_npy(X_test,  "test")

    # Persist labels
    train_df[[label_column]].to_csv(base_path / f"y_train_{task_suffix}.csv", index=False)
    val_df[[label_column]].to_csv(  base_path / f"y_val_{task_suffix}.csv",   index=False)
    test_df[[label_column]].to_csv( base_path / f"y_test_{task_suffix}.csv",  index=False)
    logger.info(f"Saved in {time.time()-t0:.1f}s")

    meta = {
        "task": task,
        "label_column": label_column,
        "rows": {"train": int(train_df.shape[0]), "val": int(val_df.shape[0]), "test": int(test_df.shape[0])},
        "tfidf_backend": TFIDF_BACKEND,
        "tfidf_max_features": TFIDF_MAX_FEATURES,
        "tfidf_ngram_range": TFIDF_NGRAM_RANGE,
        "apply_feature_selection": APPLY_FEATURE_SELECTION,
        "num_svd_components": int(NUM_SVD_COMPONENTS) if APPLY_DIM_REDUCTION else None,
        "use_textblob": USE_TEXTBLOB,
        "sent_batch": SENT_BATCH,
        "svd_outputs": {"train": train_path_out, "val": val_path_out, "test": test_path_out}
    }
    with open(base_path / f"labels_meta_{task_suffix}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"✅ Saved features/labels for TASK='{task}' ({task_suffix}).")
    logger.info(f"   train: {train_path_out}")
    logger.info(f"   val:   {val_path_out}")
    logger.info(f"   test:  {test_path_out}")

if __name__ == "__main__":
    main()
