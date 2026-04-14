"""
Preprocess movie/TV reviews into one canonical split with binary and count-task exports.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

matplotlib.use("Agg")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = PROJECT_ROOT / "datasets"
DEFAULT_INPUT = DATASETS_ROOT / "Movies_and_TV.jsonl"
DEFAULT_OUTPUT = DATASETS_ROOT / "preprocessed"


@dataclass(frozen=True)
class Config:
    raw_jsonl: Path
    output_dir: Path
    min_tokens: int = 10
    cap_q: float = 0.99
    bin_pos_q: float = 0.95
    bin_neg_q: float = 0.70
    apply_gray_band: bool = True
    gray_band: float = 0.00
    apply_age_filter: bool = True
    days_old_min: int = 14
    random_state: int = 42
    train_pct: float = 0.7
    val_pct_of_temp: float = 0.5
    max_workers: int = 4
    split_stratify_bins: int = 10
    extreme_tail_q: float = 0.99


def setup_logger() -> logging.LoggerAdapter:
    logger = logging.getLogger("preprocess")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] trace_id=%(trace_id)s %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logging.LoggerAdapter(logger, {"trace_id": os.getpid()})


def clean_text(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return text.lower().strip()


def parse_timestamp(ts: Any) -> pd.Timestamp | pd.NaT:
    try:
        if pd.isna(ts):
            return pd.NaT
        if isinstance(ts, (int, float, np.integer, np.floating)) or (isinstance(ts, str) and ts.isdigit()):
            ts_float = float(ts)
            if ts_float > 1e12:
                return pd.to_datetime(ts_float, unit="ms", utc=True)
            if ts_float > 1e9:
                return pd.to_datetime(ts_float, unit="s", utc=True)
            return pd.to_datetime(ts_float, utc=True, errors="coerce")
        return pd.to_datetime(ts, utc=True, errors="coerce")
    except Exception:
        return pd.NaT


def safe_path(path: Path, base_dir: Path) -> Path:
    resolved = path.expanduser().resolve()
    base = base_dir.expanduser().resolve()
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"Unsafe path detected: {resolved}") from exc
    return resolved


def save_json(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


def stream_jsonl(path: Path, logger: logging.LoggerAdapter) -> Iterator[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            try:
                yield json.loads(line.strip())
            except json.JSONDecodeError:
                logger.warning("Skipping malformed line")


def load_reviews(path: Path, logger: logging.LoggerAdapter) -> pd.DataFrame:
    rows = list(tqdm(stream_jsonl(path, logger), desc="Loading reviews"))
    df = pd.DataFrame(rows)
    df.insert(0, "review_id", np.arange(len(df), dtype=np.int64))
    return df


def compute_sentiment(
    df: pd.DataFrame,
    text_col: str,
    max_workers: int,
    logger: logging.LoggerAdapter,
) -> pd.Series:
    from nltk import download
    from nltk.sentiment import SentimentIntensityAnalyzer

    download("vader_lexicon", quiet=True)
    sia = SentimentIntensityAnalyzer()
    texts = df[text_col].fillna("").tolist()
    results = np.empty(len(texts), dtype=float)
    batch_size = 50_000

    logger.info("Computing VADER sentiment for %d reviews", len(texts))
    for start in tqdm(range(0, len(texts), batch_size), desc="Computing sentiment"):
        end = min(start + batch_size, len(texts))
        batch = texts[start:end]
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            batch_scores = list(executor.map(lambda text: sia.polarity_scores(text)["compound"], batch))
        results[start:end] = batch_scores

    return pd.Series(results, index=df.index)


def build_features(df: pd.DataFrame, cfg: Config, logger: logging.LoggerAdapter) -> pd.DataFrame:
    logger.info("Cleaning and building shared base features")
    df = df.copy()

    text_series = df["text"] if "text" in df.columns else pd.Series("", index=df.index)
    title_series = df["title"] if "title" in df.columns else pd.Series("", index=df.index)
    helpful_votes = df["helpful_vote"] if "helpful_vote" in df.columns else pd.Series(0, index=df.index)
    rating_series = df["rating"] if "rating" in df.columns else pd.Series(np.nan, index=df.index)
    verified_series = df["verified_purchase"] if "verified_purchase" in df.columns else pd.Series(False, index=df.index)
    timestamp_series = df["timestamp"] if "timestamp" in df.columns else pd.Series(pd.NaT, index=df.index)

    df["clean_text"] = text_series.apply(clean_text)
    df["clean_title"] = title_series.apply(clean_text)
    df["full_text"] = (
        df["clean_title"].fillna("").astype(str).str.strip() + ". " + df["clean_text"].fillna("").astype(str).str.strip()
    ).str.strip(" .")
    df["helpful_vote"] = pd.to_numeric(helpful_votes, errors="coerce").fillna(0).astype(int)
    df["rating"] = pd.to_numeric(rating_series, errors="coerce").fillna(0).astype(int)
    df["is_verified"] = pd.Series(verified_series, index=df.index).fillna(False).astype(bool).astype(int)
    df["review_length"] = df["clean_text"].str.split().str.len().fillna(0).astype(int)
    df["sentiment_score"] = compute_sentiment(df, "clean_text", cfg.max_workers, logger)
    df["timestamp_parsed"] = pd.Series(timestamp_series, index=df.index).apply(parse_timestamp)
    now = pd.Timestamp.utcnow()
    df["review_age_days"] = (now - df["timestamp_parsed"]).dt.days.astype(float)
    df["timestamp_recency_days"] = df["review_age_days"].fillna(-1.0)
    df = df[df["review_length"] >= cfg.min_tokens].copy()
    logger.info("Shared base rows after min token filter: %d", len(df))
    return df


def cap_and_normalize(df: pd.DataFrame, cap_q: float) -> pd.DataFrame:
    df = df.copy()
    cap_value = float(df["helpful_vote"].quantile(cap_q))
    df["helpful_vote_capped"] = np.minimum(df["helpful_vote"], cap_value)
    df["log_helpfulness_score"] = np.log1p(df["helpful_vote_capped"])
    df["log1p_helpful_vote"] = np.log1p(df["helpful_vote"])
    max_log = float(df["log_helpfulness_score"].max()) or 1.0
    df["normalized_helpfulness_score"] = (df["log_helpfulness_score"] / max_log) * 10.0
    df["regression_score"] = df["normalized_helpfulness_score"]
    return df


def build_stratify_labels(series: pd.Series, max_bins: int) -> pd.Series | None:
    values = pd.Series(series).astype(float)
    unique_values = int(values.nunique(dropna=True))
    if unique_values < 2:
        return None

    n_bins = min(max_bins, unique_values)
    if n_bins < 2:
        return None

    ranked = values.rank(method="first")
    try:
        labels = pd.qcut(ranked, q=n_bins, labels=False, duplicates="drop")
    except ValueError:
        return None

    labels = pd.Series(labels, index=values.index)
    if labels.nunique(dropna=True) < 2:
        return None
    counts = labels.value_counts(dropna=True)
    if counts.empty or int(counts.min()) < 2:
        return None
    return labels


def split_base_dataset(df: pd.DataFrame, cfg: Config) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    stratify_main = build_stratify_labels(df["log1p_helpful_vote"], cfg.split_stratify_bins)
    train_df, temp_df = train_test_split(
        df,
        test_size=1.0 - cfg.train_pct,
        random_state=cfg.random_state,
        stratify=stratify_main,
    )

    temp_bins = max(2, cfg.split_stratify_bins // 2)
    stratify_temp = build_stratify_labels(temp_df["log1p_helpful_vote"], temp_bins)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=cfg.val_pct_of_temp,
        random_state=cfg.random_state,
        stratify=stratify_temp,
    )

    split_map = {
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }
    split_meta = {
        "random_state": cfg.random_state,
        "train_pct": cfg.train_pct,
        "val_pct_of_temp": cfg.val_pct_of_temp,
        "stratify_target": "log1p_helpful_vote",
        "stratify_bins_requested": cfg.split_stratify_bins,
        "main_split_used_stratify": stratify_main is not None,
        "temp_split_used_stratify": stratify_temp is not None,
        "rows_per_split": {name: int(part.shape[0]) for name, part in split_map.items()},
    }
    return split_map, split_meta


def annotate_shared_targets(split_map: dict[str, pd.DataFrame], extreme_tail_threshold: float) -> dict[str, pd.DataFrame]:
    annotated: dict[str, pd.DataFrame] = {}
    for split_name, df in split_map.items():
        part = df.copy()
        part["has_helpful_vote"] = (part["helpful_vote"] > 0).astype(int)
        part["is_extreme_tail"] = (part["helpful_vote"] >= extreme_tail_threshold).astype(int)
        annotated[split_name] = part
    return annotated


def save_split_definition(split_map: dict[str, pd.DataFrame], out_dir: Path, split_meta: dict[str, Any]) -> None:
    split_df = pd.concat(
        [
            df.loc[:, ["review_id"]].assign(split=split_name)
            for split_name, df in split_map.items()
        ],
        ignore_index=True,
    ).sort_values("review_id")
    split_df.to_csv(out_dir / "split_definition.csv", index=False)
    save_json(out_dir / "split_definition_metadata.json", split_meta)


def apply_age_filter(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, int]:
    if not cfg.apply_age_filter:
        return df.copy(), 0
    mask_drop = (
        (df["helpful_vote"] <= 1)
        & df["review_age_days"].notna()
        & (df["review_age_days"] < cfg.days_old_min)
    )
    return df.loc[~mask_drop].copy(), int(mask_drop.sum())


def label_and_filter(df: pd.DataFrame, neg_thr: float, pos_thr: float) -> tuple[pd.DataFrame, int]:
    scores = df["normalized_helpfulness_score"]
    labels = np.full(len(df), -1, dtype=np.int64)
    labels[scores >= pos_thr] = 1
    labels[scores <= neg_thr] = 0

    keep_mask = labels >= 0
    out = df.loc[keep_mask].copy()
    out["binary_label"] = labels[keep_mask]
    gray_rows = int((~keep_mask).sum())
    return out, gray_rows


def prevalence_stats(df: pd.DataFrame, label_col: str) -> dict[str, Any]:
    if df.empty:
        return {
            "rows": 0,
            "positive_rows": 0,
            "negative_rows": 0,
            "positive_fraction": None,
            "negative_fraction": None,
        }
    positives = int((df[label_col] == 1).sum())
    negatives = int((df[label_col] == 0).sum())
    total = int(len(df))
    return {
        "rows": total,
        "positive_rows": positives,
        "negative_rows": negatives,
        "positive_fraction": float(positives / total),
        "negative_fraction": float(negatives / total),
    }


def describe_count_distribution(series: pd.Series) -> dict[str, Any]:
    values = pd.Series(series).dropna().astype(float)
    if values.empty:
        return {
            "rows": 0,
            "zero_fraction": None,
            "mean": None,
            "variance": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "rows": int(values.shape[0]),
        "zero_fraction": float((values == 0).mean()),
        "mean": float(values.mean()),
        "variance": float(values.var(ddof=0)),
        "p90": float(values.quantile(0.90)),
        "p95": float(values.quantile(0.95)),
        "p99": float(values.quantile(0.99)),
        "max": float(values.max()),
    }


def build_binary_branch(
    split_map: dict[str, pd.DataFrame],
    cfg: Config,
    out_dir: Path,
) -> dict[str, Any]:
    filtered_splits: dict[str, pd.DataFrame] = {}
    age_filter_counts: dict[str, int] = {}
    for split_name, df in split_map.items():
        filtered, removed_rows = apply_age_filter(df, cfg)
        filtered_splits[split_name] = filtered
        age_filter_counts[split_name] = removed_rows

    train_df = filtered_splits["train"]
    if train_df.empty:
        raise ValueError("Binary training split is empty after age filtering.")

    pos_thr = float(train_df["normalized_helpfulness_score"].quantile(cfg.bin_pos_q))
    neg_thr = float(train_df["normalized_helpfulness_score"].quantile(cfg.bin_neg_q))
    if cfg.apply_gray_band and cfg.gray_band > 0:
        pos_thr += cfg.gray_band * 10.0
        neg_thr -= cfg.gray_band * 10.0

    labeled_splits: dict[str, pd.DataFrame] = {}
    gray_band_counts: dict[str, int] = {}
    for split_name, df in filtered_splits.items():
        labeled, gray_count = label_and_filter(df, neg_thr=neg_thr, pos_thr=pos_thr)
        labeled_splits[split_name] = labeled
        gray_band_counts[split_name] = gray_count
        labeled.to_csv(out_dir / f"binary_label_{split_name}.csv", index=False)

    metadata = {
        "thresholds": {"neg_thr": neg_thr, "pos_thr": pos_thr},
        "quantiles_used": {"neg_q": cfg.bin_neg_q, "pos_q": cfg.bin_pos_q},
        "gray_band": {"enabled": cfg.apply_gray_band, "width": cfg.gray_band},
        "age_filter": {
            "enabled": cfg.apply_age_filter,
            "days_old_min": cfg.days_old_min,
            "rows_removed_per_split": age_filter_counts,
            "rows_removed_total": int(sum(age_filter_counts.values())),
        },
        "dropped_gray_band_rows": {
            "per_split": gray_band_counts,
            "total": int(sum(gray_band_counts.values())),
        },
        "class_prevalence_per_split": {
            split_name: prevalence_stats(df, "binary_label")
            for split_name, df in labeled_splits.items()
        },
    }
    save_json(out_dir / "binary_label_metadata.json", metadata)
    return metadata


def build_count_branch(
    split_map: dict[str, pd.DataFrame],
    out_dir: Path,
    extreme_tail_threshold: float,
) -> dict[str, Any]:
    stage1_splits: dict[str, pd.DataFrame] = {}
    stage2_splits: dict[str, pd.DataFrame] = {}

    for split_name, df in split_map.items():
        stage1_df = df.copy()
        stage1_df.to_csv(out_dir / f"count_stage1_{split_name}.csv", index=False)
        stage1_splits[split_name] = stage1_df

        stage2_df = stage1_df.loc[stage1_df["helpful_vote"] > 0].copy()
        stage2_df["helpful_vote_pos"] = stage2_df["helpful_vote"]
        stage2_df.to_csv(out_dir / f"count_stage2_{split_name}.csv", index=False)
        stage2_splits[split_name] = stage2_df

    metadata = {
        "targets": {
            "stage1": "has_helpful_vote",
            "stage2": "helpful_vote_pos",
            "raw_count_target": "helpful_vote",
            "comparison_target_available": "log1p_helpful_vote",
        },
        "extreme_tail_threshold": float(extreme_tail_threshold),
        "distribution": {
            split_name: describe_count_distribution(df["helpful_vote"])
            for split_name, df in stage1_splits.items()
        },
        "positive_only_distribution": {
            split_name: describe_count_distribution(df["helpful_vote_pos"])
            for split_name, df in stage2_splits.items()
        },
        "has_helpful_vote_prevalence": {
            split_name: {
                "rows": int(df.shape[0]),
                "positive_fraction": float(df["has_helpful_vote"].mean()) if not df.empty else None,
                "zero_fraction": float(1.0 - df["has_helpful_vote"].mean()) if not df.empty else None,
            }
            for split_name, df in stage1_splits.items()
        },
    }
    save_json(out_dir / "count_target_metadata.json", metadata)
    return metadata


def summarize_regression_targets(df: pd.DataFrame, logger: logging.LoggerAdapter) -> None:
    target = df["regression_score"].astype(float)
    helpful_votes = df["helpful_vote"].astype(float)

    logger.info(
        "Regression target stats | count=%d mean=%.4f variance=%.4f std=%.4f min=%.4f median=%.4f max=%.4f",
        int(target.count()),
        float(target.mean()),
        float(target.var()),
        float(target.std()),
        float(target.min()),
        float(target.median()),
        float(target.max()),
    )
    logger.info(
        "Helpful vote stats | count=%d mean=%.4f variance=%.4f std=%.4f min=%.4f median=%.4f max=%.4f",
        int(helpful_votes.count()),
        float(helpful_votes.mean()),
        float(helpful_votes.var()),
        float(helpful_votes.std()),
        float(helpful_votes.min()),
        float(helpful_votes.median()),
        float(helpful_votes.max()),
    )


def plot_regression_distributions(
    full_df: pd.DataFrame,
    reg_train: pd.DataFrame,
    reg_val: pd.DataFrame,
    reg_test: pd.DataFrame,
    out_dir: Path,
    logger: logging.LoggerAdapter,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].hist(full_df["helpful_vote"], bins=50, color="#355070", edgecolor="white", alpha=0.9)
    axes[0, 0].set_title("Helpful Vote Distribution")
    axes[0, 0].set_xlabel("helpful_vote")
    axes[0, 0].set_ylabel("count")

    axes[0, 1].hist(full_df["regression_score"], bins=50, color="#6d597a", edgecolor="white", alpha=0.9)
    axes[0, 1].set_title("Regression Score Distribution")
    axes[0, 1].set_xlabel("regression_score")
    axes[0, 1].set_ylabel("count")

    split_specs = [
        ("Train", reg_train["regression_score"], "#2a9d8f"),
        ("Validation", reg_val["regression_score"], "#e9c46a"),
        ("Test", reg_test["regression_score"], "#e76f51"),
    ]
    for label, values, color in split_specs:
        axes[1, 0].hist(values, bins=40, alpha=0.55, label=label, color=color, edgecolor="white")
    axes[1, 0].set_title("Regression Score by Split")
    axes[1, 0].set_xlabel("regression_score")
    axes[1, 0].set_ylabel("count")
    axes[1, 0].legend()

    axes[1, 1].boxplot(
        [reg_train["regression_score"], reg_val["regression_score"], reg_test["regression_score"]],
        tick_labels=["train", "val", "test"],
        patch_artist=True,
        boxprops={"facecolor": "#84a59d"},
        medianprops={"color": "#bc4749", "linewidth": 2},
    )
    axes[1, 1].set_title("Regression Score Spread by Split")
    axes[1, 1].set_ylabel("regression_score")

    fig.suptitle("Regression Vote Distribution Diagnostics", fontsize=14)
    fig.tight_layout()
    plot_path = out_dir / "regression_vote_distribution.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved regression distribution plot to %s", plot_path)


def process_pipeline(cfg: Config, logger: logging.LoggerAdapter) -> None:
    path = safe_path(cfg.raw_jsonl, DATASETS_ROOT)
    out_dir = safe_path(cfg.output_dir, DATASETS_ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)

    reviews_df = load_reviews(path, logger)
    base = build_features(reviews_df, cfg, logger)
    base = cap_and_normalize(base, cfg.cap_q)

    logger.info("Creating canonical shared split definition")
    split_map, split_meta = split_base_dataset(base, cfg)
    extreme_tail_threshold = float(split_map["train"]["helpful_vote"].quantile(cfg.extreme_tail_q))
    split_map = annotate_shared_targets(split_map, extreme_tail_threshold=extreme_tail_threshold)

    split_meta["extreme_tail_q"] = cfg.extreme_tail_q
    split_meta["extreme_tail_threshold"] = extreme_tail_threshold
    save_split_definition(split_map, out_dir, split_meta)

    logger.info("Saving regression exports on top of the canonical split")
    for split_name, df in split_map.items():
        df.to_csv(out_dir / f"regression_score_{split_name}.csv", index=False)

    full_regression_df = pd.concat(split_map.values(), ignore_index=True)
    summarize_regression_targets(full_regression_df, logger)
    plot_regression_distributions(
        full_regression_df,
        split_map["train"],
        split_map["val"],
        split_map["test"],
        out_dir,
        logger,
    )

    logger.info("Saving reproducible binary exports and metadata")
    build_binary_branch(split_map, cfg, out_dir)

    logger.info("Saving count stage datasets and metadata")
    build_count_branch(split_map, out_dir, extreme_tail_threshold=extreme_tail_threshold)

    logger.info("Saved preprocessed datasets successfully")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess review datasets into canonical binary and count splits.")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    cfg = Config(raw_jsonl=Path(args.input), output_dir=Path(args.output))
    logger = setup_logger()
    process_pipeline(cfg, logger)


if __name__ == "__main__":
    main()
