from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error, roc_auc_score


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "datasets" / "preprocessed"
OUT = ROOT / "outputs"
PRED = OUT / "predictions"

REPORTED = {
    "xgb_binary": {"auroc": 0.913050, "auprc": 0.543307, "thr": 0.800070},
    "bert_binary": {"auroc": 0.907987, "auprc": 0.492692, "thr": 0.106897},
    "xgb_count": {"rmse": 15.0280, "mae": 1.7306, "spearman": 0.5398},
    "bert_count": {"rmse": 15.0149, "mae": 1.9395},
}


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def csv_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return next(csv.reader(fh))


def count_rows(path: Path) -> int:
    # Count lines in binary mode to avoid decoding the multi-GB files.
    with path.open("rb") as fh:
        return max(0, sum(1 for _ in fh) - 1)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def print_json_line(label: str, obj: object) -> None:
    print(f"{label}: {json.dumps(obj, sort_keys=True)}")


def split_checks() -> None:
    print("\n=== SPLITS ===")
    print_json_line("split_definition_metadata", read_json(DATA / "split_definition_metadata.json"))
    print(f"split_definition_header: {csv_header(DATA / 'split_definition.csv')}")

    count_test = pd.read_csv(DATA / "count_stage1_test.csv", usecols=["review_id"])
    bin_train = pd.read_csv(DATA / "binary_label_train.csv", usecols=["review_id"])
    bin_val = pd.read_csv(DATA / "binary_label_val.csv", usecols=["review_id"])
    bin_test = pd.read_csv(DATA / "binary_label_test.csv", usecols=["review_id"])
    xgb_bin_ids = pd.read_csv(DATA / "X_test_feature_ids_binary.csv", usecols=["review_id"])
    reg_ids = pd.read_csv(DATA / "X_test_feature_ids_regression.csv", usecols=["review_id"])

    count_ids = set(count_test["review_id"].astype(np.int64))
    train_ids = set(bin_train["review_id"].astype(np.int64))
    val_ids = set(bin_val["review_id"].astype(np.int64))

    print_json_line(
        "split_counts",
        {
            "count_stage1_test_rows": len(count_test),
            "binary_label_train_rows": len(bin_train),
            "binary_label_val_rows": len(bin_val),
            "binary_label_test_rows": len(bin_test),
            "X_test_feature_ids_binary_rows": len(xgb_bin_ids),
            "X_test_feature_ids_regression_rows": len(reg_ids),
            "count_test_overlap_binary_train": len(count_ids & train_ids),
            "count_test_overlap_binary_val": len(count_ids & val_ids),
        },
    )


def candidate_csvs() -> list[Path]:
    paths = list(PRED.glob("*.csv"))
    for folder in OUT.glob("bert_binary_*"):
        paths.extend((folder / "predictions").glob("*.csv"))
    return sorted(set(paths))


def find_probability_col(cols: list[str]) -> str | None:
    priority = [
        "prob_pos",
        "probability",
        "probabilities",
        "pred_prob",
        "positive_prob",
        "prob_has_helpful_vote",
        "predicted_probability",
    ]
    lower = {c.lower(): c for c in cols}
    for name in priority:
        if name in lower:
            return lower[name]
    for c in cols:
        lc = c.lower()
        if "prob" in lc and pd.api.types.is_numeric_dtype(pd.Series([], dtype="float64")):
            return c
    return None


def find_binary_label_col(cols: list[str]) -> str | None:
    for c in ["true_label", "binary_label", "label", "y_true"]:
        if c in cols:
            return c
    return None


def binary_metrics_for_file(path: Path, binary_labels: pd.DataFrame) -> dict | None:
    cols = csv_header(path)
    prob_col = find_probability_col(cols)
    label_col = find_binary_label_col(cols)
    if prob_col is None:
        return None

    usecols = [prob_col]
    if label_col:
        usecols.append(label_col)
    if "review_id" in cols:
        usecols.append("review_id")
    usecols = list(dict.fromkeys(usecols))

    try:
        df = pd.read_csv(path, usecols=usecols)
    except Exception as exc:
        return {"path": rel(path), "error": f"read_failed: {exc}"}

    if label_col is None:
        if "review_id" not in df.columns:
            return None
        df = df.merge(binary_labels, on="review_id", how="inner")
        label_col = "binary_label"

    labels = pd.to_numeric(df[label_col], errors="coerce")
    probs = pd.to_numeric(df[prob_col], errors="coerce")
    mask = labels.notna() & probs.notna()
    labels = labels[mask].astype(int).to_numpy()
    probs = probs[mask].astype(float).to_numpy()
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return None
    return {
        "path": rel(path),
        "rows": int(len(df)),
        "metric_rows": int(len(labels)),
        "columns": cols,
        "id_col": "review_id" if "review_id" in cols else None,
        "label_col": label_col,
        "prob_col": prob_col,
        "auroc": float(roc_auc_score(labels, probs)),
        "auprc": float(average_precision_score(labels, probs)),
    }


def count_metrics_for_file(path: Path) -> list[dict]:
    cols = csv_header(path)
    actual_col = None
    if "true_count" in cols:
        actual_col = "true_count"
    elif "y_true" in cols:
        actual_col = "y_true"
    if actual_col is None:
        return []

    pred_cols = [c for c in ["expected_count", "thresholded_count", "pred_count", "y_pred"] if c in cols]
    if not pred_cols:
        return []

    try:
        df = pd.read_csv(path, usecols=["review_id", actual_col, *pred_cols] if "review_id" in cols else [actual_col, *pred_cols])
    except Exception as exc:
        return [{"path": rel(path), "error": f"read_failed: {exc}"}]

    out = []
    y = pd.to_numeric(df[actual_col], errors="coerce").to_numpy(dtype=float)
    for pred_col in pred_cols:
        pred = pd.to_numeric(df[pred_col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(y) & np.isfinite(pred)
        yy = y[mask]
        pp = pred[mask]
        if len(yy) == 0:
            continue
        out.append(
            {
                "path": rel(path),
                "rows": int(len(df)),
                "metric_rows": int(len(yy)),
                "columns": cols,
                "id_col": "review_id" if "review_id" in cols else None,
                "actual_col": actual_col,
                "pred_col": pred_col,
                "rmse": float(math.sqrt(mean_squared_error(yy, pp))),
                "mae": float(mean_absolute_error(yy, pp)),
                "spearman": float(spearmanr(yy, pp).statistic) if len(np.unique(pp)) > 1 else float("nan"),
            }
        )
    return out


def metric_scan() -> tuple[list[dict], list[dict]]:
    print("\n=== PREDICTION CANDIDATES ===")
    binary_labels = pd.read_csv(DATA / "binary_label_test.csv", usecols=["review_id", "binary_label"])
    binary_results = []
    count_results = []
    for path in candidate_csvs():
        b = binary_metrics_for_file(path, binary_labels)
        if b is not None:
            binary_results.append(b)
        count_results.extend(count_metrics_for_file(path))

    def b_key(row: dict) -> float:
        targets = REPORTED["xgb_binary"], REPORTED["bert_binary"]
        return min(abs(row["auroc"] - t["auroc"]) + abs(row["auprc"] - t["auprc"]) for t in targets)

    def c_key(row: dict) -> float:
        targets = REPORTED["xgb_count"], REPORTED["bert_count"]
        return min(abs(row["rmse"] - t["rmse"]) + abs(row["mae"] - t["mae"]) for t in targets)

    print("binary_candidate_count:", len(binary_results))
    for row in sorted(binary_results, key=b_key)[:12]:
        print_json_line(
            "binary_candidate",
            {
                "path": row["path"],
                "rows": row["rows"],
                "metric_rows": row["metric_rows"],
                "prob_col": row["prob_col"],
                "label_col": row["label_col"],
                "id_col": row["id_col"],
                "auroc": round(row["auroc"], 9),
                "auprc": round(row["auprc"], 9),
            },
        )

    print("count_candidate_count:", len(count_results))
    for row in sorted(count_results, key=c_key)[:12]:
        print_json_line(
            "count_candidate",
            {
                "path": row["path"],
                "rows": row["rows"],
                "metric_rows": row["metric_rows"],
                "actual_col": row["actual_col"],
                "pred_col": row["pred_col"],
                "id_col": row["id_col"],
                "rmse": round(row["rmse"], 6),
                "mae": round(row["mae"], 6),
                "spearman": round(row["spearman"], 6),
            },
        )
    return binary_results, count_results


def join_and_coverage() -> None:
    print("\n=== JOINS_AND_COVERAGE ===")
    final_files = {
        "xgb_binary": PRED / "xgb_binary_test_predictions.csv",
        "bert_binary": PRED / "bert_binary_test_predictions.csv",
        "xgb_count_log1p_combined": PRED / "xgb_count_stage1_test_combined_count_stage2_log1p_predictions.csv",
        "bert_count_log1p_combined": PRED / "bert_count_combined_count_stage2_log1p_test_predictions.csv",
    }
    refs = {
        "binary_label_test": pd.read_csv(DATA / "binary_label_test.csv", usecols=["review_id", "parent_asin", "rating", "helpful_vote", "review_length"]),
        "count_stage1_test": pd.read_csv(DATA / "count_stage1_test.csv", usecols=["review_id", "parent_asin", "rating", "helpful_vote", "review_length"]),
        "X_test_feature_ids_binary": pd.read_csv(DATA / "X_test_feature_ids_binary.csv", usecols=["review_id"]),
        "X_test_feature_ids_regression": pd.read_csv(DATA / "X_test_feature_ids_regression.csv", usecols=["review_id"]),
    }

    print_json_line("legacy_test_csv_header", csv_header(DATA / "test.csv"))
    print_json_line("legacy_test_csv_rows", count_rows(DATA / "test.csv"))

    for name, path in final_files.items():
        cols = csv_header(path)
        pred = pd.read_csv(path, usecols=["review_id"] if "review_id" in cols else None)
        ref_name = "binary_label_test" if "binary" in name else "count_stage1_test"
        feature_ref_name = "X_test_feature_ids_binary" if "binary" in name else "X_test_feature_ids_regression"
        for target_name in [ref_name, feature_ref_name]:
            ref = refs[target_name]
            pred_ids = pd.Index(pred["review_id"]) if "review_id" in pred.columns else pd.Index([])
            ref_ids = pd.Index(ref["review_id"])
            print_json_line(
                "join_check",
                {
                    "prediction": name,
                    "path": rel(path),
                    "target": target_name,
                    "pred_rows": int(len(pred)),
                    "target_rows": int(len(ref)),
                    "matched": int(pred_ids.isin(ref_ids).sum()),
                    "pred_only": int((~pred_ids.isin(ref_ids)).sum()),
                    "target_only": int((~ref_ids.isin(pred_ids)).sum()),
                    "pred_dup_review_id": int(pred["review_id"].duplicated().sum()) if "review_id" in pred.columns else None,
                    "target_dup_review_id": int(ref["review_id"].duplicated().sum()),
                },
            )

    count_ids = pd.Index(refs["count_stage1_test"]["review_id"])
    xgb_ids = pd.Index(pd.read_csv(final_files["xgb_binary"], usecols=["review_id"])["review_id"])
    bert_ids = pd.Index(pd.read_csv(final_files["bert_binary"], usecols=["review_id"])["review_id"])
    no_xgb = count_ids.difference(xgb_ids)
    no_bert = count_ids.difference(bert_ids)
    no_any = count_ids.difference(xgb_ids.union(bert_ids))
    gray = refs["count_stage1_test"][refs["count_stage1_test"]["review_id"].isin(no_any)]
    vc = gray["helpful_vote"].value_counts().sort_index()
    print_json_line(
        "binary_prediction_coverage_on_count_test",
        {
            "count_test_rows": int(len(count_ids)),
            "xgb_binary_pred_rows": int(len(xgb_ids)),
            "bert_binary_pred_rows": int(len(bert_ids)),
            "count_reviews_without_xgb_binary": int(len(no_xgb)),
            "count_reviews_without_bert_binary": int(len(no_bert)),
            "count_reviews_without_either_binary": int(len(no_any)),
            "no_any_helpful_vote_min": int(gray["helpful_vote"].min()),
            "no_any_helpful_vote_max": int(gray["helpful_vote"].max()),
            "no_any_helpful_vote_counts": {str(int(k)): int(v) for k, v in vc.items()},
        },
    )


def threshold_and_title_checks() -> None:
    print("\n=== THRESHOLDS_AND_TITLES ===")
    paths = [
        OUT / "predictions" / "xgb_binary_test_metrics.json",
        OUT / "checkpoints" / "xgb_binary_thresholds.json",
        OUT / "checkpoints" / "xgb_binary_thr.json",
        OUT / "predictions" / "bert_binary_test_metrics.json",
    ]
    for path in paths:
        if path.exists():
            print_json_line(f"threshold_source:{rel(path)}", read_json(path))

    # Check a deterministic sample of count-test parent_asin values against Amazon metadata.
    sample_parent_asins = (
        pd.read_csv(DATA / "count_stage1_test.csv", usecols=["parent_asin"])["parent_asin"]
        .dropna()
        .drop_duplicates()
        .sample(n=1000, random_state=42)
        .astype(str)
    )
    wanted = set(sample_parent_asins)
    found: dict[str, str | None] = {}
    meta_path = ROOT / "datasets" / "meta_Movies_and_TV.jsonl"
    with meta_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            parent = str(row.get("parent_asin", ""))
            if parent in wanted:
                title = row.get("title")
                found[parent] = title if title is not None else None
                if len(found) == len(wanted):
                    break
    missing = sorted(wanted.difference(found))
    missing_title = sorted(k for k, v in found.items() if v in (None, ""))
    print_json_line(
        "meta_title_sample",
        {
            "sample_unique_parent_asin": len(wanted),
            "found_meta_rows": len(found),
            "missing_meta_rows": len(missing),
            "found_but_empty_title": len(missing_title),
            "first_5_titles": {k: found[k] for k in sorted(found)[:5]},
        },
    )


def environment() -> None:
    print("\n=== ENVIRONMENT ===")
    print(f"python: {sys.version}")
    modules = ["numpy", "pandas", "scipy", "xgboost", "torch", "transformers", "sklearn"]
    versions = {}
    for name in modules:
        try:
            mod = __import__(name)
            versions[name] = {"import": True, "version": getattr(mod, "__version__", "unknown")}
        except Exception as exc:
            versions[name] = {"import": False, "error": repr(exc)}
    print_json_line("imports", versions)
    try:
        import torch

        print_json_line(
            "torch_cuda",
            {
                "available": bool(torch.cuda.is_available()),
                "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
                "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            },
        )
    except Exception as exc:
        print_json_line("torch_cuda", {"error": repr(exc)})


def main() -> None:
    split_checks()
    metric_scan()
    join_and_coverage()
    threshold_and_title_checks()
    environment()


if __name__ == "__main__":
    main()
