from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse as sp
from scipy.stats import spearmanr
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

try:
    from main_figure_plots import try_generate_requested_main_figures
except ImportError:  # pragma: no cover - fallback for package-style imports
    from training.main_figure_plots import try_generate_requested_main_figures


MODE_CHOICES = (
    "binary",
    "count_stage1",
    "count_stage2_log1p",
    "count_stage2_log1p_mae",
    "count_stage2_log1p_pseudohuber",
    "count_stage2_log1p_squaredlogerror",
    "count_stage2_poisson",
    "count_stage2_tweedie",
)
COUNT_STAGE2_MODE_CHOICES = (
    "count_stage2_log1p",
    "count_stage2_log1p_mae",
    "count_stage2_log1p_pseudohuber",
    "count_stage2_log1p_squaredlogerror",
    "count_stage2_poisson",
    "count_stage2_tweedie",
)

ModeT = Literal[
    "binary",
    "count_stage1",
    "count_stage2_log1p",
    "count_stage2_log1p_mae",
    "count_stage2_log1p_pseudohuber",
    "count_stage2_log1p_squaredlogerror",
    "count_stage2_poisson",
    "count_stage2_tweedie",
]
RoleT = Literal["classifier", "regressor"]


@dataclass(frozen=True)
class ModeSpec:
    mode: ModeT
    role: RoleT
    dataset_prefix: str
    feature_dataset_prefix: str
    feature_task_suffix: str
    target_column: str
    raw_target_column: str
    prediction_transform: Literal["identity", "log1p_expm1"]
    checkpoint_stem: str
    default_objective: str
    default_eval_metric: str
    objective_params: dict[str, Any]


@dataclass
class DataBundle:
    spec: ModeSpec
    split: str
    frame: pd.DataFrame
    X: Any
    y: np.ndarray
    y_raw: np.ndarray


MODE_SPECS: dict[ModeT, ModeSpec] = {
    "binary": ModeSpec(
        mode="binary",
        role="classifier",
        dataset_prefix="binary_label",
        feature_dataset_prefix="binary_label",
        feature_task_suffix="binary",
        target_column="binary_label",
        raw_target_column="binary_label",
        prediction_transform="identity",
        checkpoint_stem="xgb_binary",
        default_objective="binary:logistic",
        default_eval_metric="aucpr",
        objective_params={},
    ),
    "count_stage1": ModeSpec(
        mode="count_stage1",
        role="classifier",
        dataset_prefix="count_stage1",
        feature_dataset_prefix="regression_score",
        feature_task_suffix="regression",
        target_column="has_helpful_vote",
        raw_target_column="has_helpful_vote",
        prediction_transform="identity",
        checkpoint_stem="xgb_count_stage1",
        default_objective="binary:logistic",
        default_eval_metric="aucpr",
        objective_params={},
    ),
    "count_stage2_poisson": ModeSpec(
        mode="count_stage2_poisson",
        role="regressor",
        dataset_prefix="count_stage2",
        feature_dataset_prefix="regression_score",
        feature_task_suffix="regression",
        target_column="helpful_vote_pos",
        raw_target_column="helpful_vote_pos",
        prediction_transform="identity",
        checkpoint_stem="xgb_count_stage2_poisson",
        default_objective="count:poisson",
        default_eval_metric="poisson-nloglik",
        objective_params={},
    ),
    "count_stage2_log1p": ModeSpec(
        mode="count_stage2_log1p",
        role="regressor",
        dataset_prefix="count_stage2",
        feature_dataset_prefix="regression_score",
        feature_task_suffix="regression",
        target_column="log1p_helpful_vote",
        raw_target_column="helpful_vote_pos",
        prediction_transform="log1p_expm1",
        checkpoint_stem="xgb_count_stage2_log1p",
        default_objective="reg:squarederror",
        default_eval_metric="rmse",
        objective_params={},
    ),
    "count_stage2_log1p_mae": ModeSpec(
        mode="count_stage2_log1p_mae",
        role="regressor",
        dataset_prefix="count_stage2",
        feature_dataset_prefix="regression_score",
        feature_task_suffix="regression",
        target_column="log1p_helpful_vote",
        raw_target_column="helpful_vote_pos",
        prediction_transform="log1p_expm1",
        checkpoint_stem="xgb_count_stage2_log1p_mae",
        default_objective="reg:absoluteerror",
        default_eval_metric="mae",
        objective_params={},
    ),
    "count_stage2_log1p_pseudohuber": ModeSpec(
        mode="count_stage2_log1p_pseudohuber",
        role="regressor",
        dataset_prefix="count_stage2",
        feature_dataset_prefix="regression_score",
        feature_task_suffix="regression",
        target_column="log1p_helpful_vote",
        raw_target_column="helpful_vote_pos",
        prediction_transform="log1p_expm1",
        checkpoint_stem="xgb_count_stage2_log1p_pseudohuber",
        default_objective="reg:pseudohubererror",
        default_eval_metric="mphe",
        objective_params={},
    ),
    "count_stage2_log1p_squaredlogerror": ModeSpec(
        mode="count_stage2_log1p_squaredlogerror",
        role="regressor",
        dataset_prefix="count_stage2",
        feature_dataset_prefix="regression_score",
        feature_task_suffix="regression",
        target_column="log1p_helpful_vote",
        raw_target_column="helpful_vote_pos",
        prediction_transform="log1p_expm1",
        checkpoint_stem="xgb_count_stage2_log1p_squaredlogerror",
        default_objective="reg:squaredlogerror",
        default_eval_metric="rmsle",
        objective_params={},
    ),
    "count_stage2_tweedie": ModeSpec(
        mode="count_stage2_tweedie",
        role="regressor",
        dataset_prefix="count_stage2",
        feature_dataset_prefix="regression_score",
        feature_task_suffix="regression",
        target_column="helpful_vote_pos",
        raw_target_column="helpful_vote_pos",
        prediction_transform="identity",
        checkpoint_stem="xgb_count_stage2_tweedie",
        default_objective="reg:tweedie",
        default_eval_metric="tweedie-nloglik@1.5",
        objective_params={"tweedie_variance_power": 1.5},
    ),
}


def get_mode_spec(mode: ModeT) -> ModeSpec:
    try:
        return MODE_SPECS[mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported XGBoost mode: {mode}") from exc


def setup_logger(name: str, log_path: Optional[str | Path] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
    if log_path is not None:
        resolved = Path(log_path).expanduser().resolve()
        if not any(isinstance(h, logging.FileHandler) and Path(h.baseFilename).resolve() == resolved for h in logger.handlers):
            resolved.parent.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(resolved, mode="a", encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
            logger.addHandler(handler)
    return logger


def safe_path(base_dir: str | Path, *parts: str) -> str:
    base = Path(base_dir).expanduser().resolve()
    out = base.joinpath(*parts).resolve()
    try:
        out.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"Unsafe path outside base dir: {out}") from exc
    out.parent.mkdir(parents=True, exist_ok=True)
    return str(out)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def checkpoint_path(out_dir: str | Path, mode: ModeT) -> str:
    spec = get_mode_spec(mode)
    return safe_path(out_dir, "checkpoints", f"{spec.checkpoint_stem}.json")


def metadata_path(out_dir: str | Path, mode: ModeT) -> str:
    spec = get_mode_spec(mode)
    return safe_path(out_dir, "checkpoints", f"{spec.checkpoint_stem}_meta.json")


def threshold_path(out_dir: str | Path, mode: ModeT) -> str:
    spec = get_mode_spec(mode)
    return safe_path(out_dir, "checkpoints", f"{spec.checkpoint_stem}_thresholds.json")


def prediction_csv_path(out_dir: str | Path, mode: ModeT, split: str) -> str:
    spec = get_mode_spec(mode)
    return safe_path(out_dir, "predictions", f"{spec.checkpoint_stem}_{split}_predictions.csv")


def metrics_json_path(out_dir: str | Path, mode: ModeT, split: str) -> str:
    spec = get_mode_spec(mode)
    return safe_path(out_dir, "predictions", f"{spec.checkpoint_stem}_{split}_metrics.json")


def _feature_meta_path(base_dir: str | Path, feature_task_suffix: str) -> Path:
    return Path(base_dir) / f"labels_meta_{feature_task_suffix}.json"


def _feature_id_path(base_dir: str | Path, split: str, feature_task_suffix: str) -> Path:
    return Path(base_dir) / f"X_{split}_feature_ids_{feature_task_suffix}.csv"


def _load_feature_matrix(
    base_dir: str | Path,
    split: str,
    feature_task_suffix: str,
    logger: logging.Logger,
) -> Any:
    base = Path(base_dir)
    npy = base / f"X_{split}_features_{feature_task_suffix}_svd.npy"
    npz = base / f"X_{split}_features_{feature_task_suffix}.npz"
    meta_path = _feature_meta_path(base, feature_task_suffix)

    if npy.exists():
        try:
            return np.load(npy, mmap_mode="r")
        except ValueError as exc:
            logger.warning("np.load failed for %s (%s); retrying as raw memmap", npy, exc)
            meta = load_json(meta_path)
            n_rows = int(meta["rows"][split])
            n_cols = int(meta.get("num_svd_components") or 0)
            return np.memmap(npy, dtype=np.float32, mode="r", shape=(n_rows, n_cols))
    if npz.exists():
        return sp.load_npz(npz)
    raise FileNotFoundError(
        f"Missing features for split='{split}', feature_task_suffix='{feature_task_suffix}'. "
        f"Expected one of: {npy} or {npz}"
    )


def _load_feature_ids(
    base_dir: str | Path,
    split: str,
    spec: ModeSpec,
) -> tuple[np.ndarray, Path]:
    id_path = _feature_id_path(base_dir, split, spec.feature_task_suffix)
    if id_path.exists():
        ids = pd.read_csv(id_path, usecols=["review_id"])["review_id"].to_numpy()
        return ids, id_path

    fallback_path = Path(base_dir) / f"{spec.feature_dataset_prefix}_{split}.csv"
    ids = pd.read_csv(fallback_path, usecols=["review_id"])["review_id"].to_numpy()
    return ids, fallback_path


def _load_csv_columns(
    base_dir: str | Path,
    prefix: str,
    split: str,
    columns: list[str],
) -> pd.DataFrame:
    path = Path(base_dir) / f"{prefix}_{split}.csv"
    unique_cols = list(dict.fromkeys(columns))
    return pd.read_csv(path, usecols=unique_cols)


def _subset_matrix(X: Any, indices: np.ndarray) -> Any:
    if indices.ndim != 1:
        indices = indices.reshape(-1)
    return X[indices]


def _stale_feature_artifacts_error(
    *,
    split: str,
    spec: ModeSpec,
    feature_id_source: Path,
    feature_id_count: int,
    matrix_rows: int,
) -> RuntimeError:
    feature_name = f"X_{split}_features_{spec.feature_task_suffix}"
    command = f"python .\\src\\feature_extraction.py --task {spec.feature_task_suffix}"
    return RuntimeError(
        f"Stale feature artifacts for split='{split}' mode='{spec.mode}'. "
        f"{feature_name} has {matrix_rows} rows, but review_id rows loaded from "
        f"'{feature_id_source.name}' are {feature_id_count}. "
        f"Regenerate the {spec.feature_task_suffix} feature bundle so the matrix and row ids are rebuilt together: {command}"
    )


def _align_features_to_targets(
    feature_ids: np.ndarray,
    target_ids: np.ndarray,
    X: Any,
    *,
    split: str,
    spec: ModeSpec,
    feature_id_source: Path,
) -> Any:
    if feature_ids.shape[0] != X.shape[0]:
        raise _stale_feature_artifacts_error(
            split=split,
            spec=spec,
            feature_id_source=feature_id_source,
            feature_id_count=int(feature_ids.shape[0]),
            matrix_rows=int(X.shape[0]),
        )

    if feature_ids.shape == target_ids.shape and np.array_equal(feature_ids, target_ids):
        return X

    indexer = pd.Index(feature_ids).get_indexer(target_ids)
    if (indexer < 0).any():
        missing = target_ids[indexer < 0][:5].tolist()
        raise RuntimeError(f"Could not align target rows to feature rows via review_id. Missing examples: {missing}")
    return _subset_matrix(X, indexer.astype(np.int64))


def _align_target_frame_to_features(
    feature_ids: np.ndarray,
    target_frame: pd.DataFrame,
    *,
    split: str,
    spec: ModeSpec,
    logger: logging.Logger,
) -> pd.DataFrame:
    target_ids = target_frame["review_id"].to_numpy()
    if feature_ids.shape == target_ids.shape and np.array_equal(feature_ids, target_ids):
        return target_frame

    feature_indexer = pd.Index(feature_ids).get_indexer(target_ids)
    dropped_mask = feature_indexer < 0
    dropped_count = int(dropped_mask.sum())
    if dropped_count:
        dropped_examples = target_ids[dropped_mask][:5].tolist()
        logger.warning(
            "Dropping %d target rows without matching %s feature rows for split='%s'. "
            "This usually means feature extraction filtered out reviews with missing/short text. "
            "Example review_id values: %s",
            dropped_count,
            spec.feature_task_suffix,
            split,
            dropped_examples,
        )
        target_frame = target_frame.iloc[~dropped_mask].reset_index(drop=True)
        target_ids = target_frame["review_id"].to_numpy()
        if target_ids.size == 0:
            raise RuntimeError(
                f"No target rows remain after aligning '{spec.mode}' split='{split}' to {spec.feature_task_suffix} features."
            )

    if feature_ids.shape == target_ids.shape and np.array_equal(feature_ids, target_ids):
        return target_frame

    target_index = pd.Index(target_ids)
    feature_index = pd.Index(feature_ids)

    # Count stage-2 targets are a positive-only subset of the broader regression feature bundle.
    # In that case the feature matrix will be subset later to the target review_id order.
    target_in_feature = feature_index.get_indexer(target_ids)
    if not (target_in_feature < 0).any():
        logger.info(
            "Aligning target subset to %s feature superset for split='%s' mode='%s' (%d target rows, %d feature rows).",
            spec.feature_task_suffix,
            split,
            spec.mode,
            int(target_ids.shape[0]),
            int(feature_ids.shape[0]),
        )
        return target_frame

    indexer = target_index.get_indexer(feature_ids)
    if (indexer < 0).any():
        missing = feature_ids[indexer < 0][:5].tolist()
        raise RuntimeError(
            f"Could not align feature rows to target rows via review_id for split='{split}' mode='{spec.mode}'. "
            f"Missing examples: {missing}"
        )

    return target_frame.iloc[indexer.astype(np.int64)].reset_index(drop=True)


def load_split_bundle(
    base_dir: str | Path,
    split: str,
    mode: ModeT,
    logger: logging.Logger,
    extra_columns: Optional[list[str]] = None,
) -> DataBundle:
    spec = get_mode_spec(mode)
    extra_columns = extra_columns or []

    feature_ids, feature_id_source = _load_feature_ids(base_dir, split, spec)
    target_columns = ["review_id", spec.target_column, spec.raw_target_column, *extra_columns]
    target_frame = _load_csv_columns(
        base_dir,
        spec.dataset_prefix,
        split,
        columns=target_columns,
    )
    target_frame = _align_target_frame_to_features(
        feature_ids,
        target_frame,
        split=split,
        spec=spec,
        logger=logger,
    )
    X_full = _load_feature_matrix(base_dir, split, spec.feature_task_suffix, logger)
    X = _align_features_to_targets(
        feature_ids,
        target_frame["review_id"].to_numpy(),
        X_full,
        split=split,
        spec=spec,
        feature_id_source=feature_id_source,
    )

    y = target_frame[spec.target_column].to_numpy()
    y_raw = target_frame[spec.raw_target_column].to_numpy()
    return DataBundle(spec=spec, split=split, frame=target_frame, X=X, y=y, y_raw=y_raw)


def to_raw_count_predictions(preds: np.ndarray, spec: ModeSpec) -> np.ndarray:
    preds = np.asarray(preds, dtype=float)
    if spec.prediction_transform == "log1p_expm1":
        preds = np.expm1(preds)
    return np.clip(preds, a_min=0.0, a_max=None)


def pick_precision_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    target_precision: float = 0.30,
    min_recall: float = 0.15,
) -> dict[str, float]:
    precs, recs, thrs = precision_recall_curve(labels, probs)
    precs = precs[:-1]
    recs = recs[:-1]

    if thrs.size == 0:
        preds = (probs >= 0.5).astype(int)
        return {
            "thr": 0.5,
            "precision": float(precision_score(labels, preds, zero_division=0)),
            "recall": float(recall_score(labels, preds, zero_division=0)),
            "f1": float(f1_score(labels, preds, zero_division=0)),
        }

    mask = (precs >= target_precision) & (recs >= min_recall)
    if np.any(mask):
        candidate_idx = np.flatnonzero(mask)
        f1s = 2 * precs[candidate_idx] * recs[candidate_idx] / np.maximum(precs[candidate_idx] + recs[candidate_idx], 1e-12)
        best_idx = int(candidate_idx[np.argmax(f1s)])
        return {
            "thr": float(thrs[best_idx]),
            "precision": float(precs[best_idx]),
            "recall": float(recs[best_idx]),
            "f1": float(np.max(f1s)),
        }

    f1s = 2 * precs * recs / np.maximum(precs + recs, 1e-12)
    best_idx = int(np.argmax(f1s))
    return {
        "thr": float(thrs[best_idx]),
        "precision": float(precs[best_idx]),
        "recall": float(recs[best_idx]),
        "f1": float(f1s[best_idx]),
    }


def compute_binary_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    target_precision: float,
    min_recall: float,
) -> dict[str, float]:
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs).astype(float)

    metrics = {
        "auroc": float(roc_auc_score(labels, probs)) if np.unique(labels).size > 1 else float("nan"),
        "auprc": float(average_precision_score(labels, probs)) if np.unique(labels).size > 1 else float("nan"),
    }

    preds_05 = (probs >= 0.5).astype(int)
    metrics.update(
        {
            "acc_05": float(accuracy_score(labels, preds_05)),
            "prec_05": float(precision_score(labels, preds_05, zero_division=0)),
            "rec_05": float(recall_score(labels, preds_05, zero_division=0)),
            "f1_05": float(f1_score(labels, preds_05, zero_division=0)),
        }
    )

    tuned = pick_precision_threshold(probs, labels, target_precision=target_precision, min_recall=min_recall)
    metrics.update(
        {
            "thr_opt": float(tuned["thr"]),
            "prec_opt": float(tuned["precision"]),
            "rec_opt": float(tuned["recall"]),
            "f1_opt": float(tuned["f1"]),
        }
    )
    return metrics


def _safe_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        return float(r2_score(y_true, y_pred))
    except ValueError:
        return float("nan")


def compute_count_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    positive_mask: Optional[np.ndarray] = None,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    metrics = {
        "rmse": _safe_rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": _safe_r2(y_true, y_pred),
    }
    if y_true.size > 1 and np.unique(y_true).size > 1 and np.unique(y_pred).size > 1:
        spearman = spearmanr(y_true, y_pred).statistic
        metrics["spearman_rho"] = float(spearman) if spearman is not None else float("nan")
    else:
        metrics["spearman_rho"] = float("nan")

    if positive_mask is None:
        positive_mask = y_true > 0
    positive_mask = np.asarray(positive_mask, dtype=bool)

    if positive_mask.any():
        metrics["positive_only_rmse"] = _safe_rmse(y_true[positive_mask], y_pred[positive_mask])
        metrics["positive_only_mae"] = float(mean_absolute_error(y_true[positive_mask], y_pred[positive_mask]))
    else:
        metrics["positive_only_rmse"] = float("nan")
        metrics["positive_only_mae"] = float("nan")
    return metrics


def save_feature_importance_plot(model: Any, out_dir: str | Path, stem: str) -> str:
    plot_path = safe_path(out_dir, "plots", f"{stem}_feature_importance.png")
    plt.figure(figsize=(8, 6))
    xgb.plot_importance(model, importance_type="gain", max_num_features=20, height=0.5)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    return plot_path


def save_binary_plots(
    out_dir: str | Path,
    stem: str,
    labels: np.ndarray,
    probs: np.ndarray,
    *,
    threshold_opt: float,
) -> dict[str, str]:
    plot_dir = Path(out_dir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    roc_path = safe_path(out_dir, "plots", f"{stem}_roc.png")
    pr_path = safe_path(out_dir, "plots", f"{stem}_pr.png")
    cm_path = safe_path(out_dir, "plots", f"{stem}_cm_05.png")
    cm_opt_path = safe_path(out_dir, "plots", f"{stem}_cm_opt.png")

    fpr, tpr, _ = roc_curve(labels, probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(roc_path)
    plt.close()

    precisions, recalls, _ = precision_recall_curve(labels, probs)
    plt.figure(figsize=(6, 5))
    plt.plot(recalls, precisions, lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(pr_path)
    plt.close()

    preds_05 = (probs >= 0.5).astype(int)
    cm = confusion_matrix(labels, preds_05)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix @0.5")
    plt.colorbar()
    ticks = np.arange(2)
    plt.xticks(ticks, [0, 1])
    plt.yticks(ticks, [0, 1])
    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = int(cm[i, j])
            plt.text(j, i, str(value), ha="center", va="center", color="white" if value > thresh else "black")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()

    preds_opt = (probs >= float(threshold_opt)).astype(int)
    cm_opt = confusion_matrix(labels, preds_opt)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm_opt, interpolation="nearest")
    plt.title(f"Confusion Matrix @{float(threshold_opt):.4f}")
    plt.colorbar()
    ticks = np.arange(2)
    plt.xticks(ticks, [0, 1])
    plt.yticks(ticks, [0, 1])
    thresh = cm_opt.max() / 2.0 if cm_opt.size else 0.0
    for i in range(cm_opt.shape[0]):
        for j in range(cm_opt.shape[1]):
            value = int(cm_opt[i, j])
            plt.text(j, i, str(value), ha="center", va="center", color="white" if value > thresh else "black")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(cm_opt_path)
    plt.close()

    return {"roc": roc_path, "pr": pr_path, "cm_05": cm_path, "cm_opt": cm_opt_path}


def save_regression_scatter(out_dir: str | Path, stem: str, y_true: np.ndarray, y_pred: np.ndarray, title: str) -> str:
    plot_path = safe_path(out_dir, "plots", f"{stem}_scatter.png")
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.25, s=6)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    return plot_path


def _booster_expected_schema(booster: xgb.Booster) -> tuple[int, Optional[list[str]]]:
    attrs = booster.attributes() or {}
    num_features = None
    if "num_feature" in attrs:
        try:
            num_features = int(attrs["num_feature"])
        except Exception:
            num_features = None

    if num_features is None:
        try:
            cfg = json.loads(booster.save_config())
            num_features = int(cfg["learner"]["learner_model_param"]["num_feature"])
        except Exception:
            num_features = 0

    feature_names = None
    try:
        cfg = json.loads(booster.save_config())
        feature_cfg = cfg["learner"].get("features", {})
        candidate = feature_cfg.get("feature_names")
        if isinstance(candidate, list) and len(candidate) == num_features:
            feature_names = candidate
    except Exception:
        feature_names = None

    return num_features, feature_names


def load_booster(model_path: str | Path) -> xgb.Booster:
    booster = xgb.Booster()
    booster.load_model(str(model_path))
    return booster


def evaluate_mode(
    *,
    mode: ModeT,
    split: str,
    base_dir: str | Path,
    out_dir: str | Path,
    target_precision: float,
    min_recall: float,
    logger: logging.Logger,
    model_path_override: Optional[str] = None,
) -> dict[str, Any]:
    spec = get_mode_spec(mode)
    model_path = model_path_override or checkpoint_path(out_dir, mode)
    meta_path = metadata_path(out_dir, mode)
    logger.info("Loading %s model from %s", mode, model_path)

    booster = load_booster(model_path)
    expected_cols, feature_names = _booster_expected_schema(booster)
    bundle = load_split_bundle(base_dir, split, mode, logger)

    if expected_cols and bundle.X.shape[1] != expected_cols:
        raise RuntimeError(
            f"Feature column mismatch for {mode}: model expects {expected_cols}, loaded {bundle.X.shape[1]}"
        )

    dsplit = xgb.DMatrix(bundle.X, feature_names=feature_names if feature_names else None)
    summary: dict[str, Any] = {
        "mode": mode,
        "split": split,
        "rows": int(bundle.frame.shape[0]),
        "model_path": model_path,
        "metadata_path": meta_path,
    }

    if spec.role == "classifier":
        probs = booster.predict(dsplit, validate_features=True).astype(float)
        metrics = compute_binary_metrics(bundle.y.astype(int), probs, target_precision=target_precision, min_recall=min_recall)
        plot_paths = save_binary_plots(
            out_dir,
            f"{spec.checkpoint_stem}_{split}",
            bundle.y.astype(int),
            probs,
            threshold_opt=float(metrics["thr_opt"]),
        )

        prediction_df = bundle.frame.loc[:, ["review_id"]].copy()
        prediction_df["true_label"] = bundle.y.astype(int)
        prediction_df["prob_pos"] = probs
        prediction_df["pred_05"] = (probs >= 0.5).astype(int)
        prediction_df["pred_opt"] = (probs >= metrics["thr_opt"]).astype(int)
    else:
        raw_preds = booster.predict(dsplit, validate_features=True).astype(float)
        preds = to_raw_count_predictions(raw_preds, spec)
        metrics = compute_count_metrics(bundle.y_raw.astype(float), preds, positive_mask=bundle.y_raw.astype(float) > 0)
        plot_paths = {
            "scatter": save_regression_scatter(
                out_dir,
                f"{spec.checkpoint_stem}_{split}",
                bundle.y_raw.astype(float),
                preds,
                title=f"{mode} predicted vs true ({split})",
            )
        }

        prediction_df = bundle.frame.loc[:, ["review_id"]].copy()
        prediction_df["y_true"] = bundle.y_raw.astype(float)
        prediction_df["y_pred"] = preds
        prediction_df["model_output"] = raw_preds

    prediction_path = prediction_csv_path(out_dir, mode, split)
    prediction_df.to_csv(prediction_path, index=False)

    payload = {
        "summary": summary,
        "metrics": metrics,
        "plots": plot_paths,
    }
    save_json(metrics_json_path(out_dir, mode, split), payload)

    for key, value in metrics.items():
        logger.info("%s=%s", key, f"{value:.6f}" if isinstance(value, float) else value)
    logger.info("Saved predictions to %s", prediction_path)
    if mode == "binary":
        try_generate_requested_main_figures(
            out_dir,
            split=split,
            logger=logger,
            include_binary=True,
            include_count=False,
            target_precision=target_precision,
            min_recall=min_recall,
        )
    return payload


def save_checkpoint_metadata(out_dir: str | Path, mode: ModeT, payload: dict[str, Any]) -> str:
    path = metadata_path(out_dir, mode)
    save_json(path, payload)
    return path


def default_model_metadata(mode: ModeT) -> dict[str, Any]:
    spec = get_mode_spec(mode)
    return {
        "mode": spec.mode,
        "role": spec.role,
        "dataset_prefix": spec.dataset_prefix,
        "feature_dataset_prefix": spec.feature_dataset_prefix,
        "feature_task_suffix": spec.feature_task_suffix,
        "target_column": spec.target_column,
        "raw_target_column": spec.raw_target_column,
        "prediction_transform": spec.prediction_transform,
        "default_objective": spec.default_objective,
        "default_eval_metric": spec.default_eval_metric,
        "objective_params": spec.objective_params,
    }
