"""Small, explicit orchestration around unchanged thesis helpers, always on CPU."""
from __future__ import annotations

from dataclasses import asdict
import gc
import hashlib
from importlib.metadata import version
import json
import logging
from pathlib import Path
import platform
import random
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
import torch
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from xgboost import XGBClassifier, XGBRegressor

from src import data_preprocessing as prep
from src import feature_extraction as features
from src.demo.config import CFG
from models.bert_model import BERTClassifier, BERTRegressor
from experiments.weighted_rating import weighted_rating as rating

# Existing command-line training scripts use sibling imports. Keep their files intact.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))
from training import baseline_common as baseline
from training.xgboost_common import compute_count_metrics, pick_precision_threshold

OUT = ROOT / "outputs" / "demo"
NOTICE = "DEMO ONLY: reduced synthetic data / one epoch; these are NOT thesis results."
AUX = ["review_length", "sentiment_score", "is_verified"]


def write_json(path: Path, value) -> None:
    def safe(obj):
        if isinstance(obj, dict):
            return {str(k): safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [safe(v) for v in obj]
        if isinstance(obj, np.generic):
            return safe(obj.item())
        if isinstance(obj, float) and not np.isfinite(obj):
            return None
        return obj
    path.write_text(json.dumps(safe(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def preprocess(offline: bool, logger) -> dict:
    # Avoid NLTK's unconditional download call in the full-scale loader on cached runs.
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        if offline:
            raise RuntimeError("VADER lexicon is not cached. Run once online without --offline.")
        logger.info("Downloading VADER sentiment lexicon (first use).")
        if not nltk.download("vader_lexicon", quiet=True):
            raise RuntimeError("Could not download VADER lexicon.")
    source = ROOT / "datasets/sample/reviews.jsonl"
    manifest = json.loads((source.parent / "manifest.json").read_text(encoding="utf-8"))
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("Bundled sample hash differs from manifest.json.")
    df = prep.load_reviews(source, logger)
    # Same text, length, sentiment, timestamp and target transforms as the thesis.
    # Fixed reference date makes repeated demo runs independent of today's date.
    df["clean_text"] = df.text.apply(prep.clean_text)
    df["clean_title"] = df.title.apply(prep.clean_text)
    df["full_text"] = (df.clean_title + ". " + df.clean_text).str.strip(" .")
    df["review_length"] = df.clean_text.str.split().str.len()
    sia = SentimentIntensityAnalyzer()
    df["sentiment_score"] = df.clean_text.map(lambda text: sia.polarity_scores(text)["compound"])
    df["is_verified"] = df.verified_purchase.astype(int)
    df["timestamp_parsed"] = df.timestamp.apply(prep.parse_timestamp)
    df["review_age_days"] = (pd.Timestamp(CFG.reference_date) - df.timestamp_parsed).dt.days.astype(float)
    df["timestamp_recency_days"] = df.review_age_days.fillna(-1.0)
    config = prep.Config(raw_jsonl=source, output_dir=OUT / "preprocessed", max_workers=1)
    df = df.loc[df.review_length >= config.min_tokens].copy()
    df = prep.cap_and_normalize(df, config.cap_q)
    splits, meta = prep.split_base_dataset(df, config)
    tail = float(splits["train"].helpful_vote.quantile(config.extreme_tail_q))
    splits = prep.annotate_shared_targets(splits, extreme_tail_threshold=tail)
    meta.update(extreme_tail_q=config.extreme_tail_q, extreme_tail_threshold=tail)
    prep.save_split_definition(splits, config.output_dir, meta)
    prep.build_binary_branch(splits, config, config.output_dir)
    prep.build_count_branch(splits, config.output_dir, extreme_tail_threshold=tail)
    for split, frame in splits.items():
        frame.to_csv(config.output_dir / f"regression_score_{split}.csv", index=False)
    binary = {s: pd.read_csv(config.output_dir / f"binary_label_{s}.csv") for s in splits}
    for split in splits:
        if set(binary[split].binary_label) != {0, 1} or set(splits[split].has_helpful_vote) != {0, 1}:
            raise ValueError(f"Both classes are required in {split}.")
        if not (splits[split].helpful_vote > 1).any():
            raise ValueError(f"Multi-vote examples are required in {split}.")
    ids = [set(splits[s].review_id) for s in ("train", "val", "test")]
    if any(ids[i] & ids[j] for i in range(3) for j in range(i)):
        raise ValueError("Train/validation/test overlap.")
    return {"count": splits, "binary": binary, "manifest": manifest}


def feature_bundles(data: dict, logger) -> dict:
    # Changes are confined to this demo process; source defaults are unchanged.
    features.TFIDF_MAX_FEATURES = CFG.tfidf_features
    features.NUM_SVD_COMPONENTS = CFG.svd_components
    arrays = {}
    for task, frames in (("binary", data["binary"]), ("regression", data["count"])):
        arrays[task] = {}
        for split, frame in frames.items():
            path = features.extract_features(frame, "full_text", split, split == "train",
                                             OUT / "preprocessed", OUT / "figures", OUT / "feature_models",
                                             task, False, logger)
            arrays[task][split] = np.load(path)
            frame[["review_id"]].to_csv(OUT / "preprocessed" / f"X_{split}_feature_ids_{task}.csv", index=False)
    return arrays


def binary_metrics(labels, probs, threshold: float) -> dict:
    labels, probs = np.asarray(labels), np.asarray(probs)
    predicted = probs >= threshold
    return {"rows": len(labels), "threshold_from_validation": threshold,
            "auprc": average_precision_score(labels, probs), "auroc": roc_auc_score(labels, probs),
            "accuracy": accuracy_score(labels, predicted), "precision": precision_score(labels, predicted, zero_division=0),
            "recall": recall_score(labels, predicted, zero_division=0), "f1": f1_score(labels, predicted, zero_division=0)}


def fit_baselines(data, metrics) -> dict:
    binary = baseline.train_binary_baseline(data["binary"]["train"], "binary_label")
    regression = baseline.train_regression_baseline(data["count"]["train"], "regression_score")
    count = baseline.train_regression_baseline(data["count"]["train"], "helpful_vote")
    for name, model in (("binary", binary), ("regression", regression), ("count", count)):
        baseline.save_model(model, OUT / "checkpoints" / f"baseline_{name}.json")
    _, reg_metrics = baseline.evaluate_regression(data["count"]["test"], regression)
    metrics["baseline_regression_score"] = reg_metrics
    probs = np.full(len(data["binary"]["test"]), binary.positive_rate)
    metrics["baseline_binary"] = binary_metrics(data["binary"]["test"].binary_label, probs, 0.5)
    counts = np.full(len(data["count"]["test"]), count.constant_value)
    metrics["baseline_count"] = compute_count_metrics(data["count"]["test"].helpful_vote, counts)
    data["binary"]["test"][["review_id", "binary_label"]].assign(prob_pos=probs).to_csv(
        OUT / "predictions/baseline_binary_test.csv", index=False)
    data["count"]["test"][["review_id", "helpful_vote", "regression_score"]].assign(
        predicted_count=counts, predicted_score=regression.constant_value).to_csv(
        OUT / "predictions/baseline_count_test.csv", index=False)
    return {"binary": probs, "count": counts}


def fit_xgboost(data, arrays, metrics, logger) -> dict:
    results = {}
    params = dict(n_estimators=CFG.xgb_estimators, max_depth=CFG.xgb_max_depth, learning_rate=0.1,
                  subsample=0.8, colsample_bytree=0.8, tree_method="hist", device="cpu",
                  n_jobs=torch.get_num_threads(), random_state=CFG.seed, reg_lambda=1.5)
    for mode in ("binary", "count_stage1", "count_stage2_log1p", "count_stage2_poisson"):
        logger.info("XGBoost: %s (%d estimators, CPU)", mode, CFG.xgb_estimators)
        frames = data["binary"] if mode == "binary" else data["count"]
        X = arrays["binary"] if mode == "binary" else arrays["regression"]
        train = frames["train"]
        if mode in ("binary", "count_stage1"):
            column = "binary_label" if mode == "binary" else "has_helpful_vote"
            y = train[column].to_numpy()
            model = XGBClassifier(**params, objective="binary:logistic", scale_pos_weight=float((y == 0).sum() / (y == 1).sum()))
            model.fit(X["train"], y)
            val_probs = model.predict_proba(X["val"])[:, 1]
            threshold = pick_precision_threshold(val_probs, frames["val"][column].to_numpy())["thr"]
            probs = model.predict_proba(X["test"])[:, 1]
            metrics[f"xgb_{mode}"] = binary_metrics(frames["test"][column], probs, threshold)
            results[mode] = {"prediction": probs, "threshold": threshold, "model": model}
        else:
            mask = train.helpful_vote.to_numpy() > 0
            y = train.helpful_vote.to_numpy()[mask].astype(float)
            log_target = mode.endswith("log1p")
            model = XGBRegressor(**params, objective="reg:squarederror" if log_target else "count:poisson")
            model.fit(X["train"][mask], np.log1p(y) if log_target else y)
            raw = model.predict(X["test"])
            pred = np.maximum(0, np.expm1(raw) if log_target else raw)
            combined = pred * results["count_stage1"]["prediction"]
            metrics[f"xgb_{mode}_combined"] = compute_count_metrics(frames["test"].helpful_vote, combined)
            positive = frames["test"].helpful_vote.to_numpy() > 0
            metrics[f"xgb_{mode}_positive_only"] = compute_count_metrics(frames["test"].helpful_vote.to_numpy()[positive], pred[positive])
            results[mode] = {"prediction": combined, "conditional": pred, "model": model}
        model.save_model(OUT / "checkpoints" / f"xgb_{mode}.json")
        target = frames["test"][["review_id", "helpful_vote"]].copy()
        target["prediction"] = results[mode]["prediction"]
        if "conditional" in results[mode]:
            target["conditional_positive_count"] = results[mode]["conditional"]
            target["stage1_probability"] = results["count_stage1"]["prediction"]
        target.to_csv(OUT / "predictions" / f"xgb_{mode}_test.csv", index=False)
    # Binary predictions on gray-band rows are needed for rating controls too.
    results["binary_all_test"] = results["binary"]["model"].predict_proba(arrays["binary_all_test"])[:, 1]
    return results


def slice_frame(frame: pd.DataFrame, count: int, column: str | None = None) -> pd.DataFrame:
    if len(frame) <= count:
        return frame.copy()
    subset, _ = train_test_split(frame, train_size=count, random_state=CFG.seed,
                                stratify=frame[column] if column else None)
    return subset.copy()


def fit_bert(data, metrics, model_path: str, logger) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    results = {}
    for mode in ("binary", "count_stage1", "count_stage2_log1p", "count_stage2_poisson"):
        random.seed(CFG.seed)
        np.random.seed(CFG.seed)
        torch.manual_seed(CFG.seed)
        classification = mode in ("binary", "count_stage1")
        frames = data["binary"] if mode == "binary" else data["count"]
        column = "binary_label" if mode == "binary" else "has_helpful_vote" if classification else "helpful_vote"
        pool = frames["train"] if classification else frames["train"].loc[frames["train"].helpful_vote > 0]
        train = slice_frame(pool, CFG.bert_train_rows, column if classification else None)
        val = slice_frame(frames["val"], CFG.bert_val_rows, column if classification else None)
        model_class = BERTClassifier if classification else BERTRegressor
        model = model_class(model_name=model_path, dropout_rate=0.3, use_mean_pooling=True,
                            aux_feat_dim=len(AUX), freeze_init=5).to("cpu")
        # Freeze embeddings as well as the first five blocks. Fine-tune block six.
        for parameter in model.bert.parameters():
            parameter.requires_grad_(False)
        for parameter in model.bert.transformer.layer[-1].parameters():
            parameter.requires_grad_(True)
        aux_raw = train[AUX].to_numpy(dtype=np.float32)
        aux_raw[:, 0] = np.log1p(aux_raw[:, 0])
        means, stds = aux_raw.mean(axis=0), np.maximum(aux_raw.std(axis=0), 1e-6)
        transformed = np.log1p(train.helpful_vote.to_numpy(dtype=np.float32))
        target_mean = float(transformed.mean())
        target_std = max(float(transformed.std()), 1e-6)
        encoded = {}
        for name, frame in (("train", train), ("val", val), ("test", data["count"]["test"])):
            tokens = tokenizer(frame.full_text.tolist(), padding="max_length", truncation=True,
                               max_length=CFG.bert_max_length, return_tensors="pt")
            aux = frame[AUX].to_numpy(dtype=np.float32).copy()
            aux[:, 0] = np.log1p(aux[:, 0])
            encoded[name] = (tokens, torch.from_numpy((aux - means) / stds))
        groups = [
            {"params": [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("bert.")], "lr": CFG.bert_base_lr},
            {"params": [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("bert.")], "lr": CFG.bert_head_lr},
        ]
        optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
        labels = torch.tensor(train[column].to_numpy(), dtype=torch.float32)
        if classification:
            positive = float(labels.sum())
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(labels) - positive) / max(1, positive)))
        elif mode.endswith("log1p"):
            labels = (torch.log1p(labels) - target_mean) / target_std
            loss_fn = torch.nn.SmoothL1Loss()
        else:
            loss_fn = torch.nn.PoissonNLLLoss(log_input=True)
        logger.info("DistilBERT %s: 1 epoch, %d rows, max length %d, fine-tuning last block + head on CPU",
                    mode, len(train), CFG.bert_max_length)
        model.train()
        losses = []
        for indices in torch.randperm(len(train)).split(CFG.bert_batch_size):
            tokens, aux = encoded["train"]
            optimizer.zero_grad(set_to_none=True)
            predicted = model(input_ids=tokens["input_ids"][indices], attention_mask=tokens["attention_mask"][indices], aux_features=aux[indices])
            if not classification and mode.endswith("poisson"):
                predicted = predicted.clamp(-10, 10)
            loss = loss_fn(predicted, labels[indices])
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss in {mode}.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        def predict(split):
            tokens, aux = encoded[split]
            batches = []
            with torch.inference_mode():
                for start in range(0, len(aux), CFG.bert_batch_size):
                    end = start + CFG.bert_batch_size
                    logits = model(input_ids=tokens["input_ids"][start:end], attention_mask=tokens["attention_mask"][start:end], aux_features=aux[start:end])
                    if classification:
                        output = logits.sigmoid()
                    elif mode.endswith("log1p"):
                        output = torch.expm1((logits * target_std + target_mean).clamp(-10, 10)).clamp_min(0)
                    else:
                        output = logits.clamp(-10, 10).exp()
                    batches.append(output.numpy())
            return np.concatenate(batches)
        test_pred = predict("test")
        if classification:
            threshold = pick_precision_threshold(predict("val"), val[column].to_numpy())["thr"]
            positions = data["count"]["test"].review_id.to_list()
            positions = {rid: i for i, rid in enumerate(positions)}
            selected = [positions[rid] for rid in frames["test"].review_id]
            metrics[f"bert_{mode}"] = binary_metrics(frames["test"][column], test_pred[selected], threshold)
            results[mode] = {"prediction": test_pred, "threshold": threshold}
        else:
            combined = test_pred * results["count_stage1"]["prediction"]
            metrics[f"bert_{mode}_combined"] = compute_count_metrics(frames["test"].helpful_vote, combined)
            positive = frames["test"].helpful_vote.to_numpy() > 0
            metrics[f"bert_{mode}_positive_only"] = compute_count_metrics(frames["test"].helpful_vote.to_numpy()[positive], test_pred[positive])
            results[mode] = {"prediction": combined, "conditional": test_pred}
        details = {"notice": NOTICE, "mode": mode, "train_rows": len(train), "validation_rows": len(val),
                   "train_review_ids": train.review_id.tolist(), "validation_review_ids": val.review_id.tolist(),
                   "mean_train_loss": np.mean(losses), "epochs": CFG.bert_epochs, "device": "cpu",
                   "aux_means": means.tolist(), "aux_stds": stds.tolist(),
                   "target_log_mean": target_mean, "target_log_std": target_std,
                   "threshold": results[mode].get("threshold"), "model_revision": CFG.model_revision}
        # Store trained parameters only; the pinned pretrained backbone stays in HF's cache.
        torch.save({n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
                   OUT / "checkpoints" / f"bert_{mode}_trained_parameters.pt")
        write_json(OUT / "checkpoints" / f"bert_{mode}_config.json", details)
        predictions = data["count"]["test"][["review_id", "helpful_vote"]].copy()
        predictions["prediction"] = results[mode]["prediction"]
        if "conditional" in results[mode]:
            predictions["conditional_positive_count"] = results[mode]["conditional"]
            predictions["stage1_probability"] = results["count_stage1"]["prediction"]
        predictions.to_csv(OUT / "predictions" / f"bert_{mode}_test.csv", index=False)
        del model, optimizer, groups, encoded
        gc.collect()
    return results


def weighted_comparison(data, xgb, bert) -> tuple:
    frame = data["count"]["test"].copy()
    frame["p_bert"] = bert["binary"]["prediction"]
    frame["p_xgb"] = xgb["binary_all_test"]
    frame["yhat_xgb"] = xgb["count_stage2_log1p"]["prediction"]
    frame["pred_source"] = "demo_inference"
    sizes = frame.groupby("parent_asin").size()
    frame = frame.loc[frame.parent_asin.isin(sizes[sizes >= CFG.rating_min_reviews].index)].reset_index(drop=True)
    if frame.parent_asin.nunique() < 2:
        raise ValueError("At least two held-out titles are required for weighting.")
    frame.to_csv(OUT / "weighted_rating/review_level_inputs.csv", index=False)
    signals = rating.signal_frame(frame, bert["binary"]["threshold"])
    base = rating.item_weighted_mean(frame, np.ones(len(frame)))
    n = frame.groupby("parent_asin").size()
    per_item = pd.DataFrame({"reviews": n, "arithmetic_mean": base})
    sweeps, summaries = [], []
    for alpha in np.linspace(0, 1, 11):
        weights = rating.weights_for_alpha(signals, float(alpha))
        for scheme, w in weights.items():
            weighted = rating.item_weighted_mean(frame, w)
            change = rating.change_summary(base, weighted, n)
            rank = rating.ranking_summary(base, weighted)
            sweeps.append({"alpha": alpha, "scheme": scheme, **change, **rank})
            if np.isclose(alpha, 0.5) or np.isclose(alpha, 1.0):
                summaries.append({"alpha": alpha, "scheme": scheme, **change, **rank})
            if np.isclose(alpha, 0.5):
                per_item[f"{scheme}_mean"] = weighted
                per_item[f"{scheme}_delta"] = weighted - base
    null, p_item = rating.permutation_null(frame, rating.weights_for_alpha(signals, 0.5)["bert"], base,
                                         per_item.bert_mean, CFG.permutations, np.random.default_rng(CFG.seed))
    per_item["bert_permutation_p"] = p_item
    per_item.to_csv(OUT / "weighted_rating/per_item.csv")
    pd.DataFrame(summaries).to_csv(OUT / "weighted_rating/comparison.csv", index=False)
    sweep = pd.DataFrame(sweeps)
    sweep.to_csv(OUT / "weighted_rating/alpha_sweep.csv", index=False)
    diagnostics = {"notice": NOTICE, "reviews": len(frame), "items": len(per_item), "n_min": CFG.rating_min_reviews,
                   "permutations": CFG.permutations, "mean_abs_delta_bert_alpha_0_5": per_item.bert_delta.abs().mean(),
                   "permutation_mean_abs_delta": null.mean(), "interpretation": "Rating changes only; no rating-accuracy ground truth."}
    write_json(OUT / "weighted_rating/diagnostics.json", diagnostics)
    return per_item, sweep, diagnostics


def figures(data, baselines, xgb, bert, per_item, sweep) -> None:
    def save(fig, name):
        fig.suptitle("CPU demo - synthetic sample (not thesis metrics)", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        for extension in ("png", "pdf"):
            fig.savefig(OUT / "figures" / f"{name}.{extension}", dpi=150)
        plt.close(fig)
    binary = data["binary"]["test"]
    all_test = data["count"]["test"]
    index = {rid: i for i, rid in enumerate(all_test.review_id)}
    b_indices = [index[rid] for rid in binary.review_id]
    probs = {"Baseline": baselines["binary"], "XGBoost": xgb["binary"]["prediction"],
             "DistilBERT": bert["binary"]["prediction"][b_indices]}
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, values in probs.items():
        precision, recall, _ = precision_recall_curve(binary.binary_label, values)
        ax.plot(recall, precision, label=name)
    ax.set(xlabel="Recall", ylabel="Precision", title="Binary helpfulness")
    ax.legend()
    save(fig, "binary_precision_recall")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    for ax, name, values, threshold in ((axes[0], "XGBoost", probs["XGBoost"], xgb["binary"]["threshold"]),
                                         (axes[1], "DistilBERT", probs["DistilBERT"], bert["binary"]["threshold"])):
        cm = confusion_matrix(binary.binary_label, values >= threshold, labels=[0, 1])
        ax.imshow(cm, cmap="Blues")
        for row in range(2):
            for column in range(2):
                color = "white" if cm[row, column] > cm.max() / 2 else "black"
                ax.text(column, row, str(cm[row, column]), ha="center", va="center", color=color)
        ax.set(xticks=[0, 1], yticks=[0, 1], xlabel="Predicted", ylabel="Actual", title=name)
    save(fig, "binary_confusion_matrices")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, family, result in ((axes[0], "XGBoost", xgb), (axes[1], "DistilBERT", bert)):
        for mode, label in (("count_stage2_log1p", "log1p"), ("count_stage2_poisson", "Poisson")):
            ax.scatter(all_test.helpful_vote, result[mode]["prediction"], label=label, s=12, alpha=0.5)
        ax.set(xlabel="Actual helpful votes", ylabel="Expected count (soft gate)", title=family)
        ax.legend()
    save(fig, "count_actual_vs_predicted")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(per_item.bert_delta, bins=min(12, len(per_item)), edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(xlabel="DistilBERT weighted mean - arithmetic mean (stars)", ylabel="Titles", title="Rating change, alpha = 0.5")
    save(fig, "rating_change_distribution")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for scheme, name in (("bert", "DistilBERT"), ("xgb_count", "XGBoost count"), ("votes_observed", "Observed votes")):
        part = sweep.loc[sweep.scheme == scheme]
        axes[0].plot(part.alpha, part.mean_abs_delta, label=name)
        axes[1].plot(part.alpha, part.kendall_tau, label=name)
    axes[0].set(xlabel="Weighting strength alpha", ylabel="Mean absolute rating change (stars)")
    axes[1].set(xlabel="Weighting strength alpha", ylabel="Kendall tau with arithmetic ranking")
    axes[0].legend(fontsize=8)
    save(fig, "weighting_alpha_sweep")


def run(threads: int, offline: bool) -> None:
    start = time.perf_counter()
    for folder in ("preprocessed", "feature_models", "checkpoints", "predictions", "figures", "weighted_rating"):
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    # Preserve earlier runs and their completion marker. A failed rerun cannot look successful.
    for name in ("metrics_summary.json", "metrics_summary.csv", "summary.txt", "output_manifest.json"):
        path = OUT / name
        if path.exists():
            previous = OUT / "previous_run"
            previous.mkdir(exist_ok=True)
            path.replace(previous / name)
    logger = logging.getLogger("demo")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(OUT / "run.log", mode="w", encoding="utf-8")):
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(handler)
    logger.info(NOTICE)
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    random.seed(CFG.seed)
    np.random.seed(CFG.seed)
    torch.manual_seed(CFG.seed)
    if torch.version.cuda is not None or torch.cuda.is_available():
        raise RuntimeError("The demo must use a CPU-only torch build.")
    timings = {}
    def stage(name, fn):
        logger.info("%s", name)
        stamp = time.perf_counter()
        result = fn()
        timings[name] = time.perf_counter() - stamp
        logger.info("Completed in %.1f s", timings[name])
        return result
    model_path = stage("1/7 Load pinned DistilBERT (cache or first download)", lambda: snapshot_download(
        repo_id=CFG.model_name, revision=CFG.model_revision, local_files_only=offline,
        allow_patterns=["config.json", "tokenizer_config.json", "tokenizer.json", "vocab.txt", "model.safetensors"]))
    data = stage("2/7 Preprocess bundled sample and create seed-42 splits", lambda: preprocess(offline, logger))
    arrays = stage("3/7 Build TF-IDF / SVD / metadata features", lambda: feature_bundles(data, logger))
    # Transform all test rows, including the binary gray band, with the binary training features.
    # Use a distinct filename so the primary binary test bundle remains aligned to its labels.
    path = features.extract_features(data["count"]["test"], "full_text", "all_test", False,
                                     OUT / "preprocessed", OUT / "figures", OUT / "feature_models", "binary", False, logger)
    arrays["binary_all_test"] = np.load(path)
    metrics = {}
    baselines = stage("4/7 Train and evaluate baselines", lambda: fit_baselines(data, metrics))
    xgb = stage("5/7 Train and evaluate XGBoost binary + two-stage counts", lambda: fit_xgboost(data, arrays, metrics, logger))
    bert = stage("6/7 Fine-tune and evaluate DistilBERT binary + two-stage counts", lambda: fit_bert(data, metrics, model_path, logger))
    def finish():
        per_item, sweep, diagnostics = weighted_comparison(data, xgb, bert)
        figures(data, baselines, xgb, bert, per_item, sweep)
        return diagnostics
    diagnostics = stage("7/7 Compare helpfulness-weighted ratings and render figures", finish)
    versions = {p: version(p) for p in ("torch", "transformers", "xgboost", "scikit-learn", "pandas", "numpy", "scipy", "nltk")}
    elapsed = time.perf_counter() - start
    payload = {"status": "complete", "notice": NOTICE, "device": "cpu", "torch_cuda_build": torch.version.cuda,
               "python": sys.version, "platform": platform.platform(), "cpu": platform.processor(), "threads": threads,
               "versions": versions, "config": asdict(CFG), "sample": data["manifest"],
               "split_rows": {name: {s: len(f) for s, f in data[name].items()} for name in ("count", "binary")},
               "timings_seconds": timings, "pipeline_seconds": elapsed, "metrics": metrics, "weighted_rating": diagnostics}
    write_json(OUT / "metrics_summary.json", payload)
    pd.DataFrame(metrics).T.rename_axis("model_task").to_csv(OUT / "metrics_summary.csv")
    summary = [NOTICE, f"Completed end to end on CPU in {elapsed:.1f} seconds ({threads} threads).",
               "Binary held-out results (thresholds selected on validation):"]
    for name in ("baseline_binary", "xgb_binary", "bert_binary"):
        summary.append(f"  {name}: AP={metrics[name]['auprc']:.4f}, F1={metrics[name]['f1']:.4f}")
    summary.append("Combined count held-out results:")
    for name in ("baseline_count", "xgb_count_stage2_log1p_combined", "bert_count_stage2_log1p_combined"):
        summary.append(f"  {name}: MAE={metrics[name]['mae']:.3f}, RMSE={metrics[name]['rmse']:.3f}")
    summary.extend([f"Weighted rating: {diagnostics['items']} held-out titles; mean absolute change "
                    f"{diagnostics['mean_abs_delta_bert_alpha_0_5']:.4f} stars (alpha=0.5).",
                    "Rating changes demonstrate aggregation, not improved rating accuracy.",
                    "Outputs: outputs/demo/metrics_summary.json, summary.txt, figures/, weighted_rating/."])
    (OUT / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    files = sorted(p.relative_to(OUT).as_posix() for p in OUT.rglob("*") if p.is_file() and "previous_run" not in p.parts)
    write_json(OUT / "output_manifest.json", {"notice": NOTICE, "files": files + ["output_manifest.json"]})
    logger.info("\n%s", "\n".join(summary))
