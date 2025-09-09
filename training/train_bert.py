
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # avoid parallel Rust tokenizer threads
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import logging
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_

from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

from src.dataset import ReviewDataset
from models.bert_model import BERTClassifier, BERTRegressor
from src.utils import (
    set_freeze_depth, build_param_groups, rdrop_ce_loss,
    EMA, autocast_context, maybe_print_batch_stats,
    AsymmetricFocalLoss, pick_precision_threshold
)

# Safer default threading on Windows and stable CUDA backends
try:
    torch.set_num_threads(1)
except Exception:
    pass
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

# ============================================================
# Logging
# ============================================================
os.makedirs("../outputs/logs", exist_ok=True)
os.makedirs("../outputs/checkpoints", exist_ok=True)
os.makedirs("../outputs/plots", exist_ok=True)

LOG_PATH = "../outputs/logs/train_bert.log"

logger = logging.getLogger("train_bert")
logger.setLevel(logging.INFO)
logger.handlers.clear()
ch = logging.StreamHandler(); ch.setLevel(logging.INFO)
fh = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8"); fh.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
ch.setFormatter(fmt); fh.setFormatter(fmt)
logger.addHandler(ch); logger.addHandler(fh)

# ============================================================
# AMP controller
# ============================================================
USE_AMP = True
AMP_DTYPE = "bf16"      # "bf16" or "fp16"
DISABLE_AMP_ON_NAN = True
REENABLE_AMP_AFTER_EPOCH = True  # only if not forced fp32
AMP_WAS_DISABLED_THIS_EPOCH = False
force_fp32_next_epoch = False

# ============================================================
# CONFIG (baseline)
# ============================================================
TASK = "binary"  # "regression" or "binary"
assert TASK in ["regression", "binary"]
PATIENCE = 3
N_SPLITS = 1
DROPOUT_RATE = 0.7
LOSS_FN = "smooth"              # "smooth" | "mse" (regression only)
USE_MEAN_POOLING = True

MAX_LEN = 256
BATCH_SIZE = 8
EPOCHS = 10

BASE_LR = 1e-5
HEAD_LR = 3e-5
GRAD_ACCUM_STEPS = 2
WEIGHT_DECAY = 0.01
HEAD_WEIGHT_DECAY = 0.10
FREEZE_LAYERS_INIT = 4
AUX_FEAT_DIM = 3

WARMUP_PCT = 0.10
UNFREEZE_SCHEDULE = {0: 6, 3: 4, 6: 2, 8: 0}

USE_R_DROP = True
R_DROP_ALPHA = 0.5
AMP_WARMUP_EPOCHS = 1           # run first N epochs without AMP

# Binary-specific regularization toggles
USE_CLASS_WEIGHTS = True
USE_WEIGHTED_SAMPLER = False
CE_LABEL_SMOOTH = 0.01          # mild smoothing (helps precision a bit)
USE_EMA = True
EMA_DECAY = 0.999

# ---- Precision-oriented knobs ----
LOSS_KIND = "ce"                 # "ce" | "afl"
AF_ALPHA_POS = 0.5
AF_GAMMA_POS = 0.0
AF_GAMMA_NEG = 2.0
CLASS_WEIGHT_POS_OVERRIDE = None
TARGET_PRECISION = 0.30
MIN_RECALL = 0.15
SAVE_PRECISION_OPT_THR = True

# Safety / stability
MAX_WARN_PRINT = 20
WARN_PRINT_EVERY = 1000
MAX_SKIP_FRAC_DISABLE_AMP = 0.005
LR_REDUCE_FACTOR = 0.5
CATASTROPHIC_SKIP_FRAC = 0.50
LR_FLOOR = 1e-7
LOGIT_CLAMP = 50.0              # safe CE/AFL range

# Optional: soften epoch-1 dynamics for the head
KICKSTART_HEAD_LR = True
HEAD_KICK_LR = 5e-6

# ============================================================
# SPEED-FIRST OVERRIDES (virtual epochs, shorter seqs, larger batch)
# Toggle this True for quick iterations; False for full training.
# ============================================================
FAST_RUN = True

TRAIN_MAX_STEPS_PER_EPOCH = 50_000 if FAST_RUN else None   # cap train batches per epoch
VAL_MAX_STEPS             = 5_000  if FAST_RUN else None    # cap val batches per eval

if FAST_RUN:
    MAX_LEN = 128
    BATCH_SIZE = 16
    GRAD_ACCUM_STEPS = 1
    USE_R_DROP = False
    USE_EMA = False
    AMP_WARMUP_EPOCHS = 0

# Optional: training-only negative downsampling (keep all positives)
APPLY_NEGATIVE_DOWNSAMPLING = True if FAST_RUN else False
NEG_TO_POS_RATIO = 5  # keep at most 5x negatives per positive

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# ============================================================
# Training & Evaluation
# ============================================================
def train_epoch(use_amp_now: bool, use_rdrop_now: bool, max_steps: int | None):
    global AMP_WAS_DISABLED_THIS_EPOCH

    model.train()
    optimizer.zero_grad()

    losses = []
    skips = 0
    warn_printed = 0
    accum_count = 0
    early_bail = False

    tbar = tqdm(train_loader, desc="Training", leave=False)
    for step, batch in enumerate(tbar, 1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["target"].to(device)
        aux_features = batch.get("aux_features", None)
        if aux_features is not None:
            aux_features = aux_features.to(device)

        with autocast_context(device, use_amp_now, AMP_DTYPE):
            outputs = model(input_ids, attention_mask, aux_features=aux_features)

        # ---- numerical safety on logits ----
        outputs = torch.nan_to_num(outputs, nan=0.0, posinf=1e4, neginf=-1e4).clamp_(-LOGIT_CLAMP, LOGIT_CLAMP)

        # compute loss in fp32
        if TASK == "regression":
            loss_val = criterion(outputs.float(), targets.float()) / GRAD_ACCUM_STEPS
        else:
            if use_rdrop_now and model.training and LOSS_KIND == "ce":
                with autocast_context(device, use_amp_now, AMP_DTYPE):
                    outputs2 = model(input_ids, attention_mask, aux_features=aux_features)
                outputs2 = torch.nan_to_num(outputs2, nan=0.0, posinf=1e4, neginf=-1e4).clamp_(-LOGIT_CLAMP, LOGIT_CLAMP)
                loss_val = rdrop_ce_loss(outputs, outputs2, targets, criterion, alpha=R_DROP_ALPHA) / GRAD_ACCUM_STEPS
            else:
                loss_val = criterion(outputs.float(), targets.long()) / GRAD_ACCUM_STEPS

        if not torch.isfinite(loss_val):
            skips += 1
            with torch.no_grad():
                lens = attention_mask.sum(dim=1).float()
                logger.warning("[TRAIN] Non-finite loss. lens[min/mean/max]=%.0f/%.1f/%.0f",
                               lens.min().item(), lens.mean().item(), lens.max().item())
                if aux_features is not None:
                    a = aux_features.detach()
                    logger.warning("[TRAIN] AUX stats per-col: min=%s max=%s mean=%s std=%s",
                                   a.min(0).values.tolist(), a.max(0).values.tolist(),
                                   a.mean(0).tolist(), a.std(0).tolist())

            if warn_printed < MAX_WARN_PRINT or step % WARN_PRINT_EVERY == 0:
                maybe_print_batch_stats(logger, outputs, targets, "[TRAIN]")
                warn_printed += 1

            if DISABLE_AMP_ON_NAN and use_amp_now:
                logger.warning("[TRAIN] Disabling AMP for the rest of this epoch.")
                AMP_WAS_DISABLED_THIS_EPOCH = True
            for g in optimizer.param_groups:
                g["lr"] = max(g["lr"] * LR_REDUCE_FACTOR, LR_FLOOR)

            optimizer.zero_grad(set_to_none=True)
            if (skips / max(1, step)) > CATASTROPHIC_SKIP_FRAC:
                logger.warning("[TRAIN] >50%% batches skipped so far. Early-bailing epoch.")
                early_bail = True
                break
            continue

        if scaler_enabled and use_amp_now:
            scaler.scale(loss_val).backward()
        else:
            loss_val.backward()
        accum_count += 1

        if (accum_count % GRAD_ACCUM_STEPS == 0) or (step == len(train_loader)):
            if scaler_enabled and use_amp_now:
                scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if USE_EMA:
                ema.update(model)
            scheduler.step()
            if scaler_enabled and use_amp_now:
                scaler.update()
            optimizer.zero_grad()

        losses.append(loss_val.item() * GRAD_ACCUM_STEPS)

        # ----- cap train steps (virtual epoch) -----
        if max_steps is not None and step >= max_steps:
            break

    skip_frac = skips / max(1, len(train_loader))
    if skip_frac > MAX_SKIP_FRAC_DISABLE_AMP and DISABLE_AMP_ON_NAN and use_amp_now:
        logger.warning(f"[TRAIN] Skips {skips}/{len(train_loader)} ({skip_frac:.2%}) > threshold. Disabling AMP next.")
        AMP_WAS_DISABLED_THIS_EPOCH = True

    avg_loss = float(np.mean(losses)) if losses else float("nan")
    return avg_loss, skips, early_bail


def _eval_with_model(eval_model, use_amp_now: bool, return_probs_labels=False, max_steps: int | None = None):
    """Internal eval used for both raw and EMA weights."""
    losses, skips = [], 0
    warn_printed = 0

    all_probs = []
    all_labels = []

    with torch.no_grad():
        vbar = tqdm(val_loader, desc="Validation", leave=False)
        for step, batch in enumerate(vbar, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["target"].to(device)
            aux_features = batch.get("aux_features", None)
            if aux_features is not None:
                aux_features = aux_features.to(device)

            with autocast_context(device, use_amp_now, AMP_DTYPE):
                outputs = eval_model(input_ids, attention_mask, aux_features=aux_features)

            outputs = torch.nan_to_num(outputs, nan=0.0, posinf=1e4, neginf=-1e4).clamp_(-LOGIT_CLAMP, LOGIT_CLAMP)

            if TASK == "regression":
                loss_val = criterion(outputs.float(), targets.float())
            else:
                loss_val = criterion(outputs.float(), targets.long())

            if not torch.isfinite(loss_val):
                skips += 1
                if warn_printed < MAX_WARN_PRINT or step % WARN_PRINT_EVERY == 0:
                    logger.warning("[VAL] Non-finite val loss encountered. Skipping batch.")
                    maybe_print_batch_stats(logger, outputs, targets, "[VAL]")
                    warn_printed += 1
                continue

            losses.append(loss_val.item())

            if TASK == "binary":
                probs = torch.softmax(outputs.float(), dim=1)[:, 1]
                all_probs.append(probs.detach().cpu().numpy())
                all_labels.append(targets.detach().cpu().numpy())

            # ----- cap validation steps -----
            if max_steps is not None and step >= max_steps:
                break

    avg_loss = float(np.mean(losses)) if losses else float("inf")
    metrics = {}
    if TASK == "binary" and len(all_probs) > 0:
        probs = np.concatenate(all_probs)
        labels = np.concatenate(all_labels).astype(int)
        try:
            auroc = roc_auc_score(labels, probs)
        except ValueError:
            auroc = float("nan")
        try:
            auprc = average_precision_score(labels, probs)
        except ValueError:
            auprc = float("nan")
        best_f1, best_thr = 0.0, 0.5
        for thr in np.linspace(0.05, 0.95, 19):
            preds = (probs >= thr).astype(int)
            f1 = f1_score(labels, preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_thr = f1, thr
        metrics = {"auroc": auroc, "auprc": auprc, "best_f1": best_f1, "best_thr": best_thr}
        if return_probs_labels:
            return avg_loss, skips, metrics, probs, labels
    return (avg_loss, skips, metrics, None, None) if return_probs_labels else (avg_loss, skips, metrics)


def eval_epoch(use_amp_now: bool, max_steps: int | None):
    """Evaluate using EMA weights if enabled; fall back to raw."""
    if USE_EMA:
        ema.apply_shadow(model)
        va_loss, va_skips, metrics, probs, labels = _eval_with_model(model, use_amp_now, return_probs_labels=True, max_steps=max_steps)
        ema.restore(model)
    else:
        va_loss, va_skips, metrics, probs, labels = _eval_with_model(model, use_amp_now, return_probs_labels=True, max_steps=max_steps)

    if TASK == "binary" and metrics:
        if probs is not None:
            thr_p, p, r, f1p = pick_precision_threshold(
                probs, labels, target_prec=TARGET_PRECISION, min_recall=MIN_RECALL
            )
            metrics["prec_thr"] = float(thr_p)
            metrics["prec_at_thr"] = float(p)
            metrics["rec_at_thr"] = float(r)
            metrics["f1_at_thr"] = float(f1p)
            logger.info(
                f"[VAL] AUROC={metrics['auroc']:.4f} | AUPRC={metrics['auprc']:.4f} | "
                f"Best-F1={metrics['best_f1']:.4f} @ {metrics['best_thr']:.2f} | "
                f"Prec-Opt thr={metrics['prec_thr']:.2f} -> Prec={p:.4f} Rec={r:.4f} F1={f1p:.4f}"
            )
        else:
            logger.info(
                f"[VAL] AUROC={metrics['auroc']:.4f} | AUPRC={metrics['auprc']:.4f} | "
                f"Best-F1={metrics['best_f1']:.4f} @ thr={metrics['best_thr']:.2f}"
            )
    return va_loss, va_skips, metrics

# ============================================================
# Data / setup
# ============================================================
label_column = "regression_score" if TASK == "regression" else "binary_label"
data_path = f"../datasets/preprocessed/{label_column}_train.csv"
df = pd.read_csv(data_path)

# Track best model across folds
global_best_val_score = -float("inf") if TASK == "binary" else float("inf")
global_best_model_path = ("../outputs/checkpoints/bert_model_bin_best.pth"
                          if TASK == "binary" else
                          "../outputs/checkpoints/bert_model_reg_best.pth")

logger.info("\n[CHECK] Target column diagnostics (pre-normalization)")
logger.info(df[label_column].describe().to_string())
logger.info(f"NaNs in target: {int(df[label_column].isna().sum())}")
logger.info(f"Infs in target: {int(np.isinf(df[label_column].values).sum())}")
logger.info(f"Min target: {float(df[label_column].min())}")
logger.info(f"Max target: {float(df[label_column].max())}")

aux_cols = ["review_length", "sentiment_score", "is_verified"]
missing_aux = [c for c in aux_cols if c not in df.columns]
if missing_aux:
    raise ValueError(f"Missing required aux columns: {missing_aux}")

def _count_bad(x): return int((~np.isfinite(x)).sum())

logger.info("\n[CHECK] Aux diagnostics (raw)")
for c in aux_cols:
    arr = df[c].values
    logger.info(f"  {c}: NaN/Inf count = {_count_bad(arr)} | min={np.nanmin(arr)} | max={np.nanmax(arr)}")

# Clean NaNs/Inf early
cols_to_check = aux_cols + [label_column, "full_text"]
before = len(df)
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=cols_to_check)
after = len(df)
if after < before:
    logger.info(f"[INFO] Dropped {before - after} rows with NaN/Inf in aux/target/text.")

# Normalize regression target
y_mean, y_std = None, None
if TASK == "regression":
    y_mean = df[label_column].mean()
    y_std = df[label_column].std()
    if y_std < 1e-12:
        logger.warning("[WARN] Target std is ~0; forcing std=1.0.")
        y_std = 1.0
    df[label_column] = (df[label_column] - y_mean) / (y_std + 1e-12)
    logger.info(f"[INFO] Normalized regression target: mean={y_mean:.4f}, std={y_std:.4f}")
    logger.info("\n[CHECK] Target diagnostics (post-normalization)")
    vals = df[label_column].values
    logger.info(pd.Series(vals).describe().to_string())
    logger.info(f"NaNs in normalized: {int(np.isnan(vals).sum())}")
    logger.info(f"Infs in normalized: {int(np.isinf(vals).sum())}")
    logger.info(f"Var(normalized): {float(np.var(vals)):.6f}")

# Split
val_scores = []
if N_SPLITS > 1:
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    folds = list(kf.split(df))
else:
    idx = np.arange(len(df)); np.random.shuffle(idx)
    cut = int(0.8 * len(df))
    folds = [(idx[:cut], idx[cut:])]

# ============================================================
# Train
# ============================================================
for fold, (train_index, val_index) in enumerate(folds, 1):
    logger.info(f"\n=== Fold {fold}/{len(folds)} ===")
    train_df = df.iloc[train_index].reset_index(drop=True)
    val_df   = df.iloc[val_index].reset_index(drop=True)

    # Optional: training-only negative downsampling
    if TASK == "binary" and APPLY_NEGATIVE_DOWNSAMPLING:
        pos_df = train_df[train_df["binary_label"] == 1]
        neg_df = train_df[train_df["binary_label"] == 0]
        n_pos = len(pos_df)
        n_neg_keep = min(len(neg_df), NEG_TO_POS_RATIO * max(1, n_pos))
        neg_sampled = neg_df.sample(n=n_neg_keep, random_state=42) if n_neg_keep < len(neg_df) else neg_df
        before_n = len(train_df)
        train_df = pd.concat([pos_df, neg_sampled], axis=0).sample(frac=1.0, random_state=42).reset_index(drop=True)
        logger.info(f"[SPEED] Downsampled train negatives: {before_n} -> {len(train_df)} rows "
                    f"(pos={len(pos_df)}, neg_kept={len(neg_sampled)})")

    # Sanity on variance
    if TASK == "regression":
        tv = float(np.var(train_df[label_column].values))
        vv = float(np.var(val_df[label_column].values))
        if tv < 1e-10: logger.warning("[WARN] Train split target variance is ~0.")
        if vv < 1e-10: logger.warning("[WARN] Val split target variance is ~0.")

    train_dataset = ReviewDataset(train_df, max_len=MAX_LEN, task=TASK, use_aux_features=True)
    val_dataset   = ReviewDataset(val_df,   max_len=MAX_LEN, task=TASK, use_aux_features=True)

    # Aux standardization (from TRAIN split)
    aux_means, aux_stds = {}, {}
    for c in aux_cols:
        v = train_df[c].to_numpy(np.float32)
        if c == "review_length":
            v = np.log1p(v)
        aux_means[c] = float(np.mean(v)); aux_stds[c] = float(np.std(v) + 1e-12)
    train_dataset.aux_means = aux_means; train_dataset.aux_stds = aux_stds
    val_dataset.aux_means   = aux_means; val_dataset.aux_stds   = aux_stds

    # ---- Dataloaders (Windows-safe settings) ----
    DL_KW = dict(batch_size=BATCH_SIZE, num_workers=0, pin_memory=False, persistent_workers=False, drop_last=True)

    if TASK == "binary" and USE_WEIGHTED_SAMPLER and (not USE_CLASS_WEIGHTS):
        y = train_df["binary_label"].to_numpy()
        pos = int((y == 1).sum()); neg = int((y == 0).sum())
        w_pos = neg / max(1, pos) if pos > 0 else 1.0
        w_neg = 1.0
        weights = np.where(y == 1, w_pos, w_neg).astype(np.float32)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(train_dataset, sampler=sampler, shuffle=False, **DL_KW)
    else:
        train_loader = DataLoader(train_dataset, shuffle=True, **DL_KW)

    val_loader = DataLoader(val_dataset, shuffle=False, **DL_KW)

    # Model & criterion
    if TASK == "regression":
        model = BERTRegressor(dropout_rate=DROPOUT_RATE, use_mean_pooling=USE_MEAN_POOLING, aux_feat_dim=AUX_FEAT_DIM)
        criterion = nn.SmoothL1Loss() if LOSS_FN == "smooth" else nn.MSELoss()
        model_path = f"../outputs/checkpoints/bert_model_reg_fold{fold}.pth"
        loss_label = "SmoothL1 Loss" if LOSS_FN == "smooth" else "MSE Loss"
        last_path  = f"../outputs/checkpoints/bert_model_reg_fold{fold}_last.pth"
    else:
        model = BERTClassifier(dropout_rate=DROPOUT_RATE, use_mean_pooling=USE_MEAN_POOLING, aux_feat_dim=AUX_FEAT_DIM)

        if USE_CLASS_WEIGHTS:
            y_train = train_df["binary_label"].to_numpy()
            pos = int((y_train == 1).sum()); neg = int((y_train == 0).sum())
            if CLASS_WEIGHT_POS_OVERRIDE is not None:
                w_pos = float(CLASS_WEIGHT_POS_OVERRIDE); w_neg = 1.0
            else:
                w_pos = neg / max(1, pos) if pos > 0 else 1.0
                w_neg = 1.0
            class_weights = torch.tensor([w_neg, w_pos], dtype=torch.float, device=device)
            logger.info(f"[INFO] Class weights: w_neg={w_neg:.3f}, w_pos={w_pos:.3f}")
        else:
            class_weights = None

        if LOSS_KIND == "afl":
            criterion = AsymmetricFocalLoss(alpha_pos=AF_ALPHA_POS,
                                            gamma_pos=AF_GAMMA_POS,
                                            gamma_neg=AF_GAMMA_NEG)
            loss_label = "AsymFocalLoss"
        else:
            criterion = nn.CrossEntropyLoss(
                weight=(class_weights.to(torch.float32) if class_weights is not None else None),
                label_smoothing=CE_LABEL_SMOOTH
            )
            loss_label = "CrossEntropy Loss"

        model_path = f"../outputs/checkpoints/bert_model_bin_fold{fold}.pth"
        last_path  = f"../outputs/checkpoints/bert_model_bin_fold{fold}_last.pth"

    model = model.to(device)

    # EMA
    ema = EMA(model, decay=EMA_DECAY) if USE_EMA else None

    # Initial freeze depth
    set_freeze_depth(model, UNFREEZE_SCHEDULE.get(0, FREEZE_LAYERS_INIT))
    model._last_freeze_depth = UNFREEZE_SCHEDULE.get(0, FREEZE_LAYERS_INIT)

    # Optimizer & Scheduler (based on EFFECTIVE steps per epoch)
    param_groups = build_param_groups(
        model, base_lr=BASE_LR, head_lr=HEAD_LR,
        weight_decay=WEIGHT_DECAY, head_weight_decay=HEAD_WEIGHT_DECAY
    )
    optimizer = AdamW(param_groups, eps=1e-8, betas=(0.9, 0.999))

    # Compute effective optimizer steps for the scheduler
    steps_per_epoch_cap = TRAIN_MAX_STEPS_PER_EPOCH if TRAIN_MAX_STEPS_PER_EPOCH is not None else len(train_loader)
    eff_steps_per_epoch = min(len(train_loader), steps_per_epoch_cap)
    optimizer_steps_per_epoch = math.ceil(eff_steps_per_epoch / max(1, GRAD_ACCUM_STEPS))
    total_optimizer_steps = max(1, optimizer_steps_per_epoch * EPOCHS)
    num_warmup_steps = int(WARMUP_PCT * total_optimizer_steps)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_optimizer_steps
    )

    # Modern GradScaler (with fallback for older PyTorch)
    scaler_enabled = (device.type == "cuda")
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=scaler_enabled)
    except Exception:
        from torch.cuda.amp import GradScaler as CudaGradScaler
        scaler = CudaGradScaler(enabled=scaler_enabled)

    train_losses, val_losses = [], []
    if TASK == "binary":
        best_val_score = -float("inf")
    else:
        best_val_score = float("inf")
    patience_counter = 0
    lr_drop_on_unfreeze_done = False

    # Task-specific metadata for checkpoint
    def _checkpoint_payload():
        payload = {
            "model_state_dict": model.state_dict(),
            "aux_means": aux_means,
            "aux_stds": aux_stds,
        }
        if TASK == "binary" and SAVE_PRECISION_OPT_THR:
            if 'decision_threshold' in locals():
                payload["decision_threshold"] = float(decision_threshold)
            elif 'va_metrics' in locals() and va_metrics is not None and "prec_thr" in va_metrics:
                payload["decision_threshold"] = float(va_metrics["prec_thr"])
        if TASK == "regression":
            payload["y_mean"] = y_mean
            payload["y_std"]  = y_std
        return payload

    for epoch in range(EPOCHS):
        # AMP gating for this epoch
        use_amp_now = (not force_fp32_next_epoch) and (epoch >= AMP_WARMUP_EPOCHS) and USE_AMP and (not AMP_WAS_DISABLED_THIS_EPOCH)
        use_rdrop_now = USE_R_DROP and (LOSS_KIND == "ce") and not (epoch == 0 and False)  # R-Drop disabled if USE_R_DROP=False

        if REENABLE_AMP_AFTER_EPOCH and AMP_WAS_DISABLED_THIS_EPOCH and not force_fp32_next_epoch:
            logger.info("[ACTION] Re-enabling AMP for this epoch.")
            AMP_WAS_DISABLED_THIS_EPOCH = False
            use_amp_now = (epoch >= AMP_WARMUP_EPOCHS) and USE_AMP

        # Staged unfreezing
        freeze_depth = None
        for k in sorted(UNFREEZE_SCHEDULE.keys()):
            if epoch >= k:
                freeze_depth = UNFREEZE_SCHEDULE[k]
        if freeze_depth is not None:
            prev_depth = getattr(model, "_last_freeze_depth", None)
            set_freeze_depth(model, freeze_depth)
            model._last_freeze_depth = freeze_depth
            if USE_EMA and ema is not None:
                ema.refresh_from_model(model)
            if prev_depth is not None and prev_depth > 0 and freeze_depth == 0 and not lr_drop_on_unfreeze_done:
                logger.info("[ACTION] Unfroze all layers -> reducing LR x0.5 for stability.")
                for g in optimizer.param_groups:
                    g["lr"] = max(g["lr"] * LR_REDUCE_FACTOR, LR_FLOOR)
                lr_drop_on_unfreeze_done = True

        lr_str = ", ".join([f"{g['lr']:.2e}" for g in optimizer.param_groups])
        logger.info(f"Epoch {epoch + 1}/{EPOCHS} | freeze_depth={freeze_depth} | AMP={'on' if use_amp_now else 'off'} | LRs=[{lr_str}]")

        if epoch == 0 and KICKSTART_HEAD_LR and len(optimizer.param_groups) > 1:
            optimizer.param_groups[1]["lr"] = max(optimizer.param_groups[1]["lr"], HEAD_KICK_LR)
            logger.info(f"[NOTE] Kickstarted head LR to {optimizer.param_groups[1]['lr']:.2e} for epoch 1 warmup.")
        if epoch == 0 and TASK == "binary":
            logger.info("[NOTE] Elevated CE in epoch 1 is expected with class weights + smoothing + deep freeze + warmup.")

        tr_loss, tr_skips, early_bail = train_epoch(
            use_amp_now=use_amp_now,
            use_rdrop_now=use_rdrop_now,
            max_steps=TRAIN_MAX_STEPS_PER_EPOCH
        )
        va_loss, va_skips, va_metrics = eval_epoch(
            use_amp_now=use_amp_now,
            max_steps=VAL_MAX_STEPS
        )

        decision_threshold = None
        if TASK == "binary" and va_metrics is not None and "prec_thr" in va_metrics:
            decision_threshold = float(va_metrics["prec_thr"])

        train_losses.append(tr_loss); val_losses.append(va_loss)

        tr_skip_pct = (tr_skips / max(1, len(train_loader))) * 100.0
        va_skip_pct = (va_skips / max(1, len(val_loader))) * 100.0

        logger.info(f"[EPOCH {epoch+1}] Train Loss: {tr_loss:.4f} | Val Loss: {va_loss:.4f} | "
                    f"Train skipped: {tr_skips} ({tr_skip_pct:.2f}%) | "
                    f"Val skipped: {va_skips} ({va_skip_pct:.2f}%)")

        if TASK == "binary":
            current_score = va_metrics.get("auroc", float("nan")) if va_metrics else float("nan")
            is_better = np.isfinite(current_score) and (current_score > best_val_score + 1e-6)
            toxic_val = (not np.isfinite(va_loss)) or (va_skips == len(val_loader))
        else:
            current_score = -va_loss
            is_better = np.isfinite(va_loss) and (-va_loss > best_val_score + 1e-6)
            toxic_val = (not np.isfinite(va_loss)) or (va_skips == len(val_loader))

        toxic_train = early_bail or (tr_skips / max(1, len(train_loader)) > CATASTROPHIC_SKIP_FRAC)

        if toxic_val or toxic_train:
            logger.warning("[TOXIC] Epoch unstable: reducing LR and forcing fp32 next epoch.")
            for g in optimizer.param_groups:
                g["lr"] = max(g["lr"] * LR_REDUCE_FACTOR, LR_FLOOR)
            force_fp32_next_epoch = True
            AMP_WAS_DISABLED_THIS_EPOCH = True

            # Try rolling back to last *saved* good checkpoint if it exists
            if os.path.exists(model_path):
                try:
                    logger.warning("[TOXIC] Rolling back to last fold-best checkpoint.")
                    best = torch.load(model_path, map_location=device, weights_only=True)
                    model.load_state_dict(best["model_state_dict"])
                    if USE_EMA and ema is not None:
                        ema.refresh_from_model(model)
                except Exception as e:
                    logger.warning(f"[TOXIC] Rollback failed: {e}")
        else:
            if is_better:
                best_val_score = current_score
                torch.save(_checkpoint_payload(), model_path)
                patience_counter = 0
                logger.info("[CHECKPOINT] Saved new best model.")
                if (TASK == "binary" and best_val_score > global_best_val_score + 1e-6) or \
                   (TASK == "regression" and -best_val_score < global_best_val_score - 1e-6):
                    global_best_val_score = best_val_score
                    torch.save(_checkpoint_payload(), global_best_model_path)
                    logger.info(f"[CHECKPOINT] New global best model saved to {global_best_model_path}")
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    logger.info("Early stopping.")
                    break

        # Always save a lightweight 'last' checkpoint for this fold at end of epoch
        try:
            torch.save(_checkpoint_payload(), last_path)
        except Exception as e:
            logger.warning(f"[WARN] Could not save last-epoch checkpoint: {e}")

    val_scores.append(best_val_score if TASK == "binary" else (-best_val_score))

    # Curves
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")
    plt.title(f"Fold {fold} Loss Curve")
    plt.xlabel("Epoch"); plt.ylabel(loss_label)
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(f"../outputs/plots/bert_loss_curve_fold{fold}.png")
    plt.close()

# Summary
if TASK == "binary":
    logger.info(f"\nBest AUROC across folds: {np.max(val_scores):.4f} ± {np.std(val_scores):.4f}")
else:
    logger.info(f"\nAvg Val Loss: {np.mean(val_scores):.4f} ± {np.std(val_scores):.4f}")
