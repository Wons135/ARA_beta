import os
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

from models.bert_model import BERTRegressor, BERTClassifier
from src.dataset import ReviewDataset

# ---------- CONFIG ----------
MODEL_TYPE = "binary"  # "regression" or "binary"
# Use the *best* checkpoints you saved during training:
CHECKPOINT_PATH = "../outputs/checkpoints/bert_model_bin_best.pth"  # or bert_model_bin_best.pth
TEST_CSV = "../datasets/preprocessed/binary_label_test.csv"         # or regression_score_test.csv
TOKENIZER_NAME = "distilbert-base-uncased"

MAX_LEN = 256
BATCH_SIZE = 64
USE_AUX = True
AUX_COLUMNS = ["review_length", "sentiment_score", "is_verified"]  # must match training
SAVE_PREFIX = "reg" if MODEL_TYPE == "regression" else "bin"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

os.makedirs("../outputs/plots", exist_ok=True)
os.makedirs("../outputs/predictions", exist_ok=True)

# ---------- Dataset with AUX standardization identical to training ----------
class ReviewDatasetEval(Dataset):
    """
    Wraps ReviewDataset and applies the same AUX standardization used in training:
      - review_length: log1p, then (x - mean) / std (means/stds preferably loaded from checkpoint)
      - sentiment_score, is_verified: (x - mean) / std
    Also supports normalizing regression targets using y_mean/y_std from checkpoint so that
    loss in eval is comparable to training loss.
    """
    def __init__(self, dataframe_or_path, tokenizer_name, max_len, task,
                 use_aux=True, aux_columns=None, aux_means=None, aux_stds=None,
                 y_mean=None, y_std=None):
        self.task = task
        self.use_aux = use_aux
        self.aux_columns = aux_columns or []
        self.aux_means = aux_means or {}
        self.aux_stds = aux_stds or {}

        self.base = ReviewDataset(
            dataframe_or_path, tokenizer_name=tokenizer_name, max_len=max_len, task=task
        )
        self.df = pd.read_csv(dataframe_or_path) if isinstance(dataframe_or_path, str) else dataframe_or_path.copy()

        # Aux: build in the same order as AUX_COLUMNS
        self.aux = None
        if self.use_aux and self.aux_columns:
            for col in self.aux_columns:
                if col not in self.df.columns:
                    raise ValueError(f"Aux column '{col}' not found in dataset.")

            aux_parts = []
            for col in self.aux_columns:
                v = self.df[col].to_numpy(np.float32)
                if col == "review_length":
                    v = np.log1p(v)
                mean = float(self.aux_means.get(col, np.mean(v)))
                std  = float(self.aux_stds.get(col,  np.std(v) + 1e-12))
                v = (v - mean) / (std + 1e-12)
                aux_parts.append(v.astype(np.float32))
            self.aux = np.stack(aux_parts, axis=1).astype(np.float32)

        # Regression target normalization for loss comparability
        self.y_mean = y_mean
        self.y_std = y_std

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        if self.use_aux and self.aux is not None:
            item["aux_features"] = torch.from_numpy(self.aux[idx])
        # If regression and we have normalization stats, transform target to normalized space
        if self.task == "regression" and (self.y_mean is not None) and (self.y_std is not None):
            t = item["target"].float()
            t = (t - float(self.y_mean)) / (float(self.y_std) + 1e-12)
            item["target"] = t
        return item

# ---------- Load checkpoint (safe) ----------
try:
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
except TypeError:
    # For older PyTorch versions that don't support weights_only:
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")

# Extract AUX stats & (for regression) label normalization
aux_means = ckpt.get("aux_means", None) if isinstance(ckpt, dict) else None
aux_stds  = ckpt.get("aux_stds",  None) if isinstance(ckpt, dict) else None
y_mean    = ckpt.get("y_mean", None)     if isinstance(ckpt, dict) else None
y_std     = ckpt.get("y_std",  None)     if isinstance(ckpt, dict) else None

if USE_AUX and (aux_means is None or aux_stds is None):
    print("[WARN] aux_means/aux_stds not found in checkpoint; standardizing AUX from test set statistics.")
else:
    print("[INFO] Using aux_means/aux_stds from checkpoint for standardization.")

# ---------- Build dataset/loader ----------
test_dataset = ReviewDatasetEval(
    TEST_CSV, tokenizer_name=TOKENIZER_NAME, max_len=MAX_LEN, task=MODEL_TYPE,
    use_aux=USE_AUX, aux_columns=AUX_COLUMNS,
    aux_means=(aux_means or {}), aux_stds=(aux_stds or {}),
    y_mean=y_mean, y_std=y_std,
)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, pin_memory=True)

# ---------- Load model ----------
if MODEL_TYPE == "regression":
    model = BERTRegressor(
        model_name=TOKENIZER_NAME, dropout_rate=0.5, use_mean_pooling=True,
        aux_feat_dim=(len(AUX_COLUMNS) if USE_AUX else 0)
    )
    criterion = nn.MSELoss()
else:
    model = BERTClassifier(
        model_name=TOKENIZER_NAME, dropout_rate=0.5, use_mean_pooling=True,
        aux_feat_dim=(len(AUX_COLUMNS) if USE_AUX else 0)
    )
    # At eval we do *not* pass class-weights/label smoothing; they were training-time tricks.
    criterion = nn.CrossEntropyLoss()

if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
else:
    missing, unexpected = model.load_state_dict(ckpt, strict=False)

if unexpected or missing:
    print("Warning: state_dict mismatches ->")
    if unexpected: print(f"  Unexpected keys: {unexpected}")
    if missing:    print(f"  Missing keys: {missing}")

model = model.to(device).eval()

# ---------- Eval ----------
@torch.no_grad()
def eval_epoch(model, loader, criterion, task, use_aux):
    losses, all_logits, all_targets = [], [], []
    pbar = tqdm(loader, desc="Testing", leave=False)
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["target"].to(device)
        aux = batch.get("aux_features", None)
        if use_aux and aux is not None:
            aux = aux.to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, aux_features=aux)
        loss = criterion(outputs.float(), targets if task == "regression" else targets.long())
        losses.append(loss.item())

        all_logits.append(outputs.detach().cpu().float().numpy())
        all_targets.append(targets.detach().cpu().float().numpy())
        pbar.set_postfix(loss=loss.item())

    logits = np.concatenate(all_logits, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    return float(np.mean(losses)), logits, targets

test_loss, test_logits, test_targets = eval_epoch(model, test_loader, criterion, MODEL_TYPE, USE_AUX)

# ---------- Metrics & Plots ----------
if MODEL_TYPE == "regression":
    preds_norm = test_logits.squeeze(-1)
    true_norm  = test_targets.squeeze(-1)

    if (y_mean is not None) and (y_std is not None):
        y_log_pred = preds_norm * y_std + y_mean
        y_log_true = true_norm  * y_std + y_mean
        print(f"[INFO] Inverse normalized to log space using mean={float(y_mean):.4f}, std={float(y_std):.4f}")
    else:
        y_log_pred = preds_norm.copy()
        y_log_true = true_norm.copy()
        print("[WARN] y_mean/y_std not in checkpoint. Assuming predictions/targets are already in log space.")

    mse_log = mean_squared_error(y_log_true, y_log_pred)
    rmse_log = np.sqrt(mse_log)
    ybar_log = np.full_like(y_log_true, fill_value=y_log_true.mean())
    mse_base_log = mean_squared_error(y_log_true, ybar_log)
    r2_log = 1.0 - (mse_log / mse_base_log if mse_base_log > 0 else np.nan)
    print(f"[LOG SPACE] RMSE: {rmse_log:.4f} | R²: {r2_log:.4f} | MSE (eval loss proxy): {test_loss:.4f}")

    y_true = np.expm1(y_log_true)
    y_pred = np.expm1(y_log_pred)
    rmse_orig = np.sqrt(mean_squared_error(y_true, y_pred))
    mae_orig = mean_absolute_error(y_true, y_pred)

    sigma2_log = mse_log
    y_pred_bc = np.expm1(y_log_pred + 0.5 * sigma2_log)
    rmse_orig_bc = np.sqrt(mean_squared_error(y_true, y_pred_bc))
    mae_orig_bc = mean_absolute_error(y_true, y_pred_bc)

    print(f"[ORIGINAL]            RMSE: {rmse_orig:.4f} | MAE: {mae_orig:.4f}")
    print(f"[ORIGINAL + biascorr] RMSE: {rmse_orig_bc:.4f} | MAE: {mae_orig_bc:.4f}")

    plt.figure(figsize=(6, 6))
    plt.scatter(y_log_true, y_log_pred, alpha=0.3)
    plt.xlabel("True (log)"); plt.ylabel("Predicted (log)")
    plt.title("Predicted vs True (log)"); plt.grid(True); plt.tight_layout()
    plt.savefig(f"../outputs/plots/bert_{SAVE_PREFIX}_test_scatter_log.png"); plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist((y_log_pred - y_log_true), bins=60, edgecolor="black")
    plt.xlabel("Prediction Error (log)"); plt.ylabel("Frequency")
    plt.title("Prediction Error Distribution (log)")
    plt.grid(True); plt.tight_layout()
    plt.savefig(f"../outputs/plots/bert_{SAVE_PREFIX}_test_error_log.png"); plt.close()

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred_bc, alpha=0.3)
    plt.xlabel("True (orig)"); plt.ylabel("Predicted (bias-corr, orig)")
    plt.title("Predicted vs True (original, bias-corr)")
    plt.grid(True); plt.tight_layout()
    plt.savefig(f"../outputs/plots/bert_{SAVE_PREFIX}_test_scatter_original.png"); plt.close()

    out = pd.read_csv(TEST_CSV)
    out["pred_norm"] = preds_norm
    out["true_norm"] = true_norm
    out["pred_log"] = y_log_pred
    out["true_log"] = y_log_true
    out["pred_original"] = y_pred
    out["pred_original_biascorr"] = y_pred_bc
    out["true_original"] = y_true
    out.to_csv(f"../outputs/predictions/bert_{SAVE_PREFIX}_predictions.csv", index=False)

else:
    logits = test_logits  # shape (N, 2)
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]
    labels = test_targets.astype(int)

    auroc = roc_auc_score(labels, probs)
    auprc = average_precision_score(labels, probs)
    print(f"AUROC: {auroc:.4f} | AUPRC: {auprc:.4f}")

    thrs = np.linspace(0.05, 0.95, 19)
    f1s = []
    for thr in thrs:
        preds = (probs >= thr).astype(int)
        f1s.append(f1_score(labels, preds, zero_division=0))
    best_idx = int(np.argmax(f1s))
    best_thr, best_f1 = float(thrs[best_idx]), float(f1s[best_idx])

    for tag, thr in [("default@0.5", 0.5), (f"best@{best_thr:.2f}", best_thr)]:
        preds = (probs >= thr).astype(int)
        acc = accuracy_score(labels, preds)
        prec = precision_score(labels, preds, zero_division=0)
        rec = recall_score(labels, preds, zero_division=0)
        f1v = f1_score(labels, preds, zero_division=0)
        print(f"[{tag}] Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1v:.4f}")

        cm = confusion_matrix(labels, preds)
        plt.figure(figsize=(6,5))
        plt.imshow(cm, interpolation='nearest')
        plt.title(f"Confusion Matrix ({tag})"); plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, [0,1]); plt.yticks(tick_marks, [0,1])
        fmt = 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], fmt),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")
        plt.ylabel('True'); plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(f"../outputs/plots/bert_{SAVE_PREFIX}_cm_{tag.replace('@','_')}.png")
        plt.close()

    fpr, tpr, _ = roc_curve(labels, probs)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, lw=2)
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC (AUROC={auroc:.4f})")
    plt.grid(True); plt.tight_layout()
    plt.savefig(f"../outputs/plots/bert_{SAVE_PREFIX}_roc.png"); plt.close()

    precs, recs, _ = precision_recall_curve(labels, probs)
    plt.figure(figsize=(6,5))
    plt.plot(recs, precs, lw=2)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR (AUPRC={auprc:.4f})")
    plt.grid(True); plt.tight_layout()
    plt.savefig(f"../outputs/plots/bert_{SAVE_PREFIX}_pr.png"); plt.close()

    out = pd.read_csv(TEST_CSV)
    out["prob_pos"] = probs
    out["true_label"] = labels
    out.to_csv(f"../outputs/predictions/bert_{SAVE_PREFIX}_predictions.csv", index=False)
