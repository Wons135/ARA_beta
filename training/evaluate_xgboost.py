# evaluate_xgboost.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse as sp
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    roc_auc_score, average_precision_score,
    mean_squared_error, mean_absolute_error, r2_score
)

# ---------- CONFIG ----------
# Set to "binary" or "regression" to match how the model was trained
TASK = "regression"   # "binary" | "regression"

BASE_DIR   = "../datasets/preprocessed"
OUT_DIR    = "../outputs"
MODEL_PATH = os.path.join(OUT_DIR, "checkpoints",
                          f"xgb_precomputed_{'binary' if TASK=='binary' else 'regression'}.json")

task_suffix = "binary" if TASK == "binary" else "regression"
X_VAL_NPZ  = os.path.join(BASE_DIR, f"X_val_features_{task_suffix}.npz")
Y_VAL_CSV  = os.path.join(BASE_DIR, f"y_val_{task_suffix}.csv")

PLOT_DIR   = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# ---------- Load data ----------
print("Loading features/labels...")
X_val = sp.load_npz(X_VAL_NPZ)                # shape: (n_val, d)
y_val = pd.read_csv(Y_VAL_CSV).squeeze()      # Series/array

# Ensure 1D numpy arrays for metrics
if hasattr(y_val, "values"):
    y_val = y_val.values
y_val = y_val.ravel()

print(f"X_val: {X_val.shape} | y_val: {y_val.shape}")

# ---------- Load model ----------
booster = xgb.Booster()
booster.load_model(MODEL_PATH)

# ---------- Predict ----------
# Build DMatrix from sparse matrix (no feature names → matches training)
dval = xgb.DMatrix(X_val)

if TASK == "binary":
    # For 'binary:logistic' objective, Booster.predict returns probabilities
    probs = booster.predict(dval, validate_features=False)
    preds = (probs >= 0.5).astype(int)

    # Metrics
    acc  = accuracy_score(y_val, preds)
    prec = precision_score(y_val, preds, zero_division=0)
    rec  = recall_score(y_val, preds,    zero_division=0)
    f1   = f1_score(y_val, preds,        zero_division=0)
    try:
        auroc = roc_auc_score(y_val, probs)
    except Exception:
        auroc = float("nan")
    try:
        auprc = average_precision_score(y_val, probs)
    except Exception:
        auprc = float("nan")

    print(f"[VAL] AUROC={auroc:.4f} | AUPRC={auprc:.4f}")
    print(f"[VAL @0.5] Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_val, preds)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title("XGBoost (binary) Confusion Matrix")
    plt.tight_layout()
    cm_path = os.path.join(PLOT_DIR, "xgb_confusion_matrix_binary.png")
    plt.savefig(cm_path); plt.close()
    print(f"Saved confusion matrix -> {cm_path}")

else:
    # Regression
    y_pred = booster.predict(dval, validate_features=False)

    rmse = mean_squared_error(y_val, y_pred, squared=False)
    mae  = mean_absolute_error(y_val, y_pred)
    r2   = r2_score(y_val, y_pred)

    print(f"[VAL] RMSE={rmse:.4f} | MAE={mae:.4f} | R²={r2:.4f}")

    plt.figure(figsize=(6,6))
    plt.scatter(y_val, y_pred, alpha=0.25, s=4)
    plt.xlabel("True"); plt.ylabel("Predicted")
    plt.title("XGBoost Regression: Predicted vs True")
    plt.grid(True, alpha=0.3)
    pv_path = os.path.join(PLOT_DIR, "xgb_scatter_regression.png")
    plt.tight_layout(); plt.savefig(pv_path); plt.close()
    print(f"Saved scatter plot -> {pv_path}")
