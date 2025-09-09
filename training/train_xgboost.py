# training/train_xgboost.py
import os, json, time, logging
import numpy as np
import pandas as pd
from scipy import sparse as sp
import matplotlib.pyplot as plt
from tqdm import tqdm
import xgboost as xgb
from xgboost.callback import TrainingCallback
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, precision_recall_curve
)

# ------------ CONFIG ------------
TASK = "regression"   # "binary" | "regression"
task_suffix = "binary" if TASK == "binary" else "regression"

X_DIR = "../datasets/preprocessed"
OUT_DIR = "../outputs"
os.makedirs(os.path.join(OUT_DIR, "logs"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "checkpoints"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "plots"), exist_ok=True)

LOG_PATH = os.path.join(OUT_DIR, "logs", f"train_xgb_precomputed_{task_suffix}.log")
MODEL_PATH = os.path.join(OUT_DIR, "checkpoints", f"xgb_precomputed_{task_suffix}.json")
THRESHOLD_PATH = os.path.join(OUT_DIR, "checkpoints", f"xgb_precomputed_{task_suffix}_decision_threshold.json")
IMPLOT_PATH = os.path.join(OUT_DIR, "plots", f"xgb_precomputed_{task_suffix}_fi.png")
CURVE_PATH = os.path.join(OUT_DIR, "plots", f"xgb_precomputed_{task_suffix}_curves.png")
HIST_PATH  = os.path.join(OUT_DIR, "logs",  f"xgb_precomputed_{task_suffix}_eval_history.csv")

X_train_npz = os.path.join(X_DIR, f"X_train_features_{task_suffix}.npz")
X_val_npz   = os.path.join(X_DIR, f"X_val_features_{task_suffix}.npz")
y_train_csv = os.path.join(X_DIR, f"y_train_{task_suffix}.csv")
y_val_csv   = os.path.join(X_DIR, f"y_val_{task_suffix}.csv")

# ---- tuned defaults for text ----
XGB_PARAMS = dict(
    n_estimators=2000,            # fewer trees, higher lr
    max_depth=6,
    learning_rate=0.05,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.7,
    tree_method="hist",
    device="cuda",
    predictor="gpu_predictor",
    reg_lambda=2.0,
    reg_alpha=0.1,
    random_state=42,
)
EARLY_STOP = 200
VERBOSE_EVAL = 50

TARGET_PRECISION = 0.30
MIN_RECALL = 0.15
SAVE_PRECISION_OPT_THR = True

# ------------ Logging ------------
logger = logging.getLogger(f"train_xgb_precomputed_{task_suffix}")
logger.setLevel(logging.INFO)
logger.handlers.clear()
ch = logging.StreamHandler(); ch.setLevel(logging.INFO)
fh = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8"); fh.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
ch.setFormatter(fmt); fh.setFormatter(fmt)
logger.addHandler(ch); logger.addHandler(fh)

class TQDMCallback(TrainingCallback):
    def __init__(self, total):
        self.pbar = tqdm(total=total, desc="Boosting")
    def after_iteration(self, model, epoch, evals_log):
        self.pbar.update(1); return False
    def after_training(self, model):
        self.pbar.close(); return model

def pick_precision_threshold(probs, labels, target_prec=0.3, min_recall=0.15):
    precs, recs, thrs = precision_recall_curve(labels, probs)
    best = dict(thr=0.5, p=0.0, r=0.0, f1=0.0)
    for p, r, t in zip(precs[:-1], recs[:-1], thrs):
        if np.isfinite(p) and p >= target_prec and r >= min_recall:
            f1 = 2*p*r/(p+r+1e-12)
            if f1 > best["f1"]:
                best = dict(thr=float(t), p=float(p), r=float(r), f1=float(f1))
    if best["f1"] == 0.0 and len(thrs):
        f1s = []
        for t in thrs:
            pr = (probs >= t).astype(int)
            P = precision_score(labels, pr, zero_division=0)
            R = recall_score(labels, pr, zero_division=0)
            f1s.append(2*P*R/(P+R+1e-12))
        i = int(np.argmax(f1s))
        best["thr"] = float(thrs[i])
        pr = (probs >= best["thr"]).astype(int)
        best["p"] = float(precision_score(labels, pr, zero_division=0))
        best["r"] = float(recall_score(labels, pr, zero_division=0))
        best["f1"] = float(f1s[i])
    return best

# ------------ Load ------------
def load_sparse(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing features file: {path}\n"
                                "Re-run feature_extraction.py for the same TASK to generate task-suffixed features.")
    return sp.load_npz(path)

logger.info("Loading precomputed features...")
X_train = load_sparse(X_train_npz)
X_val   = load_sparse(X_val_npz)
y_train = pd.read_csv(y_train_csv).squeeze().values
y_val   = pd.read_csv(y_val_csv).squeeze().values

if X_train.shape[0] != len(y_train):
    raise RuntimeError(f"Row mismatch: X_train rows={X_train.shape[0]} vs y_train rows={len(y_train)}.")
if X_val.shape[0] != len(y_val):
    raise RuntimeError(f"Row mismatch: X_val rows={X_val.shape[0]} vs y_val rows={len(y_val)}.")

if TASK == "binary":
    y_train = y_train.astype(int)
    y_val   = y_val.astype(int)
else:
    y_train = y_train.astype(np.float32)
    y_val   = y_val.astype(np.float32)

logger.info(f"X_train: {X_train.shape}, X_val: {X_val.shape}")

# ------------ Params / imbalance ------------
params = dict(XGB_PARAMS)
if TASK == "binary":
    params.update(objective="binary:logistic", eval_metric=["logloss", "aucpr"])
    pos = int((y_train == 1).sum()); neg = int((y_train == 0).sum())
    if pos > 0:
        params["scale_pos_weight"] = neg / max(1, pos)
        logger.info(f"scale_pos_weight={params['scale_pos_weight']:.3f} (neg/pos)")
elif TASK == "regression":
    params.update(objective="reg:squarederror", eval_metric=["rmse"])

model = xgb.XGBClassifier(**params) if TASK=="binary" else xgb.XGBRegressor(**params)

# Track eval history automatically
model.set_params(early_stopping_rounds=EARLY_STOP,
                 callbacks=[TQDMCallback(total=params["n_estimators"])])

# ------------ Train ------------
logger.info("Starting training...")
start = time.time()
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],   # track both
    verbose=VERBOSE_EVAL
)
best_iter = getattr(model, "best_iteration", None)
elapsed_min = (time.time() - start) / 60.0
logger.info(f"Done in {elapsed_min:.2f} min | best_iteration={best_iter}")

# ------------ Save eval history (loss tracking) ------------
evals_result = model.evals_result()
# Build a tidy DataFrame
hist = pd.DataFrame({
    "iter": np.arange(len(next(iter(evals_result.values()))["validation_0"])),
})
for name, logs in evals_result.items():
    for metric, series in logs.items():
        hist[f"{name}.{metric}"] = series
hist.to_csv(HIST_PATH, index=False)
logger.info(f"Saved eval history -> {HIST_PATH}")

# Plot curves
plt.figure(figsize=(8,6))
for col in hist.columns:
    if col == "iter": continue
    plt.plot(hist["iter"], hist[col], label=col)
plt.xlabel("Iteration"); plt.ylabel("Metric")
plt.title(f"XGBoost training history ({task_suffix})")
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(CURVE_PATH)
logger.info(f"Saved curves -> {CURVE_PATH}")

# ------------ Evaluate ------------
if TASK=="binary":
    probs = model.predict_proba(X_val)[:, 1]
    auroc = roc_auc_score(y_val, probs)
    auprc = average_precision_score(y_val, probs)
    preds05 = (probs >= 0.5).astype(int)
    logger.info(f"[VAL] AUROC={auroc:.4f} | AUPRC={auprc:.4f}")
    logger.info(f"[VAL @0.5] Acc={accuracy_score(y_val, preds05):.4f} "
                f"Prec={precision_score(y_val, preds05, zero_division=0):.4f} "
                f"Rec={recall_score(y_val, preds05, zero_division=0):.4f} "
                f"F1={f1_score(y_val, preds05, zero_division=0):.4f}")
    best = pick_precision_threshold(probs, y_val, TARGET_PRECISION, MIN_RECALL)
    logger.info(f"[VAL @PrecOpt {best['thr']:.2f}] Prec={best['p']:.4f} Rec={best['r']:.4f} F1={best['f1']:.4f}")
    if SAVE_PRECISION_OPT_THR:
        with open(THRESHOLD_PATH, "w") as f:
            json.dump({"decision_threshold": best["thr"],
                       "precision": best["p"],
                       "recall": best["r"],
                       "f1": best["f1"]}, f, indent=2)
        logger.info(f"Saved threshold -> {THRESHOLD_PATH}")
else:
    preds = model.predict(X_val)
    rmse = mean_squared_error(y_val, preds, squared=False)
    r2   = r2_score(y_val, preds)
    logger.info(f"[VAL] RMSE={rmse:.4f} | R²={r2:.4f}")

# ------------ Save model & importance ------------
model.save_model(MODEL_PATH)
logger.info(f"Saved model -> {MODEL_PATH}")

plt.figure(figsize=(8,6))
xgb.plot_importance(model, importance_type="gain", max_num_features=20, height=0.5)
plt.tight_layout(); plt.savefig(IMPLOT_PATH)
logger.info(f"Saved feature importance -> {IMPLOT_PATH}")
