import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from models.xgboost_model import XGBoostModel
import time
from tqdm import tqdm
from xgboost.callback import TrainingCallback


class TQDMCallback(TrainingCallback):
    def __init__(self, total):
        self.pbar = tqdm(total=total, desc="Training Boosting Rounds")

    def after_iteration(self, model, epoch, evals_log):
        self.pbar.update(1)
        return False

    def after_training(self, model):
        self.pbar.close()
        return model


# ---------- CONFIG ----------
TASK = "regression"  # "regression" or "classification"

CONFIG = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "tree_method": "gpu_hist",         # Enable GPU training
    "predictor": "gpu_predictor",      # Optional but improves inference
    "verbosity": 1
}

X_train_path = "../datasets/preprocessed/X_train_features.csv"
y_train_path = "../datasets/preprocessed/y_train.csv"
X_val_path = "../datasets/preprocessed/X_val_features.csv"
y_val_path = "../datasets/preprocessed/y_val.csv"
MODEL_OUTPUT = "../outputs/checkpoints/xgboost_model.json"
IMPORTANCE_PLOT = "../outputs/plots/feature_importance.png"

# ---------- Load Data ----------
print("Loading data...")
X_train = pd.read_csv(X_train_path)
y_train = pd.read_csv(y_train_path).squeeze()
X_val = pd.read_csv(X_val_path)
y_val = pd.read_csv(y_val_path).squeeze()

# ---------- Train ----------
print("Starting training...")
start_time = time.time()

model = XGBoostModel(task=TASK, **CONFIG)

with tqdm(total=CONFIG["n_estimators"], desc="Training Boosting Rounds") as pbar:
    def callback(env):
        pbar.update(1)


    model.model.set_params(
        callbacks=[TQDMCallback(CONFIG["n_estimators"])],
    )
    model.fit(X_train, y_train, X_val, y_val)

elapsed = time.time() - start_time
print(f"Training completed in {elapsed:.2f} seconds")

# ---------- Save Model ----------
model.model.save_model(MODEL_OUTPUT)
print(f"Model saved to {MODEL_OUTPUT}")

# ---------- Plot Feature Importance ----------
xgb.plot_importance(model.model)
plt.tight_layout()
plt.savefig(IMPORTANCE_PLOT)
print(f"Feature importance plot saved to {IMPORTANCE_PLOT}")
