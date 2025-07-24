import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)

# ---------- CONFIG ----------
TASK = "classification"  # "regression" or "classification"
X_val_path = "../datasets/preprocessed/X_val_features.csv"
y_val_path = "../datasets/preprocessed/y_val.csv"
MODEL_INPUT = "../outputs/models/xgboost_model.json"

# ---------- Load Data ----------
X_val = pd.read_csv(X_val_path)
y_val = pd.read_csv(y_val_path).squeeze()

# ---------- Load Model ----------
model = xgb.Booster()
model.load_model(MODEL_INPUT)
dval = xgb.DMatrix(X_val)

# ---------- Predict ----------
if TASK == "classification":
    raw_preds = model.predict(dval)
    y_pred = raw_preds.argmax(axis=1)
else:
    y_pred = model.predict(dval)

# ---------- Evaluation ----------
if TASK == "classification":
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred, average='weighted')
    rec = recall_score(y_val, y_pred, average='weighted')
    f1 = f1_score(y_val, y_pred, average='weighted')
    cm = confusion_matrix(y_val, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("XGBoost Confusion Matrix")
    plt.tight_layout()
    plt.savefig("../outputs/plots/xgboost_confusion_matrix.png")
    plt.close()

else:
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")

    plt.figure(figsize=(6, 6))
    plt.scatter(y_val, y_pred, alpha=0.3)
    plt.xlabel("True Score")
    plt.ylabel("Predicted Score")
    plt.title("XGBoost: Predicted vs True")
    plt.grid(True)
    plt.savefig("../outputs/plots/xgboost_scatter.png")
    plt.close()
