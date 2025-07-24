import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.bert_model import BERTRegressor, BERTClassifier
from src.dataset import ReviewDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
import seaborn as sns

# ---------- CONFIG ----------
MODEL_TYPE = "regression"  # set to "regression" or "classification"
CHECKPOINT_PATH = "../outputs/checkpoints/bert_model_best.pth"
MAX_LEN = 256
BATCH_SIZE = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- Load Dataset ----------
test_dataset = ReviewDataset("../datasets/preprocessed/test.csv", max_len=MAX_LEN, task=MODEL_TYPE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, pin_memory=True)

# ---------- Load Model ----------
if MODEL_TYPE == "regression":
    model = BERTRegressor()
    criterion = nn.MSELoss()
else:
    model = BERTClassifier(num_classes=5)
    criterion = nn.CrossEntropyLoss()

model.load_state_dict(torch.load(CHECKPOINT_PATH))
model = model.to(device)
model.eval()

# ---------- Evaluation Function ----------
def eval_epoch(model, loader, criterion, task):
    losses, preds, true = [], [], []
    pbar = tqdm(loader, desc="Testing", leave=False)
    with torch.no_grad():
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["target"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, targets if task == "regression" else targets.long())
            losses.append(loss.item())

            if task == "regression":
                preds.extend(outputs.cpu().numpy())
                true.extend(targets.cpu().numpy())
            else:
                preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                true.extend(targets.cpu().numpy())

            pbar.set_postfix(loss=loss.item())
    return np.mean(losses), np.array(preds), np.array(true)

# ---------- Run Evaluation ----------
test_loss, test_preds, test_true = eval_epoch(model, test_loader, criterion, MODEL_TYPE)

# ---------- Regression Metrics ----------
if MODEL_TYPE == "regression":
    test_preds_orig = np.expm1(test_preds)
    test_true_orig = np.expm1(test_true)
    rmse = np.sqrt(mean_squared_error(test_true_orig, test_preds_orig))
    mae = mean_absolute_error(test_true_orig, test_preds_orig)
    r2 = r2_score(test_true_orig, test_preds_orig)

    print(f"Test Loss (MSE log): {test_loss:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test R²: {r2:.4f}")

    # Scatter Plot
    plt.figure(figsize=(6, 6))
    plt.scatter(test_true_orig, test_preds_orig, alpha=0.3)
    plt.xlabel("True Helpfulness Score")
    plt.ylabel("Predicted Helpfulness Score")
    plt.title("BERT: Predicted vs True (Test set)")
    plt.grid(True)
    plt.savefig("../outputs/plots/bert_test_scatter.png")
    plt.close()

    # Error Distribution
    errors = test_preds_orig - test_true_orig
    plt.figure(figsize=(8, 5))
    plt.hist(errors, bins=50, edgecolor="black")
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.title("BERT: Prediction Error Distribution (Test set)")
    plt.grid(True)
    plt.savefig("../outputs/plots/bert_test_error_distribution.png")
    plt.close()

    # Bin Analysis
    bins = pd.qcut(test_true_orig, q=4, duplicates='drop')
    bin_df = pd.DataFrame({"true": test_true_orig, "pred": test_preds_orig, "bin": bins})
    bin_means = bin_df.groupby("bin", observed=False)[["true", "pred"]].mean().reset_index()
    bin_labels = bin_means["bin"].astype(str)

    plt.figure(figsize=(8, 5))
    plt.plot(bin_labels, bin_means["true"], label="Mean True")
    plt.plot(bin_labels, bin_means["pred"], label="Mean Predicted")
    plt.xticks(rotation=45)
    plt.ylabel("Helpfulness Score")
    plt.title("Mean Prediction per Bin (Quartiles)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("../outputs/plots/bert_test_bin_means.png")
    plt.close()

    # Save Predictions
    test_df = pd.read_csv("../datasets/preprocessed/test.csv")
    test_df["pred_log"] = test_preds
    test_df["pred_original"] = test_preds_orig
    test_df["true_original"] = test_true_orig
    test_df.to_csv("../outputs/predictions/bert_test_predictions.csv", index=False)

# ---------- Classification Metrics ----------
else:
    acc = accuracy_score(test_true, test_preds)
    prec = precision_score(test_true, test_preds, average='weighted')
    rec = recall_score(test_true, test_preds, average='weighted')
    f1 = f1_score(test_true, test_preds, average='weighted')
    cm = confusion_matrix(test_true, test_preds)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("../outputs/plots/bert_test_confusion_matrix.png")
    plt.close()

    # Save Predictions
    test_df = pd.read_csv("../datasets/preprocessed/test.csv")
    test_df["pred_class"] = test_preds
    test_df["true_class"] = test_true
    test_df.to_csv("../outputs/predictions/bert_test_classification_predictions.csv", index=False)
