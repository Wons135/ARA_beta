import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.bert_model import BERTRegressor
from src.dataset import ReviewDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Config
MAX_LEN = 256
BATCH_SIZE = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load test dataset
test_dataset = ReviewDataset("../datasets/preprocessed/test.csv", max_len=MAX_LEN)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, pin_memory=True)

# Load model
model = BERTRegressor()
model.load_state_dict(torch.load("../outputs/checkpoints/bert_model_best.pth"))
model = model.to(device)
model.eval()

criterion = nn.MSELoss()

def eval_epoch(model, loader, criterion):
    losses = []
    preds = []
    true = []
    pbar = tqdm(loader, desc="Testing", leave=False)
    with torch.no_grad():
        for batch in pbar:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, targets)
            losses.append(loss.item())

            preds.extend(outputs.cpu().numpy())
            true.extend(targets.cpu().numpy())

            pbar.set_postfix(loss=loss.item())
    return np.mean(losses), np.array(preds), np.array(true)

# Evaluate
print("Evaluating on test set...")
test_loss, test_preds, test_true = eval_epoch(model, test_loader, criterion)

# Inverse transform
test_preds_orig = np.expm1(test_preds)
test_true_orig = np.expm1(test_true)

# Metrics
rmse = np.sqrt(mean_squared_error(test_true_orig, test_preds_orig))
mae = mean_absolute_error(test_true_orig, test_preds_orig)
r2 = r2_score(test_true_orig, test_preds_orig)

print(f"Test Loss (MSE log): {test_loss:.4f}")
print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAE: {mae:.4f}")
print(f"Test R²: {r2:.4f}")

# Scatter plot
plt.figure(figsize=(6, 6))
plt.scatter(test_true_orig, test_preds_orig, alpha=0.3)
plt.xlabel("True Helpfulness Score")
plt.ylabel("Predicted Helpfulness Score")
plt.title("BERT: Predicted vs True (Test set)")
plt.grid(True)
plt.savefig("../outputs/plots/bert_test_scatter.png")
plt.close()

# Error distribution
errors = test_preds_orig - test_true_orig
plt.figure(figsize=(8, 5))
plt.hist(errors, bins=50, edgecolor="black")
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.title("BERT: Prediction Error Distribution (Test set)")
plt.grid(True)
plt.savefig("../outputs/plots/bert_test_error_distribution.png")
plt.close()

# Bin analysis
bins = pd.qcut(test_true_orig, q=4, duplicates='drop')
bin_df = pd.DataFrame({"true": test_true_orig, "pred": test_preds_orig, "bin": bins})
bin_means = bin_df.groupby("bin", observed=False)[["true", "pred"]].mean().reset_index()

# Convert bin intervals to strings for plotting
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

# Save predictions
test_df = pd.read_csv("../datasets/preprocessed/test.csv")
test_df["pred_log"] = test_preds
test_df["pred_original"] = test_preds_orig
test_df["true_original"] = test_true_orig
test_df.to_csv("../outputs/predictions/bert_test_predictions.csv", index=False)
