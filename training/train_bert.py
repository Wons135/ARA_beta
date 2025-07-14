import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from models.bert_model import BERTRegressor
from src.dataset import ReviewDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Config
MAX_LEN = 256
BATCH_SIZE = 8
EPOCHS = 3
LR = 2e-5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Datasets
train_dataset = ReviewDataset("../datasets/processed/train.csv", max_len=MAX_LEN)
val_dataset = ReviewDataset("../datasets/processed/val.csv", max_len=MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Model
model = BERTRegressor()
model = model.to(device)

# Optimizer & Scheduler
optimizer = AdamW(model.parameters(), lr=LR)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

criterion = nn.MSELoss()

def train_epoch(model, loader, optimizer, scheduler, criterion):
    model.train()
    losses = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["target"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
    return np.mean(losses)

def eval_epoch(model, loader, criterion):
    model.eval()
    losses = []
    preds = []
    true = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["target"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, targets)
            losses.append(loss.item())

            preds.extend(outputs.cpu().numpy())
            true.extend(targets.cpu().numpy())

    return np.mean(losses), np.array(preds), np.array(true)

train_losses = []
val_losses = []

for epoch in range(EPOCHS):
    train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion)
    val_loss, val_preds, val_true = eval_epoch(model, val_loader, criterion)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# Save model
torch.save(model.state_dict(), "../outputs/checkpoints/bert_model.pth")

# Plot loss curves
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE on log scale)")
plt.legend()
plt.grid(True)
plt.savefig("../outputs/plots/bert_loss_curve.png")
plt.close()

# Optional: Save validation predictions
val_df = pd.read_csv("../datasets/processed/val.csv")
val_df["pred_log"] = val_preds
val_df["pred_original"] = np.expm1(val_preds)
val_df["true_original"] = np.expm1(val_true)
val_df.to_csv("../outputs/predictions/bert_val_predictions.csv", index=False)
