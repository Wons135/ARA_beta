import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from torch.optim import AdamW
from models.bert_model import BERTRegressor
from src.dataset import ReviewDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Config
MAX_LEN = 256
BATCH_SIZE = 8
EPOCHS = 10
LR = 2e-5
GRAD_ACCUM_STEPS = 2
PATIENCE = 2
WEIGHT_DECAY = 0.01
DROPOUT_RATE = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Datasets
train_dataset = ReviewDataset("../datasets/preprocessed/train.csv", max_len=MAX_LEN)
val_dataset = ReviewDataset("../datasets/preprocessed/val.csv", max_len=MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, pin_memory=True)

# Model
model = BERTRegressor(dropout_rate=DROPOUT_RATE)
model = model.to(device)

# Optimizer & Scheduler
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
total_steps = len(train_loader) // GRAD_ACCUM_STEPS * EPOCHS
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

criterion = nn.MSELoss()
scaler = torch.amp.GradScaler()

def train_epoch(model, loader, optimizer, scheduler, criterion):
    model.train()
    losses = []
    optimizer.zero_grad()
    pbar = tqdm(loader, desc="Training", leave=False)

    for step, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type='cuda'):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, targets) / GRAD_ACCUM_STEPS

        scaler.scale(loss).backward()

        if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        losses.append(loss.item() * GRAD_ACCUM_STEPS)
        pbar.set_postfix(loss=(loss.item() * GRAD_ACCUM_STEPS))

    return np.mean(losses)

def eval_epoch(model, loader, criterion):
    model.eval()
    losses = []
    preds = []
    true = []
    pbar = tqdm(loader, desc="Validation", leave=False)
    with torch.no_grad():
        for batch in pbar:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, targets)
            losses.append(loss.item())

            preds.extend(outputs.cpu().numpy())
            true.extend(targets.cpu().numpy())

            pbar.set_postfix(loss=loss.item())
    return np.mean(losses), np.array(preds), np.array(true)

train_losses = []
val_losses = []

best_val_loss = float("inf")
patience_counter = 0

for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion)
    val_loss, val_preds, val_true = eval_epoch(model, val_loader, criterion)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "../outputs/checkpoints/bert_model_best.pth")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

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
