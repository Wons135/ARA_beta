# backfill_ckpt.py
import torch, numpy as np, pandas as pd

CKPT_PATH = "outputs/checkpoints/bert_model_bin_best.pth"  # change if needed
TRAIN_CSV = "datasets/preprocessed/binary_label_train.csv" # must match the checkpoint's training split!
AUX_COLS  = ["review_length", "sentiment_score", "is_verified"]

ckpt = torch.load(CKPT_PATH, map_location="cpu")

df = pd.read_csv(TRAIN_CSV)
aux_means, aux_stds = {}, {}
for c in AUX_COLS:
    v = df[c].to_numpy(np.float32)
    if c == "review_length":
        v = np.log1p(v)
    aux_means[c] = float(np.mean(v))
    aux_stds[c]  = float(np.std(v) + 1e-12)

ckpt["aux_means"] = aux_means
ckpt["aux_stds"]  = aux_stds

torch.save(ckpt, CKPT_PATH)
print("Patched:", CKPT_PATH)
print("aux_means:", aux_means)
print("aux_stds:", aux_stds)