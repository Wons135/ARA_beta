import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import numpy as np

class ReviewDataset(Dataset):
    def __init__(
        self,
        data,
        tokenizer_name: str = "distilbert-base-uncased",
        max_len: int = 256,
        task: str = "regression",
        use_aux_features: bool = False,
    ):
        if isinstance(data, str):
            self.data = pd.read_csv(data)
        else:
            self.data = data.copy()

        # --- texts ---
        self.texts = self.data["full_text"].astype(str).tolist()
        self.use_aux_features = use_aux_features

        # --- targets ---
        task_to_label_column = {
            "regression": ("regression_score", torch.float),
            "binary": ("binary_label", torch.long),
        }
        if task not in task_to_label_column:
            raise ValueError("Unsupported task type. Use 'regression' or 'binary'.")
        label_column, dtype = task_to_label_column[task]

        # coerce label dtype early (CSV may load as float)
        if task == "binary":
            self.targets = self.data[label_column].astype(int).to_numpy()
        else:
            self.targets = self.data[label_column].astype(float).to_numpy()
        self.target_dtype = dtype

        # --- tokenizer ---
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        # keep tokenizer from complaining about long inputs; we already truncate
        try:
            self.tokenizer.model_max_length = max_len
        except Exception:
            pass
        self.max_len = max_len

        # --- aux features ---
        if self.use_aux_features:
            aux_cols = ["review_length", "sentiment_score", "is_verified"]
            # ensure the dtypes are numeric (verified -> 0/1 float)
            aux_df = self.data[aux_cols].copy()
            aux_df["is_verified"] = aux_df["is_verified"].astype(float)
            self.aux_features = aux_df.values.astype(float)
        else:
            self.aux_features = None

        # set externally from train_bert.py
        self.aux_means = {}
        self.aux_stds = {}

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        target = self.targets[index]

        enc = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        out = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "target": torch.tensor(target, dtype=self.target_dtype),
        }

        if self.use_aux_features:
            raw_aux = self.aux_features[index].astype(np.float32, copy=True)

            # log1p on review_length (col 0) before standardization
            raw_aux[0] = np.log1p(max(raw_aux[0], 0.0))  # guard negatives just in case

            # standardize if stats provided
            if self.aux_means and self.aux_stds:
                cols = ["review_length", "sentiment_score", "is_verified"]
                for i, col in enumerate(cols):
                    mean = float(self.aux_means.get(col, 0.0))
                    std = float(self.aux_stds.get(col, 1.0))
                    raw_aux[i] = (raw_aux[i] - mean) / (std + 1e-12)

            # final safety: replace any NaN/Inf that might slip through
            raw_aux = np.nan_to_num(raw_aux, nan=0.0, posinf=0.0, neginf=0.0)

            out["aux_features"] = torch.tensor(raw_aux, dtype=torch.float32)

        return out
