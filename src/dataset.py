import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

class ReviewDataset(Dataset):
    def __init__(self, csv_file, tokenizer_name="bert-base-uncased", max_len=256):
        self.data = pd.read_csv(csv_file)
        self.texts = self.data["full_text"].tolist()
        self.targets = self.data["helpfulness_ratio_log"].values
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = str(self.texts[index])
        target = self.targets[index]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "target": torch.tensor(target, dtype=torch.float)
        }
