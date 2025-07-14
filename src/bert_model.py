import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class BERTRegressor(nn.Module):
    def __init__(self, model_name="bert-base-uncased", dropout_rate=0.1):
        super(BERTRegressor, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.regressor = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        dropped = self.dropout(pooled_output)
        regression_output = self.regressor(dropped)
        return regression_output.squeeze(-1)


def get_tokenizer(model_name="bert-base-uncased"):
    return BertTokenizer.from_pretrained(model_name)

