import torch.nn as nn
from transformers import BertModel


class BERTRegressor(nn.Module):
    def __init__(self, model_name="bert-base-uncased", dropout_rate=0.1):
        super(BERTRegressor, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)

        for name, param in self.bert.named_parameters():
            if "encoder.layer" in name:
                layer_num = int(name.split('.')[2])
                if layer_num < 6:
                    param.requires_grad = False

        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(self.bert.config.hidden_size)
        self.regressor = nn.Linear(self.bert.config.hidden_size, 1)
        nn.init.xavier_uniform_(self.regressor.weight)
        nn.init.constant_(self.regressor.bias, 0)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # CLS token
        dropped = self.dropout(cls_embedding)
        normalized = self.norm(dropped)
        return self.regressor(normalized).squeeze(-1)


class BERTClassifier(nn.Module):
    def __init__(self, model_name="bert-base-uncased", dropout_rate=0.1, num_classes=5):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)

        for name, param in self.bert.named_parameters():
            if "encoder.layer" in name:
                layer_num = int(name.split('.')[2])
                if layer_num < 6:
                    param.requires_grad = False

        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(self.bert.config.hidden_size)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # CLS token
        dropped = self.dropout(cls_embedding)
        normalized = self.norm(dropped)
        return self.classifier(normalized)