import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig


class BERTRegressor(nn.Module):
    """
    DistilBERT-based regressor.

    Text signal: pooled transformer embeddings (mean pooling or token 0).
    Aux signal: optional small numeric vector (e.g., [time_exposure, review_length, sentiment, is_verified]).
                Processed via a small MLP and concatenated with text features.

    Output: single scalar (helpfulness in the training space, e.g., log-score or normalized score).
    """
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        dropout_rate: float = 0.5,
        use_mean_pooling: bool = False,
        aux_feat_dim: int = 0,
    ):
        super().__init__()
        self.use_mean_pooling = use_mean_pooling
        self.aux_feat_dim = aux_feat_dim

        # Backbone
        config = DistilBertConfig.from_pretrained(model_name)
        config.dropout = 0.3              # encoder dropout
        config.attention_dropout = 0.3
        self.bert = DistilBertModel.from_pretrained(model_name, config=config)

        # Freeze early layers (first 2 transformer blocks)
        for name, param in self.bert.transformer.layer[:2].named_parameters():
            param.requires_grad = False

        # Text pathway head
        self.dropout1 = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(self.bert.config.dim)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Aux pathway head (optional)
        if aux_feat_dim > 0:
            # Light normalization + projection for stability
            self.aux_norm = nn.LayerNorm(aux_feat_dim)
            self.aux_proj = nn.Sequential(
                nn.Linear(aux_feat_dim, 32),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            )
            final_dim = self.bert.config.dim + 32
        else:
            self.aux_norm = None
            self.aux_proj = None
            final_dim = self.bert.config.dim

        # Final regressor
        self.regressor = nn.Linear(final_dim, 1)
        nn.init.xavier_uniform_(self.regressor.weight)
        nn.init.constant_(self.regressor.bias, 0)

    def forward(self, input_ids, attention_mask, aux_features=None):
        # Encode text
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        if self.use_mean_pooling:
            # Mean pooling over valid tokens
            last_hidden = outputs.last_hidden_state
            mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            summed = torch.sum(last_hidden * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            pooled = summed / counts
        else:
            # DistilBERT has no dedicated [CLS]; token 0 often works as a surrogate
            pooled = outputs.last_hidden_state[:, 0, :]

        x = self.dropout1(pooled)
        x = self.norm(x)
        x = self.dropout2(x)

        # Aux features (time exposure, length, sentiment, verified, etc.)
        if self.aux_feat_dim > 0:
            if aux_features is None:
                # If you trained with aux features, you should pass them at inference too.
                # Fall back to zeros to avoid crashes, but warn via comment/log if needed.
                aux = torch.zeros(x.size(0), self.aux_feat_dim, device=x.device, dtype=x.dtype)
            else:
                aux = aux_features
                # ensure correct dtype
                if aux.dtype != x.dtype:
                    aux = aux.to(dtype=x.dtype)
            aux = self.aux_norm(aux)
            aux = self.aux_proj(aux)
            x = torch.cat([x, aux], dim=-1)

        return self.regressor(x).squeeze(-1)


class BERTClassifier(nn.Module):
    """
    DistilBERT-based binary classifier.

    Same architecture as the regressor, but outputs logits for 2 classes.
    Supports the same aux-feature pathway.
    """
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        dropout_rate: float = 0.5,
        use_mean_pooling: bool = False,
        aux_feat_dim: int = 0,
    ):
        super().__init__()
        self.use_mean_pooling = use_mean_pooling
        self.aux_feat_dim = aux_feat_dim

        # Backbone
        config = DistilBertConfig.from_pretrained(model_name)
        config.dropout = 0.3
        config.attention_dropout = 0.3
        self.bert = DistilBertModel.from_pretrained(model_name, config=config)

        # Freeze early layers (first 2 transformer blocks)
        for name, param in self.bert.transformer.layer[:2].named_parameters():
            param.requires_grad = False

        # Text pathway head
        self.dropout1 = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(self.bert.config.dim)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Aux pathway head (optional)
        if aux_feat_dim > 0:
            self.aux_norm = nn.LayerNorm(aux_feat_dim)
            self.aux_proj = nn.Sequential(
                nn.Linear(aux_feat_dim, 32),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            )
            final_dim = self.bert.config.dim + 32
        else:
            self.aux_norm = None
            self.aux_proj = None
            final_dim = self.bert.config.dim

        # Final classifier (binary)
        self.classifier = nn.Linear(final_dim, 2)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, input_ids, attention_mask, aux_features=None):
        # Encode text
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        if self.use_mean_pooling:
            last_hidden = outputs.last_hidden_state
            mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            summed = torch.sum(last_hidden * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            pooled = summed / counts
        else:
            pooled = outputs.last_hidden_state[:, 0, :]

        x = self.dropout1(pooled)
        x = self.norm(x)
        x = self.dropout2(x)

        if self.aux_feat_dim > 0:
            if aux_features is None:
                aux = torch.zeros(x.size(0), self.aux_feat_dim, device=x.device, dtype=x.dtype)
            else:
                aux = aux_features
                if aux.dtype != x.dtype:
                    aux = aux.to(dtype=x.dtype)
            aux = self.aux_norm(aux)
            aux = self.aux_proj(aux)
            x = torch.cat([x, aux], dim=-1)

        return self.classifier(x)
