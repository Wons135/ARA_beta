# models/bert_model.py
# DistilBERT base with configurable pooling, validated aux features, runtime-freezing,
# and AMP-safe LayerNorm. Backward-compatible BERTRegressor/BERTClassifier APIs.

from __future__ import annotations
from typing import Optional, Literal, Tuple

import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig


PoolingMode = Literal["token0", "mean"]


class FP32LayerNorm(nn.LayerNorm):
    """
    AMP-safe LayerNorm: compute in fp32 for stability, cast back to input dtype.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        if x.dtype in (torch.float16, torch.bfloat16):
            x = super().forward(x.float())
            return x.to(dtype)
        return super().forward(x)


class BaseBERTWithAux(nn.Module):
    """
    DistilBERT backbone with configurable pooling and optional auxiliary feature pathway.

    Parameters
    ----------
    model_name : str
        HF model id for DistilBERT.
    dropout_rate : float
        Dropout applied around pooled text features and aux MLP.
    pooling : {"token0","mean"}
        Pooled representation. "token0" is default (CLS surrogate). "mean" = mask-aware mean.
    aux_feat_dim : int
        Dimension of optional auxiliary numeric features; 0 disables aux path.
    freeze_init : int
        Number of initial transformer blocks to freeze at construction.
    gradient_checkpointing : bool
        Enable DistilBERT gradient checkpointing to trade compute for memory.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        dropout_rate: float = 0.5,
        pooling: PoolingMode = "token0",
        aux_feat_dim: int = 0,
        freeze_init: int = 2,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        if pooling not in ("token0", "mean"):
            raise ValueError("pooling must be 'token0' or 'mean'")
        self.pooling: PoolingMode = pooling
        self.aux_feat_dim = int(aux_feat_dim)

        # Backbone
        config = DistilBertConfig.from_pretrained(model_name)
        config.dropout = 0.3
        config.attention_dropout = 0.3
        self.bert = DistilBertModel.from_pretrained(model_name, config=config)

        # Optional gradient checkpointing (HF supports on the transformer blocks)
        if gradient_checkpointing and hasattr(self.bert, "gradient_checkpointing_enable"):
            self.bert.gradient_checkpointing_enable()

        # Runtime-freezable encoder blocks
        self._freeze_depth = -1
        self.set_freeze_depth(max(0, int(freeze_init)))

        hidden = self.bert.config.dim

        # Text pathway head
        self.text_drop1 = nn.Dropout(dropout_rate)
        self.text_norm = FP32LayerNorm(hidden)
        self.text_drop2 = nn.Dropout(dropout_rate)

        # Aux pathway head (optional)
        if self.aux_feat_dim > 0:
            self.aux_norm = FP32LayerNorm(self.aux_feat_dim)
            self.aux_proj = nn.Sequential(
                nn.Linear(self.aux_feat_dim, 32),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            )
            self._final_in = hidden + 32
        else:
            self.aux_norm = None
            self.aux_proj = None
            self._final_in = hidden

    # ---------- Public control APIs ----------

    @torch.no_grad()
    def set_freeze_depth(self, depth: int) -> None:
        """
        Freeze first `depth` transformer blocks (0..n). depth=0 -> train all.
        """
        depth = max(0, int(depth))
        for i, block in enumerate(self.bert.transformer.layer):
            req_grad = not (i < depth)
            for p in block.parameters():
                p.requires_grad = req_grad
        self._freeze_depth = depth

    def get_freeze_depth(self) -> int:
        return int(self._freeze_depth)

    # ---------- Internals ----------

    def _pool(self, last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden.dtype)
            summed = (last_hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp_min(1e-9)
            return summed / counts
        # token0
        return last_hidden[:, 0, :]

    def _encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._pool(out.last_hidden_state, attention_mask)
        x = self.text_drop1(pooled)
        x = self.text_norm(x)
        x = self.text_drop2(x)
        return x

    def _encode_aux(self, aux_features: Optional[torch.Tensor], batch_size: int, dtype: torch.dtype, device: torch.device) -> Optional[torch.Tensor]:
        if self.aux_feat_dim <= 0:
            return None
        if aux_features is None:
            # Backward-compatible: zero-fill if not provided.
            aux = torch.zeros(batch_size, self.aux_feat_dim, device=device, dtype=dtype)
        else:
            if aux_features.dim() != 2 or aux_features.size(1) != self.aux_feat_dim:
                raise ValueError(f"Expected aux_features shape [B,{self.aux_feat_dim}], got {tuple(aux_features.shape)}")
            aux = aux_features.to(device=device, dtype=dtype, non_blocking=True)
        aux = self.aux_norm(aux)  # fp32-safe
        aux = self.aux_proj(aux)
        return aux

    # To be finished by subclasses (final head)
    def _forward_head(self, fused: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract by convention
        raise NotImplementedError

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, aux_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self._encode_text(input_ids, attention_mask)
        aux = self._encode_aux(aux_features, batch_size=x.size(0), dtype=x.dtype, device=x.device)
        if aux is not None:
            x = torch.cat([x, aux], dim=-1)
        return self._forward_head(x)


class BERTRegressor(BaseBERTWithAux):
    """
    DistilBERT-based regressor.
    Backward-compatible init args: `use_mean_pooling` retained; mapped to pooling mode.
    """
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        dropout_rate: float = 0.5,
        use_mean_pooling: bool = False,
        aux_feat_dim: int = 0,
        freeze_init: int = 2,
        gradient_checkpointing: bool = False,
    ):
        pooling = "mean" if use_mean_pooling else "token0"
        super().__init__(model_name=model_name,
                         dropout_rate=dropout_rate,
                         pooling=pooling,
                         aux_feat_dim=aux_feat_dim,
                         freeze_init=freeze_init,
                         gradient_checkpointing=gradient_checkpointing)
        self.regressor = nn.Linear(self._final_in, 1)
        nn.init.xavier_uniform_(self.regressor.weight)
        nn.init.constant_(self.regressor.bias, 0.0)

    def _forward_head(self, fused: torch.Tensor) -> torch.Tensor:
        return self.regressor(fused).squeeze(-1)


class BERTClassifier(BaseBERTWithAux):
    """
    DistilBERT-based binary classifier (2-logit output).
    Backward-compatible init args: `use_mean_pooling` retained; mapped to pooling mode.
    """
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        dropout_rate: float = 0.5,
        use_mean_pooling: bool = False,
        aux_feat_dim: int = 0,
        freeze_init: int = 2,
        gradient_checkpointing: bool = False,
    ):
        pooling = "mean" if use_mean_pooling else "token0"
        super().__init__(model_name=model_name,
                         dropout_rate=dropout_rate,
                         pooling=pooling,
                         aux_feat_dim=aux_feat_dim,
                         freeze_init=freeze_init,
                         gradient_checkpointing=gradient_checkpointing)
        self.classifier = nn.Linear(self._final_in, 2)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0.0)

    def _forward_head(self, fused: torch.Tensor) -> torch.Tensor:
        return self.classifier(fused)
