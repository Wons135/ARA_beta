# src/utils.py
"""
Utility functions and lightweight helpers used across model training/evaluation.
Refactored (Strategy 2) for:
  • Backbone-agnostic freezing (BERT, RoBERTa, DistilBERT, etc.)
  • Stable param grouping with no-decay handling
  • EMA robust to (un)freezing
  • Vectorized precision-threshold selection
  • Safer autocast and logging
"""
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve
from typing import Any, Dict, List, Tuple


# ----------------------------------------------------------------------
# Backbone introspection and freeze helpers
# ----------------------------------------------------------------------
class BackboneIntrospector:
    """Detect encoder/embedding layers in common Transformer variants."""

    @staticmethod
    def encoder_layers(model: nn.Module):
        for attr in ["transformer", "encoder"]:
            enc = getattr(getattr(model, "bert", None), attr, None)
            if enc is not None and hasattr(enc, "layer"):
                return enc.layer
        return None

    @staticmethod
    def embedding_module(model: nn.Module):
        for name in ["embeddings", "Embeddings"]:
            if hasattr(getattr(model, "bert", None), name):
                return getattr(model.bert, name)
        return None

    @staticmethod
    def pooler_module(model: nn.Module):
        if hasattr(getattr(model, "bert", None), "pooler"):
            return model.bert.pooler
        return None


def set_freeze_depth(
    model: nn.Module,
    freeze_layers: int,
    freeze_embed: bool = True,
    freeze_pooler: bool = False,
):
    """
    Freeze early encoder layers (and optionally embeddings/pooler).

    freeze_layers: number of initial encoder blocks to freeze.
    freeze_embed : freeze token embeddings.
    freeze_pooler: freeze pooler head if present.
    """
    if not hasattr(model, "bert"):
        return

    layers = BackboneIntrospector.encoder_layers(model)
    if layers is not None:
        L = len(layers)
        freeze_layers = max(0, min(freeze_layers, L))
        for i, layer in enumerate(layers):
            req = (i < freeze_layers)
            for p in layer.parameters():
                p.requires_grad = not req

    if freeze_embed:
        emb = BackboneIntrospector.embedding_module(model)
        if emb is not None:
            for p in emb.parameters():
                p.requires_grad = False

    if freeze_pooler:
        pool = BackboneIntrospector.pooler_module(model)
        if pool is not None:
            for p in pool.parameters():
                p.requires_grad = False


# ----------------------------------------------------------------------
# Parameter grouping with no-decay
# ----------------------------------------------------------------------
def build_param_groups(
    model: nn.Module,
    base_lr: float,
    head_lr: float,
    weight_decay: float,
    head_weight_decay: float,
) -> List[Dict[str, Any]]:
    """
    Separate encoder vs head parameters and exclude biases/LayerNorm weights from weight decay.
    """
    enc_params_decay, enc_params_no_decay = [], []
    head_params_decay, head_params_no_decay = [], []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_no_decay = any(nd in name.lower() for nd in ["bias", "norm", "bn"])
        is_encoder = name.startswith("bert.") or name.startswith("roberta.") or name.startswith("distilbert.")

        if is_encoder:
            (enc_params_no_decay if is_no_decay else enc_params_decay).append(p)
        else:
            (head_params_no_decay if is_no_decay else head_params_decay).append(p)

    return [
        {"params": enc_params_decay, "lr": base_lr, "weight_decay": weight_decay},
        {"params": enc_params_no_decay, "lr": base_lr, "weight_decay": 0.0},
        {"params": head_params_decay, "lr": head_lr, "weight_decay": head_weight_decay},
        {"params": head_params_no_decay, "lr": head_lr, "weight_decay": 0.0},
    ]


# ----------------------------------------------------------------------
# R-Drop loss
# ----------------------------------------------------------------------
def rdrop_ce_loss(logits1, logits2, targets, criterion, alpha: float = 1.0):
    ce1 = criterion(logits1.float(), targets.long())
    ce2 = criterion(logits2.float(), targets.long())
    ce = 0.5 * (ce1 + ce2)

    p_log = F.log_softmax(logits1.float(), dim=1)
    q_log = F.log_softmax(logits2.float(), dim=1)
    p = p_log.exp()
    q = q_log.exp()
    kl1 = F.kl_div(p_log, q, reduction="batchmean")
    kl2 = F.kl_div(q_log, p, reduction="batchmean")
    return ce + alpha * 0.5 * (kl1 + kl2)


# ----------------------------------------------------------------------
# EMA robust to (un)freezing
# ----------------------------------------------------------------------
class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999, device: str | None = None):
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        self.device = device or next(model.parameters()).device
        self._init_from_model(model)

    @torch.no_grad()
    def _init_from_model(self, model: nn.Module):
        self.shadow.clear()
        for name, p in model.named_parameters():
            self.shadow[name] = p.detach().clone().to(self.device)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, p in model.named_parameters():
            if name not in self.shadow:
                self.shadow[name] = p.detach().clone().to(self.device)
            else:
                self.shadow[name].mul_(self.decay).add_(p.detach().to(self.device), alpha=1 - self.decay)

    @torch.no_grad()
    def apply_shadow(self, model: nn.Module):
        self.backup.clear()
        for name, p in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = p.detach().clone()
                p.data.copy_(self.shadow[name].data)

    @torch.no_grad()
    def restore(self, model: nn.Module):
        for name, p in model.named_parameters():
            if name in self.backup:
                p.data.copy_(self.backup[name].data)
        self.backup.clear()

    @torch.no_grad()
    def refresh_from_model(self, model: nn.Module):
        """Re-initialize shadow weights (e.g., after checkpoint rollback)."""
        self._init_from_model(model)


# ----------------------------------------------------------------------
# AMP autocast helper
# ----------------------------------------------------------------------
def autocast_context(device, use_amp: bool, amp_dtype: str):
    if not use_amp:
        return torch.autocast(device_type=device.type, dtype=torch.float32, enabled=False)
    dtype = torch.bfloat16 if amp_dtype.lower() == "bf16" else torch.float16
    if device.type == "cpu" and dtype == torch.float16:
        raise RuntimeError("fp16 autocast unsupported on CPU; use bf16 or disable AMP.")
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=True)


# ----------------------------------------------------------------------
# Debug helpers
# ----------------------------------------------------------------------
@torch.no_grad()
def maybe_print_batch_stats(logger, outputs, targets, tag):
    o = outputs.detach().float()
    t = targets.detach().float()
    if torch.isfinite(o).any() and torch.isfinite(t).any():
        logger.warning(
            "%s stats | outputs: min=%.3f max=%.3f mean=%.3f | targets: min=%.3f max=%.3f mean=%.3f",
            tag, o.min().item(), o.max().item(), o.mean().item(),
            t.min().item(), t.max().item(), t.mean().item()
        )


# ----------------------------------------------------------------------
# Asymmetric Focal Loss
# ----------------------------------------------------------------------
class AsymmetricFocalLoss(nn.Module):
    """Binary softmax version. Emphasizes hard negatives via gamma_neg."""
    def __init__(self, alpha_pos=0.5, gamma_pos=0.0, gamma_neg=2.0, eps=1e-8):
        super().__init__()
        self.alpha_pos = alpha_pos
        self.alpha_neg = 1.0 - alpha_pos
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1).clamp(self.eps, 1.0 - self.eps)
        p_pos, p_neg = probs[:, 1], probs[:, 0]
        t = targets.long()
        p_t = torch.where(t == 1, p_pos, p_neg)
        alpha_t = torch.where(t == 1,
                              torch.full_like(p_pos, self.alpha_pos),
                              torch.full_like(p_neg, self.alpha_neg))
        gamma_t = torch.where(t == 1,
                              torch.full_like(p_pos, self.gamma_pos),
                              torch.full_like(p_neg, self.gamma_neg))
        focal = (1.0 - p_t) ** gamma_t
        ce = -torch.log(p_t)
        return (alpha_t * focal * ce).mean()


# ----------------------------------------------------------------------
# Precision-targeted threshold selection (vectorized)
# ----------------------------------------------------------------------
def pick_precision_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    target_prec: float = 0.3,
    min_recall: float = 0.15
) -> Tuple[float, float, float, float]:
    """
    Returns (best_thr, precision, recall, f1)
    satisfying precision >= target_prec and recall >= min_recall.
    Falls back to best-F1 if constraints not met.
    """
    probs = np.nan_to_num(probs.astype(float))
    labels = labels.astype(int)

    precs, recs, thrs = precision_recall_curve(labels, probs)
    f1s = 2 * precs * recs / np.maximum(precs + recs, 1e-12)

    mask = (precs >= target_prec) & (recs >= min_recall)
    if np.any(mask[:-1]):
        idx = int(np.nanargmax(f1s[:-1] * mask[:-1]))
    else:
        idx = int(np.nanargmax(f1s[:-1]))

    best_thr = float(thrs[max(0, min(idx, len(thrs) - 1))])
    return float(best_thr), float(precs[idx]), float(recs[idx]), float(f1s[idx])
