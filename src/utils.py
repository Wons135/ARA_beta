# src/utils.py
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score
from torch import nn

# ----------------------------
# (Un)freeze helpers
# ----------------------------
def set_freeze_depth(model, freeze_layers: int):
    if not hasattr(model, "bert"):
        return
    # DeBERTa-style
    if hasattr(model.bert, "transformer") and hasattr(model.bert.transformer, "layer"):
        layers = model.bert.transformer.layer
        L = len(layers); freeze_layers = max(0, min(freeze_layers, L))
        for i, layer in enumerate(layers):
            req = (i < freeze_layers)
            for p in layer.parameters():
                p.requires_grad = (not req)
        return
    # BERT/RoBERTa-style
    if hasattr(model.bert, "encoder") and hasattr(model.bert.encoder, "layer"):
        layers = model.bert.encoder.layer
        L = len(layers); freeze_layers = max(0, min(freeze_layers, L))
        for i, layer in enumerate(layers):
            req = (i < freeze_layers)
            for p in layer.parameters():
                p.requires_grad = (not req)
        return

def build_param_groups(model, base_lr: float, head_lr: float,
                       weight_decay: float, head_weight_decay: float):
    enc_params, head_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("bert."):
            enc_params.append(p)
        else:
            head_params.append(p)
    return [
        {"params": enc_params, "lr": base_lr, "weight_decay": weight_decay},
        {"params": head_params, "lr": head_lr, "weight_decay": head_weight_decay},
    ]

# ----------------------------
# R-Drop loss (binary CE + symmetric KL)
# ----------------------------
def rdrop_ce_loss(logits1, logits2, targets, criterion, alpha=1.0):
    ce1 = criterion(logits1.float(), targets.long())
    ce2 = criterion(logits2.float(), targets.long())
    ce = 0.5 * (ce1 + ce2)

    p_log = F.log_softmax(logits1.float(), dim=1)
    q_log = F.log_softmax(logits2.float(), dim=1)
    p = p_log.exp()
    q = q_log.exp()
    kl1 = F.kl_div(p_log, q, reduction="batchmean")
    kl2 = F.kl_div(q_log, p, reduction="batchmean")
    kl = 0.5 * (kl1 + kl2)
    return ce + alpha * kl

# ----------------------------
# EMA (robust to unfreezing/rollback)
# ----------------------------
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._init_from_model(model)

    @torch.no_grad()
    def _init_from_model(self, model):
        self.shadow.clear()
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()

    @torch.no_grad()
    def update(self, model):
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name not in self.shadow:
                self.shadow[name] = p.detach().clone()
            else:
                self.shadow[name].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_shadow(self, model):
        self.backup = {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name in self.shadow:
                self.backup[name] = p.detach().clone()
                p.data.copy_(self.shadow[name].data)

    @torch.no_grad()
    def restore(self, model):
        if not self.backup:
            return
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name in self.backup:
                p.data.copy_(self.backup[name].data)
        self.backup = {}

    @torch.no_grad()
    def refresh_from_model(self, model):
        """Call after rollback or (un)freeze changes."""
        self._init_from_model(model)

# ----------------------------
# AMP autocast helper
# ----------------------------
def autocast_context(device, use_amp: bool, amp_dtype: str):
    if not use_amp:
        return torch.autocast(device_type=device.type, dtype=torch.float32, enabled=False)
    if amp_dtype.lower() == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=True)
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True)

# ----------------------------
# Debug helper
# ----------------------------
@torch.no_grad()
def maybe_print_batch_stats(logger, outputs, targets, tag):
    o = outputs.detach().float()
    t = targets.detach().float()
    if torch.isfinite(o).any() and torch.isfinite(t).any():
        logger.warning(
            f"{tag} stats | outputs: min={o.min().item():.3f} max={o.max().item():.3f} mean={o.mean().item():.3f} | "
            f"targets: min={t.min().item():.3f} max={t.max().item():.3f} mean={t.mean().item():.3f}"
        )

class AsymmetricFocalLoss(nn.Module):
    """
    Binary softmax version (2 logits per sample).
    Emphasizes hard negatives via gamma_neg, which typically improves precision.
    """
    def __init__(self, alpha_pos=0.5, gamma_pos=0.0, gamma_neg=2.0, eps=1e-8):
        super().__init__()
        self.alpha_pos = alpha_pos
        self.alpha_neg = 1.0 - alpha_pos
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.eps = eps

    def forward(self, logits, targets):
        # logits: (B,2), targets: (B,)
        probs = torch.softmax(logits, dim=1).clamp(self.eps, 1.0 - self.eps)
        p_pos = probs[:, 1]
        p_neg = probs[:, 0]
        t = targets.long()

        # select p_t and alpha_t per class
        p_t = torch.where(t == 1, p_pos, p_neg)
        alpha_t = torch.where(t == 1,
                              torch.full_like(p_pos, self.alpha_pos),
                              torch.full_like(p_neg, self.alpha_neg))
        # asymmetric focusing
        gamma_t = torch.where(t == 1,
                              torch.full_like(p_pos, self.gamma_pos),
                              torch.full_like(p_neg, self.gamma_neg))
        focal = (1.0 - p_t) ** gamma_t
        ce = -torch.log(p_t)
        loss = alpha_t * focal * ce
        return loss.mean()


def pick_precision_threshold(probs, labels, target_prec=0.3, min_recall=0.15):
    """
    Returns (best_thr, best_prec, best_rec, best_f1) under the constraints:
      precision >= target_prec and recall >= min_recall.
    If no threshold satisfies, fall back to the overall best F1 threshold.
    """
    thrs = np.linspace(0.05, 0.95, 19)
    best = (0.5, 0.0, 0.0, 0.0)  # thr, prec, rec, f1
    # candidate pool: satisfying constraints, maximize precision; tie-break by F1
    candidates = []
    for thr in thrs:
        preds = (probs >= thr).astype(int)
        prec = precision_score(labels, preds, zero_division=0)
        rec  = recall_score(labels, preds, zero_division=0)
        f1v  = f1_score(labels, preds, zero_division=0)
        if (prec >= target_prec) and (rec >= min_recall):
            candidates.append((thr, prec, rec, f1v))

    if candidates:
        # sort by precision desc, then F1 desc
        candidates.sort(key=lambda x: (x[1], x[3]), reverse=True)
        return candidates[0]
    else:
        # fallback: best F1
        f1s = []
        for thr in thrs:
            preds = (probs >= thr).astype(int)
            f1s.append(f1_score(labels, preds, zero_division=0))
        i = int(np.argmax(f1s))
        thr = float(thrs[i])
        preds = (probs >= thr).astype(int)
        prec = precision_score(labels, preds, zero_division=0)
        rec  = recall_score(labels, preds, zero_division=0)
        f1v  = f1s[i]
        return (thr, prec, rec, f1v)