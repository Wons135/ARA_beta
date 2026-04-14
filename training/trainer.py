# trainer.py
# Production-hardened Trainer with safe checkpoint I/O, AMP/EMA orchestration,
# OS-aware DataLoader, and clean seams for evaluation and rollback.
from __future__ import annotations

import os
import sys
import math
import json
import time
import hashlib
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import clip_grad_norm_
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

# ------------------------- Utilities ------------------------- #

TaskT = Literal["regression", "binary"]

def safe_path(path: str | Path, base_dir: str | Path) -> Path:
    p = Path(path).expanduser().resolve()
    base = Path(base_dir).expanduser().resolve()
    if not str(p).startswith(str(base)):
        raise ValueError(f"Unsafe path outside base dir: {p}")
    return p

def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def set_seed(seed: int = 42) -> None:
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ------------------------- EMA ------------------------- #

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self.register(model)

    def register(self, model: nn.Module):
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items() if v.dtype.is_floating_point}

    def update(self, model: nn.Module):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k in self.shadow:
                    self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def apply(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        self._backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        new_state = {}
        for k, v in model.state_dict().items():
            if k in self.shadow:
                new_state[k] = self.shadow[k].clone()
            else:
                new_state[k] = v
        model.load_state_dict(new_state)
        return self._backup  # return backup for restore

    def restore(self, model: nn.Module, backup: Dict[str, torch.Tensor]):
        model.load_state_dict(backup)

# ------------------------- Config ------------------------- #

@dataclass(frozen=True)
class TrainConfig:
    task: TaskT = "binary"
    epochs: int = 10
    max_len: int = 256
    batch_size: int = 8
    grad_accum_steps: int = 1
    base_lr: float = 1e-5
    head_lr: float = 3e-5
    weight_decay: float = 0.01
    head_weight_decay: float = 0.10
    warmup_pct: float = 0.10
    clip_norm: float = 1.0
    patience: int = 3
    aux_feat_dim: int = 3
    # AMP
    use_amp: bool = True
    amp_dtype: Literal["bf16", "fp16"] = "bf16"
    amp_warmup_epochs: int = 0
    disable_amp_on_nan: bool = True
    lr_reduce_factor: float = 0.5
    lr_floor: float = 1e-7
    # Unfreezing schedule: epoch -> freeze_depth
    unfreeze_schedule: Mapping[int, int] = None  # type: ignore
    freeze_init: int = 4
    # Loss/metrics knobs
    loss_kind: Literal["ce", "afl", "mse", "smooth"] = "ce"
    ce_label_smooth: float = 0.01
    target_precision: float = 0.30
    min_recall: float = 0.15
    # EMA
    use_ema: bool = True
    ema_decay: float = 0.999
    # Speed caps
    train_max_steps_per_epoch: Optional[int] = None
    val_max_steps: Optional[int] = None
    # I/O
    base_dir: str = "../outputs"
    ckpt_dir: str = "../outputs/checkpoints"
    logs_dir: str = "../outputs/logs"
    plots_dir: str = "../outputs/plots"
    dataset_root: str = "../datasets"
    verify_checkpoints: bool = False
    checkpoint_extras: Optional[Mapping[str, Any]] = None
    # System
    seed: int = 42

    def __post_init__(self):
        if object.__getattribute__(self, "unfreeze_schedule") is None:
            object.__setattr__(self, "unfreeze_schedule", {0: self.freeze_init, 3: 4, 6: 2, 8: 0})

# ------------------------- DataLoader Factory ------------------------- #

def make_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
    workers_win: int = 0,
    workers_unix: Optional[int] = None,
    prefetch_factor: int = 2,
    persistent: Optional[bool] = None,
    collate_fn: Optional[Callable] = None,
) -> DataLoader:
    is_windows = os.name == "nt"
    n_workers = workers_win if is_windows else (workers_unix if workers_unix is not None else max(1, (os.cpu_count() or 2) - 1))
    if is_windows:
        n_workers = 0
    pin = device.type == "cuda"
    if persistent is None:
        persistent = bool(n_workers > 0)
    kwargs = dict(
        batch_size=batch_size, shuffle=shuffle, num_workers=n_workers,
        pin_memory=pin, persistent_workers=persistent, drop_last=True
    )
    if not is_windows and n_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    if collate_fn is not None:
        kwargs["collate_fn"] = collate_fn
    return DataLoader(dataset, **kwargs)

# ------------------------- Trainer ------------------------- #

class Trainer:
    def __init__(
        self,
        cfg: TrainConfig,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        logger: logging.Logger,
        *,
        class_weights: Optional[torch.Tensor] = None,
        ema: Optional[EMA] = None,
        device: Optional[torch.device] = None,
        rdrop_fn: Optional[Callable[[torch.Tensor, torch.Tensor, torch.Tensor, nn.Module, float], torch.Tensor]] = None,
        asymmetric_focal_loss: Optional[nn.Module] = None,
        logit_clamp: float = 50.0,
    ):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.logger = logger
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ema = ema
        self.class_weights = class_weights
        self.rdrop_fn = rdrop_fn
        self.asymmetric_focal_loss = asymmetric_focal_loss
        self.logit_clamp = logit_clamp

        self.model.to(self.device)
        self.scaler_enabled = (self.device.type == "cuda") and self.cfg.use_amp
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.scaler_enabled)
        except Exception:
            from torch.cuda.amp import GradScaler as CudaGradScaler  # type: ignore
            self.scaler = CudaGradScaler(enabled=self.scaler_enabled)

        set_seed(self.cfg.seed)
        for d in (self.cfg.ckpt_dir, self.cfg.logs_dir, self.cfg.plots_dir):
            Path(d).mkdir(parents=True, exist_ok=True)

        self.amp_was_disabled_this_epoch = False
        self.force_fp32_next_epoch = False

    def _selection_metric_name(self) -> str:
        return "auprc" if self.cfg.task == "binary" else "neg_val_loss"

    def _selection_score(self, val_loss: float, metrics: Optional[Dict[str, float]]) -> float:
        if self.cfg.task == "binary":
            return float(metrics.get("auprc", float("-inf")) if metrics is not None else float("-inf"))
        return float(-val_loss)

    def _checkpoint_payload(
        self,
        *,
        checkpoint_role: str,
        val_loss: Optional[float] = None,
        selection_score: Optional[float] = None,
        decision_threshold: Optional[float] = None,
        epoch_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model_state_dict": self.model.state_dict(),
            "task": self.cfg.task,
            "head_type": "classifier" if self.cfg.task == "binary" else "regressor",
            "checkpoint_role": checkpoint_role,
            "selection_metric": self._selection_metric_name(),
        }
        if epoch_idx is not None:
            payload["epoch"] = int(epoch_idx) + 1
        if val_loss is not None:
            payload["val_loss"] = float(val_loss)
        if selection_score is not None:
            payload["selection_score"] = float(selection_score)
        if decision_threshold is not None:
            payload["decision_threshold"] = float(decision_threshold)
        if self.cfg.checkpoint_extras:
            payload.update(dict(self.cfg.checkpoint_extras))
        return payload

    # ----------------- Checkpoint I/O ----------------- #

    def _ckpt_path(self, name: str) -> Path:
        return safe_path(Path(self.cfg.ckpt_dir) / name, self.cfg.base_dir)

    def _save_ckpt(self, name: str, payload: Dict[str, Any]) -> Path:
        p = self._ckpt_path(name)
        torch.save(payload, p)
        if self.cfg.verify_checkpoints:
            with open(str(p) + ".sha256", "w", encoding="utf-8") as f:
                f.write(sha256_file(p))
        return p

    def _load_ckpt(self, path: Path) -> Dict[str, Any]:
        path = safe_path(path, self.cfg.base_dir)
        if self.cfg.verify_checkpoints:
            sfile = Path(str(path) + ".sha256")
            if sfile.exists():
                expected = sfile.read_text().strip()
                actual = sha256_file(path)
                if expected != actual:
                    raise ValueError(f"Checkpoint hash mismatch for {path}")
        return torch.load(path, map_location=self.device, weights_only=True)  # PyTorch >=2.0

    # ----------------- Core Loops ----------------- #

    def _autocast(self, use_amp_now: bool):
        if use_amp_now and self.device.type == "cuda":
            dtype = torch.bfloat16 if self.cfg.amp_dtype == "bf16" else torch.float16
            return torch.autocast(device_type="cuda", dtype=dtype)
        from contextlib import nullcontext
        return nullcontext()

    def train_epoch(self, use_amp_now: bool, use_rdrop: bool, max_steps: Optional[int]) -> Tuple[float, int, bool]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        losses: List[float] = []
        skips, early_bail = 0, False
        accum = 0

        for step, batch in enumerate(self.train_loader, 1):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            targets = batch["target"].to(self.device)
            aux = batch.get("aux_features")
            if aux is not None:
                aux = aux.to(self.device)

            with self._autocast(use_amp_now):
                logits = self.model(input_ids, attention_mask, aux_features=aux)

            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4).clamp_(-self.logit_clamp, self.logit_clamp)

            if self.cfg.task == "regression":
                loss = self.criterion(logits.float(), targets.float()) / self.cfg.grad_accum_steps
            else:
                if use_rdrop and self.rdrop_fn is not None and self.cfg.loss_kind == "ce":
                    with self._autocast(use_amp_now):
                        logits2 = self.model(input_ids, attention_mask, aux_features=aux)
                    logits2 = torch.nan_to_num(logits2, nan=0.0, posinf=1e4, neginf=-1e4).clamp_(-self.logit_clamp, self.logit_clamp)
                    loss = self.rdrop_fn(logits, logits2, targets, self.criterion, alpha=0.5) / self.cfg.grad_accum_steps
                else:
                    loss = self.criterion(logits.float(), targets.long()) / self.cfg.grad_accum_steps

            if not torch.isfinite(loss):
                skips += 1
                if self.cfg.disable_amp_on_nan and use_amp_now:
                    self.amp_was_disabled_this_epoch = True
                # LR backoff on instability
                for g in self.optimizer.param_groups:
                    g["lr"] = max(g["lr"] * self.cfg.lr_reduce_factor, self.cfg.lr_floor)
                self.optimizer.zero_grad(set_to_none=True)
                if (skips / max(1, step)) > 0.50:
                    self.logger.warning("[TRAIN] >50%% batches skipped; bailing epoch.")
                    early_bail = True
                    break
                continue

            if self.scaler_enabled and use_amp_now:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            accum += 1

            if (accum % self.cfg.grad_accum_steps == 0) or (step == len(self.train_loader)):
                if self.scaler_enabled and use_amp_now:
                    self.scaler.unscale_(self.optimizer)
                clip_grad_norm_(self.model.parameters(), max_norm=self.cfg.clip_norm)
                self.optimizer.step()
                if self.cfg.use_ema and self.ema is not None:
                    self.ema.update(self.model)
                self.scheduler.step()
                if self.scaler_enabled and use_amp_now:
                    self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            losses.append(loss.item() * self.cfg.grad_accum_steps)

            if max_steps is not None and step >= max_steps:
                break

        avg = float(np.mean(losses)) if losses else float("nan")
        return avg, skips, early_bail

    def _eval_once(self, model: nn.Module, use_amp_now: bool, return_probs: bool, max_steps: Optional[int]):
        losses: List[float] = []
        skips = 0
        probs_all, labels_all = [], []

        model.eval()
        with torch.no_grad():
            for step, batch in enumerate(self.val_loader, 1):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                targets = batch["target"].to(self.device)
                aux = batch.get("aux_features")
                if aux is not None:
                    aux = aux.to(self.device)

                with self._autocast(use_amp_now):
                    logits = model(input_ids, attention_mask, aux_features=aux)
                logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4).clamp_(-self.logit_clamp, self.logit_clamp)

                if self.cfg.task == "regression":
                    loss = self.criterion(logits.float(), targets.float())
                else:
                    loss = self.criterion(logits.float(), targets.long())

                if not torch.isfinite(loss):
                    skips += 1
                    continue

                losses.append(loss.item())

                if self.cfg.task == "binary" and return_probs:
                    p = torch.softmax(logits.float(), dim=1)[:, 1]
                    probs_all.append(p.cpu().numpy())
                    labels_all.append(targets.cpu().numpy())

                if max_steps is not None and step >= max_steps:
                    break

        avg = float(np.mean(losses)) if losses else float("inf")
        metrics: Dict[str, float] = {}
        probs_np, labels_np = None, None
        if self.cfg.task == "binary" and probs_all:
            probs_np = np.concatenate(probs_all)
            labels_np = np.concatenate(labels_all).astype(int)
            try:
                metrics["auroc"] = roc_auc_score(labels_np, probs_np)
            except ValueError:
                metrics["auroc"] = float("nan")
            try:
                metrics["auprc"] = average_precision_score(labels_np, probs_np)
            except ValueError:
                metrics["auprc"] = float("nan")
            # Vectorized F1 sweep
            thrs = np.linspace(0.05, 0.95, 19)
            preds = (probs_np[:, None] >= thrs[None, :]).astype(int)
            tp = (preds & (labels_np[:, None] == 1)).sum(axis=0)
            fp = (preds & (labels_np[:, None] == 0)).sum(axis=0)
            fn = ((1 - preds) & (labels_np[:, None] == 1)).sum(axis=0)
            f1 = (2 * tp) / np.maximum(2 * tp + fp + fn, 1)
            best_idx = int(np.argmax(f1))
            metrics["best_f1"] = float(f1[best_idx])
            metrics["best_thr"] = float(thrs[best_idx])
        return avg, skips, metrics, probs_np, labels_np

    def _pick_precision_threshold(self, probs: np.ndarray, labels: np.ndarray) -> Tuple[float, float, float, float]:
        thrs = np.linspace(0.01, 0.99, 99)
        preds = (probs[:, None] >= thrs[None, :]).astype(int)
        tp = (preds & (labels[:, None] == 1)).sum(axis=0)
        fp = (preds & (labels[:, None] == 0)).sum(axis=0)
        fn = ((1 - preds) & (labels[:, None] == 1)).sum(axis=0)
        prec = tp / np.maximum(tp + fp, 1)
        rec = tp / np.maximum(tp + fn, 1)
        f1 = (2 * prec * rec) / np.maximum(prec + rec, 1e-12)
        mask = (prec >= self.cfg.target_precision) & (rec >= self.cfg.min_recall)
        if mask.any():
            idx = int(np.argmax(f1 * mask))
        else:
            idx = int(np.argmax(f1))
        return float(thrs[idx]), float(prec[idx]), float(rec[idx]), float(f1[idx])

    def fit(self, best_ckpt_name: str, last_ckpt_name: str) -> Dict[str, Any]:
        # Scheduler steps consistency
        steps_per_epoch_cap = self.cfg.train_max_steps_per_epoch or len(self.train_loader)
        eff = min(len(self.train_loader), steps_per_epoch_cap)
        total_steps = max(1, math.ceil(eff / max(1, self.cfg.grad_accum_steps)) * self.cfg.epochs)
        # Ensure scheduler configured accordingly (diagnostic)
        del total_steps  # informative; already configured by caller

        best_val_score = -float("inf")
        patience = 0
        global_best = best_val_score

        best_ckpt = self._ckpt_path(best_ckpt_name)
        last_ckpt = self._ckpt_path(last_ckpt_name)

        for epoch in range(self.cfg.epochs):
            use_amp_now = (not self.force_fp32_next_epoch) and (epoch >= self.cfg.amp_warmup_epochs) and self.cfg.use_amp and (not self.amp_was_disabled_this_epoch)
            use_rdrop = (self.cfg.loss_kind == "ce")

            # staged unfreezing
            freeze_depth = None
            for k in sorted(self.cfg.unfreeze_schedule.keys()):
                if epoch >= k:
                    freeze_depth = self.cfg.unfreeze_schedule[k]
            if hasattr(self.model, "set_freeze_depth") and freeze_depth is not None:
                # model must expose set_freeze_depth(int)
                self.model.set_freeze_depth(freeze_depth)

            self.logger.info(f"Epoch {epoch+1}/{self.cfg.epochs} | AMP={'on' if use_amp_now else 'off'} | freeze_depth={freeze_depth}")

            tr_loss, tr_skips, early_bail = self.train_epoch(use_amp_now, use_rdrop, self.cfg.train_max_steps_per_epoch)
            va_loss, va_skips, va_metrics, probs, labels = self._eval_once(self.model, use_amp_now, return_probs=True, max_steps=self.cfg.val_max_steps)

            decision_threshold = None
            if self.cfg.task == "binary" and probs is not None and labels is not None:
                thr, p, r, f1p = self._pick_precision_threshold(probs, labels)
                va_metrics["prec_thr"] = thr
                va_metrics["prec_at_thr"] = p
                va_metrics["rec_at_thr"] = r
                va_metrics["f1_at_thr"] = f1p
                self.logger.info(f"[VAL] AUROC={va_metrics.get('auroc', float('nan')):.4f} | AUPRC={va_metrics.get('auprc', float('nan')):.4f} | "
                                 f"Best-F1={va_metrics.get('best_f1', 0.0):.4f} @ {va_metrics.get('best_thr', 0.5):.2f} | "
                                 f"Prec-Opt thr={thr:.2f} -> P={p:.3f} R={r:.3f} F1={f1p:.3f}")
                decision_threshold = thr
            current_score = self._selection_score(va_loss, va_metrics)

            # Stability/rollback heuristic
            toxic_val = (not np.isfinite(va_loss)) or (va_skips == len(self.val_loader))
            toxic_train = early_bail or (tr_skips / max(1, len(self.train_loader)) > 0.50)
            if toxic_val or toxic_train:
                self.logger.warning("[TOXIC] Unstable epoch. LR backoff; forcing fp32 next epoch.")
                for g in self.optimizer.param_groups:
                    g["lr"] = max(g["lr"] * self.cfg.lr_reduce_factor, self.cfg.lr_floor)
                self.force_fp32_next_epoch = True
                self.amp_was_disabled_this_epoch = True
                if best_ckpt.exists():
                    try:
                        payload = self._load_ckpt(best_ckpt)
                        self.model.load_state_dict(payload["model_state_dict"])
                    except Exception as e:
                        self.logger.warning(f"[TOXIC] Rollback failed: {e}")
            else:
                # early stopping & best tracking
                improved = np.isfinite(current_score) and (current_score > best_val_score + 1e-6)
                if improved:
                    best_val_score = current_score
                    self._save_ckpt(
                        best_ckpt.name,
                        self._checkpoint_payload(
                            checkpoint_role="fold_best",
                            val_loss=va_loss,
                            selection_score=current_score,
                            decision_threshold=decision_threshold,
                            epoch_idx=epoch,
                        ),
                    )
                    patience = 0
                    if current_score > global_best + 1e-6:
                        global_best = current_score
                else:
                    patience += 1
                    if patience >= self.cfg.patience:
                        self.logger.info("Early stopping.")
                        break

            # always save last
            try:
                self._save_ckpt(
                    last_ckpt.name,
                    self._checkpoint_payload(
                        checkpoint_role="last",
                        val_loss=va_loss,
                        selection_score=current_score,
                        decision_threshold=decision_threshold,
                        epoch_idx=epoch,
                    ),
                )
            except Exception as e:
                self.logger.warning(f"Could not save last checkpoint: {e}")

        return {
            "best_score": float(global_best),
            "best_ckpt_path": str(best_ckpt),
            "config": asdict(self.cfg),
        }
