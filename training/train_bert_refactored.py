from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.bert_model import BERTClassifier, BERTRegressor
from src.dataset import ReviewDataset, make_collate
from src.utils import AsymmetricFocalLoss, rdrop_ce_loss
from training.bert_common import (
    BertModeT,
    apply_target_transform,
    checkpoint_path,
    fit_target_transform,
    get_mode_spec,
    load_mode_dataframe,
    setup_logger,
)
from training.trainer import EMA, TrainConfig, Trainer, make_dataloader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BERT baselines for binary and count-helpfulness tasks.")
    parser.add_argument(
        "--mode",
        choices=["binary", "count_stage1", "count_stage2_log1p", "count_stage2_poisson"],
        default="binary",
    )
    parser.add_argument("--base-dir", default=str(ROOT / "datasets" / "preprocessed"))
    parser.add_argument("--out-dir", default=str(ROOT / "outputs"))
    parser.add_argument("--tokenizer-name", default="distilbert-base-uncased")
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--base-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--head-weight-decay", type=float, default=0.10)
    parser.add_argument("--warmup-pct", type=float, default=0.10)
    parser.add_argument("--dropout-rate", type=float, default=0.7)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-precision", type=float, default=0.30)
    parser.add_argument("--min-recall", type=float, default=0.15)
    parser.add_argument("--train-max-steps-per-epoch", type=int, default=None)
    parser.add_argument("--val-max-steps", type=int, default=None)
    parser.add_argument("--disable-aux", action="store_true")
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--freeze-init", type=int, default=4)
    parser.add_argument("--use-mean-pooling", action="store_true", default=True)
    parser.add_argument("--no-mean-pooling", dest="use_mean_pooling", action="store_false")
    return parser.parse_args()


def compute_aux_stats(train_df, aux_cols: tuple[str, ...]) -> tuple[dict[str, float], dict[str, float]]:
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for col in aux_cols:
        values = train_df[col].astype(np.float32).to_numpy()
        if col == "review_length":
            values = np.log1p(np.maximum(values, 0.0))
        means[col] = float(np.mean(values))
        std = float(np.std(values))
        stds[col] = std if std >= 1e-12 else 1.0
    return means, stds


def build_param_groups(model: nn.Module, cfg: TrainConfig) -> list[dict[str, object]]:
    groups = [
        {"params": [], "lr": cfg.base_lr, "weight_decay": cfg.weight_decay},
        {"params": [], "lr": cfg.head_lr, "weight_decay": cfg.head_weight_decay},
    ]
    for name, param in model.named_parameters():
        if any(key in name for key in ("classifier", "regressor", "head")):
            groups[1]["params"].append(param)
        else:
            groups[0]["params"].append(param)
    return groups


def build_model_and_loss(
    mode: BertModeT,
    cfg: TrainConfig,
    *,
    device: torch.device,
    aux_feat_dim: int,
    tokenizer_name: str,
    dropout_rate: float,
    use_mean_pooling: bool,
    freeze_init: int,
    gradient_checkpointing: bool,
    train_df,
    label_column: str,
) -> tuple[nn.Module, nn.Module, torch.Tensor | None, str]:
    spec = get_mode_spec(mode)
    common_kwargs = dict(
        model_name=tokenizer_name,
        dropout_rate=dropout_rate,
        use_mean_pooling=use_mean_pooling,
        aux_feat_dim=aux_feat_dim,
        freeze_init=freeze_init,
        gradient_checkpointing=gradient_checkpointing,
    )

    if spec.head_task == "binary":
        model = BERTClassifier(**common_kwargs)
        labels = train_df[label_column].to_numpy(dtype=int)
        pos = int((labels == 1).sum())
        neg = int((labels == 0).sum())
        pos_weight = neg / max(1, pos) if pos > 0 else 1.0
        class_weights = torch.tensor([1.0, pos_weight], dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=cfg.ce_label_smooth)
        return model, criterion, class_weights, "cross_entropy"

    model = BERTRegressor(**common_kwargs)
    if mode == "count_stage2_poisson":
        criterion = nn.PoissonNLLLoss(log_input=True, full=False)
        return model, criterion, None, "poisson_nll"
    criterion = nn.SmoothL1Loss()
    return model, criterion, None, "smooth_l1"


def main() -> None:
    args = parse_args()
    mode: BertModeT = args.mode
    spec = get_mode_spec(mode)

    out_dir = Path(args.out_dir)
    logger = setup_logger("train_bert_refactored", out_dir / "logs" / "train_bert.log")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training mode=%s on device=%s", mode, device)

    aux_cols = ("review_length", "sentiment_score", "is_verified")
    use_aux = not args.disable_aux

    cfg = TrainConfig(
        task=spec.head_task,
        epochs=args.epochs,
        max_len=args.max_len,
        batch_size=args.batch_size,
        base_lr=args.base_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        head_weight_decay=args.head_weight_decay,
        warmup_pct=args.warmup_pct,
        patience=args.patience,
        aux_feat_dim=(len(aux_cols) if use_aux else 0),
        use_ema=args.use_ema,
        freeze_init=args.freeze_init,
        train_max_steps_per_epoch=args.train_max_steps_per_epoch,
        val_max_steps=args.val_max_steps,
        target_precision=args.target_precision,
        min_recall=args.min_recall,
        base_dir=str(out_dir),
        ckpt_dir=str(out_dir / "checkpoints"),
        logs_dir=str(out_dir / "logs"),
        plots_dir=str(out_dir / "plots"),
        dataset_root=str(Path(args.base_dir).resolve()),
        seed=args.seed,
    )

    train_df = load_mode_dataframe(args.base_dir, mode, "train")
    val_df = load_mode_dataframe(args.base_dir, mode, "val")
    logger.info("Loaded %d train rows and %d val rows", len(train_df), len(val_df))

    transform_metadata: dict[str, object] = {
        "mode": mode,
        "head_task": spec.head_task,
        "label_column": spec.label_column,
        "raw_target_column": spec.raw_target_column,
        "dataset_prefix": spec.dataset_prefix,
        "target_transform": {
            "name": "identity",
            "source_column": spec.label_column,
            "invert_prediction": "identity",
            "y_mean": None,
            "y_std": None,
        },
    }
    if spec.head_task == "regression":
        train_df, transform_metadata = fit_target_transform(train_df, spec)
        val_df = apply_target_transform(val_df, spec, transform_metadata["target_transform"])

    aux_means: dict[str, float] = {}
    aux_stds: dict[str, float] = {}
    if use_aux:
        aux_means, aux_stds = compute_aux_stats(train_df, aux_cols)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    collate = make_collate(
        tokenizer,
        max_len=cfg.max_len,
        task=spec.head_task,
        pad_to_max_length=False,
        return_aux=use_aux,
    )

    ds_train = ReviewDataset(
        train_df,
        task=spec.head_task,
        label_col=spec.label_column,
        use_aux_features=use_aux,
        aux_cols=aux_cols,
    )
    ds_val = ReviewDataset(
        val_df,
        task=spec.head_task,
        label_col=spec.label_column,
        use_aux_features=use_aux,
        aux_cols=aux_cols,
    )
    if use_aux:
        ds_train.set_aux_stats(aux_means, aux_stds)
        ds_val.set_aux_stats(aux_means, aux_stds)

    train_loader = make_dataloader(
        ds_train,
        batch_size=cfg.batch_size,
        shuffle=True,
        device=device,
        collate_fn=collate,
    )
    val_loader = make_dataloader(
        ds_val,
        batch_size=cfg.batch_size,
        shuffle=False,
        device=device,
        collate_fn=collate,
    )

    model, criterion, class_weights, loss_name = build_model_and_loss(
        mode,
        cfg,
        device=device,
        aux_feat_dim=cfg.aux_feat_dim,
        tokenizer_name=args.tokenizer_name,
        dropout_rate=args.dropout_rate,
        use_mean_pooling=args.use_mean_pooling,
        freeze_init=args.freeze_init,
        gradient_checkpointing=args.gradient_checkpointing,
        train_df=train_df,
        label_column=spec.label_column,
    )

    optimizer = torch.optim.AdamW(build_param_groups(model, cfg), eps=1e-8, betas=(0.9, 0.999))
    steps_per_epoch_cap = cfg.train_max_steps_per_epoch or len(train_loader)
    effective_steps = min(len(train_loader), steps_per_epoch_cap)
    total_optim_steps = max(1, math.ceil(effective_steps / max(1, cfg.grad_accum_steps)) * cfg.epochs)
    warmup_steps = int(cfg.warmup_pct * total_optim_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optim_steps,
    )
    ema = EMA(model, decay=cfg.ema_decay) if cfg.use_ema else None

    target_transform = transform_metadata["target_transform"]
    checkpoint_extras = {
        "mode": mode,
        "label_column": spec.label_column,
        "raw_target_column": spec.raw_target_column,
        "dataset_prefix": spec.dataset_prefix,
        "tokenizer_name": args.tokenizer_name,
        "model_name": args.tokenizer_name,
        "max_len": cfg.max_len,
        "use_aux_features": use_aux,
        "aux_columns": list(aux_cols),
        "aux_means": aux_means,
        "aux_stds": aux_stds,
        "target_transform": target_transform,
        "y_mean": target_transform.get("y_mean"),
        "y_std": target_transform.get("y_std"),
        "loss_name": loss_name,
        "dropout_rate": args.dropout_rate,
        "use_mean_pooling": args.use_mean_pooling,
        "gradient_checkpointing": args.gradient_checkpointing,
    }

    trainer = Trainer(
        cfg=replace(cfg, checkpoint_extras=checkpoint_extras),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        logger=logger,
        class_weights=class_weights,
        ema=ema,
        device=device,
        rdrop_fn=rdrop_ce_loss if spec.head_task == "binary" else None,
        asymmetric_focal_loss=AsymmetricFocalLoss(alpha_pos=0.5, gamma_pos=0.0, gamma_neg=2.0)
        if False
        else None,
        logit_clamp=50.0,
    )

    best_name = Path(checkpoint_path(out_dir, mode, "best")).name
    last_name = Path(checkpoint_path(out_dir, mode, "last")).name
    result = trainer.fit(best_ckpt_name=best_name, last_ckpt_name=last_name)

    selection_metric = "AUPRC" if spec.head_task == "binary" else "negative validation loss"
    logger.info("Finished mode=%s | best %s=%.6f", mode, selection_metric, result["best_score"])
    logger.info("Best checkpoint -> %s", checkpoint_path(out_dir, mode, "best"))
    logger.info("Last checkpoint -> %s", checkpoint_path(out_dir, mode, "last"))


if __name__ == "__main__":
    main()
