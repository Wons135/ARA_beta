from __future__ import annotations

import argparse
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import xgboost as xgb
from tqdm import tqdm
from xgboost.callback import TrainingCallback

from xgboost_common import (
    ModeT,
    checkpoint_path,
    compute_binary_metrics,
    compute_count_metrics,
    default_model_metadata,
    get_mode_spec,
    load_split_bundle,
    prediction_csv_path,
    safe_path,
    save_checkpoint_metadata,
    save_feature_importance_plot,
    save_json,
    setup_logger,
    sha256_file,
    threshold_path,
    to_raw_count_predictions,
)


ROOT = Path(__file__).resolve().parents[1]


class TQDMCallback(TrainingCallback):
    def __init__(self, total: int) -> None:
        self.pbar = tqdm(total=total, desc="Boosting")

    def after_iteration(self, model, epoch: int, evals_log) -> bool:
        self.pbar.update(1)
        return False

    def after_training(self, model):
        self.pbar.close()
        return model


@dataclass
class XGBConfig:
    mode: ModeT = "binary"
    base_dir: str = str(ROOT / "datasets" / "preprocessed")
    out_dir: str = str(ROOT / "outputs")
    early_stop: int = 300
    verbose_eval: int = 50
    target_precision: float = 0.30
    min_recall: float = 0.15
    save_precision_thr: bool = True
    random_seed: int = 42
    xgb_params: dict | None = None

    def __post_init__(self) -> None:
        spec = get_mode_spec(self.mode)
        if self.xgb_params is None:
            use_cuda = torch.cuda.is_available()
            params = dict(
                n_estimators=5000,
                max_depth=7,
                learning_rate=0.04,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method="hist",
                device="cuda" if use_cuda else "cpu",
                predictor="gpu_predictor" if use_cuda else "cpu_predictor",
                reg_lambda=1.5,
                reg_alpha=0.0,
                random_state=self.random_seed,
            )
            if spec.role == "classifier":
                params["objective"] = spec.default_objective
                params["eval_metric"] = spec.default_eval_metric
            else:
                params["objective"] = spec.default_objective
                params["eval_metric"] = spec.default_eval_metric
            self.xgb_params = params


class XGBTrainer:
    def __init__(self, cfg: XGBConfig):
        self.cfg = cfg
        self.spec = get_mode_spec(cfg.mode)
        self.logger = setup_logger(f"xgb_train_{cfg.mode}")
        np.random.seed(cfg.random_seed)
        xgb.set_config(verbosity=0)

        log_path = safe_path(cfg.out_dir, "logs", f"{self.spec.checkpoint_stem}.log")
        if not any(
            isinstance(handler, logging.FileHandler) and Path(handler.baseFilename).resolve() == Path(log_path).resolve()
            for handler in self.logger.handlers
        ):
            file_handler = xgb_train_file_handler(log_path)
            self.logger.addHandler(file_handler)

    def load_data(self):
        self.logger.info("Loading %s training/validation data", self.cfg.mode)
        train_bundle = load_split_bundle(self.cfg.base_dir, "train", self.cfg.mode, self.logger)
        val_bundle = load_split_bundle(self.cfg.base_dir, "val", self.cfg.mode, self.logger)

        if train_bundle.X.shape[0] != train_bundle.y.shape[0]:
            raise RuntimeError("Train feature/label row mismatch.")
        if val_bundle.X.shape[0] != val_bundle.y.shape[0]:
            raise RuntimeError("Validation feature/label row mismatch.")
        return train_bundle, val_bundle

    def _build_model(self, y_train: np.ndarray):
        params = dict(self.cfg.xgb_params or {})
        callbacks = [TQDMCallback(total=int(params["n_estimators"]))]

        if self.spec.role == "classifier":
            pos = int((y_train == 1).sum())
            neg = int((y_train == 0).sum())
            params["scale_pos_weight"] = neg / max(1, pos)
            model = xgb.XGBClassifier(**params)
            self.logger.info(
                "Classifier setup | positives=%d negatives=%d scale_pos_weight=%.4f",
                pos,
                neg,
                float(params["scale_pos_weight"]),
            )
        else:
            model = xgb.XGBRegressor(**params)
            self.logger.info("Regressor setup | objective=%s", params.get("objective"))

        model.set_params(
            early_stopping_rounds=self.cfg.early_stop,
            callbacks=callbacks,
        )
        return model

    def train(self, train_bundle, val_bundle):
        model = self._build_model(train_bundle.y)
        self.logger.info("Training mode=%s", self.cfg.mode)
        start = time.time()
        model.fit(
            train_bundle.X,
            train_bundle.y,
            eval_set=[(val_bundle.X, val_bundle.y)],
            verbose=self.cfg.verbose_eval,
        )
        elapsed = (time.time() - start) / 60.0
        self.logger.info("Training complete in %.2f min | best_iteration=%s", elapsed, getattr(model, "best_iteration", None))
        self.model = model
        return model

    def evaluate(self, val_bundle):
        if self.spec.role == "classifier":
            probs = self.model.predict_proba(val_bundle.X)[:, 1]
            metrics = compute_binary_metrics(
                val_bundle.y.astype(int),
                probs,
                target_precision=self.cfg.target_precision,
                min_recall=self.cfg.min_recall,
            )
            self.val_predictions = probs
            self.logger.info(
                "[VAL] AUPRC=%.4f | AUROC=%.4f | F1@0.5=%.4f | F1*=%.4f @ thr=%.4f",
                metrics["auprc"],
                metrics["auroc"],
                metrics["f1_05"],
                metrics["f1_opt"],
                metrics["thr_opt"],
            )
        else:
            raw_preds = self.model.predict(val_bundle.X)
            preds = to_raw_count_predictions(raw_preds, self.spec)
            metrics = compute_count_metrics(
                val_bundle.y_raw.astype(float),
                preds,
                positive_mask=val_bundle.y_raw.astype(float) > 0,
            )
            self.val_predictions = preds
            self.val_model_outputs = raw_preds
            self.logger.info(
                "[VAL] RMSE=%.4f | MAE=%.4f | R2=%.4f | PosRMSE=%.4f | PosMAE=%.4f",
                metrics["rmse"],
                metrics["mae"],
                metrics["r2"],
                metrics["positive_only_rmse"],
                metrics["positive_only_mae"],
            )
        return metrics

    def save_artifacts(self, train_bundle, val_bundle, metrics):
        model_path = checkpoint_path(self.cfg.out_dir, self.cfg.mode)
        self.model.save_model(model_path)
        model_sha = sha256_file(model_path)
        self.logger.info("Saved model -> %s (SHA256=%s...)", model_path, model_sha[:8])

        fi_path = save_feature_importance_plot(self.model, self.cfg.out_dir, self.spec.checkpoint_stem)
        self.logger.info("Saved feature importance -> %s", fi_path)

        metadata = {
            "config": asdict(self.cfg),
            "mode": default_model_metadata(self.cfg.mode),
            "model_path": model_path,
            "model_sha256": model_sha,
            "feature_importance_plot": fi_path,
            "rows": {
                "train": int(train_bundle.frame.shape[0]),
                "val": int(val_bundle.frame.shape[0]),
            },
            "metrics_val": metrics,
        }
        meta_path = save_checkpoint_metadata(self.cfg.out_dir, self.cfg.mode, metadata)
        self.logger.info("Saved model metadata -> %s", meta_path)

        if self.spec.role == "classifier" and self.cfg.save_precision_thr:
            thr_payload = {
                "mode": self.cfg.mode,
                "threshold": float(metrics["thr_opt"]),
                "precision": float(metrics["prec_opt"]),
                "recall": float(metrics["rec_opt"]),
                "f1": float(metrics["f1_opt"]),
                "auprc": float(metrics["auprc"]),
                "auroc": float(metrics["auroc"]),
            }
            thr_path = threshold_path(self.cfg.out_dir, self.cfg.mode)
            save_json(thr_path, thr_payload)
            self.logger.info("Saved threshold metadata -> %s", thr_path)

        pred_path = prediction_csv_path(self.cfg.out_dir, self.cfg.mode, "val")
        pred_df = val_bundle.frame.loc[:, ["review_id"]].copy()
        if self.spec.role == "classifier":
            pred_df["true_label"] = val_bundle.y.astype(int)
            pred_df["prob_pos"] = self.val_predictions
            pred_df["pred_05"] = (self.val_predictions >= 0.5).astype(int)
            pred_df["pred_opt"] = (self.val_predictions >= metrics["thr_opt"]).astype(int)
        else:
            pred_df["y_true"] = val_bundle.y_raw.astype(float)
            pred_df["y_pred"] = self.val_predictions
            pred_df["model_output"] = self.val_model_outputs
        pred_df.to_csv(pred_path, index=False)
        self.logger.info("Saved validation predictions -> %s", pred_path)


def xgb_train_file_handler(log_path: str):
    handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    return handler


def parse_args() -> XGBConfig:
    parser = argparse.ArgumentParser(description="Train XGBoost models for binary and count tasks.")
    parser.add_argument(
        "--mode",
        choices=["binary", "count_stage1", "count_stage2_poisson", "count_stage2_log1p"],
        default="binary",
    )
    parser.add_argument("--base-dir", default=str(ROOT / "datasets" / "preprocessed"))
    parser.add_argument("--out-dir", default=str(ROOT / "outputs"))
    parser.add_argument("--early-stop", type=int, default=300)
    parser.add_argument("--verbose-eval", type=int, default=50)
    parser.add_argument("--target-precision", type=float, default=0.30)
    parser.add_argument("--min-recall", type=float, default=0.15)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    return XGBConfig(
        mode=args.mode,
        base_dir=args.base_dir,
        out_dir=args.out_dir,
        early_stop=args.early_stop,
        verbose_eval=args.verbose_eval,
        target_precision=args.target_precision,
        min_recall=args.min_recall,
        random_seed=args.random_seed,
    )


def main() -> None:
    cfg = parse_args()
    trainer = XGBTrainer(cfg)
    train_bundle, val_bundle = trainer.load_data()
    trainer.train(train_bundle, val_bundle)
    metrics = trainer.evaluate(val_bundle)
    trainer.save_artifacts(train_bundle, val_bundle, metrics)


if __name__ == "__main__":
    main()
