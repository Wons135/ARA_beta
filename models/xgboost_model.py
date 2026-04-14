# models/xgb_model.py
from __future__ import annotations

import os
import hashlib
import logging
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import xgboost as xgb


def _has_cuda() -> bool:
    """Detect if the installed XGBoost build supports CUDA."""
    try:
        return bool(xgb.has_cuda_support())
    except Exception:
        return False


def _safe_path(path: Union[str, os.PathLike], base_dir: Optional[Union[str, os.PathLike]] = None) -> str:
    """
    Canonicalize `path` and, if base_dir is provided, ensure the result stays under base_dir.
    Creates parent directories as needed.
    """
    p = os.path.abspath(os.path.expanduser(str(path)))
    if base_dir is not None:
        base = os.path.abspath(os.path.expanduser(str(base_dir)))
        if not p.startswith(base + os.sep) and p != base:
            raise ValueError(f"Unsafe path outside base_dir: {p}")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


class XGBoostModel:
    """
    Hardened, reproducible wrapper around XGBoost (binary classification / regression).

    API compatibility:
    - Constructor signature preserved (task, **kwargs).
    - fit(X_train, y_train, X_val=None, y_val=None, early_stopping_rounds=50) preserved.
    - predict / predict_proba / save / load preserved.

    Enhancements:
    - Auto device fallback (GPU -> CPU) if CUDA is unavailable.
    - Deterministic runs via random_state; NumPy seeding inside fit (idempotent).
    - Optional auto-imbalance handling: scale_pos_weight computed in fit if not set.
    - Safe save/load with base_dir constraint and optional SHA-256 sidecar verification.
    - Structured logging instead of verbose prints.
    - Introspection helpers: get_config(), get_feature_importances().
    """

    def __init__(self, task: str = "binary", logger: Optional[logging.Logger] = None, **kwargs: Any):
        """
        Parameters
        ----------
        task : {"binary","regression"}
            Learning task.
        logger : logging.Logger, optional
            Injected logger. If None, a module logger is used.
        **kwargs :
            Passed to xgb.XGBClassifier / xgb.XGBRegressor (e.g., n_estimators, max_depth).
            Safe defaults are applied when not provided:
              - tree_method: "hist" (CPU) or "gpu_hist" (GPU build)
              - device: "cuda" if available, else "cpu"
              - predictor: "gpu_predictor" if CUDA, else "cpu_predictor"
              - random_state: 42 (can be overridden)
        """
        if task not in {"binary", "regression"}:
            raise ValueError("Unsupported task. Use 'binary' or 'regression'.")
        self.task = task
        self.logger = logger or logging.getLogger(__name__)

        # ----- Safe defaults with device auto-detect -----
        use_cuda = _has_cuda()
        default_params: Dict[str, Any] = {
            "tree_method": "gpu_hist" if use_cuda else "hist",
            "device": "cuda" if use_cuda else "cpu",
            "predictor": "gpu_predictor" if use_cuda else "cpu_predictor",
            "random_state": 42,
        }
        # Merge user kwargs last (user overrides win)
        base_params = {**default_params, **kwargs}

        # Objective & eval_metric defaults by task (user can override in kwargs)
        if task == "binary":
            obj = base_params.pop("objective", "binary:logistic")
            eval_metric = base_params.pop("eval_metric", "auc")  # better default than logloss for monitoring
            self.model = xgb.XGBClassifier(objective=obj, eval_metric=eval_metric, **base_params)
        else:
            obj = base_params.pop("objective", "reg:squarederror")
            eval_metric = base_params.pop("eval_metric", "rmse")
            self.model = xgb.XGBRegressor(objective=obj, eval_metric=eval_metric, **base_params)

    # ------------------------ Training ------------------------ #

    def _maybe_set_pos_weight(self, y_train: np.ndarray) -> None:
        """Auto-set scale_pos_weight if task=binary and param not already provided."""
        if self.task != "binary":
            return
        params = self.model.get_xgb_params()
        if "scale_pos_weight" in params:
            return
        # compute neg/pos safely
        y = np.asarray(y_train)
        pos = int((y == 1).sum())
        neg = int((y == 0).sum())
        if pos > 0 and neg > 0:
            spw = neg / max(1, pos)
            self.model.set_params(scale_pos_weight=spw)
            self.logger.info("Auto-set scale_pos_weight=%.4f (neg/pos)", spw)

    def fit(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        early_stopping_rounds: int = 50,
    ) -> None:
        """
        Fit the model. When X_val/y_val provided, uses early stopping.

        Notes
        -----
        - Seeds NumPy using model.random_state (if present) for reproducibility.
        - Replaces xgboost's verbose prints with model-native logging (set via logging config).
        """
        # Reproducibility: seed NumPy to align stochastic subsampling & column sampling
        try:
            rs = int(self.model.get_xgb_params().get("random_state", 42))
            np.random.seed(rs)
        except Exception:
            pass

        # Class imbalance (binary)
        self._maybe_set_pos_weight(y_train)

        if X_val is not None and y_val is not None:
            self.model.set_params(early_stopping_rounds=int(early_stopping_rounds))
            # Use built-in eval_set; keep verbose False and rely on external logging if desired
            self.model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        else:
            self.model.fit(X_train, y_train, verbose=False)

    # ------------------------ Inference ------------------------ #

    def predict(self, X):
        """Predict labels (regression scores or class labels)."""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Predict class probabilities for binary classification."""
        if self.task != "binary":
            raise RuntimeError("predict_proba is only available for binary classification.")
        return self.model.predict_proba(X)

    # ------------------------ I/O (safe) ------------------------ #

    def save(self, path: Union[str, os.PathLike], *, base_dir: Optional[Union[str, os.PathLike]] = None, with_sha256: bool = False) -> str:
        """
        Save model to JSON. If `base_dir` is provided, enforce path to reside under it.
        Returns the canonical path.
        """
        p = _safe_path(path, base_dir=base_dir)
        self.model.save_model(p)
        if with_sha256:
            digest = _sha256(p)
            with open(p + ".sha256", "w", encoding="utf-8") as f:
                f.write(digest)
        return p

    def load(self, path: Union[str, os.PathLike], *, base_dir: Optional[Union[str, os.PathLike]] = None, verify_sha256: bool = False) -> str:
        """
        Load model from JSON. If `verify_sha256` and a sidecar exists, verify integrity.
        Returns the canonical path loaded.
        """
        p = _safe_path(path, base_dir=base_dir)
        if verify_sha256 and os.path.exists(p + ".sha256"):
            expected = open(p + ".sha256", "r", encoding="utf-8").read().strip()
            actual = _sha256(p)
            if expected != actual:
                raise ValueError(f"SHA256 mismatch for model at {p}")
        self.model.load_model(p)
        return p

    # ------------------------ Introspection ------------------------ #

    def get_config(self) -> Dict[str, Any]:
        """Return a JSON-serializable snapshot of model parameters for reproducibility logs."""
        return {
            "task": self.task,
            "xgb_params": self.model.get_xgb_params(),
        }

    def get_feature_importances(self, importance_type: str = "gain") -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (feature_indices, importances) arrays. Works for both classifier/regressor.
        Note: indices correspond to the feature order provided to XGBoost.
        """
        booster = self.model.get_booster()
        scores = booster.get_score(importance_type=importance_type)
        if not scores:
            return np.array([], dtype=int), np.array([], dtype=float)
        # XGBoost keys are like "f0","f1",...
        indices = np.array([int(k[1:]) for k in scores.keys()], dtype=int)
        values = np.array(list(scores.values()), dtype=float)
        order = np.argsort(indices)
        return indices[order], values[order]
