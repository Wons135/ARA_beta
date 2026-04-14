from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

TaskT = Literal["regression", "binary"]


def _canonicalize(path: Union[str, Path]) -> Path:
    return Path(path).expanduser().resolve()


def _ensure_under(base: Optional[Path], p: Path) -> None:
    if base is None:
        return
    base = base.resolve()
    if not str(p).startswith(str(base)):
        raise ValueError(f"Unsafe path outside allowed root: {p}")


class ReviewDataset(Dataset):
    """
    A lightweight PyTorch Dataset that yields raw text + target (+ optional aux features).
    Tokenization is intentionally deferred to a batch collate_fn for performance.

    Parameters
    ----------
    data : Union[str, Path, pd.DataFrame]
        CSV path (read with pandas) or an in-memory DataFrame.
    task : {"regression","binary"}
        Controls label column and dtype.
    use_aux_features : bool
        If True, additionally returns 'aux_features' tensor in each item.
    aux_cols : Sequence[str]
        Auxiliary feature columns to extract when use_aux_features=True.
    allowed_root : Optional[Union[str, Path]]
        If set and `data` is a path, enforce canonicalization under this root (path hygiene).
    """

    _TASK_TO_LABEL = {"regression": ("regression_score", torch.float32),
                      "binary": ("binary_label", torch.long)}

    def __init__(
        self,
        data: Union[str, Path, pd.DataFrame],
        *,
        task: TaskT = "regression",
        label_col: Optional[str] = None,
        use_aux_features: bool = False,
        aux_cols: Sequence[str] = ("review_length", "sentiment_score", "is_verified"),
        allowed_root: Optional[Union[str, Path]] = None,
    ) -> None:
        if task not in self._TASK_TO_LABEL:
            raise ValueError("Unsupported task type. Use 'regression' or 'binary'.")
        self.task: TaskT = task
        self.use_aux_features = use_aux_features
        self.aux_cols: Tuple[str, ...] = tuple(aux_cols)
        self.label_col = str(label_col) if label_col is not None else None

        # Load dataframe (with minimal copies)
        if isinstance(data, (str, Path)):
            p = _canonicalize(data)
            _ensure_under(_canonicalize(allowed_root) if allowed_root else None, p)
            df = pd.read_csv(p)
        else:
            df = data

        # Schema checks
        resolved_label_col = self.label_col or self._TASK_TO_LABEL[task][0]
        required_cols = {"full_text", resolved_label_col}
        if self.use_aux_features:
            required_cols |= set(self.aux_cols)
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        # Normalize essential views (avoid extra lists to reduce RAM)
        self._texts = df["full_text"].astype(str)  # pandas Series view
        label_col, target_dtype_torch = resolved_label_col, self._TASK_TO_LABEL[task][1]
        self._target_torch_dtype = target_dtype_torch

        # Use NumPy views where possible; coerce dtype explicitly
        if task == "binary":
            self._targets = pd.to_numeric(df[label_col], errors="raise").astype("int64").to_numpy(copy=False)
        else:
            self._targets = pd.to_numeric(df[label_col], errors="raise").astype("float32").to_numpy(copy=False)

        # Aux features (as float32 matrix) — standardized later via provided stats
        self._aux: Optional[np.ndarray] = None
        if self.use_aux_features:
            aux_df = df.loc[:, list(self.aux_cols)].copy()
            if "is_verified" in aux_df.columns:
                aux_df["is_verified"] = aux_df["is_verified"].astype(float)
            self._aux = aux_df.to_numpy(dtype=np.float32, copy=False)

        # Standardization stats (set via setter)
        self._aux_means: Dict[str, float] = {}
        self._aux_stds: Dict[str, float] = {}

    # ---------------- Public API ---------------- #

    def set_aux_stats(self, means: Mapping[str, float], stds: Mapping[str, float]) -> None:
        """
        Provide per-column standardization stats for aux features.
        Columns must match `aux_cols`. Missing stats fall back to 0/1.
        """
        self._aux_means = {k: float(means.get(k, 0.0)) for k in self.aux_cols}
        self._aux_stds = {k: float(stds.get(k, 1.0)) for k in self.aux_cols}

    def __len__(self) -> int:
        return len(self._texts)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        # Raw text and target only; tokenization happens in the collate.
        text = self._texts.iloc[index]
        target_val = self._targets[index]
        item: Dict[str, Any] = {
            "text": text,
            "target": torch.tensor(target_val, dtype=self._target_torch_dtype),
        }

        if self.use_aux_features and self._aux is not None:
            raw_aux = self._aux[index].astype(np.float32, copy=True)

            # Column-wise transforms: log1p on 'review_length' if present as first column by default.
            # We do a name-based transform to be robust to column order.
            try:
                idx_rl = self.aux_cols.index("review_length")
                raw_aux[idx_rl] = np.log1p(max(float(raw_aux[idx_rl]), 0.0))
            except ValueError:
                # 'review_length' not present; ignore
                pass

            # Standardize if stats present
            if self._aux_means and self._aux_stds:
                for i, col in enumerate(self.aux_cols):
                    mean = self._aux_means.get(col, 0.0)
                    std = self._aux_stds.get(col, 1.0)
                    raw_aux[i] = (raw_aux[i] - mean) / (std + 1e-12)

            raw_aux = np.nan_to_num(raw_aux, nan=0.0, posinf=0.0, neginf=0.0)
            item["aux_features"] = torch.from_numpy(raw_aux)

        return item


# ---------------- Collate Factory ---------------- #

def make_collate(
    tokenizer,
    *,
    max_len: int = 256,
    pad_to_max_length: bool = False,
    return_aux: bool = True,
    task: TaskT = "regression",
):
    """
    Create a high-performance collate function that batch-tokenizes texts.

    Parameters
    ----------
    tokenizer : transformers.PreTrainedTokenizerBase
        Already-initialized tokenizer (use_fast=True recommended).
    max_len : int
        Max sequence length (applied with truncation).
    pad_to_max_length : bool
        If True, pads every batch to `max_len`; else dynamic padding per batch.
    return_aux : bool
        If False, drops aux_features from the batch output.
    task : {"regression","binary"}
        Controls dtype of stacked targets.

    Returns
    -------
    Callable[[List[Dict[str,Any]]], Dict[str, torch.Tensor]]
    """
    if task not in ("regression", "binary"):
        raise ValueError("task must be 'regression' or 'binary'.")

    def _collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [ex["text"] for ex in batch]
        enc = tokenizer(
            texts,
            add_special_tokens=True,
            max_length=max_len,
            padding=("max_length" if pad_to_max_length else True),
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        # Targets
        if task == "binary":
            targets = torch.stack([ex["target"].to(torch.long) for ex in batch])
        else:
            targets = torch.stack([ex["target"].to(torch.float32) for ex in batch])

        out: Dict[str, torch.Tensor] = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "target": targets,
        }

        # Optional aux
        if return_aux and "aux_features" in batch[0]:
            aux_list = [ex["aux_features"] for ex in batch]
            out["aux_features"] = torch.stack(aux_list)  # [B, d_aux]
        return out

    return _collate


# ---------------- Backward-Compatible Adapter (Deprecated) ---------------- #

class LegacyTokenizingReviewDataset(Dataset):
    """
    Deprecated adapter that mirrors the original per-item tokenization behavior.

    Notes
    -----
    - Preserves legacy output keys: input_ids, attention_mask, target, (optional) aux_features.
    - Prefer using `ReviewDataset` + `make_collate` for significantly better throughput.
    - This adapter will be removed after one minor release (90-day deprecation window).
    """
    def __init__(
        self,
        data: Union[str, Path, pd.DataFrame],
        tokenizer,
        *,
        max_len: int = 256,
        task: TaskT = "regression",
        label_col: Optional[str] = None,
        use_aux_features: bool = False,
        aux_cols: Sequence[str] = ("review_length", "sentiment_score", "is_verified"),
        allowed_root: Optional[Union[str, Path]] = None,
    ) -> None:
        self._inner = ReviewDataset(
            data,
            task=task,
            label_col=label_col,
            use_aux_features=use_aux_features,
            aux_cols=aux_cols,
            allowed_root=allowed_root,
        )
        self._tokenizer = tokenizer
        self._max_len = int(max_len)
        self._task = task

    def set_aux_stats(self, means: Mapping[str, float], stds: Mapping[str, float]) -> None:
        self._inner.set_aux_stats(means, stds)

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        ex = self._inner[index]
        enc = self._tokenizer.encode_plus(
            ex["text"],
            add_special_tokens=True,
            max_length=self._max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        out = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "target": ex["target"] if self._task == "binary" else ex["target"].to(torch.float32),
        }
        if "aux_features" in ex:
            out["aux_features"] = ex["aux_features"]
        return out
