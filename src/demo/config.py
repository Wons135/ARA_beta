"""Demo settings only. Full-scale modules retain all original defaults."""
from dataclasses import dataclass


@dataclass(frozen=True)
class DemoConfig:
    seed: int = 42
    tfidf_features: int = 1500
    svd_components: int = 24
    xgb_estimators: int = 32
    xgb_max_depth: int = 3
    bert_epochs: int = 1
    bert_train_rows: int = 96
    bert_val_rows: int = 48
    bert_batch_size: int = 8
    bert_max_length: int = 64
    bert_base_lr: float = 1e-5
    bert_head_lr: float = 3e-4
    model_name: str = "distilbert-base-uncased"
    model_revision: str = "12040accade4e8a0f71eabdb258fecc2e7e948be"
    reference_date: str = "2023-10-01T00:00:00Z"
    rating_min_reviews: int = 3
    permutations: int = 20


CFG = DemoConfig()
