# Demo validation record

Validated on 2026-10-29, Windows x64, Python 3.11.0, AMD Ryzen 7 7700X,
about 32 GB physical RAM, **4 CPU threads**. Torch was **2.5.1+cpu**,
`torch.version.cuda` was `None`, and CUDA visibility was disabled. The installed
NVIDIA GPU was not used. NumPy was 1.26.4.

**These are reduced synthetic-data demo runs, not thesis results.**

## Measured execution

| Scenario | Pipeline time | Whole command |
|---|---:|---:|
| First workspace run, CPU environment already installed | 32.7 s | 41.4 s |
| Windows wrapper, cached resources, offline | 32.1 s | 40.5 s |
| Extracted ZIP, new environment installed automatically | 32.4 s | 185.6 s |
| Final Windows wrapper verification, offline | 32.7 s | 41.5 s |

The extracted-package installation reused pip's package cache and the existing
Hugging Face snapshot and VADER caches. This verifies automatic environment setup
and independence from the full dataset, not the network download time for a new
machine. A first model download is about 268 MB plus tokenizer files. Laptop CPU
speed and internet access will change these timings.

Final wrapper command: `run_demo.bat --offline`, with `DEMO_NO_PAUSE=1` for automated
capture. Verified process exit status: **0**. The equivalent recording command is
`python run_demo.py --offline`.

## Checks passed

- All 36 retained original Python source files matched their pre-cleanup SHA-256 hashes.
- `pip check` found no broken demo requirements.
- All nine existing weighted-rating unit tests passed.
- All retained and added Python files compiled.
- Train, validation and test IDs were disjoint; both binary/count gate classes and multi-vote examples existed in each split.
- Feature matrix rows and saved feature IDs aligned with their task CSVs; features and predictions were finite.
- Each combined count equalled gate probability times conditional positive count.
- Fine-tuning train/validation IDs stayed within the correct splits; no held-out IDs were used.
- Each of the four DistilBERT runs changed actual final-block pretrained weights; embeddings and the first five blocks were absent from the trainable-parameter checkpoint.
- Alpha zero reproduced arithmetic means; all weighted means stayed within each title's observed rating range.
- The packaged run had exactly the same metrics and weighted diagnostics as workspace/offline reruns.
- Extracted sample checksum matched its manifest after Git line-ending normalization.
- All generated files in the output manifest existed; five PNG/PDF figure pairs were rendered. Annotation contrast was checked visually.
- Staged submission scanning checked prohibited paths, large artifacts and credential patterns; no hits were found.
- The ZIP contained no environments, raw data, generated outputs, review pile, IDE/agent files or Git history.

No full-scale pipeline was run. Unix/macOS execution was not tested here.

## Current generated files

83 files from the current run, relative to `outputs/demo/`:

```text
checkpoints/baseline_binary.json
checkpoints/baseline_count.json
checkpoints/baseline_regression.json
checkpoints/bert_binary_config.json
checkpoints/bert_binary_trained_parameters.pt
checkpoints/bert_count_stage1_config.json
checkpoints/bert_count_stage1_trained_parameters.pt
checkpoints/bert_count_stage2_log1p_config.json
checkpoints/bert_count_stage2_log1p_trained_parameters.pt
checkpoints/bert_count_stage2_poisson_config.json
checkpoints/bert_count_stage2_poisson_trained_parameters.pt
checkpoints/xgb_binary.json
checkpoints/xgb_count_stage1.json
checkpoints/xgb_count_stage2_log1p.json
checkpoints/xgb_count_stage2_poisson.json
feature_models/aux_meta_binary.pkl
feature_models/aux_meta_regression.pkl
feature_models/aux_prep_regression.pkl
feature_models/scaler_binary.pkl
feature_models/scaler_regression.pkl
feature_models/svd_binary.pkl
feature_models/svd_regression.pkl
feature_models/tfidf_binary.pkl
feature_models/tfidf_regression.pkl
figures/binary_confusion_matrices.pdf
figures/binary_confusion_matrices.png
figures/binary_precision_recall.pdf
figures/binary_precision_recall.png
figures/count_actual_vs_predicted.pdf
figures/count_actual_vs_predicted.png
figures/rating_change_distribution.pdf
figures/rating_change_distribution.png
figures/weighting_alpha_sweep.pdf
figures/weighting_alpha_sweep.png
metrics_summary.csv
metrics_summary.json
predictions/baseline_binary_test.csv
predictions/baseline_count_test.csv
predictions/bert_binary_test.csv
predictions/bert_count_stage1_test.csv
predictions/bert_count_stage2_log1p_test.csv
predictions/bert_count_stage2_poisson_test.csv
predictions/xgb_binary_test.csv
predictions/xgb_count_stage1_test.csv
predictions/xgb_count_stage2_log1p_test.csv
predictions/xgb_count_stage2_poisson_test.csv
preprocessed/X_all_test_features_binary_svd.npy
preprocessed/X_test_feature_ids_binary.csv
preprocessed/X_test_feature_ids_regression.csv
preprocessed/X_test_features_binary_svd.npy
preprocessed/X_test_features_regression_svd.npy
preprocessed/X_train_feature_ids_binary.csv
preprocessed/X_train_feature_ids_regression.csv
preprocessed/X_train_features_binary_svd.npy
preprocessed/X_train_features_regression_svd.npy
preprocessed/X_val_feature_ids_binary.csv
preprocessed/X_val_feature_ids_regression.csv
preprocessed/X_val_features_binary_svd.npy
preprocessed/X_val_features_regression_svd.npy
preprocessed/binary_label_metadata.json
preprocessed/binary_label_test.csv
preprocessed/binary_label_train.csv
preprocessed/binary_label_val.csv
preprocessed/count_stage1_test.csv
preprocessed/count_stage1_train.csv
preprocessed/count_stage1_val.csv
preprocessed/count_stage2_test.csv
preprocessed/count_stage2_train.csv
preprocessed/count_stage2_val.csv
preprocessed/count_target_metadata.json
preprocessed/regression_score_test.csv
preprocessed/regression_score_train.csv
preprocessed/regression_score_val.csv
preprocessed/split_definition.csv
preprocessed/split_definition_metadata.json
run.log
summary.txt
weighted_rating/alpha_sweep.csv
weighted_rating/comparison.csv
weighted_rating/diagnostics.json
weighted_rating/per_item.csv
weighted_rating/review_level_inputs.csv
output_manifest.json
```
