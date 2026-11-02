# Full thesis reproduction

The CPU demo is additive. Existing source files in `src/` (except new `src/demo/`),
`models/`, `training/`, and the retained experiments have not been edited.
Full-scale defaults and seeds remain intact. The demo changes only its own process
configuration and writes only below `outputs/demo/`.

## Environment and provenance

Use **Python 3.11.0** and `requirements.txt` in an environment separate from
`.venv-demo`. The full environment pins torch **2.5.1+cu121**, transformers 4.41.1,
XGBoost 2.0.3, scikit-learn 1.4.2, pandas 2.2.2, SciPy 1.16.0, NLTK 3.9.1,
joblib 1.5.1, and matplotlib 3.9.0. The torch index is included in requirements.
The original machine-wide package freeze is preserved locally in
`_to_review/original_requirements.txt`; unrelated cloud clients and type stubs were
left out of the submission's full requirements.

**Known NumPy discrepancy:** the intended requirements pin is **1.26.4**, but the
executed local environment and saved `outputs/weighted_rating/run_info.json` record
**2.4.4**. This discrepancy is documented rather than presented as an exact match.
For historical environment reconstruction, create a separate copy of the full
requirements and change its NumPy pin to 2.4.4; preserve that environment's resolved
versions. The demo was tested with 1.26.4. No full-scale rerun was performed during
submission preparation, and neither environment guarantees bitwise reproduction
across hardware and library builds.

Example setup (Windows; use `bin/python` instead of `Scripts/python.exe` on Unix):

```powershell
py -3.11 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

Activate that environment for the commands below. CUDA execution requires a
compatible NVIDIA GPU and driver. The audited workstation has an AMD Ryzen 7 7700X
and about 32 GB RAM; the exact original training hardware/CLI history is not fully
recorded in the source, so these are not asserted as historical training specifications.
Save model/checkpoint metadata and environment versions when reproducing a run.

## Data and cost

Download `Movies_and_TV` review JSONL and its metadata from
[Amazon Reviews 2023](https://amazon-reviews-2023.github.io/main.html).
Extract them as `datasets/Movies_and_TV.jsonl` and
`datasets/meta_Movies_and_TV.jsonl`. These legacy paths are retained deliberately;
`datasets/raw/` is also ignored for optional local storage.

The thesis corpus has approximately 11 million reviews. The audited workspace
contained about 90 GiB of raw/preprocessed data and 25 GiB of outputs. Allow more
than 120 GiB of working disk space, with extra room for intermediate files.
The loader materializes the corpus in RAM and intermediate copies can be large.
The original settings use up to 5,000 XGBoost trees and 10 DistilBERT epochs per
mode. Budget substantial GPU time (potentially hours/days, hardware dependent);
an end-to-end full-run runtime was not measured in this cleanup. This is not the
reviewer's quick start.

## Existing full-scale commands

Run from the repository root. The scripts use direct file execution and some
sibling imports, so use the forms below rather than `python -m training...`.
The seed remains **42**, including `random_state=42` for splitting and SVD.

```text
python src/data_preprocessing.py
python src/feature_extraction.py --task binary --headless-plots
python src/feature_extraction.py --task regression --headless-plots

python training/train_baseline.py --mode binary
python training/train_baseline.py --mode regression
python training/evaluate_baseline.py --mode binary --split test
python training/evaluate_baseline.py --mode regression --split test
```

For each mode `binary`, `count_stage1`, `count_stage2_log1p`, and
`count_stage2_poisson`, substitute its name for `MODE`:

```text
python training/train_xgboost.py --mode MODE
python training/train_bert_refactored.py --mode MODE
python training/evaluate_bert.py --mode MODE --split test
```

```text
python training/evaluate_xgb_binary.py --split test
python training/evaluate_xgb_count_stage1.py --split test
python training/evaluate_xgb_count_stage2.py --mode count_stage2_log1p --split test
python training/evaluate_xgb_count_stage2.py --mode count_stage2_poisson --split test
python training/evaluate_xgb_count_combined.py --stage2-mode count_stage2_log1p --split test
python training/evaluate_xgb_count_combined.py --stage2-mode count_stage2_poisson --split test
python training/evaluate_bert_count_combined.py --stage2-mode count_stage2_log1p --split test
python training/evaluate_bert_count_combined.py --stage2-mode count_stage2_poisson --split test
python training/generate_main_figures.py --split test
python experiments/weighted_rating/run_all.py
```

The weighted-rating runner expects the historical evaluation layout, inferred
gray-band probabilities, and strict saved coverage checks. Its settings are in
`experiments/weighted_rating/config.py`: minimum 20 reviews per title, 100
permutations, alpha 0.5/1.0, plus the existing alpha sweep and minimum-50 sensitivity.
The demo instead uses minimum 3 reviews and 20 permutations on held-out sample rows.
Full outputs stay in `outputs/`; demo outputs stay in `outputs/demo/`.

Additional XGBoost objectives remain available in `training/train_xgboost.py --help`.
`experiments/thesis_figures/make_figures.py` redraws saved thesis figures and checks
rounded historical metrics before writing; it is not a replacement for training.

The full preprocessing code uses the current date for review age and computes the
cap on the loaded corpus. This existing behavior is preserved. Exact historical
reproduction needs the original data exports, checkpoints, thresholds, CLI choices,
and environment provenance; the submitted code cannot supply missing historical
run metadata. The demo fixes its reference date to 2023-10-01 for repeatability.
