# Enhanced media rating estimation using machine learning

Code accompanying the completed master's thesis. The research estimates Amazon
review helpfulness with baselines, XGBoost over TF-IDF/SVD and metadata, and
DistilBERT, then compares arithmetic ratings with helpfulness-weighted ratings.
This submission includes the original full-scale scripts and a separate CPU demo.
**Demo numbers differ from the thesis:** reduced synthetic data, trees, sequence
length, trainable layers, and epochs demonstrate execution and the same kinds of
outputs, not the reported research metrics.

## Reviewer's quick start — one-click CPU demo

Install **Python 3.11** first. Extract the submission ZIP into a writable directory.
On Windows, **double-click `run_demo.bat`**. From a terminal, use:

```powershell
python run_demo.py
```

Windows can also use `py -3.11 run_demo.py`; on Unix use `sh run_demo.sh` (or
`python3.11 run_demo.py`). The launcher creates `.venv-demo` and installs the CPU
requirements automatically. No CUDA or full review dataset is needed.

Expect roughly **1–3 minutes on a laptop CPU after setup**. On the tested Windows
Ryzen 7 7700X, four CPU threads, the pipeline took **33 seconds**, about **41 seconds**
including launcher checks. Initial installation and the roughly 268 MB DistilBERT
download take extra time; VADER may also download once. These resources need
internet on first use, with no account or token required.

The run preprocesses → builds features → trains baselines, XGBoost and DistilBERT
(binary plus two-stage log1p and Poisson count tasks) → evaluates → compares weighted
ratings → saves figures. Open:

- `outputs/demo/summary.txt` — readable results and measured runtime.
- `outputs/demo/metrics_summary.json` / `.csv` — metrics, configuration and provenance.
- `outputs/demo/figures/` — five PNG/PDF figure pairs.
- `outputs/demo/weighted_rating/` — rating/ranking comparisons and alpha sweep.
- `outputs/demo/output_manifest.json` — complete generated file list.

**Screen recording:** prepare the environment once online, then run:

```powershell
python run_demo.py --offline
```

The reviewer can run that exact command after first online setup too. It uses the
same pipeline, default settings and bundled input, retraining all demo models.
See [demo details and troubleshooting](docs/DEMO.md).

## Repository layout

```text
run_demo.py / run_demo.bat / run_demo.sh   CPU quick start
requirements-demo.txt                    pinned CPU demo environment
requirements.txt                         pinned full CUDA environment
src/                                     thesis preprocessing, features, datasets
  demo/                                  additive orchestration and quick settings
models/                                  original DistilBERT and XGBoost models
training/                                thesis training/evaluation/plot scripts
experiments/weighted_rating/              thesis rating experiment and tests
experiments/thesis_figures/               saved-result verification and figure redraw
scripts/                                 sample generator and submission packager
datasets/sample/                         bundled synthetic reviews and source note
docs/                                    demo and reproduction instructions
outputs/                                 generated results (ignored)
```

Local `_to_review/`, raw/preprocessed datasets, environments, IDE settings and model
caches are excluded from the submission. The thesis manuscript goes to the university
archive separately.

## Requirements and setup: demo versus full run

| | CPU demo | Full thesis-scale path |
|---|---|---|
| Python | 3.11 (tested 3.11.0) | 3.11.0 |
| Install | Automatic isolated `.venv-demo` | Separate environment, `pip install -r requirements.txt` |
| Torch | 2.5.1+cpu on Windows/Linux; native macOS 2.5.1 | 2.5.1+cu121, CUDA 12.1 index included |
| Data | 1,024 bundled synthetic reviews | Approximately 11 million Amazon reviews |
| Compute | CPU, 4 threads; optional `--threads 2` | Compatible NVIDIA GPU/driver; substantial RAM and disk |
| Seed | 42 | Original defaults, including `random_state=42` |

Both paths pin transformers 4.41.1, XGBoost 2.0.3, scikit-learn 1.4.2, pandas 2.2.2,
SciPy 1.16.0, NLTK 3.9.1, joblib 1.5.1, and matplotlib 3.9.0.
**NumPy discrepancy:** intended pin **1.26.4**, but the recorded executed research
environment had **2.4.4**. The demo uses 1.26.4. See the
[reproduction notes](docs/REPRODUCIBILITY.md) before comparing numbers.
Windows x64 is tested; Unix wrappers and native macOS CPU installation are provided
but were not executed on those operating systems.

## Data

The full thesis data is public **Amazon Reviews 2023**, `Movies_and_TV`, from
[McAuley Lab](https://amazon-reviews-2023.github.io/main.html)
([Hugging Face mirror](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)).
Full-run files belong at `datasets/Movies_and_TV.jsonl` and
`datasets/meta_Movies_and_TV.jsonl`; generated exports go to `datasets/preprocessed/`.

The demo never downloads review data. Its bundled records are **original synthetic
movie reviews in the Amazon field schema, not real Amazon excerpts**. The source
maintainers do not assign a redistribution licence to Amazon review text, so that
text is not bundled. The [sample README](datasets/sample/README.md) provides
provenance, permission for the original synthetic sample, the Amazon source/citation,
and rebuild instructions.

## Reproduce the full thesis-scale results

Follow [the complete command sequence](docs/REPRODUCIBILITY.md). It retains the
existing preprocessing, task definitions, seed 42, feature settings and model
defaults. Full settings live in `src/data_preprocessing.py`,
`src/feature_extraction.py`, `training/train_xgboost.py`,
`training/train_bert_refactored.py`, and `experiments/weighted_rating/config.py`.

The full run uses up to 5,000 XGBoost trees and 10 DistilBERT epochs per mode.
Allow substantial GPU time (hours/days depending on hardware) and over 120 GiB of
working storage, with additional space for intermediates. This cleanup did not
rerun the full corpus. Historical checkpoints, CLI choices, environment differences,
and date-dependent preprocessing matter for exact metric reproduction; these limits
are documented in the reproduction notes.

## Outputs and their thesis counterparts

| Demo output under `outputs/demo/` | Corresponding thesis result type / full-run location |
|---|---|
| `metrics_summary.*`, binary PR/confusion figures | Binary helpfulness tables and figures; `outputs/predictions/`, `outputs/plots/` |
| Count metrics, `figures/count_actual_vs_predicted.*` | Two-stage log1p/Poisson count comparison and actual/predicted figures |
| `weighted_rating/comparison.csv`, `per_item.csv` | Rating/ranking-change tables; full `outputs/weighted_rating/table_*.csv` |
| `figures/rating_change_distribution.*` | Distribution of weighted-minus-arithmetic rating changes |
| `weighted_rating/alpha_sweep.csv`, `figures/weighting_alpha_sweep.*` | Weighting-strength sensitivity and rank stability |

Mappings use result names because the thesis manuscript and its numbered table/
figure labels are not in this repository. **Every demo metric uses reduced data and
training and therefore differs from the thesis.** Rating shifts demonstrate
aggregation behavior; they do not establish improved rating accuracy.
