# CPU demo configuration and outputs

**All demo numbers differ from thesis results.** The bundled data is synthetic;
reduced data, trees, sequence length, trainable layers, and epochs make this an
execution demonstration, not a model benchmark.

`python run_demo.py` checks Python 3.11, creates `.venv-demo`, and installs
`requirements-demo.txt` if the pinned dependencies are absent or mismatched.
It never installs into the thesis environment. `python run_demo.py --offline`
requires the demo dependencies, pinned DistilBERT snapshot, and VADER lexicon
already cached. Both commands execute the same seven pipeline stages; offline
mode only changes resource loading.

| Setting | Demo |
|---|---|
| Input | Bundled 1,024 original synthetic reviews, 8 fictional titles |
| Split | Shared 70/15/15, `random_state=42`; binary gray band excluded only from binary metrics |
| Features | TF-IDF up to 1,500 features, 24 SVD components, original task-specific metadata |
| Baselines | Majority class, mean normalized helpfulness score, mean raw vote count |
| XGBoost | 32 trees, depth 3, `device=cpu`, seed 42 |
| DistilBERT | Actual pretrained base model; 1 epoch, at most 96 train / 48 validation rows per mode |
| Fine-tuning | Last transformer block and original aux/head modules; embeddings and first 5 blocks frozen |
| Text / batch | Maximum 64 tokens, batch 8, 4 CPU threads by default |
| Tasks | Binary, count gate, positive-only log1p regression and Poisson regression |
| Thresholds | Selected on validation predictions; test data is used only for reporting |
| Combined counts | Soft gate probability multiplied by conditional positive count |
| Weighting | Held-out reviews, minimum 3/title; alpha 0.5/1.0, 0..1 sweep, 20 within-title permutations |

All settings are in `src/demo/config.py`; actual settings, IDs, environment
versions, sample hash, and timing are saved with the run. The model snapshot is
pinned to `12040accade4e8a0f71eabdb258fecc2e7e948be`. No fallback replaces DistilBERT
if its download fails. The model download is about 268 MB, plus tokenizer files;
first dependency installation takes additional time and disk space. VADER may
also require a small first download. No credentials or Hugging Face login are
needed, and implicit token sending is disabled in the demo process.

The five figures are saved as PNG/PDF pairs in `outputs/demo/figures/`:

- `binary_precision_recall`
- `binary_confusion_matrices`
- `count_actual_vs_predicted`
- `rating_change_distribution`
- `weighting_alpha_sweep`

`metrics_summary.json` is the completion marker and contains CPU status, settings,
versions, metrics, and stage timings; `metrics_summary.csv` is the compact table;
`summary.txt` is the readable report; `run.log` is the progress log.
`output_manifest.json` lists all generated files. Preprocessed task CSVs, shared
split IDs, feature models/matrices, baseline/XGBoost checkpoints, DistilBERT trained
parameters/configurations, held-out predictions, and weighted-rating CSVs/diagnostics
are also saved. The pretrained backbone remains in Hugging Face's cache rather
than being duplicated in each checkpoint. Trained-parameter files need that exact
backbone plus their JSON configuration to reconstruct the fine-tuned models.

Reruns retrain everything with the same configuration. Previous summary files move
to `outputs/demo/previous_run/`; current task artifacts are regenerated. A failed
rerun leaves no current success marker. Deterministic CPU algorithms and seed 42
make repeats stable on the same software/hardware; cross-platform bitwise identity
is not promised. A few titles and a few positive binary examples make these demo
metrics noisy. Weighted rating shifts measure changes, not improved rating accuracy.

Windows: double-click `run_demo.bat`, or run `.\run_demo.bat` from PowerShell.
The wrapper pauses at the end for reading; set `DEMO_NO_PAUSE=1` to suppress this.
Unix: `sh run_demo.sh`; Python 3.11 must already be installed. Both wrappers change
to the repository directory. Apple Silicon uses the native macOS torch wheel,
still explicitly on CPU; Windows x64 is the tested platform. Older Intel Macs or
other architectures may lack compatible pinned wheels and have not been tested.

For a screen recording, run the default command once online to prepare caches,
then record `python run_demo.py --offline` and open `summary.txt` and the PNGs.
The reviewer can use the same offline command after their first online setup.
For slow CPUs, keep the same sample/settings; `--threads 2` changes only parallelism.
