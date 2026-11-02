# Thesis figure redraw

Run from the repository root:

```powershell
.venv/Scripts/python.exe experiments/thesis_figures/make_figures.py
```

Reads only saved binary predictions, combined count predictions, and the saved
`outputs/weighted_rating/alpha_sweep_models.csv`. No training or inference.
Binary labels are `true_label`, scores are `prob_pos`; count columns are
`true_count` and `expected_count`.

Before plotting, checks binary sample sizes and metrics rounded to the thesis's
four decimal places at fixed thresholds 0.1006 and 0.8095, the 25-review subset
difference, matching labels, finite valid inputs, paired count IDs/labels, and
alpha sweep structure. Any failed check stops execution.

Writes four PDF/300-dpi PNG pairs and a report with source SHA-256 hashes under
`outputs/thesis_figures/`. Refuses to overwrite any of its output files.
Confusion matrix shading is row-normalised with raw count annotations.
Count scatter points are rasterised in the otherwise vector PDFs; all rows are
plotted. XGBoost Poisson shares the first panel's x-axis range and has a separate
0–75 y-axis. Alpha sweep results are not recomputed.

To apply only the requested count-axis correction to existing generated outputs:

```powershell
.venv/Scripts/python.exe experiments/thesis_figures/fix_count_axes.py
```

This verifies the count source hashes against `verification.json`, then replaces
only `actual_vs_predicted.pdf` and `.png` and updates the report.
