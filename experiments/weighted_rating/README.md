# Weighted Rating Experiment

Reproduce the DistilBERT-helpfulness weighted rating analysis from the repository root:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s experiments\weighted_rating\tests
.\.venv\Scripts\python.exe experiments\weighted_rating\run_all.py
```

The pipeline writes review-level inputs, summary CSVs, PDF figures, LaTeX tables, and `run_info.json` to `outputs/weighted_rating/`.
