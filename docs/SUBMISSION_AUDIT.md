# Submission audit and review checklist

Prepared on 2026-10-29 on `thesis-submission`, branched from `weighted-rating-experiment` at `1783f5e`. No commit, push, or history rewrite was performed.

## Findings before cleanup

- `datasets/`: 92,243 MiB, including 8,391,445,499-byte reviews, 1,285,744,674-byte metadata and large derived CSV/feature bundles.
- `outputs/`: 25,949 MiB, mostly checkpoints/predictions and historical generated figures/tables.
- `.venv/`: 5,221 MiB. `.git/`: 830 MiB. Neither belongs in the submission archive.
- Core: `src/`, `models/`, `training/`, weighted-rating analysis/tests, and thesis figure verification/redraw scripts.
- Scratch: checkpoint-patching `main.py`, empty Python-2-era notebook, scratch/gate audits, follow-up dumps, smoke artifacts, file listing and report dumps.
- Caches: project `__pycache__/` directories; dependency caches remain inside excluded environments.
- Thesis manuscript: no manuscript `.tex` or thesis PDF found. Existing `.tex` and `.pdf` files are generated tables/figures in `outputs/`, now excluded.
- Initial source/config scan found no apparent credentials. A separate staged-file scan checks credential patterns, prohibited paths and large files before packaging.

The complete pre-cleanup file tree with byte sizes is in [REPOSITORY_INVENTORY.md](REPOSITORY_INVENTORY.md). Audit-only files are excluded from the reviewer ZIP.

## Preserved for the author to confirm

Everything below was moved into ignored `_to_review/`, preserving the original relative path. Nothing ambiguous was discarded. Directory contents are listed in local `_to_review/manifest.json` and remain available for inspection.

| Original path | Preserved destination | Bytes |
|---|---|---:|
| `.claude` | `_to_review/.claude` | 621 |
| `.idea` | `_to_review/.idea` | 32,420 |
| `main.py` | `_to_review/main.py` | 804 |
| `files.txt` | `_to_review/files.txt` | 79,874 |
| `weighted_rating_report.txt` | `_to_review/weighted_rating_report.txt` | 59,624 |
| `notebooks` | `_to_review/notebooks` | 615 |
| `experiments/weighted_rating/scratch` | `_to_review/experiments/weighted_rating/scratch` | 20,043,832 |
| `experiments/weighted_rating/followup` | `_to_review/experiments/weighted_rating/followup` | 6,610 |
| `datasets/_tmp_preprocess_smoke.jsonl` | `_to_review/datasets/_tmp_preprocess_smoke.jsonl` | 104,363 |
| `docs/CLEANUP_AUDIT.md` | `_to_review/docs/CLEANUP_AUDIT.md` | 5,869 |
| `outputs/bert_binary_diagnostics` | `_to_review/outputs/bert_binary_diagnostics` | 39,784,011 |
| `outputs/bert_binary_retry_fast` | `_to_review/outputs/bert_binary_retry_fast` | 531,018,427 |
| `outputs/bert_binary_smoke_fix` | `_to_review/outputs/bert_binary_smoke_fix` | 570,715,108 |
| `datasets/_tmp_preprocessed_smoke` | `_to_review/datasets/_tmp_preprocessed_smoke` | 810,814 |
| `outputs/weighted_rating/followup` | `_to_review/outputs/weighted_rating/followup` | 19,185 |

Also preserved as copies: `_to_review/original_README.md`, `_to_review/original_requirements.txt`, and `_to_review/original_source_hashes.json`.

Raw/full preprocessed data and historical thesis outputs stay at their original local paths so existing scripts still find them. They are ignored and removed from the Git index, not deleted from disk. Tracked IDE metadata and scratch artifacts have likewise been removed from the index. Git history is retained, so ship the exported ZIP rather than the entire workspace or `.git/`.

## Pipeline and data decisions

- All 36 retained original Python files match the pre-cleanup SHA-256 hashes byte for byte. Existing full-scale defaults and seed 42 are untouched.
- Previously untracked baseline/evaluation/common/figure helpers are included as core source, without edits.
- `requirements.txt` is a UTF-8 list of the pinned full-run dependencies with the CUDA wheel index; unrelated cloud clients/type stubs from the local freeze are excluded. The original freeze is preserved for review. The requested NumPy 1.26.4 pin and executed 2.4.4 discrepancy are documented.
- Redistribution of Amazon review text could not be established from the source maintainers. The demo bundles 1,024 original synthetic reviews with explicit provenance and permission instead. This is a deliberate deviation from a real Amazon excerpt, not a claim that the synthetic data was drawn from Amazon.
- The sample SHA-256 uses canonical LF line endings so Windows checkouts and ZIP extraction agree. Synthetic sample rows have no real user identifiers.
- Demo-only settings: CPU torch 2.5.1+cpu, 32 XGBoost trees, DistilBERT 96-row training slices / one epoch / 64 tokens / last-block-and-head fine-tuning, 20 rating permutations and minimum 3 reviews per title.

## Packaging for review

After reviewing staged changes, run `python scripts/package_submission.py` to refresh `outputs/submission/ARA-thesis-submission.zip`. It exports the staged index, checks paths/sizes/credential patterns, respects `export-ignore`, and includes neither Git history nor untracked/ignored files. No commit is required. The author-only audit and inventory documents are not exported. A clean pattern scan is evidence of the performed checks, not a proof that arbitrary secrets cannot exist.

Proposed commit message: `Package thesis submission with additive CPU demo and reviewer documentation`.

## Project caches removed from the source tree

Caches were moved to ignored `_to_review/caches/` rather than discarded. Dependency caches inside excluded environments are retained.

- `src/__pycache__`
- `src/demo/__pycache__`
- `models/__pycache__`
- `training/__pycache__`
- `experiments/thesis_figures/__pycache__`
- `experiments/weighted_rating/__pycache__`
- `experiments/weighted_rating/tests/__pycache__`
- `__pycache__`

## Validation handover

See [DEMO_TEST_RUN.md](DEMO_TEST_RUN.md) for measured CPU runtimes, checks, and the complete generated-file list. The final Windows wrapper returned exit status 0. The sample-policy clarification was optional and unanswered during implementation; the documented synthetic sample was used.
