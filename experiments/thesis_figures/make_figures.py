"""Verify saved binary predictions against the fixed thesis results."""
from pathlib import Path
import json
import hashlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, precision_recall_curve,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/thesis_figures"
COLORS = ["#355C7D", "#C06C84", "#6C757D"]


def verify_binary():
    specs = [
        ("DistilBERT", "bert", 0.1006, [1435602, .9080, .4927, .4414, .5779, .5005]),
        ("XGBoost", "xgb", 0.8095, [1435577, .9131, .5433, .5054, .5599, .5313]),
    ]
    failures = []
    results = []
    for name, key, threshold, expected in specs:
        path = ROOT / "outputs/predictions" / f"{key}_binary_test_predictions.csv"
        data = pd.read_csv(path, usecols=["review_id", "true_label", "prob_pos"])
        assert not data.isna().any().any(), f"Missing values: {path}"
        assert data.review_id.is_unique, f"Duplicate review IDs: {path}"
        y, score = data.true_label, data.prob_pos
        assert y.isin([0, 1]).all() and score.between(0, 1).all(), path
        predicted = score >= threshold
        actual = [len(data), roc_auc_score(y, score), average_precision_score(y, score),
                  precision_score(y, predicted), recall_score(y, predicted), f1_score(y, predicted)]
        print(f"{name}, threshold={threshold:.4f}, positive share={y.mean():.10f}")
        for metric, value, reference in zip(["n", "AUROC", "AUPRC", "Precision", "Recall", "F1"], actual, expected):
            matches = value == reference if metric == "n" else round(value, 4) == reference
            print(f"  {metric}: {value} (thesis {reference}; {'PASS' if matches else 'FAIL'})")
            if not matches:
                failures.append(f"{name} {metric}: {value} != {reference}")
        print(f"  confusion matrix: {confusion_matrix(y, predicted).tolist()}")
        assert round(y.mean(), 4) == .0690, "Positive-class share mismatch"
        results.append((name, threshold, data, actual, confusion_matrix(y, predicted)))
    if failures:
        raise RuntimeError("Verification failed; no figures generated.\n" + "\n".join(failures))
    ids_bert = set(results[0][2].review_id)
    ids_xgb = set(results[1][2].review_id)
    assert ids_xgb <= ids_bert and len(ids_bert - ids_xgb) == 25
    joined = results[0][2].merge(results[1][2], on="review_id", suffixes=("_d", "_x"))
    assert (joined.true_label_d == joined.true_label_x).all(), "Binary label disagreement"
    return results


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_counts(counts):
    labels = ["DistilBERT + log1p", "DistilBERT + Poisson", "XGBoost + log1p", "XGBoost + Poisson"]
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 6.0), layout="constrained")
    ranges = []
    for i, (ax, d, label) in enumerate(zip(axes.flat, counts, labels)):
        x, y = np.log1p(d.true_count.to_numpy()), np.log1p(d.expected_count.to_numpy())
        hi = float(max(x.max(), y.max()))
        ax.scatter(x, y, s=8, alpha=.25, color=COLORS[i // 2], rasterized=True)
        reference_hi = float(x.max()) if i == 3 else hi
        ax.plot([0, reference_hi], [0, reference_hi], "--", color="black", lw=1)
        if i == 3:
            ax.set_xlim(axes.flat[0].get_xlim())
            ax.set_ylim(0, 75)
        ax.set(xlabel="log(1 + actual count)", ylabel="log(1 + predicted count)")
        ax.set_title(label)
        ax.grid(alpha=.2)
        ranges.append({"panel": label, "n": len(d), "max_log1p_actual": float(x.max()), "max_log1p_predicted": float(y.max())})
    save(fig, "actual_vs_predicted")

    return ranges


def main():
    stems = ["pr_curves", "confusion_matrices", "actual_vs_predicted", "alpha_sweep"]
    targets = [OUT / f"{s}.{e}" for s in stems for e in ("pdf", "png")]
    targets += [OUT / "report.md", OUT / "verification.json"]
    assert not any(p.exists() for p in targets), "Output exists; refusing to overwrite"
    binary = verify_binary()
    count_names = [
        "bert_count_combined_count_stage2_log1p_test_predictions.csv",
        "bert_count_combined_count_stage2_poisson_test_predictions.csv",
        "xgb_count_stage1_test_combined_count_stage2_log1p_predictions.csv",
        "xgb_count_stage1_test_combined_count_stage2_poisson_predictions.csv",
    ]
    counts = []
    sources = [ROOT / "outputs/predictions" / f"{key}_binary_test_predictions.csv" for key in ("bert", "xgb")]
    for filename in count_names:
        path = ROOT / "outputs/predictions" / filename
        d = pd.read_csv(path, usecols=["review_id", "true_count", "expected_count"])
        assert d.review_id.is_unique and not d.isna().any().any(), path
        values = d[["true_count", "expected_count"]].to_numpy()
        assert np.isfinite(values).all() and (values >= 0).all(), path
        counts.append(d)
        sources.append(path)
    for i in (0, 2):
        assert counts[i][["review_id", "true_count"]].equals(counts[i+1][["review_id", "true_count"]]), "Count variant labels/IDs differ"
    sweep_path = ROOT / "outputs/weighted_rating/alpha_sweep_models.csv"
    sweep = pd.read_csv(sweep_path)
    assert set(sweep.columns) == {"alpha", "scheme", "mean_abs_delta", "kendall_tau"}
    assert set(sweep.scheme) == {"bert", "xgb_count", "votes_observed"}
    assert not sweep.isna().any().any() and not sweep.duplicated(["alpha", "scheme"]).any()
    assert np.isfinite(sweep[["alpha", "mean_abs_delta", "kendall_tau"]]).all().all()
    assert sweep.alpha.between(0, 1).all() and (sweep.mean_abs_delta >= 0).all() and sweep.kendall_tau.between(-1, 1).all()
    grids = [tuple(g.sort_values("alpha").alpha) for _, g in sweep.groupby("scheme")]
    assert len(set(grids)) == 1, "Different alpha grids"
    sources.append(sweep_path)
    # Finish all input verification before creating any figure.
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
                         "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(6.3, 4.2), layout="constrained")
    for i, (name, threshold, d, metrics, cm) in enumerate(binary):
        precision, recall, _ = precision_recall_curve(d.true_label, d.prob_pos)
        ax.plot(recall, precision, color=COLORS[i], linestyle=["-", "-."][i], lw=1.4,
                label=f"{name} (AUPRC = {metrics[2]:.4f})", rasterized=False)
    ax.hlines(.0690, 0, 1, colors=COLORS[2], linestyles="--", label="Majority-class baseline (AUPRC = 0.0690)")
    for i, (name, threshold, d, metrics, cm) in enumerate(binary):
        ax.plot(metrics[4], metrics[3], marker=["o", "s"][i], color=COLORS[i], linestyle="none",
                label=f"{name}, t = {threshold:.4f}")
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="upper right")
    save(fig, "pr_curves")

    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.5), layout="constrained")
    for ax, (name, threshold, d, metrics, cm) in zip(axes, binary):
        shading = cm / cm.sum(axis=1, keepdims=True)
        im = ax.imshow(shading, cmap="Blues", vmin=0, vmax=1)
        for row in range(2):
            for col in range(2):
                ax.text(col, row, f"{cm[row, col]:,}", ha="center", va="center",
                        color="white" if shading[row, col] > .5 else "black", fontsize=10)
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Not helpful", "Helpful"],
               yticklabels=["Not helpful", "Helpful"], xlabel="Predicted class", ylabel="True class")
        ax.set_title(f"{name} (t = {threshold:.4f})")
    fig.colorbar(im, ax=axes, shrink=.65, label="Proportion within true class")
    save(fig, "confusion_matrices")

    ranges = render_counts(counts)

    fig, left = plt.subplots(figsize=(6.3, 4.2), layout="constrained")
    right = left.twinx()
    handles = []
    for i, (key, label) in enumerate(zip(["bert", "xgb_count", "votes_observed"], ["DistilBERT", "XGBoost count", "Votes"])):
        part = sweep[sweep.scheme == key].sort_values("alpha")
        handles += left.plot(part.alpha, part.mean_abs_delta, color=COLORS[i], marker=["o", "s", "^"][i],
                             linestyle="-", markersize=4, label=label + r" mean $|\Delta|$")
        handles += right.plot(part.alpha, part.kendall_tau, color=COLORS[i], marker=["o", "s", "^"][i],
                              linestyle="--", markersize=4, label=label + r" Kendall's $\tau$")
    left.set(xlabel=r"Weighting strength $\alpha$", ylabel=r"Mean $|\Delta|$ (stars)")
    right.set_ylabel(r"Kendall's $\tau$ with arithmetic ranking")
    left.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, 1.30), ncol=2)
    save(fig, "alpha_sweep")

    audit = {"binary": [{"model": name, "threshold": t, "n": metrics[0], "auroc": metrics[1],
                         "auprc": metrics[2], "precision": metrics[3], "recall": metrics[4],
                         "f1": metrics[5], "confusion_matrix": cm.tolist()} for name, t, d, metrics, cm in binary],
             "count_panels": ranges, "sources": [{"path": str(p.relative_to(ROOT)),
                 "sha256": hashlib.file_digest(p.open("rb"), "sha256").hexdigest()} for p in sources]}
    (OUT / "verification.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    report = ["# Thesis figures", "", "All binary metrics match the thesis to four decimal places at the fixed validation thresholds. No threshold selection, training, inference, or prediction generation was performed.",
              "", "Confusion matrices use row-normalised shading with the same 0–1 scale; annotations are raw counts. The PR baseline is horizontal at 0.0690.",
              "", "The scatter uses every saved row, true_count and expected_count, transformed with log1p. Points are rasterised; axes and text remain vector in PDF. XGBoost Poisson shares the first panel's x-axis range and uses a separate y-axis range of 0–75.", ""]
    for r in ranges:
        report.append(f"- {r['panel']}: {r['n']:,} rows; maximum log(1 + predicted count) = {r['max_log1p_predicted']:.4f}.")
    report += ["", "Alpha sweep values are read directly from alpha_sweep_models.csv. Source hashes and full binary checks are recorded in verification.json.",
               "", "No AGENTS.md or CLAUDE.md was found in the repository or its checked parent directories."]
    (OUT / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Saved four PDF/PNG pairs and verification report to {OUT}")


if __name__ == "__main__":
    main()
