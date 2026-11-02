"""Redraw only the count figure with corrected XGBoost Poisson axes."""
import hashlib
import json

import matplotlib.pyplot as plt
import pandas as pd

from make_figures import ROOT, OUT, render_counts


def main():
    audit = json.loads((OUT / "verification.json").read_text(encoding="utf-8"))
    counts = []
    for source in audit["sources"]:
        if "count" not in source["path"] or not source["path"].endswith(".csv"):
            continue
        path = ROOT / source["path"]
        with path.open("rb") as fh:
            assert hashlib.file_digest(fh, "sha256").hexdigest() == source["sha256"], path
        counts.append(pd.read_csv(path, usecols=["review_id", "true_count", "expected_count"]))
    assert len(counts) == 4
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
                         "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    render_counts(counts)
    report_path = OUT / "report.md"
    report = report_path.read_text(encoding="utf-8")
    report = report.replace("Each panel retains its own axis range, including XGBoost Poisson.",
                            "XGBoost Poisson shares the first panel's x-axis range and uses a separate y-axis range of 0–75. Its y = x reference line ends at the maximum actual log count.")
    report_path.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
