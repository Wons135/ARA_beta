from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiments.weighted_rating.config import CFG
    from experiments.weighted_rating.weighted_rating import SCHEME_LABELS, SCHEME_ORDER
else:
    from .config import CFG
    from .weighted_rating import SCHEME_LABELS, SCHEME_ORDER


SCHEME_COLUMNS = {name: SCHEME_LABELS[name] for name in SCHEME_ORDER}


def _format_metric(name: str, value: object) -> str:
    if pd.isna(value):
        return ""
    number = float(value)
    if "%" in name:
        return f"{number:.1f}"
    if abs(number - round(number)) < 1e-12 and ("items" in name or name.startswith("top")):
        return str(int(round(number)))
    return f"{number:.4f}"


def _escape_latex(text: object) -> str:
    value = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    return value


def _write_table(df: pd.DataFrame, path: Path) -> None:
    columns = ["Metric", *SCHEME_COLUMNS.values()]
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\hline",
        " & ".join(columns) + r" \\",
        r"\hline",
    ]
    for metric, row in df.iterrows():
        values = [_escape_latex(metric)]
        for source_col in SCHEME_COLUMNS:
            values.append(_format_metric(str(metric), row[source_col]))
        lines.append(" & ".join(values) + r" \\")
        lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_largest_changes(out_dir: Path, path: Path) -> None:
    decreases = pd.read_csv(out_dir / "largest_decreases_alpha_0_5.csv").head(3)
    increases = pd.read_csv(out_dir / "largest_increases_alpha_0_5.csv").tail(3).sort_values("bert_delta", ascending=False)
    combined = pd.concat(
        [
            increases.assign(direction="Increase"),
            decreases.assign(direction="Decrease"),
        ],
        ignore_index=True,
    )
    lines = [
        r"\begin{tabular}{llrrrrrrrrr}",
        r"\hline",
        r"Direction & Title & n & 1 star & 2 stars & 3 stars & 4 stars & 5 stars & Arithmetic mean & Weighted mean & Delta \\",
        r"\hline",
    ]
    for _, row in combined.iterrows():
        lines.append(
            " & ".join(
                [
                    _escape_latex(row["direction"]),
                    _escape_latex(row["title"]),
                    str(int(row["n"])),
                    str(int(row["stars_1"])),
                    str(int(row["stars_2"])),
                    str(int(row["stars_3"])),
                    str(int(row["stars_4"])),
                    str(int(row["stars_5"])),
                    f"{float(row['arith_mean']):.4f}",
                    f"{float(row['bert_mean']):.4f}",
                    f"{float(row['bert_delta']):.4f}",
                ]
            )
            + r" \\"
        )
        lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_tables(out_dir: Path = CFG.out_dir) -> None:
    CFG.tables_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ["alpha_0_5", "alpha_1_0", "alpha_0_5_nmin50"]:
        rating_path = out_dir / f"table_rating_change_{suffix}.csv"
        ranking_path = out_dir / f"table_ranking_change_{suffix}.csv"
        if not rating_path.exists() or not ranking_path.exists():
            continue
        rating = pd.read_csv(rating_path, index_col=0)
        ranking = pd.read_csv(ranking_path, index_col=0)
        _write_table(rating.loc[:, list(SCHEME_COLUMNS)], CFG.tables_dir / f"table_rating_change_{suffix}.tex")
        _write_table(ranking.loc[:, list(SCHEME_COLUMNS)], CFG.tables_dir / f"table_ranking_change_{suffix}.tex")
    # Backward-compatible headline names point to alpha=0.5.
    rating = pd.read_csv(out_dir / "table_rating_change_alpha_0_5.csv", index_col=0)
    ranking = pd.read_csv(out_dir / "table_ranking_change_alpha_0_5.csv", index_col=0)
    _write_table(rating.loc[:, list(SCHEME_COLUMNS)], CFG.tables_dir / "table_rating_change.tex")
    _write_table(ranking.loc[:, list(SCHEME_COLUMNS)], CFG.tables_dir / "table_ranking_change.tex")
    _write_largest_changes(out_dir, CFG.tables_dir / "table_largest_changes_alpha_0_5.tex")
    _write_largest_changes(out_dir, CFG.tables_dir / "table_largest_changes.tex")
    print(f"Exported LaTeX tables -> {CFG.tables_dir}")


def main() -> None:
    export_tables()


if __name__ == "__main__":
    main()
