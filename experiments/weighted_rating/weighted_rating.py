from __future__ import annotations

import argparse
import os
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiments.weighted_rating.config import CFG
else:
    from .config import CFG


SCHEME_ORDER = ["length", "rank_length", "bert", "xgb_count", "filtered", "votes_observed"]
SCHEME_LABELS = {
    "length": "Length",
    "rank_length": "Rank-length",
    "bert": "DistilBERT",
    "xgb_count": "XGBoost count",
    "filtered": "Filtered",
    "votes_observed": "Votes",
}


@dataclass(frozen=True)
class AnalysisResult:
    diagnostics: dict[str, float | int]
    rating: pd.DataFrame
    ranking: pd.DataFrame


def item_weighted_mean(df: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    tmp = pd.DataFrame({"item": df["parent_asin"].values, "wr": weights * df["rating"].values, "w": weights})
    grouped = tmp.groupby("item", sort=True)[["wr", "w"]].sum()
    return grouped["wr"] / grouped["w"].replace(0, np.nan)


def kish_neff(df: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    tmp = pd.DataFrame({"item": df["parent_asin"].values, "w": weights, "w2": weights**2})
    grouped = tmp.groupby("item", sort=True)[["w", "w2"]].sum()
    return grouped["w"] ** 2 / grouped["w2"].replace(0, np.nan)


def max_single_review_share_by_title(df: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    tmp = pd.DataFrame({"item": df["parent_asin"].values, "w": weights})
    grouped = tmp.groupby("item", sort=True)["w"]
    return grouped.max() / grouped.sum().replace(0, np.nan)


def change_summary(base: pd.Series, new: pd.Series, n: pd.Series) -> dict[str, float]:
    ok = new.notna()
    delta = (new - base)[ok]
    display_changed = base[ok].round(1) != new[ok].round(1)
    return {
        "items_compared": int(ok.sum()),
        "coverage_%": float(100 * ok.mean()),
        "mean_delta": float(delta.mean()),
        "median_delta": float(delta.median()),
        "mean_abs_delta": float(delta.abs().mean()),
        "p95_abs_delta": float(delta.abs().quantile(0.95)),
        "max_abs_delta": float(delta.abs().max()),
        "share_up_%": float(100 * (delta > 0).mean()),
        "share_down_%": float(100 * (delta < 0).mean()),
        "share_abs_ge_0.1_%": float(100 * (delta.abs() >= 0.1).mean()),
        "share_abs_ge_0.25_%": float(100 * (delta.abs() >= 0.25).mean()),
        "share_display_changed_%": float(100 * display_changed.mean()),
    }


def ranking_summary(base: pd.Series, new: pd.Series, top_k: tuple[int, ...] = (10, 100)) -> dict[str, float]:
    ok = new.notna()
    b, weighted = base[ok], new[ok]
    rank_base = b.rank(ascending=False, method="min")
    rank_weighted = weighted.rank(ascending=False, method="min")
    result = {
        "spearman_rho": float(spearmanr(b, weighted).statistic),
        "kendall_tau": float(kendalltau(b, weighted).statistic),
        "share_items_moved_%": float(100 * (rank_base != rank_weighted).mean()),
        "mean_abs_rank_shift": float((rank_base - rank_weighted).abs().mean()),
        "median_abs_rank_shift": float((rank_base - rank_weighted).abs().median()),
        "max_abs_rank_shift": float((rank_base - rank_weighted).abs().max()),
    }
    for k in top_k:
        top_base = set(b.sort_values(ascending=False).index[:k])
        top_weighted = set(weighted.sort_values(ascending=False).index[:k])
        result[f"top{k}_overlap"] = int(len(top_base & top_weighted))
    return result


def alpha_mix(signal: np.ndarray, alpha: float) -> np.ndarray:
    return (1.0 - float(alpha)) + float(alpha) * np.asarray(signal, dtype=float)


apply_alpha = alpha_mix


def per_title_max_normalize(df: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    tmp = pd.DataFrame({"parent_asin": df["parent_asin"].values, "value": np.asarray(values, dtype=float)})
    max_per_title = tmp.groupby("parent_asin")["value"].transform("max").to_numpy(dtype=float)
    out = np.divide(tmp["value"].to_numpy(dtype=float), max_per_title, out=np.zeros(len(tmp), dtype=float), where=max_per_title > 0)
    return np.clip(out, 0.0, 1.0)


def rank_matched_length_signal(df: pd.DataFrame) -> np.ndarray:
    length_order = np.lexsort((df["review_id"].to_numpy(), df["review_length"].to_numpy(dtype=float)))
    p_sorted = np.sort(df["p_bert"].to_numpy(dtype=float), kind="stable")
    out = np.empty(len(df), dtype=float)
    out[length_order] = p_sorted
    return out


def signal_frame(df: pd.DataFrame, tau: float) -> pd.DataFrame:
    raw_votes = 1.0 + np.log1p(df["helpful_vote"].to_numpy(dtype=float))
    raw_length = np.log1p(df["review_length"].to_numpy(dtype=float))
    signals = pd.DataFrame(index=df.index)
    signals["length"] = per_title_max_normalize(df, raw_length)
    signals["rank_length"] = np.clip(rank_matched_length_signal(df), 0.0, 1.0)
    signals["bert"] = np.clip(df["p_bert"].to_numpy(dtype=float), 0.0, 1.0)
    signals["xgb_count"] = per_title_max_normalize(df, df["yhat_xgb"].to_numpy(dtype=float))
    signals["filtered"] = (df["p_bert"].to_numpy(dtype=float) >= float(tau)).astype(float)
    signals["votes_observed"] = per_title_max_normalize(df, raw_votes)
    return signals.loc[:, SCHEME_ORDER]


def weights_for_alpha(signals: pd.DataFrame, alpha: float) -> OrderedDict[str, np.ndarray]:
    return OrderedDict((name, alpha_mix(signals[name].to_numpy(dtype=float), alpha)) for name in SCHEME_ORDER)


def permuted_weights_by_item(df: pd.DataFrame, weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    order = np.argsort(df["parent_asin"].values, kind="stable")
    sorted_weights = weights[order]
    key = np.lexsort((rng.random(len(df)), df["parent_asin"].values))
    out = np.empty_like(weights)
    out[key] = sorted_weights
    return out


def permutation_null(
    df: pd.DataFrame,
    weights: np.ndarray,
    base: pd.Series,
    observed: pd.Series,
    n_perm: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, pd.Series]:
    exceed = np.zeros(len(base))
    perm_mean_abs = []
    observed_abs = (observed - base).abs().values
    for _ in range(n_perm):
        weighted = item_weighted_mean(df, permuted_weights_by_item(df, weights, rng))
        delta_perm = (weighted - base).abs()
        perm_mean_abs.append(float(delta_perm.mean()))
        exceed += delta_perm.values >= observed_abs
    p_item = (exceed + 1) / (n_perm + 1)
    return np.array(perm_mean_abs), pd.Series(p_item, index=base.index)


def _delta_relation(prefix: str, left: pd.Series, right: pd.Series) -> dict[str, float]:
    ok = left.notna() & right.notna()
    return {
        f"spearman_delta_{prefix}": float(spearmanr(left[ok], right[ok]).statistic),
        f"sign_agreement_delta_{prefix}_%": float(100 * (np.sign(left[ok]) == np.sign(right[ok])).mean()),
    }


def _p_bert_quantiles(df: pd.DataFrame) -> dict[str, float]:
    qs = df["p_bert"].quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    return {f"p_bert_q{int(q * 100):02d}": float(value) for q, value in qs.items()}


def _rating_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.pivot_table(index="parent_asin", columns="rating", values="review_id", aggfunc="count", fill_value=0)
    for rating in range(1, 6):
        if rating not in counts.columns:
            counts[rating] = 0
    return counts.loc[:, [1, 2, 3, 4, 5]].rename(columns={r: f"stars_{r}" for r in range(1, 6)})


def _prepare_df(path: Path, n_min: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    keep = df.groupby("parent_asin").size()
    keep = keep[keep >= int(n_min)].index
    return df[df["parent_asin"].isin(keep)].reset_index(drop=True)


def analyze_setting(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    alpha: float,
    tau: float,
    out_dir: Path,
    n_min: int,
    n_perm: int,
    seed: int,
    write_artifacts: bool,
) -> AnalysisResult:
    rng = np.random.default_rng(seed + int(round(alpha * 1000)) + n_min)
    suffix = f"alpha_{alpha:.1f}".replace(".", "_")
    if n_min != CFG.n_min:
        suffix = f"{suffix}_nmin{n_min}"
    weights = weights_for_alpha(signals, alpha)
    n = df.groupby("parent_asin", sort=True).size()
    rating_sd = df.groupby("parent_asin", sort=True)["rating"].std()
    base = item_weighted_mean(df, np.ones(len(df), dtype=float))

    per_item = pd.DataFrame({"n": n, "rating_sd": rating_sd, "arith_mean": base})
    change_rows: dict[str, dict[str, float]] = {}
    ranking_rows: dict[str, dict[str, float]] = {}
    share_stats: dict[str, float] = {}
    for name, w in weights.items():
        weighted = item_weighted_mean(df, w)
        per_item[f"{name}_mean"] = weighted
        per_item[f"{name}_delta"] = weighted - base
        per_item[f"{name}_neff"] = kish_neff(df, w)
        change_rows[name] = change_summary(base, weighted, n)
        change_rows[name]["mean_neff_over_n"] = float((per_item[f"{name}_neff"] / n).mean())
        ranking_rows[name] = ranking_summary(base, weighted)
        shares = max_single_review_share_by_title(df, w)
        share_stats[f"max_review_weight_share_median_{name}"] = float(shares.median())
        share_stats[f"max_review_weight_share_p90_{name}"] = float(shares.quantile(0.90))

    rating = pd.DataFrame(change_rows).loc[:, SCHEME_ORDER]
    ranking = pd.DataFrame(ranking_rows).loc[:, SCHEME_ORDER]
    if write_artifacts:
        rating.to_csv(out_dir / f"table_rating_change_{suffix}.csv")
        ranking.to_csv(out_dir / f"table_ranking_change_{suffix}.csv")

    main_delta = per_item["bert_delta"]
    diagnostics: dict[str, float | int] = {
        "setting": suffix,
        "alpha": float(alpha),
        "n_min": int(n_min),
        "reviews_used": int(len(df)),
        "items_used": int(n.shape[0]),
        "tau": float(tau),
        "inferred_prediction_share_%": float(100 * (df["pred_source"] == "inferred").mean()),
        **_p_bert_quantiles(df),
        **share_stats,
        **_delta_relation("main_vs_votes", main_delta, per_item["votes_observed_delta"]),
        **_delta_relation("main_vs_xgb_count", main_delta, per_item["xgb_count_delta"]),
        **_delta_relation("main_vs_rank_length", main_delta, per_item["rank_length_delta"]),
    }
    if df["p_xgb"].notna().any():
        valid = df["p_xgb"].notna()
        diagnostics["subset_binary_test_rows_xgb_vs_bert"] = int(valid.sum())
        diagnostics["subset_spearman_p_xgb_vs_p_bert"] = float(spearmanr(df.loc[valid, "p_xgb"], df.loc[valid, "p_bert"]).statistic)
        diagnostics["subset_sign_agreement_xgb_binary_vs_bert_above_threshold_%"] = float(
            100 * ((df.loc[valid, "p_xgb"] >= 0.8095203042030334) == (df.loc[valid, "p_bert"] >= tau)).mean()
        )

    perm_abs, p_item = permutation_null(df, weights["bert"], base, per_item["bert_mean"], n_perm, rng)
    diagnostics.update(
        {
            "obs_mean_abs_delta_main": float(per_item["bert_delta"].abs().mean()),
            "perm_mean_abs_delta_main_mean_over_perms": float(perm_abs.mean()),
            "perm_mean_abs_delta_main_95th_pct": float(np.quantile(perm_abs, 0.95)),
            "share_items_perm_p_lt_0.05_%": float(100 * (p_item < 0.05).mean()),
        }
    )
    per_item["bert_perm_p"] = p_item

    if write_artifacts and n_min == CFG.n_min and abs(alpha - 0.5) < 1e-12:
        per_item = per_item.join(df.groupby("parent_asin")["title"].first()).join(_rating_breakdown(df))
        per_item.to_csv(out_dir / "per_item.csv")
        cols = [
            "title",
            "n",
            "stars_1",
            "stars_2",
            "stars_3",
            "stars_4",
            "stars_5",
            "arith_mean",
            "bert_mean",
            "bert_delta",
        ]
        per_item.sort_values("bert_delta").head(10)[cols].to_csv(out_dir / "largest_decreases_alpha_0_5.csv")
        per_item.sort_values("bert_delta").tail(10)[cols].to_csv(out_dir / "largest_increases_alpha_0_5.csv")

    return AnalysisResult(diagnostics=diagnostics, rating=rating, ranking=ranking)


def _alpha_sweep(df: pd.DataFrame, signals: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    n = df.groupby("parent_asin", sort=True).size()
    base = item_weighted_mean(df, np.ones(len(df), dtype=float))
    rows = []
    for alpha in np.linspace(0.0, 1.0, 11):
        weights = weights_for_alpha(signals, float(alpha))
        for scheme in ["bert", "xgb_count", "votes_observed"]:
            weighted = item_weighted_mean(df, weights[scheme])
            c = change_summary(base, weighted, n)
            r = ranking_summary(base, weighted)
            rows.append(
                {
                    "alpha": float(alpha),
                    "scheme": scheme,
                    "mean_abs_delta": c["mean_abs_delta"],
                    "kendall_tau": r["kendall_tau"],
                }
            )
    sweep = pd.DataFrame(rows)
    sweep.to_csv(out_dir / "alpha_sweep_models.csv", index=False)

    fig, ax1 = plt.subplots(figsize=(6.5, 3.6))
    colors = {"bert": "0.15", "xgb_count": "0.45", "votes_observed": "0.70"}
    for scheme in ["bert", "xgb_count", "votes_observed"]:
        part = sweep[sweep["scheme"] == scheme]
        ax1.plot(part["alpha"], part["mean_abs_delta"], "o-", color=colors[scheme], label=f"{SCHEME_LABELS[scheme]} mean |delta|")
    ax1.set_xlabel(r"weighting strength $\alpha$")
    ax1.set_ylabel(r"mean $|\Delta_j|$ (stars)")
    ax2 = ax1.twinx()
    for scheme in ["bert", "xgb_count", "votes_observed"]:
        part = sweep[sweep["scheme"] == scheme]
        ax2.plot(part["alpha"], part["kendall_tau"], "s--", color=colors[scheme], label=f"{SCHEME_LABELS[scheme]} Kendall")
    ax2.set_ylabel(r"Kendall $\tau$ with arithmetic ranking")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(handles1 + handles2, labels1 + labels2, loc="lower center", ncol=2, fontsize=8)
    fig.tight_layout(rect=(0, 0.18, 1, 1))
    fig.savefig(out_dir / "fig_alpha_sweep_models.pdf")
    plt.close(fig)
    return sweep


def _figures_for_headline(out_dir: Path) -> None:
    per_item = pd.read_csv(out_dir / "per_item.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.hist(per_item["bert_delta"], bins=80, color="0.35")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel(r"$\Delta_j$ = weighted mean $-$ arithmetic mean (stars)")
    ax.set_ylabel("number of titles")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_delta_hist_alpha_0_5.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.scatter(per_item["votes_observed_delta"], per_item["bert_delta"], s=3, alpha=0.3, color="0.2")
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel(r"$\Delta_j$, observed-vote weights")
    ax.set_ylabel(r"$\Delta_j$, predicted-helpfulness weights")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_delta_model_vs_votes_alpha_0_5.pdf")
    plt.close(fig)


def run_analysis(
    path: Path = CFG.review_level_inputs_path,
    out_dir: Path = CFG.out_dir,
    n_min: int = CFG.n_min,
    tau: float | None = None,
    n_perm: int = CFG.n_perm,
    seed: int = CFG.seed,
) -> dict[str, object]:
    tau_value = CFG.bert_validation_threshold() if tau is None else float(tau)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = _prepare_df(path, n_min)
    print(f"reviews: {len(df):,}  items with >= {n_min} reviews: {df['parent_asin'].nunique():,}")
    signals = signal_frame(df, tau_value)

    results = [
        analyze_setting(df, signals, 0.5, tau_value, out_dir, n_min, n_perm, seed, write_artifacts=True),
        analyze_setting(df, signals, 1.0, tau_value, out_dir, n_min, n_perm, seed, write_artifacts=True),
    ]
    diagnostics = pd.DataFrame([r.diagnostics for r in results])
    diagnostics.to_csv(out_dir / "diagnostics.csv", index=False)
    _alpha_sweep(df, signals, out_dir)
    _figures_for_headline(out_dir)

    if n_min == CFG.n_min:
        df50 = _prepare_df(path, 50)
        signals50 = signal_frame(df50, tau_value)
        analyze_setting(df50, signals50, 0.5, tau_value, out_dir, 50, n_perm, seed, write_artifacts=True)

    # Backward-compatible headline names point to alpha=0.5.
    results[0].rating.to_csv(out_dir / "table_rating_change.csv")
    results[0].ranking.to_csv(out_dir / "table_ranking_change.csv")
    return {
        "settings": [r.diagnostics for r in results],
        "headline_reviews_used": int(len(df)),
        "headline_items_used": int(df["parent_asin"].nunique()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run helpfulness-weighted rating analysis.")
    parser.add_argument("path", nargs="?", default=str(CFG.review_level_inputs_path))
    parser.add_argument("--out", default=str(CFG.out_dir))
    parser.add_argument("--n_min", type=int, default=CFG.n_min)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--n_perm", type=int, default=CFG.n_perm)
    parser.add_argument("--seed", type=int, default=CFG.seed)
    parser.add_argument("--main", default=CFG.main_weight)
    args = parser.parse_args()
    if args.main != "p_bert":
        raise ValueError("This experiment is configured for --main p_bert only.")
    diagnostics = run_analysis(Path(args.path), Path(args.out), args.n_min, args.tau, args.n_perm, args.seed)
    print(pd.json_normalize(diagnostics["settings"]).round(4).to_string(index=False))
    print(f"done -> {args.out}{os.sep}")


if __name__ == "__main__":
    main()
