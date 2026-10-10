"""
Helpfulness-weighted rating estimation: comparison with the arithmetic mean.

Input: ONE file (parquet or csv) with one row per TEST review and columns
    parent_asin    item identifier from Amazon Reviews 2023
    rating         star rating 1-5
    helpful_vote   observed helpful votes (used ONLY for the vote-weighted check)
    review_length  token count (same definition as in the thesis)
    p_xgb          XGBoost binary model, predicted probability of "helpful"
optional:
    p_bert         DistilBERT binary model probability
    title          item title (from meta_Movies_and_TV.jsonl), for the examples table

IMPORTANT before running:
  * use reviews that were NOT in the training/validation split of ANY model whose
    predictions are used (binary and count splits were made separately!);
  * p_xgb must be the raw model output on these reviews, no votes involved.

Usage:
    python weighted_rating_experiment.py preds.parquet --out results --tau 0.80
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------- helpers
def item_weighted_mean(df, w):
    """Weighted mean of ratings per item; items with total weight 0 -> NaN."""
    tmp = pd.DataFrame({"item": df["parent_asin"].values,
                        "wr": w * df["rating"].values, "w": w})
    g = tmp.groupby("item", sort=True)[["wr", "w"]].sum()
    out = g["wr"] / g["w"].replace(0, np.nan)
    return out


def kish_neff(df, w):
    tmp = pd.DataFrame({"item": df["parent_asin"].values, "w": w, "w2": w ** 2})
    g = tmp.groupby("item", sort=True)[["w", "w2"]].sum()
    return g["w"] ** 2 / g["w2"].replace(0, np.nan)


def change_summary(base, new, n):
    ok = new.notna()
    d = (new - base)[ok]
    disp_changed = (base[ok].round(1) != new[ok].round(1))
    return {
        "items_compared": int(ok.sum()),
        "coverage_%": 100 * ok.mean(),
        "mean_delta": d.mean(),
        "median_delta": d.median(),
        "mean_abs_delta": d.abs().mean(),
        "p95_abs_delta": d.abs().quantile(0.95),
        "max_abs_delta": d.abs().max(),
        "share_up_%": 100 * (d > 0).mean(),
        "share_down_%": 100 * (d < 0).mean(),
        "share_abs_ge_0.1_%": 100 * (d.abs() >= 0.1).mean(),
        "share_abs_ge_0.25_%": 100 * (d.abs() >= 0.25).mean(),
        "share_display_changed_%": 100 * disp_changed.mean(),
    }


def ranking_summary(base, new, top_k=(10, 100)):
    ok = new.notna()
    b, nw = base[ok], new[ok]
    rb = b.rank(ascending=False, method="min")
    rn = nw.rank(ascending=False, method="min")
    res = {
        "spearman_rho": spearmanr(b, nw).statistic,
        "kendall_tau": kendalltau(b, nw).statistic,
        "share_items_moved_%": 100 * (rb != rn).mean(),
        "mean_abs_rank_shift": (rb - rn).abs().mean(),
        "median_abs_rank_shift": (rb - rn).abs().median(),
        "max_abs_rank_shift": (rb - rn).abs().max(),
    }
    # top-k overlap; ties broken by number of reviews is NOT used, simple order
    for k in top_k:
        tb = set(b.sort_values(ascending=False).index[:k])
        tn = set(nw.sort_values(ascending=False).index[:k])
        res[f"top{k}_overlap"] = len(tb & tn)
    return res


def permutation_null(df, w, base, observed, n_perm, rng):
    """Shuffle weights WITHIN each item: keeps the weight distribution of the
    item, removes any link between weight and rating."""
    order = np.argsort(df["parent_asin"].values, kind="stable")
    items_sorted = df["parent_asin"].values[order]
    w_sorted = w[order]
    exceed = np.zeros(len(base))
    perm_mean_abs = []
    for _ in range(n_perm):
        key = np.lexsort((rng.random(len(df)), df["parent_asin"].values))
        w_perm = np.empty_like(w)
        w_perm[key] = w_sorted  # items_sorted == parent_asin[key]
        wm = item_weighted_mean(df, w_perm)
        d_perm = (wm - base).abs()
        perm_mean_abs.append(d_perm.mean())
        exceed += (d_perm.values >= (observed - base).abs().values)
    p_item = (exceed + 1) / (n_perm + 1)
    return np.array(perm_mean_abs), pd.Series(p_item, index=base.index)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--out", default="weighted_rating_results")
    ap.add_argument("--n_min", type=int, default=20)
    ap.add_argument("--tau", type=float, default=0.80,
                    help="validation-selected XGBoost threshold for filtering")
    ap.add_argument("--n_perm", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    df = (pd.read_parquet(a.path) if a.path.endswith(".parquet")
          else pd.read_csv(a.path))
    n_all = df.groupby("parent_asin").size()
    keep = n_all[n_all >= a.n_min].index
    df = df[df["parent_asin"].isin(keep)].reset_index(drop=True)
    print(f"reviews: {len(df):,}  items with >= {a.n_min} reviews: {len(keep):,}")

    n = df.groupby("parent_asin", sort=True).size()
    sd = df.groupby("parent_asin", sort=True)["rating"].std()
    base = item_weighted_mean(df, np.ones(len(df)))

    # -------- weighting schemes (alpha = 1)
    schemes = {
        "length": np.log1p(df["review_length"].values.astype(float)),
        "xgb": df["p_xgb"].values.astype(float),
        "xgb_filtered": (df["p_xgb"].values >= a.tau).astype(float),
        "votes_observed": 1 + np.log1p(df["helpful_vote"].values.astype(float)),
    }
    if "p_bert" in df:
        schemes["bert"] = df["p_bert"].values.astype(float)

    per_item = pd.DataFrame({"n": n, "rating_sd": sd, "arith_mean": base})
    rows_c, rows_r = {}, {}
    for name, w in schemes.items():
        wm = item_weighted_mean(df, w)
        per_item[f"{name}_mean"] = wm
        per_item[f"{name}_delta"] = wm - base
        per_item[f"{name}_neff"] = kish_neff(df, w)
        rows_c[name] = change_summary(base, wm, n)
        rows_c[name]["mean_neff_over_n"] = (per_item[f"{name}_neff"] / n).mean()
        rows_r[name] = ranking_summary(base, wm)

    pd.DataFrame(rows_c).to_csv(f"{a.out}/table_rating_change.csv")
    pd.DataFrame(rows_r).to_csv(f"{a.out}/table_ranking_change.csv")
    print(pd.DataFrame(rows_c).round(4).to_string())
    print(pd.DataFrame(rows_r).round(4).to_string())

    # -------- diagnostics for interpretation
    diag = {
        "spearman_p_xgb_vs_rating (review level)":
            spearmanr(df["p_xgb"], df["rating"]).statistic,
        "spearman_p_xgb_vs_length (review level)":
            spearmanr(df["p_xgb"], df["review_length"]).statistic,
        "spearman_absdelta_xgb_vs_item_rating_sd":
            spearmanr(per_item["xgb_delta"].abs(), per_item["rating_sd"],
                      nan_policy="omit").statistic,
        "spearman_absdelta_xgb_vs_n":
            spearmanr(per_item["xgb_delta"].abs(), per_item["n"]).statistic,
        "spearman_delta_xgb_vs_delta_votes":
            spearmanr(per_item["xgb_delta"], per_item["votes_observed_delta"]).statistic,
        "sign_agreement_xgb_vs_votes_%":
            100 * (np.sign(per_item["xgb_delta"])
                   == np.sign(per_item["votes_observed_delta"])).mean(),
        "spearman_delta_xgb_vs_delta_length":
            spearmanr(per_item["xgb_delta"], per_item["length_delta"]).statistic,
    }
    # mean rating of reviews above / below tau (direction explanation)
    diag["mean_rating_p>=tau"] = df.loc[df.p_xgb >= a.tau, "rating"].mean()
    diag["mean_rating_p<tau"] = df.loc[df.p_xgb < a.tau, "rating"].mean()

    # -------- permutation null for XGBoost weights
    perm_abs, p_item = permutation_null(df, schemes["xgb"], base,
                                        per_item["xgb_mean"], a.n_perm, rng)
    diag["obs_mean_abs_delta_xgb"] = per_item["xgb_delta"].abs().mean()
    diag["perm_mean_abs_delta_xgb (mean over perms)"] = perm_abs.mean()
    diag["perm_mean_abs_delta_xgb (95th pct)"] = np.quantile(perm_abs, 0.95)
    diag["share_items_p<0.05_%"] = 100 * (p_item < 0.05).mean()
    per_item["xgb_perm_p"] = p_item
    pd.Series(diag).to_csv(f"{a.out}/diagnostics.csv")
    print(pd.Series(diag).round(4).to_string())

    # -------- alpha sweep: w = (1 - alpha) + alpha * p
    sweep = []
    for alpha in np.linspace(0, 1, 11):
        w = (1 - alpha) + alpha * schemes["xgb"]
        wm = item_weighted_mean(df, w)
        s = change_summary(base, wm, n)
        r = ranking_summary(base, wm)
        sweep.append({"alpha": alpha, "mean_delta": s["mean_delta"],
                      "mean_abs_delta": s["mean_abs_delta"],
                      "share_display_changed_%": s["share_display_changed_%"],
                      "kendall_tau": r["kendall_tau"]})
    sweep = pd.DataFrame(sweep)
    sweep.to_csv(f"{a.out}/alpha_sweep.csv", index=False)

    # -------- largest differences
    cols = ["n", "rating_sd", "arith_mean", "xgb_mean", "xgb_delta",
            "votes_observed_delta"]
    if "title" in df:
        per_item = per_item.join(df.groupby("parent_asin")["title"].first())
        cols = ["title"] + cols
    per_item.sort_values("xgb_delta").head(10)[cols].to_csv(
        f"{a.out}/largest_decreases.csv")
    per_item.sort_values("xgb_delta").tail(10)[cols].to_csv(
        f"{a.out}/largest_increases.csv")
    per_item.to_csv(f"{a.out}/per_item.csv")

    # -------- figures (no titles above plots: captions go in LaTeX)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.hist(per_item["xgb_delta"], bins=80, color="0.35")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel(r"$\Delta_j$ = weighted mean $-$ arithmetic mean (stars)")
    ax.set_ylabel("number of titles")
    fig.tight_layout(); fig.savefig(f"{a.out}/fig_delta_hist.pdf"); plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(6.5, 3.6))
    ax1.plot(sweep.alpha, sweep.mean_abs_delta, "o-", color="0.2",
             label=r"mean $|\Delta_j|$")
    ax1.set_xlabel(r"weighting strength $\alpha$")
    ax1.set_ylabel(r"mean $|\Delta_j|$ (stars)")
    ax2 = ax1.twinx()
    ax2.plot(sweep.alpha, sweep.kendall_tau, "s--", color="0.55",
             label=r"Kendall $\tau$")
    ax2.set_ylabel(r"Kendall $\tau$ with arithmetic ranking")
    fig.legend(loc="lower left", bbox_to_anchor=(0.12, 0.15))
    fig.tight_layout(); fig.savefig(f"{a.out}/fig_alpha_sweep.pdf"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.scatter(per_item["votes_observed_delta"], per_item["xgb_delta"],
               s=3, alpha=0.3, color="0.2")
    ax.axhline(0, color="k", lw=0.6); ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel(r"$\Delta_j$, observed-vote weights")
    ax.set_ylabel(r"$\Delta_j$, predicted-helpfulness weights")
    fig.tight_layout(); fig.savefig(f"{a.out}/fig_delta_model_vs_votes.pdf")
    plt.close(fig)
    print(f"done -> {a.out}/")


if __name__ == "__main__":
    main()