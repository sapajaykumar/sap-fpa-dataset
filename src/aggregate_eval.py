"""
aggregate_eval.py
=================
Cell-level versus aggregate evaluation, reproducing Section 7.6 and Table 10.

Why this exists
---------------
The benchmark reports one number per model: the mean cell-level MASE across the
236 planning cells. That is the right metric for a cost-centre owner reviewing a
single planning line, because it asks whether each series was forecast well.

It is not the metric a finance director reads. Consolidated revenue is a sum over
cells, and a sum has a property no per-cell average captures: errors can cancel.
A model whose per-cell errors are small but share a systematic direction will
accumulate them on the total. A model whose per-cell errors are larger but
independent will see much of the error disappear in the sum. Cell-level accuracy
therefore carries no guarantee about aggregate accuracy, and this module measures
how far apart the two rankings actually fall.

What it computes
----------------
For every ranked model:

    cell-level MASE   the figure reported in Table 6
    aggregate WAPE    all revenue cells summed to one monthly total, then scored

and then the rank correlation between the two orderings, by Spearman and Kendall,
with and without the naive baseline (which is last under both measures and would
otherwise inflate any apparent agreement).

The rolling forecast is excluded from the ranking throughout, as elsewhere in the
benchmark: it observes within-year actuals and does not face the same information
boundary. It is reported separately for reference.

A note on power
---------------
Eleven models is a small sample for a rank-correlation test. A non-significant
result here means no relationship was detected, which is not the same as showing
there is none. The paper states the claim that way and this module prints the
p-values so a reader can see why.

Requires: out/dl_predictions.npz (nbeats.py) and out/ckpt/*.npz (classical_extra.py).

Usage:  python3 aggregate_eval.py ./out
"""

from __future__ import annotations

import sys
import pathlib
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from baselines import (load_panel, to_matrix, mase_scale, f_naive, f_snaive,
                       f_ets, f_lgbm, f_plan, KEY, TRAIN_PERIODS, HORIZON)

warnings.filterwarnings("ignore")

REVENUE_ACCOUNTS = ["400000", "410000"]


def collect(out: pathlib.Path):
    """Every ranked model's FY2025 forecast, plus the unranked rolling forecast."""
    panel, plan = load_panel(out)
    mat, keys = to_matrix(panel)
    train = mat[:, :TRAIN_PERIODS]
    test = mat[:, TRAIN_PERIODS:TRAIN_PERIODS + HORIZON]

    preds = {
        "Naive":              f_naive(train),
        "Seasonal naive":     f_snaive(train),
        "ETS (Holt-Winters)": f_ets(train),
        "LightGBM (global)":  f_lgbm(panel, keys),
        "Budget v1 (human)":  f_plan(plan, "1", keys),
    }

    for label, stem in (("Theta", "Theta"), ("STL+ETS", "STL_ETS"), ("SARIMA", "SARIMA")):
        p = out / "ckpt" / f"{stem}.npz"
        if p.exists():
            preds[label] = np.load(p)["fc"]
        else:
            print(f"  [warn] {label} checkpoint missing -- run classical_extra.py")

    dlp = out / "dl_predictions.npz"
    if dlp.exists():
        rename = {"NBEATS-G (generic)": "N-BEATS generic",
                  "NBEATS-I (interpretable)": "N-BEATS interpretable",
                  "GlobalMLP (+embeddings)": "Global MLP"}
        z = np.load(dlp)
        for k in z.files:
            preds[rename.get(k, k)] = z[k]
    else:
        print("  [warn] dl_predictions.npz missing -- run nbeats.py")

    reference = {"Rolling forecast (v2)": f_plan(plan, "2", keys)}
    return preds, reference, test, train, keys


def main(out: pathlib.Path) -> int:
    preds, reference, test, train, keys = collect(out)
    scale = mase_scale(train)
    is_rev = keys["HKONT"].isin(REVENUE_ACCOUNTS).to_numpy()
    y_agg = test[is_rev].sum(axis=0)

    def cell_mase(p):
        return float(np.nanmean(np.abs(test - np.nan_to_num(p)).mean(axis=1) / scale))

    def agg_wape(p):
        a = np.nan_to_num(p)[is_rev].sum(axis=0)
        return float(np.abs(y_agg - a).sum() / y_agg.sum())

    rows = [{"model": m, "cell_MASE": cell_mase(p), "agg_WAPE": agg_wape(p)}
            for m, p in preds.items()]
    df = pd.DataFrame(rows)
    df["cell_rank"] = df["cell_MASE"].rank().astype(int)
    df["agg_rank"] = df["agg_WAPE"].rank().astype(int)
    df["delta"] = df["agg_rank"] - df["cell_rank"]
    df = df.sort_values("cell_rank")

    print("\n" + "=" * 78)
    print("  EVALUATION LEVEL AND MODEL RANKING (Table 10)")
    print("=" * 78)
    print(f"  {'model':<24}{'cell MASE':>11}{'rank':>6}{'agg WAPE':>11}{'rank':>6}{'delta':>8}")
    print("  " + "-" * 74)
    for _, r in df.iterrows():
        print(f"  {r['model']:<24}{r.cell_MASE:>11.3f}{r.cell_rank:>6}"
              f"{r.agg_WAPE:>10.1%}{r.agg_rank:>6}{r.delta:>+8d}")
    print("  " + "-" * 74)
    for m, p in reference.items():
        print(f"  not ranked: {m:<20}{cell_mase(p):>11.3f}{'':>6}{agg_wape(p):>10.1%}")
    print("=" * 78)

    # -- rank correlation -------------------------------------------------
    print("\n  Rank correlation between the two evaluation levels:")
    for label, sub in (("all models", df),
                       ("excluding naive", df[df["model"] != "Naive"])):
        rho, p_r = stats.spearmanr(sub["cell_rank"], sub["agg_rank"])
        tau, p_t = stats.kendalltau(sub["cell_rank"], sub["agg_rank"])
        print(f"    {label:<18} n={len(sub):<3} "
              f"Spearman rho {rho:+.3f} (p={p_r:.3f})   "
              f"Kendall tau {tau:+.3f} (p={p_t:.3f})")

    moved = df[df["delta"].abs() >= 6]
    if len(moved):
        print("\n  Models moving six or more places:")
        for _, r in moved.iterrows():
            print(f"    {r['model']:<24} cell {r.cell_rank:>2} -> aggregate {r.agg_rank:>2}"
                  f"  ({r.delta:+d})")

    n = len(df)
    print(f"\n  With {n} models the correlation test has little power. A "
          f"non-significant\n  result means no relationship was detected, not that "
          f"none exists.")

    df.to_csv(out / "aggregate_revenue_results.csv", index=False)
    print(f"\n  written: {out / 'aggregate_revenue_results.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "./out")))
