"""
significance.py
===============
Statistical testing for the forecasting benchmark.

Why this exists
---------------
Section 6.4 of the report currently states that the observed ranking "is not a
formal statistical significance test and should not be described as one", and
offers a stability check against seed noise instead. That was the honest thing
to say with no test in hand. It is no longer necessary: the benchmark has 236
paired cell-level observations, which is ample for a proper test, and the
classical field is now wide enough that "is Theta distinguishable from
LightGBM?" is a question a reader will actually ask.

What is tested
--------------
The unit of observation is a forecasting cell (cost centre x G/L account), and
the observation is that cell's MASE under each model. Every model sees the same
236 cells, so the observations are PAIRED -- which is what makes a rank-based
test appropriate here rather than an unpaired comparison of aggregate scores.

  Friedman     omnibus. Are the k models drawn from the same distribution of
               per-cell ranks? Non-parametric, so it makes no normality
               assumption -- which matters, because per-cell MASE is heavily
               right-skewed (a handful of near-flat cells produce enormous
               scaled errors).

  Nemenyi      post-hoc, applied only if Friedman rejects. Gives a single
               critical difference (CD): two models differ at alpha if their
               mean ranks differ by more than CD. This is the standard
               Demsar (2006) procedure for comparing multiple methods over
               multiple datasets.

  Wilcoxon     signed-rank on the specific pairs the paper makes claims about,
               with Holm correction across that family. Friedman-Nemenyi is
               conservative with many models; the paper's claims are about a
               few named comparisons, and those deserve a direct test.

What is NOT tested
------------------
The rolling forecast (v2) is excluded. It sees within-year actuals, so it is
not competing under the same information boundary, and including it in a rank
test would be comparing models that were not given the same problem.

Interpretation, stated before the numbers are seen
--------------------------------------------------
A non-significant result between N-BEATS and LightGBM would NOT invalidate the
benchmark. It would mean the aggregate MASE gap of 0.070 is not resolvable at
236 cells, and the defensible claim becomes the disaggregated one: the
advantage is concentrated in personnel cells, which Figure 10.2 already shows.
That is a narrower claim than the current text makes, and it is the one the
evidence supports.

Usage:  python3 significance.py ./out
"""

from __future__ import annotations

import sys
import pathlib
import itertools
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from baselines import (load_panel, to_matrix, mase_scale, f_naive, f_snaive,
                       f_ets, f_lgbm, f_plan, KEY, TRAIN_PERIODS, HORIZON)

warnings.filterwarnings("ignore")

ALPHA = 0.05


# ---------------------------------------------------------------------------
def per_cell_mase(test: np.ndarray, pred: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """One MASE per cell. This is the observation the tests operate on."""
    return np.abs(test - np.nan_to_num(pred)).mean(axis=1) / scale


def collect(out: pathlib.Path) -> tuple[pd.DataFrame, pd.Series, int]:
    """Rebuild every ranked model's per-cell MASE into one aligned frame."""
    panel, plan = load_panel(out)
    mat, keys = to_matrix(panel)
    train = mat[:, :TRAIN_PERIODS]
    test = mat[:, TRAIN_PERIODS:TRAIN_PERIODS + HORIZON]
    scale = mase_scale(train)

    preds = {
        "Naive": f_naive(train),
        "Seasonal naive": f_snaive(train),
        "ETS (Holt-Winters)": f_ets(train),
        "LightGBM (global)": f_lgbm(panel, keys),
        "Budget (v1)": f_plan(plan, "1", keys),
    }

    # The three added classical methods, from their checkpoints.
    for name, fn in (("Theta", "Theta"), ("STL+ETS", "STL_ETS"), ("SARIMA", "SARIMA")):
        p = out / "ckpt" / f"{fn}.npz"
        if p.exists():
            preds[name] = np.load(p)["fc"]
        else:
            print(f"  [warn] {name} checkpoint missing -- run classical_extra.py")

    # Deep learning, seed-averaged.
    dlp = out / "dl_predictions.npz"
    if dlp.exists():
        z = np.load(dlp)
        for k in z.files:
            preds[k] = z[k]
    else:
        print("  [warn] dl_predictions.npz missing -- run nbeats.py")

    cols = {k: per_cell_mase(test, v, scale) for k, v in preds.items()}
    df = pd.DataFrame(cols)

    # Account category per cell, aligned to the same row order, so the
    # confined-deficit test below operates on exactly the same panel.
    meta = (panel[KEY + ["CATEGORY"]].drop_duplicates()
            .merge(keys, on=KEY, how="right"))
    cat = pd.Series(meta["CATEGORY"].to_numpy(), index=df.index)
    # A cell whose in-sample seasonal error is zero has undefined MASE for
    # every model. Dropping it listwise keeps the panel paired.
    before = len(df)
    keep = df.notna().all(axis=1)
    df, cat = df[keep], cat[keep]
    if len(df) < before:
        print(f"  dropped {before - len(df)} cells with undefined MASE scale")
    return df, cat, len(df)


# ---------------------------------------------------------------------------
def nemenyi_cd(k: int, n: int, alpha: float = ALPHA) -> float:
    """
    Critical difference for the Nemenyi test.

        CD = q_alpha * sqrt( k(k+1) / 6n )

    where q_alpha is the studentised range statistic at infinite degrees of
    freedom, divided by sqrt(2).
    """
    q = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2.0)
    return float(q * np.sqrt(k * (k + 1) / (6.0 * n)))


def main(out: pathlib.Path) -> int:
    df, cat, n = collect(out)
    models = list(df.columns)
    k = len(models)

    print("=" * 78)
    print("  SIGNIFICANCE TESTING -- cell-level MASE, FY2025")
    print("=" * 78)
    print(f"  {k} models x {n} paired cells")
    print(f"  rolling forecast (v2) excluded: different information set\n")

    # -- Friedman omnibus ------------------------------------------------
    stat, p = stats.friedmanchisquare(*[df[m].to_numpy() for m in models])
    print(f"  Friedman chi2 = {stat:.2f}, p = {p:.3e}")
    if p >= ALPHA:
        print("  Not significant: the models are not distinguishable overall.")
        return 0
    print(f"  Significant at alpha = {ALPHA}: at least one model differs.\n")

    # -- Nemenyi post-hoc ------------------------------------------------
    ranks = df.rank(axis=1).mean().sort_values()
    cd = nemenyi_cd(k, n)
    print("  Mean rank (lower is better):")
    for m, r in ranks.items():
        print(f"    {m:<28}{r:>7.3f}")
    print(f"\n  Nemenyi critical difference = {cd:.3f}")
    print("  Two models differ significantly if their mean ranks differ by more.\n")

    best = ranks.index[0]
    print(f"  Against the best-ranked model ({best}):")
    for m, r in ranks.items():
        if m == best:
            continue
        d = r - ranks[best]
        verdict = "significantly worse" if d > cd else "NOT distinguishable"
        print(f"    {m:<28}delta rank {d:>6.3f}   {verdict}")

    # -- Wilcoxon on the paper's named claims ----------------------------
    print("\n" + "=" * 78)
    print("  PAIRWISE WILCOXON -- the comparisons the paper claims")
    print("=" * 78)
    pairs = [
        ("NBEATS-G (generic)", "LightGBM (global)"),
        ("NBEATS-G (generic)", "Theta"),
        ("NBEATS-G (generic)", "Budget (v1)"),
        ("NBEATS-G (generic)", "NBEATS-I (interpretable)"),
        ("LightGBM (global)", "Theta"),
        ("LightGBM (global)", "Budget (v1)"),
    ]
    pairs = [(a, b) for a, b in pairs if a in df.columns and b in df.columns]

    raw = []
    for a, b in pairs:
        s, pv = stats.wilcoxon(df[a], df[b])
        raw.append((a, b, df[a].mean(), df[b].mean(), pv))

    # Holm-Bonferroni across this family.
    order = np.argsort([r[4] for r in raw])
    adj = [None] * len(raw)
    m = len(raw)
    running = 0.0
    for rank, i in enumerate(order):
        v = min(1.0, (m - rank) * raw[i][4])
        running = max(running, v)          # enforce monotonicity
        adj[i] = running

    print(f"  {'comparison':<52}{'p (Holm)':>10}{'':>4}")
    print("  " + "-" * 72)
    for (a, b, ma, mb, pv), pa in zip(raw, adj):
        mark = "*" if pa < ALPHA else " "
        better = a if ma < mb else b
        print(f"  {a} vs {b:<20}{pa:>10.4f} {mark}")
        print(f"      mean MASE {ma:.3f} vs {mb:.3f}  -> {better} lower")
    print("  " + "-" * 72)
    print(f"  * significant at alpha = {ALPHA} after Holm correction "
          f"across {m} comparisons")

    # -- Is the confined deficit significant WHERE it is confined? -------
    #
    # The aggregate test above cannot answer the question section 6.6 actually
    # asks. Averaged over 236 cells the two N-BEATS variants are not
    # distinguishable, but figure 10.2 shows the difference is not spread
    # evenly: it sits almost entirely in personnel, the category carrying the
    # December bonus spike. Testing per category asks whether that confinement
    # is real or an artefact of reading a bar chart.
    #
    # Holm is applied across the five categories, so finding personnel
    # significant is not the product of testing until something passes.
    a, b = "NBEATS-G (generic)", "NBEATS-I (interpretable)"
    if a in df.columns and b in df.columns:
        print("\n" + "=" * 78)
        print("  CONFINED DEFICIT -- generic vs interpretable N-BEATS, by category")
        print("=" * 78)
        cats = sorted(cat.unique())
        res = []
        for c in cats:
            m = cat == c
            if m.sum() < 6:
                continue
            try:
                _, pv = stats.wilcoxon(df.loc[m, a], df.loc[m, b])
            except ValueError:
                continue
            res.append((c, int(m.sum()), df.loc[m, a].mean(),
                        df.loc[m, b].mean(), pv))

        order = np.argsort([r[4] for r in res])
        adj = [None] * len(res)
        mm, running = len(res), 0.0
        for rank, i in enumerate(order):
            running = max(running, min(1.0, (mm - rank) * res[i][4]))
            adj[i] = running

        print(f"  {'category':<16}{'cells':>6}{'NBEATS-G':>11}{'NBEATS-I':>11}"
              f"{'p (Holm)':>11}")
        print("  " + "-" * 60)
        for (c, nn, ma, mb, _), pa in zip(res, adj):
            mark = "*" if pa < ALPHA else " "
            print(f"  {c:<16}{nn:>6}{ma:>11.3f}{mb:>11.3f}{pa:>11.4f} {mark}")
        print("  " + "-" * 60)
        print(f"  * significant at alpha = {ALPHA} after Holm correction "
              f"across {mm} categories")
        print("\n  Read with the aggregate result above: if personnel is the only")
        print("  significant category, the claim is that the deficit is CONFINED,")
        print("  not that the interpretable variant is worse overall.")

        pd.DataFrame(res, columns=["category", "cells", a, b, "p_raw"]) \
          .assign(p_holm=adj).to_csv(out / "significance_by_category.csv",
                                      index=False)

    pd.DataFrame({"model": ranks.index, "mean_rank": ranks.values}) \
      .to_csv(out / "significance_ranks.csv", index=False)
    print(f"\n  written: {out / 'significance_ranks.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "./out")))
