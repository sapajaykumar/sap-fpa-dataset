"""
baselines.py
============
Forecasting baselines for the FP&A cycle, evaluated before any deep model is
written. If a neural network cannot beat these, it has no claim to be in the
dissertation.

Task
----
At the close of FY2024, forecast all 12 months of FY2025 for every
(company code, cost centre, GL account) cell. This mirrors how an annual plan
is actually cut: one origin, a 12-month horizon, no peeking.

    train   2022-01 .. 2024-12   (36 periods)
    test    2025-01 .. 2025-12   (12 periods)

Baselines
---------
    Naive             last observed value, carried flat
    Seasonal naive    the same month last year
    ETS               Holt-Winters, additive trend + seasonal, fit per series
    LightGBM          one global model, lag>=12 features only

    Budget            the entity's own budget (VERSION 1)  <- THE baseline

Budget is the baseline that matters. It is locked before FY2025 opens, so it
sees exactly the information the models see: everything up to the FY2024 close,
and nothing after. A model that beats seasonal-naive but loses to the finance
team's own plan has not earned its place in an FP&A pipeline. This mirrors
Kureljusic & Reisch (2022), who benchmark machine learning against sell-side
analysts rather than against statistical baselines alone.

    Rolling forecast (VERSION 2)  -- REPORTED, NOT RANKED

The rolling forecast is re-cut quarterly and shrinks toward realised actuals,
so by construction it sees within-year information the models do not. It is
NOT a fair comparison at a single FY2024-close origin and must not be reported
as a baseline the models were measured against. It is included only as an
information-advantaged reference: roughly, the accuracy a model could reach if
it were re-fit each quarter. Phase 3, if it adopts a rolling-origin evaluation,
can compare against it fairly.

Design notes
------------
* Amounts use the NATURAL sign convention: a posting on the account's normal
  side is positive, its reversal negative. Netting accruals against reversals
  is mandatory -- both carry positive DMBTR and only SHKZG separates them.
  Unsigned aggregation overstates COGS by ~5%.
* Special periods 13-16 (year-end close) fold into December: they carry no
  calendar month of their own but belong to the fiscal year's result.
* LightGBM uses only lags >= 12, so every horizon 1..12 is predictable from
  data available at the forecast origin. No recursion, therefore no error
  compounding and no leakage.
* MASE is the primary metric: scale-free, so cells of wildly different
  magnitude (revenue vs utilities) can be pooled. Denominator is the in-sample
  seasonal-naive MAE, the standard convention for seasonal series.

Usage:  python3 baselines.py ./out
"""

from __future__ import annotations

import sys
import pathlib
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

TRAIN_PERIODS = 36
HORIZON = 12
SEASON = 12
KEY = ["BUKRS", "KOSTL", "HKONT"]


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def load_panel(out: pathlib.Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Monthly panel of P&L amounts per (BUKRS, KOSTL, HKONT)."""
    act = pd.read_csv(
        out / "acdoca_actuals.csv",
        dtype={"BUKRS": str, "HKONT": str, "KOSTL": str, "BELNR": str,
               "STBLG": str, "VERSION": str},
        keep_default_na=False, na_values=[],
    )
    gl = pd.read_csv(out / "dim_gl_account.csv", dtype={"HKONT": str})
    cc = pd.read_csv(out / "dim_cost_center.csv", dtype=str)

    pl = act[act["BUZEI"] == 1].merge(gl[["HKONT", "CATEGORY", "NORMAL_SIDE"]],
                                      on="HKONT")

    # Natural sign: normal-side postings positive, reversals negative.
    pl["amt"] = pl["DMBTR"] * np.where(pl["SHKZG"] == pl["NORMAL_SIDE"], 1.0, -1.0)

    # Special periods 13-16 belong to the fiscal year's December result.
    pl["MONAT"] = pl["MONAT"].clip(upper=12)

    panel = (pl.groupby(KEY + ["GJAHR", "MONAT"])["amt"].sum()
               .rename("actual").reset_index())

    # Dense grid: a cell that posted nothing in a period is a real zero.
    years = sorted(panel["GJAHR"].unique())
    grid = (panel[KEY].drop_duplicates()
            .merge(pd.MultiIndex.from_product([years, range(1, 13)],
                                              names=["GJAHR", "MONAT"]).to_frame(index=False),
                   how="cross"))
    panel = grid.merge(panel, on=KEY + ["GJAHR", "MONAT"], how="left").fillna({"actual": 0.0})

    panel["t"] = (panel["GJAHR"] - min(years)) * 12 + panel["MONAT"] - 1
    panel = panel.merge(gl[["HKONT", "CATEGORY"]], on="HKONT")
    panel = panel.merge(cc[["KOSTL", "CC_TYPE"]], on="KOSTL", how="left")

    plan = pd.read_csv(out / "plan_versions.csv",
                       dtype={"BUKRS": str, "HKONT": str, "KOSTL": str,
                              "VERSION": str})
    plan["t"] = (plan["GJAHR"] - min(years)) * 12 + plan["MONAT"] - 1
    return panel.sort_values(KEY + ["t"]).reset_index(drop=True), plan


def to_matrix(panel: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """(n_series, n_periods) matrix plus its key frame."""
    wide = panel.pivot_table(index=KEY, columns="t", values="actual",
                             aggfunc="sum").fillna(0.0)
    return wide.to_numpy(), wide.index.to_frame(index=False)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def mase_scale(train: np.ndarray) -> np.ndarray:
    """In-sample seasonal-naive MAE, per series. The MASE denominator."""
    d = np.abs(train[:, SEASON:] - train[:, :-SEASON]).mean(axis=1)
    return np.where(d < 1e-9, np.nan, d)


def score(y: np.ndarray, yhat: np.ndarray, scale: np.ndarray) -> dict:
    err = y - yhat
    ae = np.abs(err)
    return {
        "MAE": ae.mean(),
        "RMSE": float(np.sqrt((err ** 2).mean())),
        # WAPE, not MAPE: FP&A cells include near-zero months (bonus accounts),
        # where MAPE explodes. WAPE weights by magnitude, which is what a
        # finance function actually cares about.
        "WAPE": ae.sum() / np.abs(y).sum(),
        "MASE": float(np.nanmean(ae.mean(axis=1) / scale)),
    }


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
def f_naive(train: np.ndarray) -> np.ndarray:
    return np.repeat(train[:, [-1]], HORIZON, axis=1)


def f_snaive(train: np.ndarray) -> np.ndarray:
    return train[:, -SEASON:][:, :HORIZON]


def f_ets(train: np.ndarray) -> np.ndarray:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    out = np.zeros((train.shape[0], HORIZON))
    n_fail = 0
    for i in range(train.shape[0]):
        s = train[i]
        try:
            if np.all(s > 0):
                m = ExponentialSmoothing(s, trend="add", seasonal="add",
                                         seasonal_periods=SEASON,
                                         initialization_method="estimated").fit()
                out[i] = m.forecast(HORIZON)
            else:
                raise ValueError("non-positive series")
        except Exception:
            n_fail += 1
            out[i] = f_snaive(s[None, :])[0]   # fall back, do not drop
    if n_fail:
        print(f"      ETS fell back to seasonal-naive on {n_fail}/{train.shape[0]} series")
    return out


def f_lgbm(panel: pd.DataFrame, keys: pd.DataFrame) -> np.ndarray:
    """One global model. Only lags >= 12, so all 12 horizons are direct."""
    import lightgbm as lgb

    df = panel.copy()
    df = df.sort_values(KEY + ["t"])
    g = df.groupby(KEY, observed=True)["actual"]
    for L in (12, 13, 14, 15, 18, 24):
        df[f"lag{L}"] = g.shift(L)
    df["roll12"] = (df.groupby(KEY, observed=True)["actual"]
                      .transform(lambda x: x.shift(12).rolling(12, min_periods=6).mean()))
    df["roll3"] = (df.groupby(KEY, observed=True)["actual"]
                     .transform(lambda x: x.shift(12).rolling(3, min_periods=2).mean()))
    df["month"] = df["MONAT"]
    for c in ("HKONT", "CATEGORY", "CC_TYPE", "BUKRS"):
        df[c + "_c"] = df[c].astype("category")

    feats = ([f"lag{L}" for L in (12, 13, 14, 15, 18, 24)]
             + ["roll12", "roll3", "month"]
             + [c + "_c" for c in ("HKONT", "CATEGORY", "CC_TYPE", "BUKRS")])

    tr = df[(df.t < TRAIN_PERIODS) & df["lag24"].notna()]
    te = df[df.t >= TRAIN_PERIODS]

    model = lgb.LGBMRegressor(
        n_estimators=400, learning_rate=0.05, num_leaves=31,
        min_child_samples=20, subsample=0.9, colsample_bytree=0.9,
        random_state=42, verbose=-1,
    )
    model.fit(tr[feats], tr["actual"])
    te = te.assign(pred=model.predict(te[feats]))

    wide = te.pivot_table(index=KEY, columns="t", values="pred")
    return wide.reindex(pd.MultiIndex.from_frame(keys)).to_numpy()


def f_plan(plan: pd.DataFrame, version: str, keys: pd.DataFrame) -> np.ndarray:
    p = plan[(plan["VERSION"] == version) & (plan["t"] >= TRAIN_PERIODS)]
    wide = p.pivot_table(index=KEY, columns="t", values="DMBTR", aggfunc="sum")
    return wide.reindex(pd.MultiIndex.from_frame(keys)).fillna(0.0).to_numpy()


# ---------------------------------------------------------------------------
def main(out: pathlib.Path) -> int:
    panel, plan = load_panel(out)
    mat, keys = to_matrix(panel)
    train, test = mat[:, :TRAIN_PERIODS], mat[:, TRAIN_PERIODS:TRAIN_PERIODS + HORIZON]
    scale = mase_scale(train)

    n_series = mat.shape[0]
    print(f"  series {n_series}  periods {mat.shape[1]}  "
          f"train {TRAIN_PERIODS}  horizon {HORIZON}")
    print(f"  origin: close of FY2024 -> forecasting FY2025\n")

    print("  fitting...")
    fc = {
        "Naive": f_naive(train),
        "Seasonal naive": f_snaive(train),
        "ETS (Holt-Winters)": f_ets(train),
        "LightGBM (global)": f_lgbm(panel, keys),
        "Budget (v1)  [human]": f_plan(plan, "1", keys),
    }
    # Reported separately: sees within-year actuals, so not a like-for-like
    # comparison at this forecast origin.
    reference = {"Rolling forecast (v2)": f_plan(plan, "2", keys)}

    rows = []
    for name, yhat in fc.items():
        yhat = np.nan_to_num(yhat)
        rows.append({"model": name, **score(test, yhat, scale)})
    res = pd.DataFrame(rows).set_index("model")

    ref_rows = [{"model": n, **score(test, np.nan_to_num(v), scale)}
                for n, v in reference.items()]
    ref = pd.DataFrame(ref_rows).set_index("model")

    print("\n" + "=" * 78)
    print(f"  CELL-LEVEL, FY2025 ({n_series} cells x {HORIZON} months)".ljust(78))
    print("  forecast origin: FY2024 close. All models below see identical data.")
    print("=" * 78)
    print(f"  {'model':<24}{'MASE':>9}{'WAPE':>9}{'MAE':>14}{'RMSE':>16}")
    print("  " + "-" * 74)
    for m, r in res.sort_values("MASE").iterrows():
        print(f"  {m:<24}{r.MASE:>9.3f}{r.WAPE:>8.1%}{r.MAE:>14,.0f}{r.RMSE:>16,.0f}")
    print("  " + "-" * 74)
    print("  NOT RANKED -- sees within-year actuals, different information set:")
    for m, r in ref.iterrows():
        print(f"  {m:<24}{r.MASE:>9.3f}{r.WAPE:>8.1%}{r.MAE:>14,.0f}{r.RMSE:>16,.0f}")
    print("=" * 78)

    # Aggregate view: what the reporting agent actually narrates.
    print("\n  TOTAL REVENUE, FY2025 (EUR m) -- the number the board sees")
    is_rev = keys["HKONT"].isin(["400000", "410000"]).to_numpy()
    y_agg = test[is_rev].sum(axis=0)
    print(f"  {'model':<24}{'WAPE':>9}{'MAE (EUR m)':>16}")
    print("  " + "-" * 49)
    agg = []
    for name, yhat in {**fc, **reference}.items():
        p_agg = np.nan_to_num(yhat)[is_rev].sum(axis=0)
        agg.append((name, np.abs(y_agg - p_agg).sum() / y_agg.sum(),
                    np.abs(y_agg - p_agg).mean() / 1e6))
    for name, w, mae in sorted(agg, key=lambda x: x[1]):
        print(f"  {name:<24}{w:>8.1%}{mae:>16,.1f}")

    pd.concat([res.assign(ranked=True), ref.assign(ranked=False)]) \
      .to_csv(out / "baseline_results.csv")
    print(f"\n  written: {out / 'baseline_results.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "./out")))
