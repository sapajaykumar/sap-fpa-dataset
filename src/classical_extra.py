"""
classical_extra.py
==================
The classical baselines added after review: SARIMA, Theta and STL+ETS.

Why these three
---------------
The existing table carried naive, seasonal-naive and ETS. That is a thin
classical field for a paper that claims a deep-learning advantage: a reader is
entitled to ask whether the advantage survives against methods that actually
compete. Theta won the M3 competition and is strong on short series; SARIMA is
the standard seasonal benchmark; STL+ETS separates the seasonal component
before smoothing, which is the natural fit for a process built as
base x trend x seasonality.

The data constraint, stated up front
------------------------------------
36 training periods at seasonal period 12 is THREE seasonal cycles. This is
thin for every method here and it bites unevenly:

  SARIMA    seasonal differencing consumes 12 observations, leaving ~24 to fit
            (p,d,q)(P,D,Q)_12. Orders are therefore fixed and small rather than
            searched -- an auto_arima over 236 series at this length returns
            degenerate models often enough that the search is not meaningful.
  STL       needs two full cycles minimum. Three is workable, marginal.
  Theta     designed for short series. The one most likely to compete.

Non-positive series
-------------------
Some series contain zeros or negatives -- dense-grid fills plus accrual
reversals on the bonus account. ETS rejects these (it falls back on 3/236).
Theta and STL are given an additive path rather than a multiplicative one when
the series is not strictly positive, and on the released dataset that is
sufficient: all three methods here fit 236/236 with ZERO fallbacks. The guard
and the counter are retained because a regenerated dataset under different
parameters may not be so well behaved, and a silent fallback would corrupt the
comparison. The counts are printed and belong in the paper as measured.

Checkpointing
-------------
Free-tier sessions die. This writes a checkpoint every CKPT_EVERY series, so a
disconnect costs a handful of fits rather than the whole run. Re-running the
same command resumes from the last checkpoint. Delete out/ckpt/ to force a
clean run.

Usage:  python3 classical_extra.py ./out
"""

from __future__ import annotations

import sys
import pathlib
import warnings

import numpy as np
import pandas as pd

from baselines import (load_panel, to_matrix, mase_scale, score,
                       f_snaive, KEY, TRAIN_PERIODS, HORIZON, SEASON)

warnings.filterwarnings("ignore")

CKPT_EVERY = 25          # series between checkpoint writes


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
class Checkpoint:
    """
    Per-model forecast matrix plus a cursor, persisted to .npz.

    The cursor is the number of series COMPLETED, not the index in progress:
    a checkpoint written mid-series would resume from a partially written row.
    """

    def __init__(self, out: pathlib.Path, name: str, n_series: int):
        self.dir = out / "ckpt"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"{name}.npz"
        self.name = name
        self.n = n_series

    def load(self) -> tuple[np.ndarray, int, int]:
        """Returns (forecasts, cursor, n_fallback)."""
        if self.path.exists():
            z = np.load(self.path)
            cur = int(z["cursor"])
            print(f"      resuming {self.name} from series {cur}/{self.n}")
            return z["fc"], cur, int(z["n_fallback"])
        return np.zeros((self.n, HORIZON)), 0, 0

    def save(self, fc: np.ndarray, cursor: int, n_fallback: int) -> None:
        np.savez(self.path, fc=fc, cursor=cursor, n_fallback=n_fallback)

    def done(self) -> None:
        """Keep the file: a completed checkpoint is what makes a re-run cheap."""
        pass


# ---------------------------------------------------------------------------
# Per-series forecasters
# ---------------------------------------------------------------------------
def _fit_theta(s: np.ndarray) -> np.ndarray:
    from statsmodels.tsa.forecasting.theta import ThetaModel
    idx = pd.date_range("2022-01-31", periods=len(s), freq="ME")
    y = pd.Series(s, index=idx)
    # Multiplicative deseasonalisation needs strictly positive input.
    method = "mul" if np.all(s > 0) else "add"
    m = ThetaModel(y, period=SEASON, deseasonalize=True, method=method).fit()
    return np.asarray(m.forecast(HORIZON))


def _fit_stl_ets(s: np.ndarray) -> np.ndarray:
    from statsmodels.tsa.forecasting.stl import STLForecast
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    idx = pd.date_range("2022-01-31", periods=len(s), freq="ME")
    y = pd.Series(s, index=idx)
    # STL strips the seasonal component; ETS then models the remainder with
    # trend only. Seasonal is already removed, so seasonal=None here.
    m = STLForecast(
        y, ExponentialSmoothing,
        model_kwargs={"trend": "add", "seasonal": None,
                      "initialization_method": "estimated"},
        period=SEASON,
    ).fit()
    return np.asarray(m.forecast(HORIZON))


def _fit_sarima(s: np.ndarray) -> np.ndarray:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    # Fixed small orders. With 36 observations and D=1 there are ~24 effective
    # points; anything larger is fitting noise. (1,0,0)(0,1,0)_12 is the
    # seasonal-random-walk-with-AR-drift that a short seasonal series supports.
    m = SARIMAX(s, order=(1, 0, 0), seasonal_order=(0, 1, 0, SEASON),
                enforce_stationarity=False, enforce_invertibility=False)
    r = m.fit(disp=False, maxiter=200)
    return np.asarray(r.forecast(HORIZON))


FITTERS = {
    "Theta": _fit_theta,
    "STL+ETS": _fit_stl_ets,
    "SARIMA": _fit_sarima,
}


def run_model(name: str, train: np.ndarray, out: pathlib.Path) -> tuple[np.ndarray, int]:
    """Fit one model across all series, checkpointing as it goes."""
    n = train.shape[0]
    ck = Checkpoint(out, name.replace("+", "_"), n)
    fc, cursor, n_fb = ck.load()
    fit = FITTERS[name]

    for i in range(cursor, n):
        s = train[i]
        try:
            p = fit(s)
            if not np.all(np.isfinite(p)):
                raise ValueError("non-finite forecast")
            fc[i] = p
        except Exception:
            n_fb += 1
            fc[i] = f_snaive(s[None, :])[0]     # fall back, never drop
        if (i + 1) % CKPT_EVERY == 0:
            ck.save(fc, i + 1, n_fb)
            print(f"      {name}: {i + 1}/{n} series")

    ck.save(fc, n, n_fb)
    if n_fb:
        print(f"      {name} fell back to seasonal-naive on {n_fb}/{n} series")
    return fc, n_fb


# ---------------------------------------------------------------------------
def main(out: pathlib.Path) -> int:
    panel, _ = load_panel(out)
    mat, keys = to_matrix(panel)
    train = mat[:, :TRAIN_PERIODS]
    test = mat[:, TRAIN_PERIODS:TRAIN_PERIODS + HORIZON]
    scale = mase_scale(train)

    print(f"  series {mat.shape[0]}  train {TRAIN_PERIODS}  horizon {HORIZON}")
    print(f"  origin: close of FY2024 -> forecasting FY2025")
    print(f"  checkpoints: {out / 'ckpt'}  (delete to force a clean run)\n")
    print("  fitting...")

    rows, fallbacks = [], {}
    for name in ("Theta", "STL+ETS", "SARIMA"):
        fc, n_fb = run_model(name, train, out)
        fallbacks[name] = n_fb
        rows.append({"model": name, **score(test, np.nan_to_num(fc), scale)})

    res = pd.DataFrame(rows).set_index("model")

    # Merge with the existing table so one CSV carries the full field.
    prev_path = out / "baseline_results.csv"
    if prev_path.exists():
        prev = pd.read_csv(prev_path, index_col=0)
        prev = prev[~prev.index.isin(res.index)]          # idempotent re-run
        full = pd.concat([prev, res.assign(ranked=True)])
    else:
        full = res.assign(ranked=True)

    ranked = full[full.get("ranked", True) == True].sort_values("MASE")
    unranked = full[full.get("ranked", True) == False]

    print("\n" + "=" * 78)
    print(f"  CELL-LEVEL, FY2025 -- FULL CLASSICAL FIELD".ljust(78))
    print("  forecast origin: FY2024 close. All models below see identical data.")
    print("=" * 78)
    print(f"  {'model':<24}{'MASE':>9}{'WAPE':>9}{'MAE':>14}{'RMSE':>16}")
    print("  " + "-" * 74)
    for m, r in ranked.iterrows():
        print(f"  {m:<24}{r.MASE:>9.3f}{r.WAPE:>8.1%}{r.MAE:>14,.0f}{r.RMSE:>16,.0f}")
    if len(unranked):
        print("  " + "-" * 74)
        print("  NOT RANKED -- sees within-year actuals:")
        for m, r in unranked.iterrows():
            print(f"  {m:<24}{r.MASE:>9.3f}{r.WAPE:>8.1%}{r.MAE:>14,.0f}{r.RMSE:>16,.0f}")
    print("=" * 78)

    print("\n  Seasonal-naive fallbacks (report these in the paper):")
    for k, v in fallbacks.items():
        print(f"    {k:<12} {v}/{mat.shape[0]}")

    # Aggregate revenue view -- the ranking that differs from cell level.
    print("\n  TOTAL REVENUE, FY2025 -- aggregate, not cell-level")
    is_rev = keys["HKONT"].isin(["400000", "410000"]).to_numpy()
    y_agg = test[is_rev].sum(axis=0)
    agg = []
    for name in ("Theta", "STL+ETS", "SARIMA"):
        z = np.load(out / "ckpt" / f"{name.replace('+', '_')}.npz")["fc"]
        p_agg = np.nan_to_num(z)[is_rev].sum(axis=0)
        agg.append((name, np.abs(y_agg - p_agg).sum() / y_agg.sum()))
    for name, w in sorted(agg, key=lambda x: x[1]):
        print(f"    {name:<12} WAPE {w:>6.1%}")

    full.to_csv(prev_path)
    print(f"\n  written: {prev_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "./out")))
