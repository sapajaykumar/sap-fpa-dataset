"""
refit_tuned.py
==============
Takes the configurations selected by metaheuristic.py, refits them on the full
36-period training span, and scores FY2025.

This is the only point in the tuning pipeline at which FY2025 is touched. The
search itself scored every candidate on FY2024 under an inner split, so nothing
that influenced the choice of configuration has seen the test year.

Each tuned model is fit once per seed and the seed spread is reported, matching
the protocol used for the untuned deep-learning results. A tuned configuration
that wins by less than its own seed spread has not really won.

Usage:  python3 refit_tuned.py ./out
"""

from __future__ import annotations

import sys
import json
import pathlib
import warnings

import numpy as np
import pandas as pd

from baselines import (load_panel, to_matrix, mase_scale, KEY,
                       TRAIN_PERIODS, HORIZON)
from metaheuristic import SPACE_LGBM, SPACE_NBEATS, _lgbm_frame

warnings.filterwarnings("ignore")

SEEDS = (0, 1, 2)


def fit_lgbm(panel, keys, cfg, seed=42):
    import lightgbm as lgb
    df, feats = _lgbm_frame(panel, cfg["max_lag"])
    tr = df[df.t < TRAIN_PERIODS]
    te = df[df.t >= TRAIN_PERIODS]
    m = lgb.LGBMRegressor(
        num_leaves=cfg["num_leaves"], learning_rate=cfg["learning_rate"],
        n_estimators=cfg["n_estimators"], min_child_samples=cfg["min_child_samples"],
        colsample_bytree=cfg["feature_fraction"], subsample=cfg["bagging_fraction"],
        subsample_freq=1, reg_alpha=cfg["lambda_l1"], reg_lambda=cfg["lambda_l2"],
        random_state=seed, verbose=-1, n_jobs=-1)
    m.fit(tr[feats], tr["actual"])
    te = te.assign(pred=m.predict(te[feats]))
    wide = te.pivot_table(index=KEY, columns="t", values="pred")
    return np.nan_to_num(wide.reindex(pd.MultiIndex.from_frame(keys)).to_numpy())


def fit_nbeats(mat, meta, cfg, seed):
    import torch
    import nbeats as NB

    train = mat[:, :TRAIN_PERIODS]
    a = pd.Categorical(meta["HKONT"]).codes
    c = pd.Categorical(meta["CC_TYPE"].fillna("NA")).codes
    b = pd.Categorical(meta["BUKRS"]).codes
    dev = NB.DEVICE

    Xw, Yw, sid, start = NB.make_windows(train)
    Xn, Yn, _ = NB.normalise(Xw, Yw)
    T = lambda v, dt=torch.float32: torch.tensor(v, dtype=dt, device=dev)
    tensors = {"X": T(Xn), "Y": T(Yn), "START": start,
               "A": T(a[sid], torch.long), "C": T(c[sid], torch.long),
               "B": T(b[sid], torch.long)}
    keys_enc = {"A": T(a, torch.long), "C": T(c, torch.long), "B": T(b, torch.long)}

    torch.manual_seed(seed)
    model = NB.NBeats("generic", width=cfg["width"], blocks=cfg["blocks"]).to(dev)
    model = NB.train_model(model, tensors, seed, lr=cfg["learning_rate"])
    return NB.forecast(model, mat, keys_enc)


def main(out: pathlib.Path) -> int:
    res_path = out / "metaheuristic_results.json"
    if not res_path.exists():
        print("  metaheuristic_results.json not found -- run metaheuristic.py first")
        return 1
    found = json.loads(res_path.read_text())

    panel, plan = load_panel(out)
    mat, keys = to_matrix(panel)
    meta = (panel[KEY + ["CATEGORY", "CC_TYPE"]].drop_duplicates()
            .merge(keys, on=KEY, how="right"))
    train = mat[:, :TRAIN_PERIODS]
    test = mat[:, TRAIN_PERIODS:TRAIN_PERIODS + HORIZON]
    scale = mase_scale(train)

    def mase(pred):
        return float(np.nanmean(np.abs(test - np.nan_to_num(pred)).mean(axis=1) / scale))

    print("=" * 78)
    print("  TUNED MODELS ON FY2025")
    print("=" * 78)
    print("  Configurations selected on FY2024; this is the first use of FY2025.\n")

    rows = []
    preds = {}

    # pick the best optimiser per family on the inner split
    for fam in ("LightGBM", "N-BEATS"):
        cand = {k: v for k, v in found.items() if k.startswith(fam + "_")}
        if not cand:
            continue
        best_tag = min(cand, key=lambda k: cand[k]["best_f"])
        cfg = cand[best_tag]["cfg"]
        print(f"  {fam}: best inner score from {best_tag.rsplit('_',1)[1]} "
              f"({cand[best_tag]['best_f']:.4f})")

        if fam == "LightGBM":
            p = fit_lgbm(panel, keys, cfg)
            preds[f"LightGBM (tuned)"] = p
            rows.append({"model": "LightGBM (tuned)", "MASE": mase(p), "sd": np.nan,
                         "selected_by": best_tag.rsplit("_", 1)[1]})
        else:
            ms, ps = [], []
            for s in SEEDS:
                p = fit_nbeats(mat, meta, cfg, s)
                ps.append(p); ms.append(mase(p))
            mp = np.mean(ps, axis=0)
            preds["N-BEATS generic (tuned)"] = mp
            rows.append({"model": "N-BEATS generic (tuned)", "MASE": float(np.mean(ms)),
                         "sd": float(np.std(ms)), "selected_by": best_tag.rsplit("_", 1)[1]})

    df = pd.DataFrame(rows).set_index("model")

    # untuned reference
    prev = pd.read_csv(out / "baseline_results.csv", index_col=0)
    dl = pd.read_csv(out / "dl_results.csv", index_col=0) if (out / "dl_results.csv").exists() else None

    print("\n" + "=" * 78)
    print("  TUNED vs UNTUNED, FY2025 cell-level MASE")
    print("=" * 78)
    print(f"  {'model':<30}{'MASE':>9}{'sd':>8}{'selected by':>14}")
    print("  " + "-" * 74)
    for m, r in df.iterrows():
        sd = "--" if np.isnan(r["sd"]) else f"{r['sd']:.3f}"
        print(f"  {m:<30}{r['MASE']:>9.3f}{sd:>8}{r['selected_by']:>14}")
    print("  " + "-" * 74)
    print("  untuned reference:")
    if dl is not None and "NBEATS-G (generic)" in dl.index:
        print(f"  {'N-BEATS generic (default)':<30}{dl.loc['NBEATS-G (generic)','MASE']:>9.3f}"
              f"{dl.loc['NBEATS-G (generic)','sd']:>8.3f}")
    if "LightGBM (global)" in prev.index:
        print(f"  {'LightGBM (default)':<30}{prev.loc['LightGBM (global)','MASE']:>9.3f}{'--':>8}")
    print("=" * 78)

    np.savez(out / "tuned_predictions.npz", **preds)
    df.to_csv(out / "tuned_results.csv")
    print(f"\n  written: {out / 'tuned_results.csv'}")
    print("\n  Read the tuned N-BEATS gain against its own seed s.d. before")
    print("  claiming an improvement.")
    return 0


if __name__ == "__main__":
    sys.exit(main(pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "./out")))
