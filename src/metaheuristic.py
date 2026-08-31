"""
metaheuristic.py
================
Hyperparameter optimisation for the two learned models, by genetic algorithm,
particle swarm, and a budget-matched random search.

Why
---
Review comment: the performance gap between N-BEATS and LightGBM is modest.
The obvious objection to the reported ranking is that the neural model was
tuned while the tree model ran near its defaults, so the comparison measures
effort rather than architecture. This module removes that objection by
searching both families under an identical budget.

Random search is included as a CONTROL, not as a third competitor. Without it,
a GA or PSO result shows only that searching beats not searching -- which is
uninteresting and already known. The question worth answering is whether a
population-based method finds anything a matched random draw does not.

The information boundary -- the thing that must not break
---------------------------------------------------------
FY2025 is never visible to the search. Configurations are scored on an inner
split entirely inside the training span:

    inner train    2022-01 .. 2023-12   (t < 24)
    inner validate 2024-01 .. 2024-12   (t = 24..35)

The winning configuration is then refit on the full 36-period training span and
used to forecast FY2025 exactly once. The objective function is not given the
test array at all, so selecting on FY2025 is not merely discouraged by
convention -- it is unavailable.

A note on the lag filter
------------------------
The original LightGBM baseline required lag-24 to be non-null before a row
entered training, which silently restricted training to t = 24..35: 2,832 rows
out of a possible 5,664. That also made the earlier "remove the deepest lag"
fairness check confounded, since removing the feature also removed the filter.

LightGBM handles missing values natively, so the filter was never necessary.
Dropping it fixes the confound AND makes the maximum lag searchable, which is
the more interesting question: is the reported 1.034 a property of the model or
of an incidental feature-construction choice? max_lag is therefore part of the
search space rather than a constant.

Pre-registered interpretation
-----------------------------
Written before the search was run.

  If tuned LightGBM reaches parity with N-BEATS, the paper's claim narrows to
  aggregate parity with the deep-learning advantage confined to volatile cells
  -- which Figure 3 and Section 7.4 already support, and which is the more
  defensible claim.

  If N-BEATS retains its lead after both families are tuned, the existing claim
  stands on stronger ground, because the tabular baseline can no longer be
  described as under-tuned.

  If GA and PSO do not beat matched random search, that is reported as a null
  result. At these budgets and dimensionalities it is a plausible outcome and
  suppressing it would be exactly the reporting bias this project criticises.

Usage:  python3 metaheuristic.py ./out [budget]
"""

from __future__ import annotations

import sys
import json
import time
import pathlib
import warnings

import numpy as np
import pandas as pd

from baselines import (load_panel, to_matrix, mase_scale, KEY,
                       TRAIN_PERIODS, HORIZON, SEASON)

warnings.filterwarnings("ignore")

INNER_TRAIN = 24          # t < 24  -> fit
INNER_VAL = 36            # t 24..35 -> score
DEFAULT_BUDGET = 40       # evaluations per optimiser per family
SEEDS = (0, 1, 2)


# ---------------------------------------------------------------------------
# Search spaces. Every parameter is encoded on [0, 1] so that the three
# optimisers operate on an identical geometry and the budget is comparable.
# ---------------------------------------------------------------------------
SPACE_LGBM = [
    ("max_lag",           "choice", [12, 15, 18, 24]),
    ("num_leaves",        "int",    (8, 128)),
    ("learning_rate",     "logf",   (0.01, 0.30)),
    ("n_estimators",      "int",    (100, 900)),
    ("min_child_samples", "int",    (5, 60)),
    ("feature_fraction",  "float",  (0.5, 1.0)),
    ("bagging_fraction",  "float",  (0.5, 1.0)),
    ("lambda_l1",         "logf",   (1e-3, 10.0)),
    ("lambda_l2",         "logf",   (1e-3, 10.0)),
]

SPACE_NBEATS = [
    ("width",        "choice", [64, 96, 128, 192]),
    ("blocks",       "choice", [2, 3, 4]),
    ("learning_rate","logf",   (3e-4, 5e-3)),
    ("batch_size",   "choice", [128, 256, 512]),
    ("weight_decay", "logf",   (1e-6, 1e-2)),
]


def decode(vec: np.ndarray, space) -> dict:
    """[0,1]^d -> a concrete configuration."""
    cfg = {}
    for v, (name, kind, rng) in zip(vec, space):
        v = float(np.clip(v, 0.0, 1.0))
        if kind == "choice":
            i = min(int(v * len(rng)), len(rng) - 1)
            cfg[name] = rng[i]
        elif kind == "int":
            cfg[name] = int(round(rng[0] + v * (rng[1] - rng[0])))
        elif kind == "float":
            cfg[name] = float(rng[0] + v * (rng[1] - rng[0]))
        elif kind == "logf":
            lo, hi = np.log(rng[0]), np.log(rng[1])
            cfg[name] = float(np.exp(lo + v * (hi - lo)))
    return cfg


# ---------------------------------------------------------------------------
# Objectives. Both score on FY2024 only.
# ---------------------------------------------------------------------------
def _lgbm_frame(panel: pd.DataFrame, max_lag: int) -> tuple[pd.DataFrame, list]:
    import numpy as np
    df = panel.sort_values(KEY + ["t"]).copy()
    g = df.groupby(KEY, observed=True)["actual"]
    lags = [L for L in (12, 13, 14, 15, 18, 24) if L <= max_lag]
    for L in lags:
        df[f"lag{L}"] = g.shift(L)
    df["roll12"] = g.transform(lambda x: x.shift(12).rolling(12, min_periods=6).mean())
    df["roll3"] = g.transform(lambda x: x.shift(12).rolling(3, min_periods=2).mean())
    df["month"] = df["MONAT"]
    for c in ("HKONT", "CATEGORY", "CC_TYPE", "BUKRS"):
        df[c + "_c"] = df[c].astype("category")
    feats = ([f"lag{L}" for L in lags] + ["roll12", "roll3", "month"]
             + [c + "_c" for c in ("HKONT", "CATEGORY", "CC_TYPE", "BUKRS")])
    return df, feats


def make_lgbm_objective(panel, keys, mat):
    """Fit on t < 24, score cell-level MASE on FY2024. FY2025 not referenced."""
    import lightgbm as lgb
    inner_train = mat[:, :INNER_TRAIN]
    inner_val = mat[:, INNER_TRAIN:INNER_VAL]
    scale = mase_scale(inner_train)

    def objective(vec):
        cfg = decode(vec, SPACE_LGBM)
        df, feats = _lgbm_frame(panel, cfg["max_lag"])
        # No .notna() filter: LightGBM handles missing lags natively, so every
        # in-span row is usable regardless of the deepest lag requested.
        tr = df[df.t < INNER_TRAIN]
        va = df[(df.t >= INNER_TRAIN) & (df.t < INNER_VAL)]
        m = lgb.LGBMRegressor(
            num_leaves=cfg["num_leaves"], learning_rate=cfg["learning_rate"],
            n_estimators=cfg["n_estimators"], min_child_samples=cfg["min_child_samples"],
            colsample_bytree=cfg["feature_fraction"], subsample=cfg["bagging_fraction"],
            subsample_freq=1, reg_alpha=cfg["lambda_l1"], reg_lambda=cfg["lambda_l2"],
            random_state=42, verbose=-1, n_jobs=-1)
        m.fit(tr[feats], tr["actual"])
        va = va.assign(pred=m.predict(va[feats]))
        wide = va.pivot_table(index=KEY, columns="t", values="pred")
        pred = wide.reindex(pd.MultiIndex.from_frame(keys)).to_numpy()
        pred = np.nan_to_num(pred)
        return float(np.nanmean(np.abs(inner_val - pred).mean(axis=1) / scale))

    return objective


def make_nbeats_objective(mat, meta):
    """
    Score = best validation loss on windows targeting FY2024.

    nbeats.train_model already splits validation temporally on window start,
    so the held-out windows target periods 24-35. Restricting the window pool
    to the inner training span keeps FY2025 out of both fitting and selection.
    """
    import torch
    import nbeats as NB

    inner = mat[:, :INNER_VAL]
    a_codes = pd.Categorical(meta["HKONT"]).codes
    c_codes = pd.Categorical(meta["CC_TYPE"].fillna("NA")).codes
    b_codes = pd.Categorical(meta["BUKRS"]).codes
    dev = NB.DEVICE

    Xw, Yw, sid, start = NB.make_windows(inner)
    Xn, Yn, _ = NB.normalise(Xw, Yw)
    T = lambda v, dt=torch.float32: torch.tensor(v, dtype=dt, device=dev)
    tensors = {"X": T(Xn), "Y": T(Yn), "START": start,
               "A": T(a_codes[sid], torch.long),
               "C": T(c_codes[sid], torch.long),
               "B": T(b_codes[sid], torch.long)}
    va = np.where(start == start.max())[0]

    def objective(vec):
        cfg = decode(vec, SPACE_NBEATS)
        torch.manual_seed(0)
        model = NB.NBeats("generic", width=cfg["width"], blocks=cfg["blocks"]).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"],
                               weight_decay=cfg["weight_decay"])
        tr = np.where(start < start.max())[0]
        best, patience = np.inf, 0
        for ep in range(140):                     # shorter than the final fit
            model.train()
            perm = np.random.permutation(tr)
            bs = cfg["batch_size"]
            for i in range(0, len(perm), bs):
                b = perm[i:i + bs]
                opt.zero_grad()
                loss = (model(tensors["X"][b]) - tensors["Y"][b]).abs().mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            model.eval()
            with torch.no_grad():
                v = (model(tensors["X"][va]) - tensors["Y"][va]).abs().mean().item()
            if v < best - 1e-5:
                best, patience = v, 0
            else:
                patience += 1
                if patience > 25:
                    break
        return float(best)

    return objective


# ---------------------------------------------------------------------------
# Optimisers. All three consume exactly `budget` objective evaluations.
# ---------------------------------------------------------------------------
def random_search(obj, dim, budget, seed, log):
    rng = np.random.default_rng(seed)
    best_x, best_f = None, np.inf
    for i in range(budget):
        x = rng.random(dim)
        f = obj(x)
        log(i, f, best_f)
        if f < best_f:
            best_x, best_f = x, f
    return best_x, best_f


def genetic(obj, dim, budget, seed, log, pop_size=10):
    """Tournament selection, blend crossover, Gaussian mutation, elitism of 1."""
    rng = np.random.default_rng(seed)
    pop = rng.random((pop_size, dim))
    fit = np.array([obj(p) for p in pop])
    used = pop_size
    for i in range(used):
        log(i, fit[i], fit[:i + 1].min())
    while used < budget:
        new = [pop[np.argmin(fit)].copy()]                # elitism
        while len(new) < pop_size and used + len(new) - 1 < budget:
            def pick():
                a, b = rng.integers(0, pop_size, 2)
                return pop[a] if fit[a] < fit[b] else pop[b]
            p1, p2 = pick(), pick()
            alpha = rng.random(dim) * 1.4 - 0.2           # BLX-0.2
            child = np.clip(alpha * p1 + (1 - alpha) * p2, 0, 1)
            mask = rng.random(dim) < 0.25
            child[mask] = np.clip(child[mask] + rng.normal(0, 0.15, mask.sum()), 0, 1)
            new.append(child)
        pop = np.array(new)
        f_new = []
        for p in pop[1:]:
            if used >= budget:
                break
            f = obj(p); used += 1
            f_new.append(f)
            log(used - 1, f, min(fit.min(), min(f_new)))
        fit = np.concatenate([[fit.min()], np.array(f_new)])
        pop = pop[:len(fit)]
    return pop[np.argmin(fit)], float(fit.min())


def pso(obj, dim, budget, seed, log, n_particles=10, w=0.7, c1=1.4, c2=1.4):
    rng = np.random.default_rng(seed)
    x = rng.random((n_particles, dim))
    v = rng.normal(0, 0.1, (n_particles, dim))
    f = np.array([obj(p) for p in x])
    used = n_particles
    for i in range(used):
        log(i, f[i], f[:i + 1].min())
    pbest_x, pbest_f = x.copy(), f.copy()
    gi = int(np.argmin(f)); gbest_x, gbest_f = x[gi].copy(), float(f[gi])
    while used < budget:
        r1, r2 = rng.random((n_particles, dim)), rng.random((n_particles, dim))
        v = w * v + c1 * r1 * (pbest_x - x) + c2 * r2 * (gbest_x - x)
        v = np.clip(v, -0.3, 0.3)
        x = np.clip(x + v, 0, 1)
        for i in range(n_particles):
            if used >= budget:
                break
            fi = obj(x[i]); used += 1
            if fi < pbest_f[i]:
                pbest_x[i], pbest_f[i] = x[i].copy(), fi
            if fi < gbest_f:
                gbest_x, gbest_f = x[i].copy(), fi
            log(used - 1, fi, gbest_f)
    return gbest_x, gbest_f


OPTIMISERS = {"Random": random_search, "GA": genetic, "PSO": pso}


# ---------------------------------------------------------------------------
def main(out: pathlib.Path, budget: int = DEFAULT_BUDGET) -> int:
    panel, plan = load_panel(out)
    mat, keys = to_matrix(panel)
    meta = (panel[KEY + ["CATEGORY", "CC_TYPE"]].drop_duplicates()
            .merge(keys, on=KEY, how="right"))

    ck = out / "ckpt_meta"
    ck.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("  METAHEURISTIC HYPERPARAMETER SEARCH")
    print("=" * 78)
    print(f"  inner split: fit t<{INNER_TRAIN}, score FY2024 (t={INNER_TRAIN}..{INNER_VAL-1})")
    print(f"  FY2025 is not referenced by any objective function")
    print(f"  budget: {budget} evaluations per optimiser per family\n")

    families = [
        ("LightGBM", SPACE_LGBM, make_lgbm_objective(panel, keys, mat)),
        ("N-BEATS",  SPACE_NBEATS, make_nbeats_objective(mat, meta)),
    ]

    results = {}
    for fam_name, space, obj in families:
        dim = len(space)
        for opt_name, opt in OPTIMISERS.items():
            tag = f"{fam_name}_{opt_name}"
            path = ck / f"{tag}.json"
            if path.exists():
                results[tag] = json.loads(path.read_text())
                print(f"  {tag:<22} cached  best {results[tag]['best_f']:.4f}")
                continue

            t0 = time.time()
            trace = []
            def log(i, f, best, _tr=trace):
                _tr.append(float(f))
                if (i + 1) % 10 == 0:
                    print(f"    {tag:<22} {i+1:>3}/{budget}  best {best:.4f}")

            x, f = opt(obj, dim, budget, seed=42, log=log)
            cfg = decode(x, space)
            results[tag] = {"best_f": f, "cfg": cfg, "trace": trace,
                            "seconds": time.time() - t0}
            path.write_text(json.dumps(results[tag], indent=1))
            print(f"  {tag:<22} done    best {f:.4f}  ({time.time()-t0:.0f}s)")

    # -- summary on the inner split -------------------------------------
    print("\n" + "=" * 78)
    print("  INNER-SPLIT SCORES (FY2024). Lower is better.")
    print("=" * 78)
    print(f"  {'family':<12}{'optimiser':<12}{'best':>10}{'evals':>8}{'time (s)':>10}")
    print("  " + "-" * 74)
    for tag, r in results.items():
        fam, opt = tag.rsplit("_", 1)
        print(f"  {fam:<12}{opt:<12}{r['best_f']:>10.4f}{len(r['trace']):>8}"
              f"{r['seconds']:>10.0f}")

    print("\n  Did the population-based methods beat matched random search?")
    for fam in ("LightGBM", "N-BEATS"):
        rnd = results.get(f"{fam}_Random", {}).get("best_f")
        if rnd is None:
            continue
        for opt in ("GA", "PSO"):
            v = results.get(f"{fam}_{opt}", {}).get("best_f")
            if v is None:
                continue
            delta = rnd - v
            verdict = "better" if delta > 0 else ("worse" if delta < 0 else "tie")
            print(f"    {fam:<10} {opt:<4} vs Random: {delta:+.4f}  ({verdict})")

    print("\n  Best configurations found:")
    for tag, r in results.items():
        print(f"    {tag}")
        for k, v in r["cfg"].items():
            vv = f"{v:.5f}" if isinstance(v, float) else v
            print(f"      {k:<20} {vv}")

    json.dump({k: {"best_f": v["best_f"], "cfg": v["cfg"]}
               for k, v in results.items()},
              open(out / "metaheuristic_results.json", "w"), indent=1)
    print(f"\n  written: {out / 'metaheuristic_results.json'}")
    print("\n  Next: refit_tuned.py takes the winning configurations, refits on the")
    print("  full 36-period span, and reports FY2025 test MASE. That is the only")
    print("  point at which FY2025 is touched.")
    return 0


if __name__ == "__main__":
    o = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "./out")
    b = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_BUDGET
    sys.exit(main(o, b))
