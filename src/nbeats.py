"""
nbeats.py
=========
Deep learning forecasters for the FP&A cycle, measured against the baseline
table in baselines.py. Same panel, same split, same MASE denominator, so the
numbers are directly comparable.

    train   2022-01 .. 2024-12   (36 periods)
    test    2025-01 .. 2025-12   (12 periods)
    origin  close of FY2024

The data scale, stated up front
-------------------------------
236 series x 36 training periods. With lookback L=12 and horizon H=12 that is
36 - 12 - 12 + 1 = 13 windows per series, or 3,068 training samples total.
L=24 would give ONE window per series. This is the entire budget, and it is
roughly two orders of magnitude below what N-BEATS was designed for (M4 monthly
series carry 100+ observations each).

Three models, for three different reasons:

  NBEATS-G   generic N-BEATS (Oreshkin et al., 2020). The default configuration.
             Included because it is what "N-BEATS" means without qualification,
             and because its parameter count against 3,068 samples is itself a
             result worth reporting.

  NBEATS-I   interpretable N-BEATS: a trend stack on a polynomial basis and a
             seasonality stack on a Fourier basis. Far fewer effective
             parameters, and the decomposition matches the known generating
             process (base x trend x seasonality x AR(1)). This is the
             configuration that should work, if any does.

  GlobalMLP  lookback window + learned embeddings for HKONT, CC_TYPE and BUKRS.
             Tests the actual hypothesis: does cross-series learning help on the
             sparse, spiky cells, even where it ties on revenue?

Three seeds each. At this data scale a single-seed number is noise, and
reporting one against LightGBM's 1.034 would be a comparison of luck.

Usage:  python3 nbeats.py ./out
"""

from __future__ import annotations

import sys
import pathlib
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from baselines import load_panel, to_matrix, mase_scale, KEY, TRAIN_PERIODS, HORIZON

warnings.filterwarnings("ignore")

LOOKBACK = 12
SEEDS = (0, 1, 2)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------
def make_windows(train: np.ndarray):
    """All (lookback, horizon) pairs inside the training span. No leakage:
    every window ends at or before period 35.

    Returns the window START time as well as the series id. The start time is
    required for a temporally honest validation split -- see train_model."""
    xs, ys, sid, start = [], [], [], []
    n_win = train.shape[1] - LOOKBACK - HORIZON + 1
    for i in range(train.shape[0]):
        for t in range(n_win):
            xs.append(train[i, t:t + LOOKBACK])
            ys.append(train[i, t + LOOKBACK:t + LOOKBACK + HORIZON])
            sid.append(i)
            start.append(t)
    return np.array(xs), np.array(ys), np.array(sid), np.array(start)


def normalise(x: np.ndarray, y: np.ndarray | None = None):
    """Per-window scaling by the lookback mean. Amounts span orders of
    magnitude across accounts; without this the loss is dominated by revenue."""
    s = x.mean(axis=1, keepdims=True)
    s = np.where(np.abs(s) < 1e-6, 1.0, s)
    return (x / s, None if y is None else y / s, s)


# ---------------------------------------------------------------------------
# N-BEATS
# ---------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, width: int, theta: int, basis_b, basis_f, layers: int = 3):
        super().__init__()
        seq = []
        d = LOOKBACK
        for _ in range(layers):
            seq += [nn.Linear(d, width), nn.ReLU()]
            d = width
        self.fc = nn.Sequential(*seq)
        self.theta_b = nn.Linear(width, theta, bias=False)
        self.theta_f = nn.Linear(width, theta, bias=False)
        self.register_buffer("Bb", basis_b)
        self.register_buffer("Bf", basis_f)

    def forward(self, x):
        h = self.fc(x)
        return self.theta_b(h) @ self.Bb, self.theta_f(h) @ self.Bf


def poly_basis(deg: int, n: int) -> torch.Tensor:
    t = torch.linspace(0, 1, n)
    return torch.stack([t ** i for i in range(deg + 1)])


def fourier_basis(harmonics: int, n: int) -> torch.Tensor:
    t = torch.linspace(0, 1, n)
    rows = [torch.ones(n)]
    for i in range(1, harmonics + 1):
        rows += [torch.cos(2 * np.pi * i * t), torch.sin(2 * np.pi * i * t)]
    return torch.stack(rows)


def identity_basis(n: int) -> torch.Tensor:
    return torch.eye(n)


class NBeats(nn.Module):
    def __init__(self, mode: str = "generic", width: int = 128, blocks: int = 3):
        super().__init__()
        self.blocks = nn.ModuleList()
        if mode == "generic":
            # Linear (identity) basis: the model must learn the basis itself.
            for _ in range(blocks * 2):
                self.blocks.append(Block(width, LOOKBACK,
                                         identity_basis(LOOKBACK),
                                         torch.eye(HORIZON)[:LOOKBACK, :]
                                         if LOOKBACK >= HORIZON else
                                         torch.eye(LOOKBACK, HORIZON)))
        else:
            # Interpretable: trend stack then seasonality stack.
            deg, harm = 3, 4
            for _ in range(blocks):
                self.blocks.append(Block(width, deg + 1,
                                         poly_basis(deg, LOOKBACK),
                                         poly_basis(deg, HORIZON)))
            for _ in range(blocks):
                self.blocks.append(Block(width, 2 * harm + 1,
                                         fourier_basis(harm, LOOKBACK),
                                         fourier_basis(harm, HORIZON)))

    def forward(self, x):
        res, fc = x, 0.0
        for b in self.blocks:
            bc, f = b(res)
            res = res - bc          # doubly-residual stacking
            fc = fc + f
        return fc


class GlobalMLP(nn.Module):
    """Lookback + categorical embeddings. Tests cross-series learning."""

    def __init__(self, n_acct, n_cctype, n_bukrs, emb=8, hidden=128):
        super().__init__()
        self.e_a = nn.Embedding(n_acct, emb)
        self.e_c = nn.Embedding(n_cctype, emb)
        self.e_b = nn.Embedding(n_bukrs, emb)
        self.net = nn.Sequential(
            nn.Linear(LOOKBACK + emb * 3, hidden), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, HORIZON),
        )

    def forward(self, x, a, c, b):
        return self.net(torch.cat([x, self.e_a(a), self.e_c(c), self.e_b(b)], 1))


# ---------------------------------------------------------------------------
def train_model(model, tensors, seed, epochs=300, lr=1e-3, is_mlp=False):
    """
    Validation is split TEMPORALLY, on the window start time.

    The original split was a random 85/15 over all windows. Windows overlap at
    stride 1: window t=5 shares 11 of its 12 lookback periods AND 11 of its 12
    target periods with window t=6. A random split therefore puts near-identical
    windows on both sides, and early stopping selects on memorisation rather
    than generalisation.

    The last window start (t=12, targeting periods 24-35) is held out instead.
    Those targets are observed training-span data and later become the forecast
    input, which is not leakage -- they are inside the training region either way.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    X, Y = tensors["X"], tensors["Y"]
    start = tensors["START"]
    va = np.where(start == start.max())[0]
    tr = np.where(start < start.max())[0]

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=15, factor=0.5)
    best, best_state, patience = np.inf, None, 0

    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(tr)
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]
            opt.zero_grad()
            out = (model(X[b], tensors["A"][b], tensors["C"][b], tensors["B"][b])
                   if is_mlp else model(X[b]))
            loss = (out - Y[b]).abs().mean()      # MAE: matches the MASE metric
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            out = (model(X[va], tensors["A"][va], tensors["C"][va], tensors["B"][va])
                   if is_mlp else model(X[va]))
            v = (out - Y[va]).abs().mean().item()
        sched.step(v)
        if v < best - 1e-5:
            best, best_state, patience = v, {k: t.clone() for k, t in
                                             model.state_dict().items()}, 0
        else:
            patience += 1
            if patience > 40:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model


def forecast(model, mat, keys_enc, is_mlp=False) -> np.ndarray:
    """Predict FY2025 from the last LOOKBACK periods of the training span."""
    ctx = mat[:, TRAIN_PERIODS - LOOKBACK:TRAIN_PERIODS]
    xn, _, s = normalise(ctx)
    model.eval()
    with torch.no_grad():
        x = torch.tensor(xn, dtype=torch.float32, device=DEVICE)
        out = (model(x, keys_enc["A"], keys_enc["C"], keys_enc["B"])
               if is_mlp else model(x))
    return out.cpu().numpy() * s


def main(out: pathlib.Path) -> int:
    panel, _ = load_panel(out)
    mat, keys = to_matrix(panel)
    train, test = mat[:, :TRAIN_PERIODS], mat[:, TRAIN_PERIODS:TRAIN_PERIODS + HORIZON]
    scale = mase_scale(train)

    meta = (panel[KEY + ["CATEGORY", "CC_TYPE"]].drop_duplicates()
            .merge(keys, on=KEY, how="right"))
    cat = meta["CATEGORY"].to_numpy()
    a_codes = pd.Categorical(meta["HKONT"]).codes
    c_codes = pd.Categorical(meta["CC_TYPE"].fillna("NA")).codes
    b_codes = pd.Categorical(meta["BUKRS"]).codes

    Xw, Yw, sid, start = make_windows(train)
    Xn, Yn, _ = normalise(Xw, Yw)
    print(f"  series {mat.shape[0]}  |  training windows {len(Xn):,}  "
          f"(lookback {LOOKBACK}, horizon {HORIZON})")
    print(f"  device {DEVICE}\n")

    T = lambda v, dt=torch.float32: torch.tensor(v, dtype=dt, device=DEVICE)
    tensors = {"X": T(Xn), "Y": T(Yn), "START": start,
               "A": T(a_codes[sid], torch.long),
               "C": T(c_codes[sid], torch.long),
               "B": T(b_codes[sid], torch.long)}
    keys_enc = {"A": T(a_codes, torch.long), "C": T(c_codes, torch.long),
                "B": T(b_codes, torch.long)}

    specs = [
        ("NBEATS-G (generic)", lambda: NBeats("generic"), False),
        ("NBEATS-I (interpretable)", lambda: NBeats("interp"), False),
        ("GlobalMLP (+embeddings)", lambda: GlobalMLP(
            a_codes.max() + 1, c_codes.max() + 1, b_codes.max() + 1), True),
    ]

    rows, percat, saved_preds = [], {}, {}
    for name, build, is_mlp in specs:
        preds, mases = [], []
        for seed in SEEDS:
            torch.manual_seed(seed)
            m = build().to(DEVICE)
            n_par = sum(p.numel() for p in m.parameters())
            m = train_model(m, tensors, seed, is_mlp=is_mlp)
            p = forecast(m, mat, keys_enc, is_mlp)
            preds.append(p)
            mases.append(np.nanmean(np.abs(test - p).mean(axis=1) / scale))
        mean_pred = np.mean(preds, axis=0)
        rows.append({"model": name, "params": n_par,
                     "MASE": np.mean(mases), "sd": np.std(mases),
                     "WAPE": np.abs(test - mean_pred).sum() / np.abs(test).sum()})
        saved_preds[name] = mean_pred
        ae = np.abs(test - mean_pred).mean(axis=1) / scale
        percat[name] = pd.Series(ae).groupby(cat).mean()

    res = pd.DataFrame(rows).set_index("model")
    print("=" * 76)
    print(f"  DEEP LEARNING vs BASELINES  ({len(SEEDS)} seeds, mean +- sd)")
    print("=" * 76)
    print(f"  {'model':<28}{'params':>10}{'MASE':>10}{'sd':>8}{'WAPE':>9}")
    print("  " + "-" * 72)
    for m, r in res.sort_values("MASE").iterrows():
        print(f"  {m:<28}{r['params']:>10,.0f}{r.MASE:>10.3f}{r.sd:>8.3f}{r.WAPE:>8.1%}")
    print("  " + "-" * 72)
    # Never hardcode a baseline: the generator is re-run whenever it is
    # recalibrated, and a stale constant silently invalidates the comparison.
    try:
        bl = pd.read_csv(out / "baseline_results.csv", index_col=0)
        for m, r in bl.sort_values("MASE").iterrows():
            tag = "" if r.get("ranked", True) else "  [not ranked: sees FY2025]"
            print(f"  {m:<28}{'--':>10}{r.MASE:>10.3f}{'--':>8}"
                  f"{r.WAPE:>8.1%}{tag}")
    except FileNotFoundError:
        print("  baseline_results.csv missing -- run baselines.py first")
    print("=" * 76)

    print("\n  MASE BY ACCOUNT CATEGORY -- the hypothesis worth testing")
    pc = pd.DataFrame(percat)
    print(pc.to_string(float_format=lambda v: f"{v:.3f}"))
    print("\n  (does a global model help most on the sparse, spiky cells?)")

    res.to_csv(out / "dl_results.csv")
    pc.to_csv(out / "dl_results_by_category.csv")

    # Persist the seed-averaged predictions. The significance tests need
    # per-cell errors, not the aggregate MASE, and re-running three seeds x
    # three models just to recover them would be wasteful.
    np.savez(out / "dl_predictions.npz", **saved_preds)
    return 0


if __name__ == "__main__":
    sys.exit(main(pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "./out")))
