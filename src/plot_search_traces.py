"""
plot_search_traces.py -- Figure 4 and the search-trace facts behind Section 7.5.

Reads the per-evaluation traces that metaheuristic.py writes to out/ckpt_meta/
and reports, for each of the six searches, the best inner score, the evaluation
at which that score was first reached, and whether it was reached inside the
initial random population (the first POP evaluations of GA and PSO). Then draws
best-so-far against evaluation count, one panel per model family.

Usage:  python src/plot_search_traces.py ./out
Writes: out/search_trace_summary.csv, out/fig4_search_traces.png (needs matplotlib)
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

POP = 10                                  # pop_size / n_particles in metaheuristic.py
FAMILIES = ("LightGBM", "N-BEATS")
OPTIMISERS = (("Random", "Random search (control)"),
              ("GA", "Genetic algorithm"),
              ("PSO", "Particle swarm"))


def load(out: pathlib.Path) -> dict[str, np.ndarray]:
    ck = out / "ckpt_meta"
    traces = {}
    for fam in FAMILIES:
        for opt, _ in OPTIMISERS:
            p = ck / f"{fam}_{opt}.json"
            if not p.exists():
                sys.exit(f"missing {p} -- run metaheuristic.py first")
            traces[f"{fam}_{opt}"] = np.asarray(json.loads(p.read_text())["trace"], float)
    return traces


def summarise(traces: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for fam in FAMILIES:
        base = traces[f"{fam}_Random"].min()
        for opt, label in OPTIMISERS:
            tr = traces[f"{fam}_{opt}"]
            best = tr.min()
            first = int(np.argmax(tr <= best)) + 1            # 1-based evaluation index
            best_init = tr[:POP].min()
            rows.append({
                "family": fam, "optimiser": label, "evals": len(tr),
                "best": round(best, 4),
                "gain_over_random": round(base - best, 4) if opt != "Random" else np.nan,
                "best_at_eval": first,
                "best_within_initial_pop": first <= POP if opt != "Random" else np.nan,
                "best_of_initial_pop": round(best_init, 4),
                "improvement_after_init": round(best_init - best, 4) if opt != "Random" else np.nan,
            })
    return pd.DataFrame(rows)


def plot(traces: dict[str, np.ndarray], path: pathlib.Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    styles = {"Random": ("0.35", "--"), "GA": ("tab:blue", "-"), "PSO": ("tab:orange", "-")}
    for ax, fam in zip(axes, FAMILIES):
        ax.axvspan(0.5, POP + 0.5, color="0.9", zorder=0, label="initial population")
        for opt, label in OPTIMISERS:
            bsf = np.minimum.accumulate(traces[f"{fam}_{opt}"])
            c, ls = styles[opt]
            ax.step(np.arange(1, len(bsf) + 1), bsf, where="post", color=c, ls=ls, label=label)
        ax.set_title(fam)
        ax.set_xlabel("evaluation")
        ax.set_ylabel("best-so-far inner score (lower is better)")
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)


def main(out: pathlib.Path) -> int:
    traces = load(out)
    df = summarise(traces)
    pd.set_option("display.width", 140)
    print(df.to_string(index=False))
    df.to_csv(out / "search_trace_summary.csv", index=False)
    print(f"\n  written: {out / 'search_trace_summary.csv'}")
    try:
        plot(traces, out / "fig4_search_traces.png")
        print(f"  written: {out / 'fig4_search_traces.png'}")
    except ImportError:
        print("  matplotlib not installed -- summary written, figure skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main(pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "out")))
