# Reference outputs (v1.1.2)

Outputs of a clean clone of this release, run on a Tesla T4 (Kaggle), September 2026.
They let a reader check any table or figure in the paper against a file, without
re-running anything. Regenerating them is described in the main README.

| File | Produced by | Paper |
|---|---|---|
| `baseline_results.csv` | `baselines.py`, `classical_extra.py` | Table 6 |
| `dl_results.csv`, `dl_results_by_category.csv` | `nbeats.py` | Table 6, §7.3 |
| `significance_ranks.csv`, `significance_by_category.csv` | `significance.py` | §7.2, Table 7 |
| `../aggregate_revenue_results.csv` | `aggregate_eval.py` | §7.6, Table 10 |
| `metaheuristic_results.json`, `search_traces/` | `metaheuristic.py` (budget 40) | §7.5, Table 8 |
| `tuned_results.csv` | `refit_tuned.py` | §7.5 |
| `search_trace_summary.csv`, `fig4_search_traces.png` | `plot_search_traces.py` | §7.5, Figure 4 |
| `logs/` | console output of each script | — |

Two runs are recorded. The `verify_*` logs and the Table 6, 7 and 10 files come from
a full clean-clone run (torch 2.10.0+cu128). The `search_*` logs and the §7.5 files come
from a second run after one change to `metaheuristic.py`: the N-BEATS search objective
now seeds numpy as well as torch, so minibatch order is fixed and identical
configurations score identically. Nothing outside §7.5 depends on that change.

Numbers from the deterministic models reproduce exactly on CPU. The deep-learning
models shift in the third decimal on CPU (for example, generic N-BEATS 0.958 on CPU
against 0.964 here); the T4 values are the ones reported in the paper.
