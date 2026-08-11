"""
validate_dataset.py
===================
Applies to the generated dataset exactly the acceptance tests that were used
to reject the public candidates, plus integrity and calibration checks.

A generator that cannot pass its own acceptance criteria is not evidence.

Usage:  python3 validate_dataset.py ./out
"""

from __future__ import annotations

import sys
import pathlib

import numpy as np
import pandas as pd

SCHREYER_MEDIAN = 486_445.0
SCHREYER_MEAN = 922_668.0

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str) -> None:
    results.append((name, PASS if ok else FAIL, detail))


def main(out: pathlib.Path) -> int:
    act = pd.read_csv(out / "acdoca_actuals.csv",
                      dtype={"BUKRS": str, "HKONT": str, "KOSTL": str,
                             "BELNR": str, "STBLG": str, "VERSION": str},
                      keep_default_na=False, na_values=[])
    plan = pd.read_csv(out / "plan_versions.csv",
                       dtype={"BUKRS": str, "HKONT": str, "KOSTL": str,
                              "VERSION": str})
    act["BUDAT"] = pd.to_datetime(act["BUDAT"])

    # -- TEST 1 -----------------------------------------------------------
    required = {"BUKRS", "BELNR", "BUZEI", "GJAHR", "BLART", "BSCHL",
                "HKONT", "SHKZG", "DMBTR", "WRBTR", "WAERS", "KOSTL",
                "PRCTR", "BUDAT", "BLDAT", "MONAT", "STBLG"}
    missing = required - set(act.columns)
    check("T1  FI/CO schema (BKPF/BSEG fields)", not missing,
          f"{len(required - missing)}/{len(required)} required fields present"
          + (f"; MISSING {sorted(missing)}" if missing else ""))

    # -- TEST 2 -----------------------------------------------------------
    regular = act[act["MONAT"] <= 12]
    n_periods = regular.groupby(["GJAHR", "MONAT"]).ngroups
    check("T2  >= 36 monthly periods", n_periods >= 36,
          f"{n_periods} regular fiscal periods "
          f"({act['GJAHR'].min()}-{act['GJAHR'].max()})")

    # Seasonality must be detectable, not merely asserted.
    rev = regular[regular["HKONT"] == "400000"]
    m = rev.groupby("MONAT")["DMBTR"].sum()
    m_idx = m / m.mean()
    amp = m_idx.max() - m_idx.min()
    check("T2b Seasonality detectable in revenue", amp > 0.25,
          f"monthly index range {m_idx.min():.2f}-{m_idx.max():.2f} "
          f"(amplitude {amp:.2f}); peak month = {int(m_idx.idxmax())}")

    # -- INTEGRITY --------------------------------------------------------
    sgn = np.where(act["SHKZG"] == "S", 1.0, -1.0)
    act = act.assign(_signed=act["DMBTR"] * sgn)
    bal = act.groupby(["BUKRS", "GJAHR", "BELNR"])["_signed"].sum().abs()
    n_bad = int((bal > 0.01).sum())
    check("I1  Every document balances (debit == credit)", n_bad == 0,
          f"{bal.size:,} documents checked, {n_bad} unbalanced")

    _s = act["STBLG"].fillna("")
    accr = act[(act["BLART"] == "AB") & (_s.str.len() == 0)]
    revs = act[_s.str.len() > 0]
    accr_docs = set(accr["BELNR"].unique())
    rev_targets = set(revs["STBLG"].unique())
    orphan = accr_docs - rev_targets
    # Accruals raised in the final period have no following period to reverse into.
    last = act[["GJAHR", "MONAT"]].max()
    tail = set(accr[(accr["GJAHR"] == last["GJAHR"])
                    & (accr["MONAT"] >= 12)]["BELNR"].unique())
    real_orphans = orphan - tail
    check("I2  Accruals reversed via STBLG", len(real_orphans) == 0,
          f"{len(accr_docs):,} accruals, {len(rev_targets):,} reversals, "
          f"{len(real_orphans)} unreversed (excl. {len(orphan & tail)} in final period)")

    _k = act["KOSTL"].fillna("")
    fk = set(_k[_k.str.len() > 0]) - set(
        pd.read_csv(out / "dim_cost_center.csv", dtype=str)["KOSTL"])
    check("I3  Cost-centre referential integrity", not fk,
          f"{len(fk)} orphan KOSTL values")

    late = act[pd.to_datetime(act["BLDAT"]) > act["BUDAT"]]
    check("I4  Document date <= posting date", late.empty,
          f"{len(late)} documents dated after posting")

    # -- REALISM ----------------------------------------------------------
    reg = act[act["MONAT"] <= 12].copy()
    reg["dom"] = reg["BUDAT"].dt.day
    reg["dim"] = reg["BUDAT"].dt.days_in_month
    me = (reg["dom"] > reg["dim"] - 3).mean()
    check("R1  Month-end posting concentration", 0.15 < me < 0.60,
          f"{me:.1%} of lines post in the last 3 days of the period "
          f"(uniform would be ~10%)")

    sp = act[act["MONAT"] > 12]
    check("R2  Special periods 13-16 present", not sp.empty,
          f"{len(sp):,} lines in year-end close periods "
          f"{sorted(sp['MONAT'].unique().tolist())}")

    # -- CALIBRATION ------------------------------------------------------
    # Level is NOT calibrated to Schreyer (their amounts inherit PaySim's
    # anonymised scale). It is checked for plausibility against the stated
    # enterprise profile instead.
    pl = act[(act["BUZEI"] == 1) & (act["DMBTR"] > 0)]["DMBTR"]
    check("C1  DMBTR level plausible for profile", 15_000 < pl.median() < 150_000,
          f"median line {pl.median():,.0f} on a ~EUR 3bn entity "
          f"(p95 {pl.quantile(0.95):,.0f}, max {pl.max():,.0f})")
    # Schreyer's POOLED skew (1.90) is not a valid target: their account-level
    # heterogeneity ratio is 1.0x, i.e. all 73 HKONT values share one amount
    # distribution -- an artifact of the homogeneous PaySim base. The fair
    # comparison is within-account, where both are lognormal.
    SCHREYER_WITHIN_ACCOUNT_SKEW = 1.87
    g = act[(act["BUZEI"] == 1) & (act["DMBTR"] > 0)].groupby("HKONT")["DMBTR"]
    within = (g.mean() / g.median()).median()
    check("C2  Within-account skew vs Schreyer",
          abs(within - SCHREYER_WITHIN_ACCOUNT_SKEW) < 0.6,
          f"mean/median {within:.2f} vs Schreyer {SCHREYER_WITHIN_ACCOUNT_SKEW:.2f} "
          f"(pooled skew is NOT a valid target -- see DATASET_DESIGN.md)")

    med_by_acct = g.median()
    het = med_by_acct.max() / med_by_acct.min()
    check("C3  Account-level heterogeneity", het > 10,
          f"{het:.0f}x between largest and smallest account median "
          f"(Schreyer: 1.0x -- the defect this generator corrects)")

    # -- PLAN -------------------------------------------------------------
    a = (act[(act["BUZEI"] == 1) & (act["MONAT"] <= 12)]
         .groupby(["BUKRS", "GJAHR", "MONAT", "KOSTL", "HKONT"])["DMBTR"].sum()
         .rename("actual").reset_index())
    b = plan[plan["VERSION"] == "1"].rename(columns={"DMBTR": "budget"})
    j = a.merge(b, on=["BUKRS", "GJAHR", "MONAT", "KOSTL", "HKONT"], how="inner")
    j["var_pct"] = (j["actual"] - j["budget"]) / j["budget"]
    mape = j["var_pct"].abs().median()
    check("P1  Budget variance is non-trivial and bounded",
          0.03 < mape < 0.60,
          f"median |actual-vs-budget| = {mape:.1%} across {len(j):,} cells")

    f = plan[plan["VERSION"] == "2"].rename(columns={"DMBTR": "forecast"})
    jf = a.merge(f, on=["BUKRS", "GJAHR", "MONAT", "KOSTL", "HKONT"], how="inner")
    fmape = ((jf["actual"] - jf["forecast"]).abs() / jf["actual"]).median()
    check("P2  Rolling forecast beats budget", fmape < mape,
          f"forecast {fmape:.1%} vs budget {mape:.1%} median abs error")

    # -- P&L COHERENCE ----------------------------------------------------
    # The defect this catches: P&L ratios held only in expectation, so revenue
    # (posted by ~10 cost centres) drifted 22% low while costs (36 cost centres)
    # sat on target -- yielding a company with a -27% operating margin.
    gl = pd.read_csv(out / "dim_gl_account.csv", dtype={"HKONT": str})
    pl_lines = act[act["BUZEI"] == 1].copy()
    pl_lines["signed"] = pl_lines["DMBTR"] * np.where(pl_lines["SHKZG"] == "S", 1, -1)
    fy = pl_lines.merge(gl, on="HKONT")
    fy = fy[fy["GJAHR"] == fy["GJAHR"].max()]
    cat = fy.groupby("CATEGORY")["signed"].sum().abs()
    rev = cat.get("REVENUE", 0.0)

    gm = (rev - cat.get("COGS", 0)) / rev
    check("P3  Gross margin on target", abs(gm - 0.45) < 0.05,
          f"{gm:.1%} vs 45.0% target")

    om = (rev - sum(cat.get(k, 0) for k in
                    ["COGS", "PERSONNEL", "OPEX", "DEPRECIATION"])) / rev
    check("P4  Operating margin plausible", 0.0 < om < 0.20,
          f"{om:.1%} (a generator that quietly emits a loss-making entity "
          f"is not a credible FP&A testbed)")

    # -- REPORT -----------------------------------------------------------
    w = max(len(n) for n, _, _ in results)
    print("=" * (w + 62))
    print("ACCEPTANCE REPORT".center(w + 62))
    print("=" * (w + 62))
    for name, status, detail in results:
        print(f"  [{status}] {name:<{w}}  {detail}")
    n_fail = sum(1 for _, s, _ in results if s == FAIL)
    print("=" * (w + 62))
    print(f"  {len(results) - n_fail}/{len(results)} checks passed")
    print(f"  actuals: {len(act):,} lines | plan: {len(plan):,} rows")
    print("=" * (w + 62))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main(pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "./out")))
