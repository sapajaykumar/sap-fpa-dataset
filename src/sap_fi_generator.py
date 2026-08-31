"""
sap_fi_generator.py
===================
A parameterised generator for SAP-structured financial accounting data,
built for FP&A research: planning, forecasting, variance analysis and
narrative reporting.

Why this exists
---------------
No public dataset provides SAP FI general-ledger data with a temporal
dimension. The only openly available SAP-FI-structured dataset --
Schreyer et al. (arXiv:1709.05254), distributed via github.com/GitiHubi/deepAI
-- carries the BKPF/BSEG attribute schema but contains NO date, fiscal
year or period field of any kind, because it was constructed for anomaly
detection rather than forecasting.

This generator follows the precedent Schreyer et al. established (deriving
SAP-structured financial data from a synthetic base) and adds the fields
their use case did not require:

    BUDAT, BLDAT, GJAHR, MONAT   temporal dimension
    KOSTL + cost centre hierarchy  planning granularity
    STBLG                          accrual/reversal linkage
    plan versions (0/1/2)          budget and rolling forecast

Design decisions are documented in DATASET_DESIGN.md.

Author: Ajay Kumar (25MDA009), M.Tech DSAI, IIIT Dharwad
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Calibration constants
# --------------------------------------------------------------------------
# Moment-matched to the published DMBTR distribution of Schreyer et al.
# (n=533,009; median 486,445; mean 922,668).
#
#   median = exp(mu)                  -> mu    = ln(486445)      = 13.0949
#   mean/median = exp(sigma^2 / 2)    -> sigma = sqrt(2*ln(1.897)) = 1.1315
#
# Moment-matching is preferred over MLE here: the MLE fit (mu=12.97,
# sigma=1.30) reproduces the tail more faithfully but understates the
# median by ~12%. FP&A works on period aggregates, where central tendency
# dominates. The resulting tail is slightly lighter than Schreyer's; this
# is a deliberate, documented trade-off.
SCHREYER_LOGNORM_MU = 13.0949
SCHREYER_LOGNORM_SIGMA = 1.1315

# Reference for the report:
#   M. Schreyer, T. Sattarov, D. Borth, A. Dengel and B. Reimer,
#   "Detection of Anomalies in Large Scale Accounting Data using Deep
#   Autoencoder Networks," arXiv:1709.05254, 2017.
#   Dataset artifact: github.com/GitiHubi/deepAI (GPL-3.0), retrieved 2026-07-16.


# --------------------------------------------------------------------------
# Master data definitions
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class GLAccount:
    """A general ledger account and its behavioural profile."""
    hkont: str            # GL account number
    txt: str              # description
    category: str         # REVENUE | COGS | PERSONNEL | OPEX | DEPRECIATION
    sign: str             # 'S' (debit / cost) or 'H' (credit / revenue)
    seasonality: str      # key into SEASONALITY
    pl_ratio: float       # share of total revenue (revenue accounts share 1.0)
    accrual_prone: bool   # does this account attract month-end accruals?


# Monthly seasonal indices, Jan..Dec. These encode the domain knowledge that
# distinguishes this generator from a generic time-series simulator: quarter-end
# revenue pushes, the March and December bonus runs, the European summer travel
# collapse, the Q1 audit-fee spike, and winter utilities.
SEASONALITY: Dict[str, np.ndarray] = {
    "revenue":    np.array([0.85, 0.88, 1.05, 0.95, 0.98, 1.10,
                            0.90, 0.85, 1.08, 1.00, 1.05, 1.31]),
    "cogs":       np.array([0.87, 0.90, 1.04, 0.96, 0.99, 1.08,
                            0.92, 0.88, 1.06, 1.00, 1.04, 1.26]),
    "flat":       np.ones(12),
    "bonus":      np.array([0.05, 0.05, 3.00, 0.05, 0.05, 0.05,
                            0.05, 0.05, 0.05, 0.05, 0.05, 5.00]),
    "travel":     np.array([0.90, 1.00, 1.10, 1.05, 1.10, 1.00,
                            0.60, 0.50, 1.10, 1.20, 1.15, 0.85]),
    "marketing":  np.array([0.80, 0.85, 1.00, 0.95, 1.00, 1.05,
                            0.80, 0.75, 1.10, 1.20, 1.30, 1.30]),
    "utilities":  np.array([1.30, 1.25, 1.10, 0.95, 0.85, 0.80,
                            0.85, 0.85, 0.90, 1.05, 1.20, 1.30]),
    "audit":      np.array([1.40, 1.50, 1.20, 0.90, 0.85, 0.90,
                            0.80, 0.80, 0.95, 1.00, 1.10, 1.30]),
}

# P&L ratios target a mid-size manufacturing enterprise:
#   gross margin 45%, personnel 23% of revenue, opex 11.2%, depreciation 4.0%
#   -> operating margin ~6.8%
# Revenue accounts partition 1.0 between them; cost accounts are expressed as a
# share of total revenue. These ratios -- not per-account magnitudes -- are what
# make the generated P&L coherent, because the number of cost centres posting to
# each account differs by a factor of four.
# Normalise every profile to mean 1.0. Seasonality must redistribute an
# account's annual total across months, never change it. Without this, 'bonus'
# (two large spikes, ten near-zero months) runs at 0.708 of its P&L target.
SEASONALITY = {k: v / v.mean() for k, v in SEASONALITY.items()}


CHART_OF_ACCOUNTS: List[GLAccount] = [
    GLAccount("400000", "Revenue - Product",        "REVENUE",      "H", "revenue",   0.700, False),
    GLAccount("410000", "Revenue - Services",       "REVENUE",      "H", "revenue",   0.300, False),
    GLAccount("500000", "COGS - Material",          "COGS",         "S", "cogs",      0.420, True),
    GLAccount("510000", "COGS - Subcontracting",    "COGS",         "S", "cogs",      0.130, True),
    GLAccount("600000", "Personnel - Salaries",     "PERSONNEL",    "S", "flat",      0.175, False),
    GLAccount("610000", "Personnel - Bonus",        "PERSONNEL",    "S", "bonus",     0.020, True),
    GLAccount("620000", "Personnel - Benefits",     "PERSONNEL",    "S", "flat",      0.035, False),
    GLAccount("700000", "Travel & Entertainment",   "OPEX",         "S", "travel",    0.015, True),
    GLAccount("710000", "Marketing",                "OPEX",         "S", "marketing", 0.030, True),
    GLAccount("720000", "Facilities & Rent",        "OPEX",         "S", "flat",      0.025, False),
    GLAccount("730000", "Utilities",                "OPEX",         "S", "utilities", 0.008, True),
    GLAccount("740000", "IT & Software",            "OPEX",         "S", "flat",      0.022, True),
    GLAccount("750000", "Professional Services",    "OPEX",         "S", "audit",     0.012, True),
    GLAccount("780000", "Depreciation",             "DEPRECIATION", "S", "flat",      0.040, False),
]

# Balance-sheet counter-accounts. Every document balances against one of these.
OFFSET_ACCOUNTS = {
    "REVENUE":      "140000",   # Accounts Receivable
    "COGS":         "160000",   # Accounts Payable
    "PERSONNEL":    "170000",   # Payroll Clearing
    "OPEX":         "160000",   # Accounts Payable
    "DEPRECIATION": "199000",   # Accumulated Depreciation
    "ACCRUAL":      "490000",   # Accrued Liabilities
}

# Which cost centre types post to which account categories.
CC_TYPE_ACCOUNTS = {
    "SALES":      ["400000", "410000", "700000", "710000", "600000", "620000", "740000"],
    "PRODUCTION": ["500000", "510000", "600000", "620000", "720000", "730000", "780000"],
    "ADMIN":      ["600000", "610000", "620000", "720000", "740000", "750000", "780000"],
    "RND":        ["600000", "620000", "740000", "700000", "780000"],
}


@dataclass
class GeneratorConfig:
    """Every knob. Change these, not the code."""
    seed: int = 42

    # Temporal extent. 4 fiscal years = 48 regular periods, which supports a
    # 36-period training window and a 12-period holdout.
    start_year: int = 2022
    n_years: int = 4

    # Organisational structure: company code -> (country, currency, n cost centres)
    company_codes: Dict[str, Tuple[str, str, int]] = field(default_factory=lambda: {
        "1000": ("DE", "EUR", 14),
        "2000": ("US", "USD", 12),
        "3000": ("IN", "INR", 10),
    })

    # Document volume per (cost centre, account, period)
    docs_per_cell_lambda: float = 7.0
    min_docs_per_cell: int = 1

    # Posting-date behaviour
    month_end_weight: float = 5.0      # relative weight on the last 3 working days
    weekend_weight: float = 0.08       # postings do occur at weekends, but rarely

    # Accruals
    accrual_rate: float = 0.06         # share of month-end docs that are accruals
    reversal_lag_days: int = 2         # reversed early in the following period

    # Special periods 13-16 (SAP year-end close adjustments)
    special_period_rate: float = 0.02

    # Underlying process
    annual_growth_mu: float = 0.06     # mean YoY growth per cell
    annual_growth_sd: float = 0.05
    ar1_phi: float = 0.35              # month-to-month persistence of the shock
    noise_sd: float = 0.14

    # Plan versions
    budget_bias_sd: float = 0.09       # budgets are set with imperfect foresight
    budget_optimism: float = 0.025     # and are systematically slightly optimistic
    forecast_shrink: float = 0.55      # rolling forecast pulls toward realised actuals

    # Amount distribution (calibrated -- see constants above)
    lognorm_mu: float = SCHREYER_LOGNORM_MU
    lognorm_sigma: float = SCHREYER_LOGNORM_SIGMA

    # Controls how unevenly a cell's period total splits across documents,
    # and therefore the within-account skew of DMBTR. Lowered from 1.6 (skew
    # 1.47) to 0.68 after two independent references converged:
    #   Schreyer et al.  73 accounts            within-account skew 1.87
    #   Google Cortex     6 accounts (USD, P&L)  within-account skew 1.85
    # Both are synthetic and neither is authoritative, but they agree, they
    # were generated by unrelated processes, and both say the original value
    # was too flat. Calibrated to ~1.86; validator check C2 enforces it.
    dirichlet_alpha: float = 0.68

    # Cell-level dispersion: cost centres of the same type are not identical.
    # Median multiplier is 1.0, so this does not shift the P&L structure.
    cell_dispersion_sigma: float = 0.45

    # THE one free scale parameter. P&L ratios fix the *structure*; this fixes
    # the *level*: a EUR 3bn-revenue enterprise across three company codes.
    #
    # Deliberately NOT calibrated to Schreyer's absolute amounts. Their DMBTR
    # derives from PaySim mobile-money transfers with the currency field
    # anonymised (WAERS values are 'C1', 'C3', ...), so its magnitude carries no
    # accounting meaning; matching it would force an implausible EUR 28.6bn
    # entity onto 36 cost centres. What IS inherited from Schreyer is the
    # distributional SHAPE -- lognormal, heavy right skew -- which is a property
    # of GL postings rather than of PaySim. See check C2.
    target_annual_revenue: float = 3.0e9


class SAPFinanceGenerator:
    """Generates ACDOCA-style actuals plus plan versions."""

    def __init__(self, cfg: GeneratorConfig | None = None):
        self.cfg = cfg or GeneratorConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.accounts = {a.hkont: a for a in CHART_OF_ACCOUNTS}
        self._belnr_counter: Dict[Tuple[str, int], int] = {}

    # ------------------------------------------------------------------
    # Master data
    # ------------------------------------------------------------------
    def build_master_data(self) -> Dict[str, pd.DataFrame]:
        cc_rows, ccg_rows, pc_rows = [], [], []
        cc_types = list(CC_TYPE_ACCOUNTS.keys())

        for bukrs, (land, waers, n_cc) in self.cfg.company_codes.items():
            for cctype in cc_types:
                ccg_rows.append({
                    "BUKRS": bukrs,
                    "KOSTL_GROUP": f"{bukrs}-{cctype}",
                    "GROUP_TXT": f"{cctype.title()} ({bukrs})",
                })
            for i in range(n_cc):
                cctype = cc_types[i % len(cc_types)]
                kostl = f"{bukrs}{i + 1:04d}"
                prctr = f"P{bukrs}{(i % 4) + 1:02d}"
                cc_rows.append({
                    "BUKRS": bukrs,
                    "KOSTL": kostl,
                    "KOSTL_GROUP": f"{bukrs}-{cctype}",
                    "CC_TYPE": cctype,
                    "PRCTR": prctr,
                    "KTEXT": f"{cctype.title()} CC {i + 1} ({land})",
                })
                pc_rows.append({"BUKRS": bukrs, "PRCTR": prctr,
                                "PTEXT": f"Profit Centre {prctr}"})

        dim_cc = pd.DataFrame(cc_rows)
        dim_ccg = pd.DataFrame(ccg_rows)
        dim_pc = pd.DataFrame(pc_rows).drop_duplicates().reset_index(drop=True)
        dim_bukrs = pd.DataFrame([
            {"BUKRS": k, "LAND1": v[0], "WAERS": v[1], "BUTXT": f"Company Code {k} ({v[0]})"}
            for k, v in self.cfg.company_codes.items()
        ])
        dim_gl = pd.DataFrame([{
            "HKONT": a.hkont, "TXT50": a.txt, "CATEGORY": a.category,
            "NORMAL_SIDE": a.sign, "SEASONALITY": a.seasonality, "PL_RATIO": a.pl_ratio,
        } for a in CHART_OF_ACCOUNTS] + [
            {"HKONT": h, "TXT50": t, "CATEGORY": "BALANCE_SHEET",
             "NORMAL_SIDE": "H", "SEASONALITY": "flat", "PL_RATIO": 0.0}
            for h, t in [("140000", "Accounts Receivable"),
                         ("160000", "Accounts Payable"),
                         ("170000", "Payroll Clearing"),
                         ("199000", "Accumulated Depreciation"),
                         ("490000", "Accrued Liabilities")]
        ])

        self.dim_cc = dim_cc
        return {"dim_company_code": dim_bukrs, "dim_cost_center": dim_cc,
                "dim_cost_center_group": dim_ccg, "dim_profit_center": dim_pc,
                "dim_gl_account": dim_gl}

    # ------------------------------------------------------------------
    # The latent process: one monthly series per (cost centre, account)
    # ------------------------------------------------------------------
    def _build_cells(self) -> pd.DataFrame:
        """
        Every (KOSTL, HKONT) pair that posts, with its latent parameters.

        A cell's base monthly level is derived top-down from the target P&L,
        not chosen per account:

            cell_base = target_revenue * pl_ratio / 12 / n_cells(account)

        so an account posted by 36 cost centres and one posted by 9 both land
        on their intended share of revenue. This is what keeps the generated
        income statement coherent.
        """
        pairs = [(cc, h) for _, cc in self.dim_cc.iterrows()
                 for h in CC_TYPE_ACCOUNTS[cc["CC_TYPE"]]]

        n_cells: Dict[str, int] = {}
        for _, h in pairs:
            n_cells[h] = n_cells.get(h, 0) + 1

        # Dispersion is drawn per account and normalised to mean 1.0 across that
        # account's cells, so it redistributes between cost centres without
        # moving the account total. Revenue posts from only ~10 cost centres;
        # an unnormalised draw leaves the realised revenue level to chance and
        # the P&L ratios hold only in expectation.
        disp: Dict[str, np.ndarray] = {}
        for h, n in n_cells.items():
            d = np.exp(self.rng.normal(0, self.cfg.cell_dispersion_sigma, n))
            disp[h] = d / d.mean()
        seen: Dict[str, int] = {}

        rows = []
        for cc, hkont in pairs:
            acct = self.accounts[hkont]
            i = seen.get(hkont, 0)
            seen[hkont] = i + 1
            monthly_target = (self.cfg.target_annual_revenue
                              * acct.pl_ratio / 12.0 / n_cells[hkont])
            rows.append({
                "BUKRS": cc["BUKRS"], "KOSTL": cc["KOSTL"], "PRCTR": cc["PRCTR"],
                "HKONT": hkont, "CATEGORY": acct.category,
                "base": monthly_target * disp[hkont][i],
                "growth": self.rng.normal(self.cfg.annual_growth_mu,
                                          self.cfg.annual_growth_sd),
            })
        return pd.DataFrame(rows)

    def _latent_series(self, cells: pd.DataFrame, n_periods: int) -> np.ndarray:
        """AR(1)-perturbed trend x seasonality. Shape (n_cells, n_periods)."""
        n = len(cells)
        out = np.zeros((n, n_periods))
        shock = np.zeros(n)
        seas = np.stack([SEASONALITY[self.accounts[h].seasonality]
                         for h in cells["HKONT"]])
        base = cells["base"].to_numpy()
        growth = cells["growth"].to_numpy()

        # exp(shock) has mean exp(var/2) > 1, which would inflate every account
        # by ~1%. Subtract half the stationary variance to keep it mean-preserving.
        stat_var = self.cfg.noise_sd ** 2 / (1 - self.cfg.ar1_phi ** 2)
        for t in range(n_periods):
            shock = (self.cfg.ar1_phi * shock
                     + self.rng.normal(0, self.cfg.noise_sd, n))
            trend = (1 + growth) ** (t / 12.0)
            out[:, t] = base * trend * seas[:, t % 12] * np.exp(shock - stat_var / 2)
        return np.clip(out, 1.0, None)

    # ------------------------------------------------------------------
    # Posting-date model
    # ------------------------------------------------------------------
    def _day_weights(self, year: int, month: int) -> np.ndarray:
        n_days = calendar.monthrange(year, month)[1]
        w = np.ones(n_days)
        for d in range(1, n_days + 1):
            if calendar.weekday(year, month, d) >= 5:
                w[d - 1] = self.cfg.weekend_weight
        # Month-end close: the last three days carry the accrual and
        # adjustment traffic. This is the pattern a forecasting model must
        # not mistake for signal.
        w[-3:] *= self.cfg.month_end_weight
        return w / w.sum()

    def _next_belnr(self, bukrs: str, gjahr: int) -> str:
        key = (bukrs, gjahr)
        self._belnr_counter[key] = self._belnr_counter.get(key, 100000) + 1
        return f"{self._belnr_counter[key]:010d}"

    # ------------------------------------------------------------------
    # Actuals
    # ------------------------------------------------------------------
    def generate_actuals(self, cells: pd.DataFrame, latent: np.ndarray) -> pd.DataFrame:
        cfg = self.cfg
        n_periods = cfg.n_years * 12
        waers_map = {k: v[1] for k, v in cfg.company_codes.items()}
        lines: List[dict] = []
        pending_reversals: List[dict] = []

        for t in range(n_periods):
            gjahr = cfg.start_year + t // 12
            monat = t % 12 + 1

            # Reversals of last period's accruals post first.
            for rev in pending_reversals:
                lines.extend(self._emit_reversal(rev, gjahr, monat))
            pending_reversals = []

            day_w = self._day_weights(gjahr, monat)
            n_days = len(day_w)

            for ci in range(len(cells)):
                cell = cells.iloc[ci]
                target = latent[ci, t]
                acct = self.accounts[cell["HKONT"]]

                n_docs = max(cfg.min_docs_per_cell,
                             self.rng.poisson(cfg.docs_per_cell_lambda))
                # Split the period total across documents via a Dirichlet, so
                # document sizes vary but the period aggregate is preserved.
                splits = self.rng.dirichlet(np.full(n_docs, cfg.dirichlet_alpha)) * target
                days = self.rng.choice(np.arange(1, n_days + 1), size=n_docs, p=day_w)

                for amt, day in zip(splits, days):
                    if amt < 1.0:
                        continue
                    is_accrual = (acct.accrual_prone
                                  and day >= n_days - 2
                                  and self.rng.random() < cfg.accrual_rate)
                    belnr = self._next_belnr(cell["BUKRS"], gjahr)
                    budat = pd.Timestamp(gjahr, monat, int(day))
                    # Document date precedes posting date -- invoices arrive
                    # before they are posted.
                    bldat = budat - pd.Timedelta(days=int(self.rng.integers(0, 12)))

                    period = monat
                    if (monat == 12 and self.rng.random() < cfg.special_period_rate):
                        period = int(self.rng.integers(13, 17))  # year-end close

                    doc = self._emit_document(
                        cell=cell, acct=acct, belnr=belnr, gjahr=gjahr,
                        monat=period, budat=budat, bldat=bldat,
                        dmbtr=float(amt), waers=waers_map[cell["BUKRS"]],
                        is_accrual=is_accrual,
                    )
                    lines.extend(doc)
                    if is_accrual:
                        pending_reversals.append({
                            "cell": cell, "acct": acct, "belnr": belnr,
                            "gjahr": gjahr, "dmbtr": float(amt),
                            "waers": waers_map[cell["BUKRS"]],
                        })

        df = pd.DataFrame(lines)
        return df.sort_values(["BUKRS", "GJAHR", "MONAT", "BELNR", "BUZEI"]) \
                 .reset_index(drop=True)

    def _emit_document(self, cell, acct, belnr, gjahr, monat, budat, bldat,
                       dmbtr, waers, is_accrual) -> List[dict]:
        """Two balanced lines: the P&L posting and its offset."""
        if is_accrual:
            blart, offset = "AB", OFFSET_ACCOUNTS["ACCRUAL"]
        elif acct.category == "REVENUE":
            blart, offset = "DR", OFFSET_ACCOUNTS["REVENUE"]
        elif acct.category == "DEPRECIATION":
            blart, offset = "AF", OFFSET_ACCOUNTS["DEPRECIATION"]
        elif acct.category == "PERSONNEL":
            blart, offset = "SA", OFFSET_ACCOUNTS["PERSONNEL"]
        else:
            blart, offset = "KR", OFFSET_ACCOUNTS[acct.category]

        pl_side = acct.sign                       # 'S' cost, 'H' revenue
        off_side = "H" if pl_side == "S" else "S"
        bschl_pl = "40" if pl_side == "S" else "50"
        bschl_off = "50" if off_side == "H" else "40"

        common = {
            "BUKRS": cell["BUKRS"], "BELNR": belnr, "GJAHR": gjahr,
            "MONAT": monat, "BUDAT": budat.date().isoformat(),
            "BLDAT": bldat.date().isoformat(), "BLART": blart,
            "WAERS": waers, "VERSION": "0",
            "USNAM": f"USER{self.rng.integers(1, 40):03d}",
            "STBLG": "",
        }
        return [
            {**common, "BUZEI": 1, "BSCHL": bschl_pl, "SHKZG": pl_side,
             "HKONT": acct.hkont, "KOSTL": cell["KOSTL"], "PRCTR": cell["PRCTR"],
             "DMBTR": round(dmbtr, 2), "WRBTR": round(dmbtr, 2)},
            {**common, "BUZEI": 2, "BSCHL": bschl_off, "SHKZG": off_side,
             "HKONT": offset, "KOSTL": "", "PRCTR": cell["PRCTR"],
             "DMBTR": round(dmbtr, 2), "WRBTR": round(dmbtr, 2)},
        ]

    def _emit_reversal(self, rev, gjahr, monat) -> List[dict]:
        """Reverse an accrual early in the following period, linked via STBLG."""
        cell, acct = rev["cell"], rev["acct"]
        belnr = self._next_belnr(cell["BUKRS"], gjahr)
        day = min(self.cfg.reversal_lag_days, calendar.monthrange(gjahr, monat)[1])
        budat = pd.Timestamp(gjahr, monat, day)

        pl_side = "H" if acct.sign == "S" else "S"   # mirror of the original
        off_side = "S" if pl_side == "H" else "H"
        common = {
            "BUKRS": cell["BUKRS"], "BELNR": belnr, "GJAHR": gjahr,
            "MONAT": monat, "BUDAT": budat.date().isoformat(),
            "BLDAT": budat.date().isoformat(), "BLART": "AB",
            "WAERS": rev["waers"], "VERSION": "0",
            "USNAM": "BATCH_REV", "STBLG": rev["belnr"],   # links to the accrual
        }
        return [
            {**common, "BUZEI": 1, "BSCHL": "50" if pl_side == "H" else "40",
             "SHKZG": pl_side, "HKONT": acct.hkont, "KOSTL": cell["KOSTL"],
             "PRCTR": cell["PRCTR"], "DMBTR": round(rev["dmbtr"], 2),
             "WRBTR": round(rev["dmbtr"], 2)},
            {**common, "BUZEI": 2, "BSCHL": "40" if off_side == "S" else "50",
             "SHKZG": off_side, "HKONT": OFFSET_ACCOUNTS["ACCRUAL"], "KOSTL": "",
             "PRCTR": cell["PRCTR"], "DMBTR": round(rev["dmbtr"], 2),
             "WRBTR": round(rev["dmbtr"], 2)},
        ]

    # ------------------------------------------------------------------
    # Plan versions
    # ------------------------------------------------------------------
    def generate_plan(self, cells: pd.DataFrame, latent: np.ndarray) -> pd.DataFrame:
        """
        Version 1 -- Budget. Locked before the year starts, using the prior
                     year's realised level plus a growth assumption. It cannot
                     see the year it is budgeting, and it is mildly optimistic:
                     the systematic bias real budgets carry.
        Version 2 -- Rolling forecast. Re-cut each quarter; shrinks toward
                     realised actuals for elapsed periods.
        """
        cfg = self.cfg
        rows = []

        for ci in range(len(cells)):
            cell = cells.iloc[ci]
            acct = self.accounts[cell["HKONT"]]
            seas = SEASONALITY[acct.seasonality]

            for y in range(cfg.n_years):
                gjahr = cfg.start_year + y
                if y == 0:
                    prior = latent[ci, 0:12].mean() / seas.mean()
                else:
                    prior = latent[ci, (y - 1) * 12:y * 12].mean() / seas.mean()

                assumed_growth = cell["growth"] + self.rng.normal(
                    cfg.budget_optimism, cfg.budget_bias_sd)

                for m in range(12):
                    t = y * 12 + m
                    budget = (prior * (1 + assumed_growth) * seas[m]
                              * np.exp(self.rng.normal(0, cfg.budget_bias_sd)))

                    quarter_start = (m // 3) * 3
                    if quarter_start == 0:
                        fc = budget
                    else:
                        realised = latent[ci, y * 12:y * 12 + quarter_start].mean()
                        implied = realised / seas[:quarter_start].mean() * seas[m]
                        fc = (cfg.forecast_shrink * implied
                              + (1 - cfg.forecast_shrink) * budget)

                    for version, val in (("1", budget), ("2", fc)):
                        rows.append({
                            "BUKRS": cell["BUKRS"], "GJAHR": gjahr, "MONAT": m + 1,
                            "VERSION": version,
                            "VERSION_TXT": "Budget" if version == "1" else "Rolling Forecast",
                            "KOSTL": cell["KOSTL"], "PRCTR": cell["PRCTR"],
                            "HKONT": cell["HKONT"], "CATEGORY": acct.category,
                            "WAERS": cfg.company_codes[cell["BUKRS"]][1],
                            "DMBTR": round(float(val), 2),
                        })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    def run(self) -> Dict[str, pd.DataFrame]:
        dims = self.build_master_data()
        cells = self._build_cells()
        latent = self._latent_series(cells, self.cfg.n_years * 12)
        actuals = self.generate_actuals(cells, latent)
        plan = self.generate_plan(cells, latent)
        return {**dims, "acdoca_actuals": actuals, "plan_versions": plan}


# --------------------------------------------------------------------------
# Universal-journal view
# --------------------------------------------------------------------------
def to_universal_journal(act: pd.DataFrame) -> pd.DataFrame:
    """
    Project the generated postings onto S/4HANA ACDOCA field names.

    The generator produces classic FI documents, so its native schema is
    BKPF/BSEG: HKONT, DMBTR, SHKZG, BUZEI. The universal journal renames most
    of those and, more importantly, changes one convention: ACDOCA carries a
    SIGNED amount in HSL rather than an unsigned figure qualified by a
    debit/credit indicator. DRCRK is retained, but the sign lives in the
    amount. Consumers who aggregate HSL directly therefore get the correct
    net figure without needing to interpret SHKZG, which is the property that
    makes account-based reporting straightforward in S/4HANA.

    This is a view, not a second generation. It is derived row-for-row from
    the postings file and carries no independent randomness, so the two
    artefacts cannot drift apart.
    """
    sign = np.where(act["SHKZG"] == "S", 1.0, -1.0)
    out = pd.DataFrame({
        "RLDNR":  "0L",                             # leading ledger
        "RBUKRS": act["BUKRS"],
        "GJAHR":  act["GJAHR"],
        "BELNR":  act["BELNR"],
        "DOCLN":  act["BUZEI"].astype(int).map(lambda v: f"{v:06d}"),
        "RYEAR":  act["GJAHR"],
        "POPER":  act["MONAT"].astype(int).map(lambda v: f"{v:03d}"),
        "BUDAT":  act["BUDAT"],
        "BLDAT":  act["BLDAT"],
        "BLART":  act["BLART"],
        "RACCT":  act["HKONT"],                     # G/L account
        "RCNTR":  act["KOSTL"],                     # cost centre
        "PRCTR":  act["PRCTR"],
        "DRCRK":  act["SHKZG"],
        "RWCUR":  act["WAERS"],
        "HSL":    (act["DMBTR"] * sign).round(2),   # signed, company code currency
        "WSL":    (act["WRBTR"] * sign).round(2),   # signed, transaction currency
        "USNAM":  act["USNAM"],
        "STBLG":  act["STBLG"],
    })
    return out


if __name__ == "__main__":
    import pathlib
    import sys

    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "./out")
    out.mkdir(parents=True, exist_ok=True)

    gen = SAPFinanceGenerator(GeneratorConfig())
    tables = gen.run()
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False)
        print(f"{name:26} {df.shape[0]:>9,} rows x {df.shape[1]:>2} cols")

    # Derived view. Written after the primary artefact so that the postings
    # file, and therefore its digest, is unaffected by this addition.
    uj = to_universal_journal(tables["acdoca_actuals"])
    uj.to_csv(out / "acdoca_universal_journal.csv", index=False)
    print(f"{'acdoca_universal_journal':26} {uj.shape[0]:>9,} rows x {uj.shape[1]:>2} cols  (view)")
