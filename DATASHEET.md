# Datasheet — Synthetic SAP-Structured FP&A Dataset

Following the *Datasheets for Datasets* format (Gebru et al., 2021).

**Version 1.1.0.** See Maintenance for changes since 1.0.

---

## Motivation

**For what purpose was the dataset created?**
To support research on automated financial planning and analysis (FP&A) — specifically
forecasting and grounded report generation over enterprise ledger data. No public dataset
combines (a) SAP-style double-entry transactional structure, (b) enough temporal depth to
learn seasonality, and (c) plan/budget versions alongside actuals. Existing candidates fail
on different axes: the Schreyer et al. ERP dataset carries the SAP attribute schema but has
no date, fiscal year or period field of any kind, and no cost centre — so neither a time
series nor a cost-centre × account planning cell can be formed from it. The Google Cortex
sample data has ECC structure and 51 months, but its monthly variation reflects load
artifacts rather than business seasonality, and it carries no plan tables. Both are
themselves synthetic. This dataset was built to fill that gap.

**Who created the dataset?**
Ajay Kumar (IIIT Dharwad), as part of an M.Tech dissertation, supervised by
Dr. Utkarsh Khaire.

**Who funded the creation of the dataset?**
No external funding; produced as part of postgraduate academic work.

---

## Composition

**What do the instances represent?**
Three tables, of which two are independent and one is derived:

| File | Rows | Represents |
|---|---|---|
| `acdoca_actuals.csv` | 159,934 | Individual accounting line items (postings), BKPF/BSEG field names |
| `acdoca_universal_journal.csv` | 159,934 | The same postings projected onto S/4HANA universal-journal field names (derived view) |
| `plan_versions.csv` | 22,656 | Monthly plan amounts per cost centre × account × version |

Plus five dimension tables: company codes, cost centres, cost centre groups, GL accounts,
profit centres.

**How many instances are there in total?**
159,934 posting lines across 79,967 accounting documents, plus 22,656 plan rows. The
universal-journal file is a re-expression of the same 159,934 lines, not additional data.

**What data does each instance consist of?**
`acdoca_actuals.csv` carries 19 fields using classic FI field names: `BUKRS` (company code),
`BELNR` (document number), `GJAHR` (fiscal year), `MONAT` (period), `BUDAT`/`BLDAT`
(posting/document date), `BLART` (document type), `WAERS` (currency), `VERSION`,
`USNAM` (user), `STBLG` (reversal document), `BUZEI` (line item), `BSCHL` (posting key),
`SHKZG` (debit/credit indicator), `HKONT` (GL account), `KOSTL` (cost centre),
`PRCTR` (profit centre), `DMBTR` (amount in local currency), `WRBTR` (document currency
amount).

`acdoca_universal_journal.csv` carries 19 fields in universal-journal names: `RLDNR`
(ledger, `0L`), `RBUKRS`, `GJAHR`, `BELNR`, `DOCLN` (line, six digits), `RYEAR`, `POPER`
(period, three digits), `BUDAT`, `BLDAT`, `BLART`, `RACCT` (GL account), `RCNTR` (cost
centre), `PRCTR`, `DRCRK` (debit/credit), `RWCUR`, `HSL` (amount in company-code currency),
`WSL` (amount in transaction currency), `USNAM`, `STBLG`.

**Note on the amount convention.** The two files differ in more than field names. Classic FI
carries an unsigned amount qualified by a debit/credit indicator; the universal journal
carries a **signed** amount in `HSL`. Summing `HSL` therefore yields the correct net figure
without interpreting `SHKZG`, and all 79,967 documents balance on `HSL` alone. Consumers
working from the view need no additional sign logic; consumers working from
`acdoca_actuals.csv` do.

**Note on the file name.** `acdoca_actuals.csv` uses BKPF/BSEG field names despite its name.
The name is retained from version 1.0 so that existing references and the published MD5
digest remain valid. Users wanting genuine universal-journal field names should read
`acdoca_universal_journal.csv`.

`plan_versions.csv` carries 11 fields including `VERSION`, `VERSION_TXT`, `CATEGORY`
and `DMBTR`.

**Version semantics (SAP convention):**
- `VERSION = 0` → actuals. Carried on the transactional records, **not** in
  `plan_versions.csv`.
- `VERSION = 1` → Budget (locked before the year opens; mildly optimistic)
- `VERSION = 2` → Rolling Forecast (re-cut quarterly)

`plan_versions.csv` therefore contains versions 1 and 2 only: 236 planning cells × 48
months × 2 versions = 22,656 rows.

**Coverage:** 4 fiscal years (2022–2025), 48 monthly periods, 3 company codes,
36 cost centres, 19 GL accounts, 5 account categories (Revenue, COGS, Personnel,
Opex, Depreciation). Currency: EUR.

**Is any information missing?**
The dataset models the FP&A slice only. It does not include master data beyond the
dimensions listed, sub-ledger detail (AR/AP open items), tax, consolidation, or
intercompany elimination.

**Does the dataset contain data that might be considered confidential?**
No. All records are synthetically generated. **No real company, client, or SAP customer
data was used at any stage**, and the generator was not fitted to any proprietary extract.
Distributional *shape* parameters were calibrated against publicly available synthetic
sources (Cortex sample data, the Schreyer et al. dataset), never against real actuals.

**Does the dataset relate to people?**
No. The `USNAM` field contains synthetic user identifiers (e.g. `BATCH01`) that do not
correspond to real individuals.

---

## Collection Process

**How was the data collected/generated?**
Programmatically, by `src/sap_fi_generator.py`. The generator produces balanced
double-entry documents with: month-end posting concentration, account-specific
seasonality, accrual–reversal pairs linked via `STBLG`, special year-end periods,
and a plan locked before the fiscal year with a rolling forecast re-cut quarterly.

**Sampling strategy:**
Amounts are drawn from account-category-specific distributions with a Dirichlet
allocation across cost centres (`dirichlet_alpha = 0.68`, calibrated so that
within-account skew resembles observed public synthetic data). Calibration targets
*shape*, not level.

**Over what timeframe was the data collected?**
Not applicable — generated in a single deterministic pass. The simulated period is
FY2022–FY2025.

**Were any ethical review processes conducted?**
Not applicable; no human subjects, no real organisational data.

---

## Preprocessing / Cleaning / Labeling

**Was any preprocessing done?**
No post-hoc cleaning. The generator emits the final form directly. Three
mean-preservation defects were identified and fixed *during development* (a seasonality
profile that did not average to one; a dispersion term that held P&L ratios only in
expectation; an autoregressive shock that inflated every account) — each would have
silently distorted downstream results.

**How is the derived view produced?**
`acdoca_universal_journal.csv` is generated row-for-row from the postings file in the same
deterministic pass, after the postings file has been written. It introduces no independent
randomness and involves no sampling, so the two files cannot drift apart, and its presence
does not affect the digest of `acdoca_actuals.csv`. It should be treated as a re-expression
of the same data rather than as a second dataset.

**Is the software available?**
Yes. `src/sap_fi_generator.py` (generation), `src/validate_dataset.py` (validation), and
the benchmark modules listed under Uses.

---

## Uses

**What tasks could the dataset be used for?**
- Time-series forecasting at cost-centre × account granularity (with budget and rolling
  forecast as human baselines).
- Grounded/verifiable natural-language report generation over a transactional ledger.
- Evaluation of retrieval and aggregation logic under SAP semantics (signed aggregation,
  accrual netting, currency decimal handling, materiality thresholds).
- Variance analysis and plan-vs-actual reasoning.

**Has the dataset been used already?**
Yes, for the forecasting benchmark reported in the accompanying paper. The following
modules reproduce it:

| Module | Produces |
|---|---|
| `src/baselines.py` | naive, seasonal naive, ETS, LightGBM, Budget v1 |
| `src/classical_extra.py` | Theta, SARIMA, STL+ETS |
| `src/nbeats.py` | N-BEATS generic and interpretable, global MLP |
| `src/significance.py` | Friedman, Nemenyi, Wilcoxon with Holm correction |
| `src/metaheuristic.py` | GA, PSO, and a budget-matched random-search control |
| `src/refit_tuned.py` | refits the selected configurations and scores FY2025 |

**Is there anything that should NOT be used?**
The dataset must not be used to make claims about any real organisation's finances, nor
as evidence about real-world ERP data distributions. It is a *testbed*, not an observation.

**What (if any) limitations should users be aware of?**
1. **Synthetic.** Realism is asserted by construction and checked by the validation suite;
   it is not established against a real ledger, because no such public ledger exists.
2. **Scale.** A single mid-sized enterprise, 236 forecastable series — far below the regimes
   where large forecasting models are usually evaluated.
3. **Scope.** One reporting use case (variance commentary); no multi-entity consolidation.
4. **Calibration provenance.** Shape parameters derive from other *synthetic* public
   datasets, so any bias they carry may propagate.
5. **Accounting simplification.** One chart-of-accounts profile, a constrained set of
   posting-key behaviours, no tax, consolidation or intercompany elimination.

---

## Distribution

**How will the dataset be distributed?**
Public GitHub repository with a DOI-minted archival snapshot on Zenodo.

**Licence:**
MIT for code (`LICENSE`); CC BY 4.0 for data (`LICENSE-DATA`).

**Will the dataset be updated?**
The released snapshot is fixed and versioned. Because generation is deterministic, any
future variant is published as a new tagged release rather than by overwriting. Users
should cite the version DOI of the release they used, not the concept DOI, so that the
artefact behind a given result stays identifiable.

---

## Maintenance

**Who supports/maintains the dataset?**
The authors.

**How can the dataset be regenerated?**
```bash
pip install -r requirements.txt
python src/sap_fi_generator.py ./out      # deterministic, seed = 42
python src/validate_dataset.py ./out      # expect 16/16 checks passed
```

**Verified reproducibility:** the transactional output has reproduced byte-identically on
five executions across three platforms — Google Colab, Kaggle, and a separate Linux
environment — carrying different operating-system images and different builds of the
underlying numerical libraries. `acdoca_actuals.csv` MD5
`95466e9fc36db89f43f4516614169c0d`, matching the snapshot in `data/` exactly. Cross-platform
byte identity is a stronger property than determinism on one machine, since it shows the
output does not depend on the floating-point or serialisation behaviour of a particular
installation.

**Is there an erratum?**
Yes, recorded here rather than silently corrected.

*Version 1.1.0.* The reversal-linkage check (I2) compared accrual and reversal documents by
document number alone. SAP restarts the document-number sequence for each company code and
fiscal year, so a document number is not by itself an identifier, and the comparison
conflated distinct records. The check also did not account for a reversal posting in the
period *following* its accrual, which places a December accrual and its January reversal in
different fiscal years. Rekeying on (company code, fiscal year, document number) and
recovering the accrual year from the reversal's own period changes the reported counts from
663 accruals / 647 reversals to 689 / 672. The verdict is unchanged — no accrual inside the
span is unreversed — and **the generated data are unaffected**; only the check was wrong.
The corrected check additionally tests for reversals pointing at no accrual.

Three further checks were widened in the same release after the same style of review: I3
tested only the cost centre while its label implied the dimensional layer as a whole; R2
asserted only that special-period lines existed, which a single line would have satisfied;
and T2b tested one revenue account pooled across all four years. Checks P1 and P2 expressed
their errors over different denominators — the budget error over the budget, the forecast
error over the actual — and then compared the two directly; both are now expressed over the
realised value, which changes the reported budget variance from 14.2% to 14.3%.

---

## Validation

The dataset ships with its own acceptance suite (`src/validate_dataset.py`), 16 checks in
three families:

- **Integrity** — every document balances; every accrual inside the span is reversed and no
  reversal points at a non-existent document; no orphan references across cost centre, GL
  account, profit centre or company code.
- **Realism** — gross margin 44.2% (45% target), operating margin 4.3%; seasonality
  detectable in every revenue account in every fiscal year (amplitude ≥ 0.45 across all
  eight account-year combinations).
- **Plan** — versions present and coherent; budget variance non-trivial and bounded
  (median 14.3% of actual across 11,328 cells); the rolling forecast beats the budget
  (12.3% vs 14.3% median absolute error, both expressed over actual), as it should.

All 16 pass on the released snapshot.

The suite is designed to fail the release rather than warn, and has been used as a
development instrument as well as a certification one: three generator defects were found
by checks intended to certify it, and the suite's own reversal check was found defective and
corrected in 1.1.0 (see Maintenance).
