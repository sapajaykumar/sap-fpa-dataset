# Datasheet — Synthetic SAP-Structured FP&A Dataset

Following the *Datasheets for Datasets* format (Gebru et al., 2021).

---

## Motivation

**For what purpose was the dataset created?**
To support research on automated financial planning and analysis (FP&A) — specifically
forecasting and grounded report generation over enterprise ledger data. No public dataset
combines (a) SAP-style double-entry transactional structure, (b) enough temporal depth to
learn seasonality, and (c) plan/budget versions alongside actuals. Existing candidates fail
on at least one axis: the Schreyer et al. ERP fraud dataset has SAP schema but no usable
time dimension; the Google Cortex sample data has ECC structure and 51 months but its
temporal variation reflects load artifacts rather than business seasonality, and it carries
no plan tables. Both are themselves synthetic. This dataset was built to fill that gap.

**Who created the dataset?**
Ajay Kumar (IIIT Dharwad), as part of an M.Tech dissertation, supervised by
Dr. Utkarsh Khaire.

**Who funded the creation of the dataset?**
No external funding; produced as part of postgraduate academic work.

---

## Composition

**What do the instances represent?**
Two linked tables:

| File | Rows | Represents |
|---|---|---|
| `acdoca_actuals.csv` | 159,934 | Individual accounting line items (postings), ACDOCA-style |
| `plan_versions.csv` | 22,656 | Monthly plan amounts per cost centre × account × version |

Plus five dimension tables: company codes, cost centres, cost centre groups, GL accounts,
profit centres.

**How many instances are there in total?**
159,934 posting lines across 79,967 accounting documents, plus 22,656 plan rows.

**What data does each instance consist of?**
`acdoca_actuals.csv` carries 19 fields using SAP field names: `BUKRS` (company code),
`BELNR` (document number), `GJAHR` (fiscal year), `MONAT` (period), `BUDAT`/`BLDAT`
(posting/document date), `BLART` (document type), `WAERS` (currency), `VERSION`,
`USNAM` (user), `STBLG` (reversal document), `BUZEI` (line item), `BSCHL` (posting key),
`SHKZG` (debit/credit indicator), `HKONT` (GL account), `KOSTL` (cost centre),
`PRCTR` (profit centre), `DMBTR` (amount in local currency), `WRBTR` (document currency amount).

`plan_versions.csv` carries 11 fields including `VERSION`, `VERSION_TXT`, `CATEGORY`
and `DMBTR`.

**Version semantics (SAP convention):**
- `VERSION = 0` → actuals (in `acdoca_actuals.csv`)
- `VERSION = 1` → Budget (locked before the year opens; mildly optimistic)
- `VERSION = 2` → Rolling Forecast (re-cut quarterly)

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

**Is the software available?**
Yes — `src/sap_fi_generator.py` (generation) and `src/validate_dataset.py` (validation).

---

## Uses

**What tasks could the dataset be used for?**
- Time-series forecasting at cost-centre × account granularity (with budget and rolling
  forecast as human baselines).
- Grounded/verifiable natural-language report generation over a transactional ledger.
- Evaluation of retrieval and aggregation logic under SAP semantics (signed aggregation,
  accrual netting, currency decimal handling, materiality thresholds).
- Variance analysis and plan-vs-actual reasoning.

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

---

## Distribution

**How will the dataset be distributed?**
Public GitHub repository with a DOI-minted archival snapshot on Zenodo.

**Licence:**
MIT for code (`LICENSE`); CC BY 4.0 for data (`LICENSE-DATA`).

**Will the dataset be updated?**
The released snapshot is fixed and versioned. Because generation is deterministic, any
future variant should be published as a new tagged release rather than by overwriting.

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

**Verified reproducibility:** two independent runs at `seed = 42` produce
byte-identical CSVs, and these match the snapshot in `data/` exactly
(`acdoca_actuals.csv` MD5 `95466e9fc36db89f43f4516614169c0d`). Every figure reported in
the accompanying paper is therefore regenerable from this repository alone.

---

## Validation

The dataset ships with its own acceptance suite (`src/validate_dataset.py`), 16 checks in
three families:

- **Integrity** — every document balances; all accruals reverse; no orphan references.
- **Realism** — gross margin 44.2% (45% target), operating margin 4.3%, seasonality is
  learnable signal (a seasonal-naive forecast beats naive by >2× on total revenue).
- **Plan** — versions present and coherent; budget variance non-trivial and bounded
  (median 14.2% across 11,328 cells); the rolling forecast beats the budget (12.3% vs 14.2%
  median absolute error), as it should.

All 16 pass on the released snapshot.
