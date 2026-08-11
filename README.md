# A Synthetic SAP-Structured FP&A Dataset

A reproducible, SAP-structured synthetic dataset for research on **financial planning and
analysis** — combining double-entry transactional actuals with budget and rolling-forecast
plan versions across four fiscal years.

## Why this exists

No public dataset combines SAP-style transactional structure, enough temporal depth to
learn seasonality, *and* plan versions alongside actuals. The two closest candidates each
fail on one axis — the Schreyer et al. ERP dataset has the schema but no usable time
dimension; Google's Cortex sample data has ECC structure and 51 months, but its temporal
variation reflects load artifacts rather than business seasonality, and it has no plan
tables. Both are themselves synthetic. This dataset fills that gap.

## What's here

```
data/     fixed-seed snapshot used in the paper (~19 MB)
src/      generator + 16-check validation suite
DATASHEET.md   full datasheet (Gebru et al. format)
```

| File | Rows | Contents |
|---|---|---|
| `acdoca_actuals.csv` | 159,934 | ACDOCA-style posting lines, 79,967 documents |
| `plan_versions.csv` | 22,656 | Monthly plan by cost centre × account × version |
| `dim_*.csv` | — | Company codes, cost centres, groups, GL accounts, profit centres |

**Coverage:** FY2022–FY2025 · 48 periods · 3 company codes · 36 cost centres ·
19 GL accounts · 5 categories · EUR.

**Version semantics (SAP convention):** `0` = actuals · `1` = Budget · `2` = Rolling Forecast.

## Reproducing it

```bash
pip install -r requirements.txt
python src/sap_fi_generator.py ./out     # deterministic, seed = 42
python src/validate_dataset.py ./out     # expect: 16/16 checks passed
```

Generation is deterministic. Two independent runs at seed 42 produce **byte-identical**
output, and that output matches the snapshot in `data/` exactly — so every number in the
accompanying paper is regenerable from this repository alone.

## Validation

The dataset ships with its own acceptance suite — 16 checks across integrity (documents
balance, accruals reverse), realism (44.2% gross margin, learnable seasonality), and plan
coherence (rolling forecast beats budget, as it should). All pass on the released snapshot.

## Honest limitations

- **Synthetic.** Realism is asserted by construction and verified by the acceptance suite,
  not established against a real ledger — because no public real ledger exists.
- **Single enterprise, 236 forecastable series** — small relative to typical forecasting
  benchmarks.
- **Shape parameters were calibrated against other public *synthetic* sources**, never
  against real or client data.

No real company, client, or SAP customer data was used at any stage.

## Citation

See `CITATION.cff`. A DOI will be minted on archival release.

## Licence

Code: MIT (`LICENSE`) · Data: CC BY 4.0 (`LICENSE-DATA`).
