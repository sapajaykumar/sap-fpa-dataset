# A Synthetic SAP-Structured FP&A Dataset

A reproducible, SAP-structured synthetic dataset for research on **financial planning and
analysis** — combining double-entry transactional actuals with budget and rolling-forecast
plan versions across four fiscal years.

## Why this exists

No public dataset combines SAP-style transactional structure, enough temporal depth to
learn seasonality, *and* plan versions alongside actuals. The two closest candidates each
fail on a different axis. The Schreyer et al. ERP dataset carries the accounting attribute
schema but has no date, fiscal year or period field of any kind, and no cost centre — so
neither a time series nor a planning cell can be formed from it. Google's Cortex sample
data has ECC structure and 51 months, but its monthly variation reflects load artifacts
rather than business seasonality, and it has no plan tables. Both are themselves synthetic.
This dataset fills that gap.

## What's here

```
data/            fixed-seed snapshot used in the paper (~19 MB)
src/             generator, 16-check validation suite, benchmark modules
DATASHEET.md     full datasheet (Gebru et al. format)
```

| File | Rows | Contents |
|---|---|---|
| `acdoca_actuals.csv` | 159,934 | Posting lines in BKPF/BSEG field names, 79,967 documents |
| `acdoca_universal_journal.csv` | 159,934 | The same postings in S/4HANA universal-journal names |
| `plan_versions.csv` | 22,656 | Monthly plan by cost centre × account × version |
| `dim_*.csv` | — | Company codes, cost centres, groups, GL accounts, profit centres |

**Coverage:** FY2022–FY2025 · 48 periods · 3 company codes · 36 cost centres ·
19 GL accounts · 5 categories · EUR.

**Version semantics (SAP convention):** `0` = actuals · `1` = Budget · `2` = Rolling
Forecast. Version 0 is carried on the transactional records; `plan_versions.csv` holds
versions 1 and 2 only.

### Two schemas for the same postings

The generator produces classic FI documents, so its native field names are those of BKPF
and BSEG: `HKONT` for the account, `DMBTR` for the amount, `SHKZG` for the debit/credit
indicator. `acdoca_universal_journal.csv` projects the same rows onto S/4HANA universal-
journal names — `RACCT`, `RCNTR`, `POPER`, `HSL`.

The substantive difference is not the renaming. The universal journal carries a **signed**
amount in `HSL` rather than an unsigned figure qualified by an indicator, so summing `HSL`
gives the correct net figure without interpreting `SHKZG`. All 79,967 documents balance on
`HSL` alone. The view is derived row-for-row from the postings file and introduces no
independent randomness, so the two cannot drift apart.

## Reproducing it

```bash
pip install -r requirements.txt
python src/sap_fi_generator.py ./out     # deterministic, seed = 42
python src/validate_dataset.py ./out     # expect: 16/16 checks passed
```

Generation is deterministic. The transactional output has reproduced **byte-identically**
on five executions across three platforms — Google Colab, Kaggle, and a separate Linux
environment — with different operating-system images and different builds of the
underlying numerical libraries. MD5 of `acdoca_actuals.csv`:
`95466e9fc36db89f43f4516614169c0d`.

That output matches the snapshot in `data/` exactly, so every number in the accompanying
paper is regenerable from this repository alone.

### Benchmark modules

The forecasting benchmark reported in the paper is reproduced by four further scripts:

```bash
python src/baselines.py ./out          # naive, seasonal naive, ETS, LightGBM, Budget v1
python src/classical_extra.py ./out    # Theta, SARIMA, STL+ETS
python src/nbeats.py ./out             # N-BEATS generic and interpretable, global MLP
python src/significance.py ./out       # Friedman, Nemenyi, Wilcoxon with Holm correction
python src/metaheuristic.py ./out 40   # GA, PSO and a budget-matched random control
python src/refit_tuned.py ./out        # refits the winners and scores FY2025
```

Deterministic methods reproduce exactly. Deep-learning results agree to within the reported
seed standard deviations rather than exactly, because GPU reduction orders are not
deterministic: generic N-BEATS returns 0.964 on a Tesla T4 and 0.958 on CPU.

`classical_extra.py` and `metaheuristic.py` checkpoint as they run, into `out/ckpt/` and
`out/ckpt_meta/`. Re-running resumes; delete those folders to force a clean run.

## Validation

The dataset ships with its own acceptance suite — 16 checks across integrity (documents
balance, accruals reverse and their reversals point at real documents), realism (44.2%
gross margin, seasonality detectable in every revenue account in every year), and plan
coherence (the rolling forecast beats the locked budget, as it should). All pass on the
released snapshot.

The suite is designed to fail the release rather than warn. It has also been used as a
development instrument: three mean-preservation bugs in the generator were found by
checks that were supposed to certify it, and the reversal check itself was corrected in
v1.1.0 after it was found to be comparing document numbers without qualifying them by
company code and fiscal year.

## Honest limitations

- **Synthetic.** Realism is asserted by construction and verified by the acceptance suite,
  not established against a real ledger — because no public real ledger exists.
- **Single enterprise, 236 forecastable series** — small relative to typical forecasting
  benchmarks.
- **Shape parameters were calibrated against other public *synthetic* sources**, never
  against real or client data.
- **The accounting model is deliberately simplified.** One chart-of-accounts profile, a
  constrained set of posting-key behaviours, and no consolidation, intercompany
  elimination or tax treatment.

No real company, client, or SAP customer data was used at any stage.

## Citation

See `CITATION.cff`. Archived at <https://doi.org/10.5281/zenodo.21888648>.

## Licence

Code: MIT (`LICENSE`) · Data: CC BY 4.0 (`LICENSE-DATA`).
