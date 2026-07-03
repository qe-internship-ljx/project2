# Experiment 5 — Capacity scaling

Re-evaluates the project's factor books with **market-cap-aware position
sizing** instead of equal weighting, asking how much of each premium survives
when the book is tilted toward the names that can actually absorb capital.

The standard quintile/tertile book puts as much money into a $150M microcap as
into a $150B megacap — uninvestable at size; pure cap weighting swings to the
opposite extreme. `capacity_scaling.py` tests two monotone middle grounds,
applied **within each leg** (buckets, orientation, benchmark and every
downstream statistic are unchanged, so each table reads directly against the
standard equal-weighted book):

```
sqrt:  w_i ∝ sqrt(mcap_usd_i)      # mild tilt toward larger, more liquid names
log6:  w_i ∝ log(mcap_usd_i)^6     # steep tilt, concentrates on the largest names
```

## Two pipelines

**Univariate (`run`).** Re-evaluates **exactly the same factor universe as
Experiment 2's `tertile.py`** (Experiment 1's general factors + every
Experiment 2 software subexperiment; `tertile.SOURCES` is imported by path as
the single source of truth). For each factor the even-quintile sort is
Experiment 1's `prepare_slice`; only the within-leg weights change. Turnover
cost is charged on the **actual capacity-weighted legs** (`cost.turnover_cost`),
and each factor is scored with the standard alpha-table statistics (full sample
and 2016+), ranked by combined alpha t-stat. Output: one alpha table per
weighting scheme —

```
output/capacity_scaling/long_short_market_alpha.png        # sqrt
output/capacity_scaling/log6_long_short_market_alpha.png   # log6
```

**Bivariate (`run_bivariate`).** Applies the sqrt weighting to Experiment 3's
double-sort corners: the T3∩T3 / T1∩T1 legs come from
`bivariate_tertile.double_sorted` (imported by path), sqrt-cap-weighted within
each corner. Default pairs: `return_stability × gross_profitability` and
`revenue_stability × gross_profitability` (or pass pairs on the CLI). Each pair
is benchmarked against its constituents' standalone books and reports the
largest single-name ownership for a $100M book. Output: one performance table
per pair under `output/bivariate_tertile/`.

## Reuse

Experiment 1's `factors` / `cost` / `regression` are imported off `sys.path`
(the generic engine, no library injection); Experiment 2's `tertile.py` and
Experiment 3's `bivariate_tertile.py` are loaded by file path for the factor
universe and the double-sort mechanics. This experiment adds **only** the
within-leg weighting schemes.

## Run

```bash
python capacity_scaling.py                                        # univariate (sqrt + log6) + default bivariate pairs
```

Requires Experiments 1–3 to have been run first (it reads their factor panels
and alpha CSVs).
