# Software & Services Factor Pipeline

Tests eight equity style factors on the **Software & Services** GICS industry
group. For every security each factor is computed monthly, standardised into a
**z-score relative to the industry cross-section**, and evaluated against the
next month's return two ways.

## Files

| File | Role |
|------|------|
| `factors.py` | Loads the raw data, builds the monthly panel, computes each factor and its industry z-score, and writes `output/factor_panel.csv`. Also the shared library (`load_panel`, `prepare_slice`, `assign_quintiles`, `ols`). |
| `quintile.py` | **Approach 1** — sorts each month into 5 even (equal-count) z-score quintiles and computes each bucket's next-period mean return. |
| `regression.py` | **Approach 2** — each month, regresses next-period return on the factor z-score across all securities (OLS), producing one monthly beta series (the cross-sectional factor premium). |
| `banks_insurance.py` | Thin driver that runs the **identical** pipeline on the **Banks + Insurance** cross-section (`gics_industry_name in {Banks, Insurance}`), writing to `output/banks_insurance/`. |
| `commodity_producers.py` | Thin driver that runs the **identical** pipeline on the **Commodity Producers** cross-section (`gics_industry_name in {Metals & Mining, Oil, Gas & Consumable Fuels}`), writing to `output/commodity_producers/`. |

Run in order (each step caches to `output/`):

```bash
python factors.py      # build output/factor_panel.csv
python quintile.py     # approach 1 -> output/quintile/
python regression.py   # approach 2 -> output/regression/
```

`quintile.py` / `regression.py` auto-build the panel if it is missing.

### Other universes

The data loaders are parameterised by a `factors.Universe`, so the same code
runs against any GICS cross-section. Three are predefined: `SOFTWARE_SERVICES`
(the default), `BANKS_INSURANCE` and `COMMODITY_PRODUCERS`. To run the whole
pipeline on Banks + Insurance or Commodity Producers:

```bash
python banks_insurance.py          # -> output/banks_insurance/{factor_panel.csv, quintile/, regression/}
python commodity_producers.py      # -> output/commodity_producers/{factor_panel.csv, quintile/, regression/}
```

Each module also accepts a universe slug on the command line, e.g.
`python factors.py banks_insurance`, `python quintile.py banks_insurance`.

## Factors (main signal of each family)

| Factor (`name`) | Family | Definition |
|------|--------|------------|
| `earnings_yield` | Value | `earnings_ltm / security_mcap` (E/P) |
| `momentum_12m` | Momentum | cumulative total return over months `[t-12, t-1]` (skips the most recent month) |
| `reversal_1m` | Short-term reversal | most recent 1-month total return |
| `gross_profitability` | Profitability/quality | `gross_income_ltm / assets` (GP/A) |
| `beta` | Low-risk | 36-month trailing beta vs. the equal-weighted universe return |
| `asset_growth` | Investment | `assets_t / assets_{t-12m} - 1` |
| `net_issuance` | Net issuance | diluted-share-count YoY growth |
| `sue` | Earnings momentum / PEAD | YoY change in LTM earnings, scaled by its trailing std |

## Methodology notes

- **Universe / industry mean.** The cross-section is the whole Software &
  Services group, so the z-score is `(x - mean) / std` taken across all
  in-group names each month.
- **Point-in-time.** `date_fundamental` is already a month-end PIT snapshot, so
  fundamentals join to month-end prices with no look-ahead.
- **No FX.** Every factor is a ratio or a return and so is currency-neutral.
- **Winsorisation.** Both the factor (before z-scoring) and `next_return`
  (before averaging/regression) are winsorised at the 1st/99th percentile each
  month, so extreme microcap moves do not dominate the equal-weighted means.

## Outputs

```
output/
  factor_panel.csv                      # tidy: date, stock_id, factor, value, zscore, next_return
  quintile/
    <factor>_quintile_returns.csv       # months x Q1..Q5 + Q5-Q1 spread
    <factor>_quintile_returns.png       # 5 monthly-return curves (per spec)
    <factor>_quintile_cumulative.png    # 5 cumulative-growth curves (readability)
    summary.csv                         # per-factor quintile means + long-short stats
  regression/
    <factor>_regression.csv             # months x {beta, alpha, tstat, r2, n}
    <factor>_beta.png                   # beta over time (with time-series mean)
    summary.csv                         # per-factor mean beta + FM t-stat (full & 2016+)
    summary_table.png                   # rendered table of the above
```
