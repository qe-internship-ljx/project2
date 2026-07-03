# Software & Services Factor Pipeline

Tests nine equity style factors on the **Software & Services** GICS industry
group. For every security each factor is computed monthly, standardised into a
**z-score relative to the industry cross-section**, and evaluated against the
next month's return two ways — an even-quintile portfolio sort and a monthly
cross-sectional regression — each book also scored for its industry-neutral
alpha and its average turnover cost under the project's trading-cost model.

## Files

| File | Role |
|------|------|
| `factors.py` | Loads the raw data, builds the monthly panel, computes each factor and its industry z-score, and writes `output/software/factor_panel.csv`. Also the shared library (`load_panel`, `prepare_slice`, `assign_quintiles`, `ols`, `attach_pit_fundamentals`, the `Universe` dataclass). |
| `quintile.py` | **Approach 1** — sorts each month into 5 even (equal-count) z-score quintiles and computes each bucket's next-period mean return. |
| `regression.py` | **Approach 2** — each month, regresses next-period return on the factor z-score across all securities (OLS), producing one monthly beta series (the cross-sectional factor premium). Also runs a **pooled** time-and-cross-section regression of the industry-relative return on the z-score (month-clustered SEs → `normalized_regression.csv`), signs each factor's dollar-neutral long/short book, regresses it on the industry return for the **industry-neutral alpha** table, and tags each book with its average turnover cost and **net-of-cost Sharpe** (raw and β-neutral). |
| `cost.py` | The project's trading-cost model (plan §1) and the **average turnover cost** of each long/short quintile book. Imported by `regression.py`; runnable standalone. |
| `software_service.py` | Driver for the default **Software & Services** universe: builds the panel once, runs `quintile.run` + `regression.run` on it, and adds two universe diagnostics — `universe_size.png` (names per month) and `smallest_mcap.png` (effective size floor before/after the cap screen). |
| `banks_insurance.py` | Thin driver that runs the **identical** pipeline on the **Banks + Insurance** cross-section (`gics_industry_name in {Banks, Insurance}`), writing to `output/banks_insurance/`. |
| `commodity_producers.py` | Thin driver that runs the **identical** pipeline on the **Commodity Producers** cross-section (`gics_industry_name in {Metals & Mining, Oil, Gas & Consumable Fuels}`), writing to `output/commodity_producers/`. |

Run (each step caches under `output/software/`):

```bash
python software_service.py   # full pipeline: panel -> quintile/ -> regression/ (+ diagnostics)
python factors.py            # or: only (re)build output/software/factor_panel.csv
```

`software_service.py` auto-builds the panel if it is missing, then runs both
approaches on the shared panel; `regression.py`'s run also computes the
long/short alpha table, turnover cost and net-of-cost Sharpe. `quintile.py`
and `regression.py` are libraries (no CLI) — they are driven by the
universe drivers and, downstream, by Experiments 2–5.

### Other universes

The data loaders are parameterised by a `factors.Universe`, so the same code
runs against any GICS cross-section. Three are predefined: `SOFTWARE_SERVICES`
(the default), `BANKS_INSURANCE` and `COMMODITY_PRODUCERS`. To run the whole
pipeline on Banks + Insurance or Commodity Producers:

```bash
python banks_insurance.py          # -> output/banks_insurance/{factor_panel.csv, quintile/, regression/}
python commodity_producers.py      # -> output/commodity_producers/{factor_panel.csv, quintile/, regression/}
```

`factors.py` and `cost.py` also accept a universe slug on the command line,
e.g. `python factors.py banks_insurance`.

## Factors (main signal of each family)

| Factor (`name`) | Family | Definition |
|------|--------|------------|
| `earnings_yield` | Value | `earnings_ltm / security_mcap` (E/P) |
| `momentum_12m` | Momentum | cumulative total return over months `[t-12, t-1]` (skips the most recent month) |
| `reversal_1m` | Short-term reversal | most recent 1-month total return |
| `gross_profitability` | Profitability/quality | `gross_income_ltm / assets` (GP/A) |
| `beta` | Low-risk | 36-month trailing beta vs. the market-cap-weighted universe return |
| `asset_growth` | Investment | `assets_t / assets_{t-12m} - 1` |
| `net_issuance` | Net issuance | diluted-share-count YoY growth |
| `sue` | Earnings momentum / PEAD | YoY change in LTM earnings, scaled by its trailing std |
| `accruals` | Accruals (Sloan) | `(earnings_ltm − operating_cash_flow_ltm) / assets` |

## Methodology notes

- **Universe / industry mean.** The cross-section is the whole Software &
  Services group, so the z-score is `(x - mean) / std` taken across all
  in-group names each month.
- **Point-in-time.** Fundamentals are attached on `observation_date` (when the
  report became observable), *not* `date_fundamental` (the fiscal-period stamp,
  which for ~a third of software stock-months precedes publication). A backward
  as-of join admits a report into month-end `t` only once `observation_date ≤ t`
  and forward-fills the last observed figure, so the join carries no look-ahead.
  Aligning on `date_fundamental` instead would let not-yet-reported earnings into
  the formation date — see `attach_pit_fundamentals` in `factors.py`.
- **FX.** Every factor is a ratio or a return and so is currency-neutral, so no
  FX conversion enters the signals. The market-cap **weights** are not
  currency-neutral, however, so for the market/industry return each name's cap is
  converted to USD (`security_mcap_local × fx_to_usd`, the security's trading
  currency at month-end) before weighting — see `attach_usd_market_cap` in
  `factors.py`.
- **Market/industry return.** The within-industry "market" return (the `beta`
  benchmark and the industry-neutral-alpha benchmark) is **market-cap-weighted**:
  each name's return is weighted by its USD market cap at the *start* of the
  return period (the prior month-end cap), so the index is look-ahead free. The
  quintile portfolio legs themselves remain equal-weighted within each leg.
- **Winsorisation.** Both the factor (before z-scoring) and `next_return`
  (before averaging/regression) are winsorised at the 1st/99th percentile each
  month, so extreme microcap moves do not dominate the cross-sectional means.

## Long/short books, industry-neutral alpha & trading cost

Each factor's directional dollar-neutral **Q5−Q1 book** is signed by its
canonical literature direction (`higher_is_bullish` → long Q5 / short Q1, else
long Q1 / short Q5) and regressed on the market-cap-weighted industry return:

```
ls_t = α + β·industry_t + ε_t
```

`α` is the book's **industry-neutral monthly return** and `α t-stat` tests it
against zero — the project's standard "alpha t-stat", reused unchanged by
Experiments 2 and 3. The book also gets a **β-neutral Sharpe**: it is hedged with
a `−β·industry` overlay, but `β` is re-estimated **every month on an expanding
window of only the data available up to that month** (a walk-forward hedge, free
of look-ahead) rather than one full-sample slope — so the Sharpe reflects a hedge
an investor could actually have put on. The industry `β` itself is no longer
reported as a column. All of these are estimated over the full sample and,
separately, the **past decade (2016+)** — where the 2016+ β-neutral Sharpe simply
windows the same walk-forward hedged series, so its monthly betas still use all
history back to sample start.

The **trading-cost model** (`cost.py`, plan §1) charges a one-way cost
`= 3·(11/log₁₀ Mff)⁶ + 3` bps on `Mff = market cap × FX × free float` (floored at
$50M, so ≤ ~0.285% one way). Cost is charged on **turnover, not holdings** — a
name carried at the same weight pays nothing — and each book's **average monthly
turnover cost** is reported alongside its alpha.

## Headline results (Software & Services, full sample ≈ 1998–2026)

Each factor's canonically-signed Q5−Q1 book regressed on the industry return.
`α` is the industry-neutral monthly return; avg cost is the book's mean monthly
turnover cost; FM t is the Fama–MacBeth t-stat of the cross-sectional premium
(from `regression/summary.csv`). Sorted by `α` t-stat.

α is now measured against the **market-cap-weighted** industry return; the turnover
cost and Fama–MacBeth t-stats do not reference the benchmark and are unchanged.

| Factor | L/S dir | α/mo | α t-stat | Avg cost (pp/mo) | FM t (full) |
|---|:--:|---:|---:|---:|---:|
| `gross_profitability` | Q5−Q1 | +1.13% | **+4.33** | 0.037 | +3.83 |
| `net_issuance` | Q1−Q5 | +0.78% | **+3.33** | 0.081 | −3.15 |
| `earnings_yield` | Q5−Q1 | +0.81% | **+2.52** | 0.066 | −1.04 |
| `accruals` | Q1−Q5 | +0.48% | **+2.17** | 0.079 | −1.34 |
| `asset_growth` | Q1−Q5 | +0.52% | +1.91 | 0.087 | −1.99 |
| `momentum_12m` | Q5−Q1 | +0.73% | +1.85 | 0.164 | +1.12 |
| `sue` | Q5−Q1 | +0.41% | +1.85 | 0.101 | +1.17 |
| `beta` | Q1−Q5 | +0.44% | +1.54 | 0.070 | +0.83 |
| `reversal_1m` | Q1−Q5 | +0.28% | +0.83 | 0.581 | −1.74 |

**Takeaways.**
- **Quality, capital-discipline and value lead.** `gross_profitability`,
  `net_issuance` (long low issuance), `earnings_yield` and `accruals` all carry
  significant industry-neutral alpha (t = 2.2–4.3) at low turnover cost
  (~0.04–0.08 pp/month round-trip), so the cost barely dents the gross premium.
- **Fast signals pay for it.** `reversal_1m` is the only book with material
  turnover cost (~0.58 pp/month) — it rebalances almost entirely each month — and
  its alpha is insignificant once that is charged.
- **The alpha sign follows the trade, not the premium.** Several factors with a
  negative Fama–MacBeth premium (`net_issuance`, `asset_growth`, `accruals`)
  are traded in their canonical low-minus-high direction, so the
  signed book's alpha is positive — the two columns are consistent, not in
  conflict.
- **Carried into Experiment 3.** `gross_profitability` (alpha t = 4.3) is one of
  the constituents of the multifactor composite; the others come from the
  software-specific factors of Experiment 2.

## Outputs

The default Software & Services run writes under `output/software/`; the other
universes mirror the same layout under `output/banks_insurance/` and
`output/commodity_producers/`.

```
output/software/
  factor_panel.csv                      # tidy: date, stock_id, factor, value, zscore,
                                        #   next_return, weight (formation-date USD mcap)
  universe_size.png                     # names per month (software driver only)
  smallest_mcap.png                     # effective size floor, before/after cap screen
  quintile/
    <factor>/
      quintile_returns.csv              # months x Q1..Q5 + Q5-Q1 spread
      quintile_cumulative.png           # 5 cumulative-growth curves (log scale)
      long_short.png                    # the signed Q5-Q1 book's cumulative growth of $1
    summary.csv                         # per-factor quintile means + long-short stats
    long_short_market_alpha.csv         # per-factor industry-neutral alpha, Sharpe,
    long_short_market_alpha.png         #   avg turnover cost, walk-forward β-neutral &
                                        #   net-of-cost Sharpe (full & 2016+)
  regression/
    <factor>/
      regression.csv                    # months x {beta, alpha, tstat, r2, n}
      beta.png                          # monthly beta over time (with time-series mean)
    summary.csv                         # per-factor mean beta + FM t-stat + L/S stats (full & 2016+)
    summary_table.png                   # rendered table of the above
    normalized_regression.csv           # pooled industry-relative return per 1σ of factor
    normalized_regression_table.png     #   (month-clustered t-stats), full & 2016+
```
