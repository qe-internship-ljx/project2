# Experiment 4 — Factor timing

Tests whether the software factors' long/short books earn their premia when
traded **only in favourable months**. The base book each overlay gates is the
factor's **quarterly-repositioned** top-minus-bottom book (the project-wide
convention); only the in/out gate is a monthly decision. Two timing signals are
tried, one factor-specific and one market-wide, each applied to **every** ranked
factor:

| Module | Signal | Enter the book when | Candidates |
|---|---|---|---|
| `spread_timing.py` | the factor's own **value spread** — the monthly top-minus-bottom gap in raw (un-z-scored) factor values | the spread is **above its trailing 12-month average** (wide dispersion = more to be paid for ranking on the factor) | **every** ranked factor (`quarter_position.ranked_factors`, `n=None`) |
| `vol_timing.py` | the market's **volatility regime** — the VVIX index, 10-trading-day moving average (from `data/VolatilityIndexData.csv`) | the VVIX MA is **above 95** (elevated vol-of-vol regime) | **every** ranked factor (`quarter_position.ranked_factors`, `n=None`) |

`spread_timing.py` runs its rule at **two bucket granularities** — the headline
quintile (Q5−Q1) book and, identically, the tertile (Q3−Q1) book — threading a
single bucket count `n` through the spread signal, the traded book and the table.
Each granularity reads its own `n`-bucket ranking as the candidate set (matching
`composite.py`'s quintile/tertile convention).

Both signals are formation-date characteristics (the spread and its trailing
average, or the last VVIX MA on or before the rebalance date), so entry into
month `t+1` uses only information known at month-end `t` — no look-ahead.

## What each script does

For each candidate factor:

1. **Timed book.** The bullish-oriented quarterly top-minus-bottom spread is the
   shared `quarter_position.quarter_held_spread` (the primitive behind
   `factor_momentum.signed_spread`), formed on `n` buckets — no return is recomputed
   — then zeroed out in months where the timing flag is off (the book sits in cash).
2. **Cost.** `cost.turnover_cost` charges turnover on the factor's quarterly-held
   legs, passing the timing flag as the `active` mask so exits and re-entries are
   priced (a month spent in cash is free, but leaving and re-entering the book is not).
   `cost.active_month_cost` then folds each run's exit-month liquidation back onto
   the run's **last active month**, so the reported cost lives entirely on the
   activated months — and a book entered and exited within a single month is charged
   the full **round-trip (double) cost** on that one month.
3. **Evaluation.** The timed book is scored over the full sample and 2016+ on the
   common sample (months where both the return and the signal are defined). **Every
   reported quantity — the raw Sharpe, the average cost, the net-of-cost Sharpe, the
   industry-neutral alpha and the walk-forward beta-neutral Sharpe — is measured over
   the activated (in-market) months only** (`composite.book_stats`; the hedge β is
   re-estimated on an expanding, look-ahead-free window). The exact-zero cash months
   are excluded throughout: including them would thin the mean but not the volatility,
   mechanically diluting the alpha and dragging both the raw and neutralised Sharpe
   below what a significant alpha implies. The `n` column is thus the count of
   activated months, and `% activation` is their share of the common sample.
4. **Output.** One consolidated alpha table with a row per factor (the timed
   book): `output/vol_timing_quintile_performance.png`, and for `spread_timing.py` one table
   per granularity — `output/spread_timing_quintile_performance.png` (quintile, headline) and
   `output/spread_timing_tertile_performance.png` (tertile).

## Reuse

Both scripts import Experiment 3's `composite` (which loads Experiment 1's
engine and analysis modules by path) and `factor_momentum`; the cost model is
Experiment 1's `cost.py`, and the table renderer is Experiment 1's
`regression.render_alpha_table`. This experiment adds **only** the two timing
signals and their timed-book evaluation.

## Run

```bash
python spread_timing.py    # every ranked factor, dispersion-timed (quintile + tertile)
python vol_timing.py       # every ranked factor, VVIX-regime-timed
```

Requires Experiment 2's factor libraries to have been built (the candidates come
from `quarter_position.ranked_factors`), and (for vol timing)
`data/VolatilityIndexData.csv` (VVIX history starts 2006-03, so its sample is
shorter).
