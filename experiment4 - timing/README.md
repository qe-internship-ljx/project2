# Experiment 4 — Factor timing

Tests whether the software factors' long/short books earn their premia when
traded **only in favourable months**. The base book each overlay gates is the
factor's **quarterly-repositioned** Q5−Q1 book (the project-wide convention); only
the in/out gate is a monthly decision. Two timing signals are tried, one
factor-specific and one market-wide, each applied to **every** ranked factor:

| Module | Signal | Enter the book when | Candidates |
|---|---|---|---|
| `spread_timing.py` | the factor's own **value spread** — the monthly Q5−Q1 gap in raw (un-z-scored) factor values | the spread is **above its trailing 6-month average** (wide dispersion = more to be paid for ranking on the factor) | **every** ranked factor (`quarter_position.ranked_factors`, `n=None`) |
| `vol_timing.py` | the market's **volatility regime** — the VVIX index, 10-trading-day moving average (from `data/VolatilityIndexData.csv`) | the VVIX MA is **above 95** (elevated vol-of-vol regime) | **every** ranked factor (`quarter_position.ranked_factors`, `n=None`) |

Both signals are formation-date characteristics (the spread and its trailing
average, or the last VVIX MA on or before the rebalance date), so entry into
month `t+1` uses only information known at month-end `t` — no look-ahead.

## What each script does

For each candidate factor:

1. **Timed book.** The bullish-oriented quarterly Q5−Q1 spread is the shared
   `factor_momentum.signed_spread` (`quarter_position.quarter_held_spread`) — no
   return is recomputed — then zeroed out in months where the timing flag is off
   (the book sits in cash).
2. **Cost.** `cost.turnover_cost` charges turnover on the factor's quarterly-held
   legs, passing the timing flag as the `active` mask so exits and re-entries are
   priced (a month spent in cash is free, but leaving and re-entering the book is not).
3. **Evaluation.** The timed book is scored over the full sample and 2016+ on the
   common sample (months where both the return and the signal are defined):
   gross and net-of-cost mean / t-stat / Sharpe, industry-neutral alpha, and the
   walk-forward beta-neutral Sharpe (`composite.book_stats`; the hedge β is
   re-estimated on an expanding, look-ahead-free window). The timed book's
   **alpha regression and beta-neutral Sharpe run over in-market months only** —
   the exact-zero cash months would mechanically dilute the alpha and drag the
   neutralised Sharpe below what a significant alpha implies (they thin the mean
   but not the volatility); the raw mean/t/Sharpe still cover the full timed series.
4. **Output.** One consolidated alpha table with a row per factor (the timed
   book): `output/spread_timing_performance.png` and
   `output/vol_timing_performance.png`.

## Reuse

Both scripts import Experiment 3's `composite` (which loads Experiment 1's
engine and analysis modules by path) and `factor_momentum`; the cost model is
Experiment 1's `cost.py`, and the table renderer is Experiment 1's
`regression.render_alpha_table`. This experiment adds **only** the two timing
signals and their timed-book evaluation.

## Run

```bash
python spread_timing.py    # every ranked factor, dispersion-timed
python vol_timing.py       # every ranked factor, VVIX-regime-timed
```

Requires Experiment 2's factor libraries to have been built (the candidates come
from `quarter_position.ranked_factors`), and (for vol timing)
`data/VolatilityIndexData.csv` (VVIX history starts 2006-03, so its sample is
shorter).
