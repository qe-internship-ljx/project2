# Experiment 4 — Factor timing

Tests whether the software factors' long/short books earn more when traded
**only in favourable months** than when held always-on. Two timing signals are
tried, one factor-specific and one market-wide:

| Module | Signal | Enter the book when | Candidates |
|---|---|---|---|
| `spread_timing.py` | the factor's own **value spread** — the monthly Q5−Q1 gap in raw (un-z-scored) factor values | the spread is **above its trailing 6-month average** (wide dispersion = more to be paid for ranking on the factor) | **every** ranked factor (`top_factors/all_factors_ranked.csv`) |
| `vol_timing.py` | the market's **volatility regime** — the VVIX index, 10-trading-day moving average (from `data/VolatilityIndexData.csv`) | the VVIX MA is **above 95** (elevated vol-of-vol regime) | the **top 5** factors (`top_factors/top_factors.csv`) |

Both signals are formation-date characteristics (the spread and its trailing
average, or the last VVIX MA on or before the rebalance date), so entry into
month `t+1` uses only information known at month-end `t` — no look-ahead.

## What each script does

For each candidate factor:

1. **Always-on book.** The bullish-oriented Q5−Q1 spread is read from the
   factor's published `quintile_returns.csv` (via
   `factor_momentum.signed_spread`) — no return is recomputed.
2. **Timed book.** The same spread zeroed out in months where the timing flag is
   off (the book sits in cash).
3. **Cost.** `cost.long_short_cost` charges turnover on both books; the timed
   book passes its flag as the `active` mask so exits and re-entries are priced
   (a month spent in cash is free, but leaving and re-entering the book is not).
4. **Evaluation.** Both books are scored over the full sample and 2016+ on the
   common sample (months where both the return and the signal are defined):
   gross and net-of-cost mean / t-stat / Sharpe, industry-neutral alpha and
   beta, and beta-neutral Sharpe (`composite.book_stats`). The timed book's
   **alpha regression runs over in-market months only** — the exact-zero cash
   months would mechanically dilute the alpha; the mean/t/Sharpe still cover
   the full timed series.
5. **Output.** One consolidated alpha table with a row per factor × book
   (always-on / timed):
   `output/spread_timing_performance.png` and `output/vol_timing_performance.png`.

## Reuse

Both scripts import Experiment 3's `composite` (which loads Experiment 1's
engine and analysis modules by path) and `factor_momentum`; the cost model is
Experiment 1's `cost.py`, and the table renderer is Experiment 1's
`regression.render_alpha_table`. This experiment adds **only** the two timing
signals and the timed-vs-always-on comparison.

## Run

```bash
python spread_timing.py    # every ranked factor, dispersion-timed
python vol_timing.py       # top-5 factors, VVIX-regime-timed
```

Requires Experiment 2's collection step (`main.py collect`) to have written
`top_factors/all_factors_ranked.csv` / `top_factors.csv`, and (for vol timing)
`data/VolatilityIndexData.csv` (VVIX history starts 2006-03, so its sample is
shorter).
