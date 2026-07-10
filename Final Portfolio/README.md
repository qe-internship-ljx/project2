# Final Portfolio — weighted composite with ownership & trading-volume constraints

The project's deliverable long/short book: one **quarterly-repositioned composite**
that longs / shorts the Software & Services industry on the **weighted sum of five
factor z-scores**, held under a **0.5% single-name ownership cap** and charged an
explicit **average-daily-volume trading drag**, so the book is scored the way it would
actually trade at $100M of capital.

```
score_{i,t} = Σ_f  sign_f · w_f(t) · zscore_{f,i,t}
```

over

```
gross_profitability, rd_stability, revenue_growth_stability,
revenue_growth_skewness, beta
```

Every factor is weighted **1** except `rd_stability`, which carries a
**state-dependent** weight (below). The industry is sorted into equal-count buckets
on the composite and the membership is **repositioned quarterly** (the project-wide
convention — buckets formed only end of Feb/May/Aug/Nov and held three months, via
`quarter_position.quarter_held_membership`); the top-minus-bottom dollar-neutral book
is held under a **0.5% single-name ownership cap at all times** (below) and measured
full sample and past decade (2016+). Both a **quintile** (Q5−Q1) and a **tertile**
(Q3−Q1) version are produced.

`sign_f` orients each factor to its bullish direction, read from the factor's
standalone alpha table exactly as `composite.py` does (`gross_profitability`,
`rd_stability`, `revenue_growth_stability` long the high z-score names; `beta` and
`revenue_growth_skewness` reversed).

## State-dependent `rd_stability` weight

`rd_stability` is normally weighted **0.5**, boosted to **5.0** whenever its
cross-sectional **factor-value spread is unusually wide**:

```
w_rd(t) = 5.0   if  spread_rd(t)  >  trailing-12m mean of spread_rd
          0.5   otherwise
```

The spread signal is **Experiment 4's factor-spread-timing pipeline, reused
verbatim** (`spread_timing.value_spread` / `spread_timing.timing_flags`): the raw
`rd_stability` value of the top bucket minus the bottom bucket of the *same*
n-bucket sort the book trades, compared to its own trailing-12-month average. Both
the spread at `t` and its trailing average are formation-date characteristics known
at `t`, so the boost is strictly look-ahead free. Because the book repositions
quarterly, the weight that fixes a held quarter is the one at that quarter's
reposition month; the score is nonetheless formed every month (the quarterly
primitive holds the reposition-date membership), matching `composite.py`. The boost
is active in ~44% (quintile) / ~43% (tertile) of months.

## 0.5% single-name ownership cap (enforced at each reposition)

No position may own more than **0.5% of a name's own market cap** at formation. At
each quarterly reposition the book starts from equal weights inside each leg; any name
whose position ($50M leg capital × its within-leg weight) would breach 0.5% of its
market cap is **pinned at its cap weight** (`0.005 · mcap / leg_capital`) and the
freed weight is **redistributed pro-rata across the still-uncapped names**,
water-filled until nothing breaches the cap. This is Experiment 5's
`capacity_scaling._cap_ownership` logic — the project's one implementation of the 0.5%
cap — reused verbatim.

The cap is **binding**: the largest **formation** ownership sits at exactly 0.5000% in
both books. The book is genuinely traded at the capped weights, so the long/short
**spread** is the capped-weight top-minus-bottom return (not the equal-weighted bucket
mean), and the turnover cost, the volume drag and the ownership diagnostic all consume
the same capped positions.

**Weights are computed once per reposition and held fixed across the quarter.** The
book only trades at the quarterly reposition dates, so the capped weights are formed
there and carried unchanged through the two following months (via the shared
`quarter_position.quarter_hold`). A name held across a month therefore keeps its
reposition-date weight (`Δw = 0`) and is charged **no cost or drag for simply
holding** — only the reposition rebalance trades. (Recomputing the cap every month
would let the weights drift with market cap and spuriously charge a "trade" every held
month, which was a bug in the first cut.) A name that leaves the panel mid-quarter is
not replaced (its weight goes to cash rather than triggering a compensating trade), so
held-month leg weights can sum to slightly below one — the honest consequence of not
retrading a held book. The reported "Max formation ownership" uses the formation cap
the water-fill enforces; a name's *mark-to-market* ownership can drift above 0.5%
intra-quarter as its cap moves, without any trade.

## Trading-volume drag (the capacity constraint)

Under a **$100M dollar-neutral** book ($50M per leg) we assume we may trade at most
**10% of a name's trailing-12-month average daily USD volume (ADV) per day**. The drag
is charged **only on the weight actually traded** — the reposition-date rebalance (a
held position has `Δw = 0` and pays nothing). When a reposition trade cannot clear in
one day, the excess tail is **delayed**, and we suffer the *opportunity cost* of the
return foregone on the delayed notional over the days needed to complete the trade — an
**untradable-tail** drag, not a fixed slippage rate:

```
trade$_{i,t} = |pos$_{i,t} − pos$_{i,t−1}|   = LEG_CAPITAL · |Δw|_{i,t}
cap$_{i,t}   = 0.10 · ADV_usd_{i,t}          (one day's tradable notional)
days_{i,t}   = trade$_{i,t} / cap$_{i,t}
```

If `days ≤ 1` the trade clears on the reposition day and costs nothing. If
`days > 1` the position is legged in linearly over `days` trading days, so on
average half the delayed notional is out of the market for the ramp. The ramp is
**capped at the quarterly holding horizon** (63 trading days ≈ 3 months) — the book
repositions every quarter, so a name can never be legged in over more than the ~63
days it is actually held; a name whose trade needs longer (large capped position but
thin volume) is treated as never fully established during the hold. The drag on that
name is the fraction of **its own realised next-period return** the delayed notional
foregoes — **not** a fraction of the whole L/S spread (a position being built misses
the return of the name it is building):

```
ramp_frac    = min(days, 63) / 21                 (ramp length in months, capped at one quarter)
drag_{i,t}   = 0.5 · |Δw|_{i,t} · ramp_frac · next_return_{i,t}
```

The book-level drag is the **long-leg foregone return minus the short-leg foregone
return** (a long name's missed return reduces the spread; a short name's — whose
return we are short — adds to it), charged as a return **subtracted from that month's
gross long/short spread** (the footing Experiment 4 nets its turnover cost on). It is
reported as an **average monthly drag** and folded into a **net-of-drag mean and
Sharpe** in the performance table.

> This corrects an earlier version that discounted by `|gross_spread|` rather than
> each name's own return — which overstated the drag by ~18× (0.22%/mo → 0.01%/mo).
> The corrected magnitude matches Experiment 6's independent ADV-participation model
> (`liquidity.py`), which reports ~0.02%/mo on a comparable book.

The **ADV panel** is the trailing-252-trading-day mean of daily
`volume · price_local · fx_to_usd` (USD dollar volume) from the same price feather
and FX table the cost model uses, sampled at each formation month-end — no new data
source is introduced. Names with no ADV contribute no drag.

## Reuse — nothing generic is re-implemented

`final_portfolio.py` is a thin driver over the existing spine:

- **Experiment 3 `composite.py`** supplies the constituent z-score matrix
  (`load_exposures`), the score → tidy-panel shaping (`scored_frame` /
  `as_factor_panel`), the quarterly-held n-bucket sort (`bucket_returns` /
  `quintile_legs`), the industry-neutral alpha and windowed performance
  (`industry_return` / `book_stats`), the turnover cost and net-of-cost Sharpe, the
  largest-single-name ownership diagnostic, and the performance-table + cumulative
  rendering (`render_performance` / `plot_cumulative`). `composite.py` in turn loads
  **Experiment 1's engine + `cost.py` + `regression.py`** and **Experiment 2's
  quarterly primitives** by path (the project's dependency-injection convention).
- **Experiment 4 `spread_timing.py`** supplies the `rd_stability` spread-timing
  signal (`value_spread` / `timing_flags`).
- **Experiment 5 `capacity_scaling.py`** supplies the 0.5% ownership-cap water-fill
  (`_cap_ownership`, reused verbatim as `cap_ownership`).
- **Experiment 2 `quarter_position.quarter_hold`** holds the per-reposition capped
  weights fixed across the quarter, so held months incur no trading.

This module adds **only** (a) the state-dependent `rd_stability` column weighting
(`weighted_composite`), (b) the ownership-capped, weighted long/short book with
reposition-held weights (`capped_legs` / `capped_spread` / `formation_ownership`, on
top of the reused cap water-fill and `quarter_hold`), and (c) the ADV panel +
untradable-tail drag (`adv_panel` / `volume_drag` / `attach_drag`), and (d) the
per-factor return contribution (`benchmark_spread` / `_ols` /
`per_factor_contribution`, ported from Experiment 6's weighted composite). No
Experiment 6 (scratch) module is imported.

## Per-factor return contribution

The realised portfolio long/short spread is attributed to its five constituents by a
**multivariate OLS** of the spread on each factor's standalone univariate quarterly
L/S spread simultaneously (same attribution as Experiment 6's weighted composite,
re-implemented here — nothing is imported from that gitignored experiment):

```
port_t = alpha + Σ_f b_f · factor_f_t + eps_t
```

Each regressor `factor_f_t` is that factor's own bullish-oriented, quarterly-held
univariate L/S spread at the portfolio's **same** n-bucket sort
(`quarter_position.quarter_held_spread` via `benchmark_spread`), so the coefficient
`b_f` is the portfolio's loading on that constituent's return holding the other four
fixed — an attribution of the portfolio's realised return to its building blocks.
`alpha` is the portfolio return not spanned by any of the five. Written per sort as
`<sort>_factor_contribution.png` (a compact `coef` / `t-stat` figure, t-stats shaded
by significance).

## Outputs

```
output/quarter_quintile/quintile_cumulative.png     buckets as cumulative growth of $1 (log)
output/quarter_quintile/quintile_performance.png    the Q5−Q1 book's performance + volume drag
output/quarter_quintile/quintile_factor_contribution.png   per-factor return contribution (multivariate OLS)
output/quarter_tertile/tertile_cumulative.png       (tertile counterpart)
output/quarter_tertile/tertile_performance.png      the Q3−Q1 book's performance + volume drag
output/quarter_tertile/tertile_factor_contribution.png     (tertile counterpart)
```

Each `*_performance.png` uses the **same compact row set as Experiment 6's
liquidity-capped table**
(`experiment6 - auto components/software/weighted_quality_composite/tertile_liquidity_capped_performance.png`),
full sample and 2016+:

1. Industry-neutral α (monthly)
2. α t-stat (shaded by significance)
3. β-neutral Sharpe
4. **Sharpe net of cost/drag (β-neut)** (bold — the turnover cost **and** the
   trading-volume drag netted from the spread before the beta-neutral Sharpe)
5. Avg monthly cost (turnover)
6. Largest single-name ownership ($100M total)
7. Trading-volume drag (monthly)

The turnover cost and the volume drag are also reported **on their own** (rows 5 and
7) for transparency, and are never netted from the gross alpha. The one Sharpe that
incorporates friction is the net-of-cost/drag β-neutral Sharpe (row 4): both the
turnover cost and the drag — each a per-month return cost on the same monthly index —
are summed and subtracted from the gross spread before the beta-neutral Sharpe. (The
Experiment 6 reference nets only the turnover cost into that row; here it nets cost +
drag, since the drag is a genuine frictional cost of trading the book at size.)

## Headline results

Ownership-capped (0.5%, enforced per reposition), drag charged only on reposition
trades with the ramp capped at the quarterly holding horizon:

| Book | α (full) | α t | β-neut Sharpe | Sharpe net cost/drag (β-neut) | Max own. | Avg cost | Vol drag |
|---|---|---|---|---|---|---|---|
| Quintile (Q5−Q1) | +1.19%/mo | +5.61 | +1.58 | +1.49 | 0.500% | 0.062%/mo | 0.012%/mo |
| Tertile (Q3−Q1)  | +0.89%/mo | +5.19 | +1.42 | +1.35 | 0.500% | 0.049%/mo | 0.002%/mo |

The industry-neutral alpha survives both capacity constraints in each book. The volume
drag is **tiny** — the 0.5% ownership cap already keeps positions modest relative to
each name's daily volume, so most reposition trades clear in a day or two — and is
charged **only at the quarterly repositions** (holding a position across a month costs
nothing). The magnitude agrees with Experiment 6's independent ADV model (~0.02%/mo);
because it is so small it barely moves the Sharpe, which is why it is reported as an
information row rather than folded into a net-of-drag Sharpe.

## Run

```bash
cd "Final Portfolio" && python final_portfolio.py    # quintile + tertile
```

Requires the upstream factor panels / rankings to exist — run Experiments 1–2 (and
the R&D extension) first, exactly as `composite.py` requires.
