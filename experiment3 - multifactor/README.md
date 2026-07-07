# Experiment 3 — Composite multifactor model

Combine a **set of factors** into a single cross-sectional score and trade it as
a quintile long/short — the same workflow as Experiment 1, but on a *composite*
signal instead of one raw factor. For each stock each month,

```
composite_{i,t} = Σ_f  sign_f · zscore_{f,i,t}
```

is the **sum of the stock's cross-sectional z-scores** across the chosen factors.
The industry is sorted into five equal-count buckets on this composite and the
membership is **repositioned quarterly** (the project-wide convention — buckets
formed only end of Feb/May/Aug/Nov and held three months, via
`quarter_position.quarter_held_membership`), and the Q5−Q1 dollar-neutral book's
performance is measured (mean, t-stat, Sharpe, and the industry-neutral alpha used
throughout the project). The constituents themselves are the **top five of
Experiment 2's quarterly-repositioned ranking** (`quarter_position.ranked_factors`,
which reads the ranking Experiment 2 persists to
`factor_ranking/quarter_{label}_ranked.csv`).

There is **no significance gate and no regression model**: the caller passes the
factor set explicitly, and the factors are combined by standardised aggregation —
the textbook "composite signal" construction. (This replaces the earlier
t-stat-gated pooled *regression*; see [Changed from the regression model](#changed-from-the-regression-model).)

Five ways to combine the factors, sharing one spine (plus the redundancy tool):

| Module | Combination | Evaluation |
|---|---|---|
| [`composite.py`](composite.py) | equal-weighted z-score sum (signs `±1`), quarterly-repositioned quintile sort (plus a tertile variant for the tertile-ranked top-5 and a half variant for the half-ranked top-5) | full sample + past decade |
| [`weighted_composite.py`](weighted_composite.py) | coefficient-weighted score — **expanding-window** regression premia, refit **quarterly** and held for the quarter | **walk-forward out-of-sample (2007+)** |
| [`bivariate_tertile.py`](bivariate_tertile.py) | independent 3×3 **double sort** on two factors, repositioned quarterly — long T3∩T3, short T1∩T1 (plus a coarser median-split "half" rule) | full sample + past decade |
| [`factor_momentum.py`](factor_momentum.py) | **rotation**: hold the factor (univariate) or double-sort the **top two** factors (bivariate) with the best trailing-12m quarterly book return — re-selected every 3 months | full sample + past decade |
| [`portfolio_overlay.py`](portfolio_overlay.py) | equal-capital **overlay** of the top-5 standalone quarterly books (1/5 each, turnover netted per name across books) | full sample + past decade |
| [`factor_correlation.py`](factor_correlation.py) | redundancy tool — R² of an industry factor on Experiment 1's general factors | — |

[`main.py`](main.py) orchestrates all five pipelines, each in its own
subprocess (they share the dependency-injected Experiment 1 engine, so they must
not share an interpreter), in dependency order: `composite` →
`weighted_composite` → `bivariate_tertile` (twice: the default
`return_stability × gross_profitability` pair, then
`revenue_stability × gross_profitability`, so both books stay current) →
`factor_momentum` → `portfolio_overlay`.

The weighted variant is described under
[Coefficient-weighted variant](#coefficient-weighted-variant-expanding-window-walk-forward--weighted_compositepy);
the double sort, rotation and overlay under
[Other combinations](#other-combinations--bivariate_tertilepy-factor_momentumpy-portfolio_overlaypy);
the rest of this section covers the equal-weighted `composite.py`.

## What it does (`composite.py`)

1. **Resolve** (`resolve_factors`). Look each requested factor up in the
   libraries' alpha tables (`…/quintile/long_short_market_alpha.csv`) to find its
   source panel, family, and **bullish sign**: `+1` when the factor's standalone
   book is long the top z-score quintile (`direction == Q5-Q1`), `−1` otherwise.
   Nothing is hard-coded — orientation is read at run time.

2. **Aggregate** (`load_composite`). Pull each constituent's `zscore` from its
   source `factor_panel.csv`, orient it by its sign so a **higher score is always
   more bullish**, align the constituents on the shared `(date, stock_id)` key,
   require **every** factor present (so the sum is comparable across stocks), and
   sum across factors. A high composite is "attractive across the whole set", so
   the book is unambiguously long Q5 / short Q1.

3. **Sort & hold quarterly** (`bucket_returns`, reusing Experiment 1's
   `prepare_slice` + `quarter_position.quarter_held_membership`). The composite is
   shaped as a one-factor panel; even-count buckets are formed at each reposition
   date (end Feb/May/Aug/Nov) and held fixed for the quarter, giving the months ×
   {Q1..Q5, Q5−Q1} table of mean next-period returns — Experiment 1's winsorisation
   and equal-count sort, repositioned quarterly instead of monthly.

4. **Measure** (Experiment 1's `regression.py` helpers). The Q5−Q1 book is scored
   over the full sample and the past decade (2016+): mean, t-stat, annualised
   Sharpe, the **industry-neutral alpha and its t-stat** (regressing the book on
   the market-cap-weighted Software & Services return), and the **β-neutral
   Sharpe** — the book hedged with a `−β·industry` overlay whose β is re-estimated
   every month on an expanding, look-ahead-free window of only prior data (a
   walk-forward hedge, not one full-sample slope). Same alpha definition as every
   other long/short book in the project.

## Design — maximal reuse, zero duplication

Everything generic is reused from Experiment 1, loaded by path (its folder name
contains spaces, so it cannot be imported normally) and wired with the project's
dependency-injection convention — register the engine as `sys.modules["factors"]`
before importing the analysis modules, exactly as Experiment 2's `main.py` does:

| Concern | Source | Reuse |
|---|---|---|
| Winsorisation + equal-count sort; quarterly holding | `experiment1 - general factors/quintile.py` + `factors.py` + `quarter_position.quarter_held_membership` | sort reused **unmodified**, membership held quarterly |
| Industry return + industry-neutral alpha regression | `experiment1 - general factors/regression.py` | helpers reused (same alpha as the rest of the project) |
| Factor **exposures** (the z-scores) | Exp 1, Exp 2 & R&D `output/.../factor_panel.csv` | read as written |
| Factor **orientation** (bullish sign) | Exp 1, Exp 2 & R&D `output/.../long_short_market_alpha.csv` | read as written |
| Composite construction + per-set reporting | `composite.py` | **new (this experiment)** |
| Expanding-window premium regression + walk-forward | `weighted_composite.py` | **new (this experiment)** |

This experiment therefore adds only the composite construction and its reporting;
every input it consumes was already produced upstream. The pipeline is factor-set
agnostic — any factor produced by Experiments 1–2 (or the R&D extension) can be
combined by name. `composite.py` exposes the shared spine (`load_exposures`,
`scored_frame`, `as_factor_panel`, `industry_return`, `book_stats`,
`plot_cumulative` / `render_performance`); the weighted
variant reuses all of it and adds only the equal-weighted benchmark and the
expanding-window walk-forward.

## Run

```bash
python main.py                                               # all five pipelines, in order (bivariate runs both pairs)

python composite.py                                          # equal-weighted, default set (below)
python composite.py buyback_quality gross_profitability rd_stability
python -c "from composite import run; run(['earnings_yield','sue','beta'])"

python weighted_composite.py                                 # coefficient-weighted, expanding-window walk-forward
python weighted_composite.py buyback_quality gross_profitability rd_stability

python bivariate_tertile.py                                  # double sort, default pair (return_stability x gross_profitability)
python bivariate_tertile.py revenue_stability gross_profitability
python factor_momentum.py                                    # univariate rotation + bivariate top-two double sort
python portfolio_overlay.py                                  # equal-capital overlay of the top-5 books
```

Requires Experiments 1 and 2 (and, for R&D factors, the R&D extension) to have
been run first — it reads their `factor_panel.csv` and `long_short_market_alpha.csv`.

## Outputs (`output/composite/<ranking>/`)

`<ranking>` is `quarter_quintile`, `quarter_tertile` or `quarter_half` (the
quarterly-repositioned hand-off the composite is built from; `<word>` below is
`quintile` / `tertile` / `half`).

```
<word>_cumulative.png       the buckets as cumulative growth of $1 (log scale)
<word>_performance.png      mean / t / Sharpe / industry-neutral alpha / largest
                            single-name ownership for a $100M book (full + 2016+)
```

## Headline result — the top-five factor composite

The default composite blends Experiment 2's five top-ranked factors — the top
five of the quarterly-repositioned `quarter_position.ranked_factors` hand-off that
`factor_momentum.py` also rotates across: `gross_profitability`, `fscore`,
`rd_stability`, `revenue_stability`, `buyback_quality`. Summing their sign-oriented
z-scores fuses weakly-correlated quality / R&D-commitment / capital-discipline /
durability signals into one score, so they carry largely **additive** information.

α below is the industry-neutral alpha against the **market-cap-weighted** Software &
Services return; the Q5−Q1 spread, t-stat and Sharpe do not reference the benchmark
and are unchanged.

| Metric | Full sample (2000–2025, 301 mo) | Past decade (2016+, 120 mo) |
|---|---:|---:|
| Q5−Q1 mean monthly | +0.937% | +0.847% |
| t-stat | +4.11 | +2.53 |
| Sharpe (annualised) | +0.82 | +0.80 |
| **Industry-neutral α (monthly)** | **+1.155%** | **+1.075%** |
| **α t-stat** | **+5.34** | **+3.15** |
| β-neutral Sharpe (walk-forward β) | +1.31 | +1.18 |

**Takeaways.**
- **The composite beats every constituent.** Its industry-neutral alpha t-stat
  (5.34) exceeds each standalone factor's — the strongest constituent,
  `gross_profitability`, reaches 4.40 (`fscore` 4.11, `rd_stability` 4.08,
  `buyback_quality` 3.20, `revenue_stability` 3.06) — the diversification benefit of
  combining weakly-correlated signals (plan §1). The alpha (1.16%/mo) is larger
  than any single factor's too.
- **Near-monotonic sort.** Cumulative growth is ordered with Q5 highest and Q1
  lowest across the whole sample (`quintile_cumulative.png`); the ranking power is
  not a tail effect.
- **Defensive by construction.** The book carries a *negative* industry beta — its
  quality/stability tilt outperforms in down-industry months — so β-hedging lifts
  the Sharpe from 0.82 to **1.31** (full) / 0.80 to **1.18** (2016+). The hedge β is
  re-estimated every month on an **expanding, look-ahead-free window** (walk-forward),
  not one full-sample slope.
- **Robust across the past decade.** The alpha is essentially unchanged in the
  2016+ re-estimation, not a pre-2010 artifact. (For a genuine walk-forward holdout
  with weights refit quarterly, see the weighted variant below.)

## Coefficient-weighted variant (expanding-window walk-forward) — `weighted_composite.py`

Instead of an equal-weighted straight sum, weight each constituent by its
**estimated return premium**, re-estimated **quarterly on an expanding window** and
traded strictly walk-forward, with the resulting quintile membership held for the
quarter. The default set here is **`revenue_stability` + `gross_profitability`**,
with an initial training period through **2006** and the book traded from **2007
onwards** (configurable via `INITIAL_TRAIN_END`).

For each reposition date `t` (end Feb/May/Aug/Nov) after the initial training period:

1. **Expanding-window premia.** Pool every stock-month **strictly before `t`** —
   look-ahead free, only returns already realised by `t` enter — and regress the
   **normalised return** on the formation-date factor z-scores:
   `(r_{i,t+1} − market_{t+1}) = a + Σ b_f·z_{f,i,t} + ε`. The slopes `b_f(t)` are
   that month's premia, with OLS and month-clustered t-stats. The subtracted
   `market` is the **unweighted (equal-weighted)** industry average, so the
   normalisation is not dominated by the few mega-cap software names.
2. **Weighted score.** Score month `t`'s cross-section with those weights,
   `score_{i,t} = Σ b_f(t)·z_{f,i,t}` — the model's predicted industry-relative
   return (intercept dropped; it doesn't affect the cross-sectional ranking). Sort
   into quintiles, long Q5 / short Q1, and **hold that membership for the quarter**.
3. **Walk-forward.** The window expands one quarter and step 1 repeats, so **every
   traded month is out-of-sample**: the weights forming a quarter's book never saw
   that quarter's (or any later) return. There is no single fixed train/test split —
   the whole 2007+ path is the holdout.

### Full-sample premia (reference)

The whole-period regression (t-stats over the **entire** sample) is printed to
the console for reference; the traded book uses the expanding-window weights, not
these.

| Term | Coef (= premium, ind-rel %/mo per 1σ) | t (OLS) | t (cluster) |
|---|---:|---:|---:|
| intercept | +0.043% | +1.09 | +1.02 |
| `revenue_stability` | +0.074% | +1.83 | +0.90 |
| `gross_profitability` | +0.264% | +6.41 | **+4.66** |

`gross_profitability` carries the dominant, strongly significant premium;
`revenue_stability` adds a smaller positive tilt. How each weight and its t-stat
evolve as the window grows is plotted in `paths.png` (weights on top, clustered
t-stats below) — the
`gross_profitability` weight drifts down from ~0.57%→~0.26% as more (lower-premium)
history accrues, but its clustered t-stat stays firmly above 4 throughout.

### Walk-forward Q5−Q1 book (2007+)

| Metric | **Walk-forward OOS (2007+, 228 mo)** |
|---|---:|
| Mean monthly | +0.620% |
| t-stat | +3.17 |
| Sharpe (ann.) | +0.73 |
| Industry-neutral α (monthly) | +0.721% |
| α t-stat | **+3.64** |
| β-neutral Sharpe (walk-forward β) | +0.80 |
| α vs `revenue_stability` book | +0.462% (t = **+3.02**) |
| α vs `gross_profitability` book | +0.110% (t = +1.08) |
| Avg monthly cost (turnover) | +0.044% |

**Takeaways.**
- **The weighting holds up out-of-sample.** Refitting the premia every month on
  only prior data still yields a significant industry-neutral α of 0.72%/mo
  (t = 3.64) across the full 228-month walk-forward — no fixed-split cherry-picking.
- **Near industry-neutral outright.** The book carries only a small negative
  industry β (−0.09), so the raw mean (+0.62%/mo, t = 3.17) is itself significant;
  β-hedging (with a walk-forward, expanding-window β) lifts the Sharpe modestly
  from 0.73 to 0.80.
- **Stable weights.** `paths.png` shows both weights are smooth and never flip
  sign; the ranking is driven throughout by `gross_profitability`, with
  `revenue_stability` a steady secondary tilt.
- **Adds alpha over the weaker leg, not the stronger.** Regressed on each
  constituent's standalone book, the weighted composite earns a significant
  +0.46%/mo (t = 3.02) above `revenue_stability` alone, but only an insignificant
  +0.11%/mo (t = 1.08) above `gross_profitability` alone — so the blend mostly
  tracks its dominant leg and adds little beyond simply holding it.

Outputs land under `output/weighted/<slug>/`: `beta_path.png` (the expanding-window
weights and t-stats over time — both series in one stacked figure) and
`performance.png` (the walk-forward summary, including the α earned above each
constituent's standalone book). No quintile files are written — the sort is an
internal step.

## Other combinations — `bivariate_tertile.py`, `factor_momentum.py`, `portfolio_overlay.py`

All three reuse `composite.py`'s spine (`resolve_factors` / `load_exposures`,
`industry_return`, `book_stats`, `attach_net_cost_sharpe`, `leg_ownership` /
`attach_ownership`, `render_performance`), so "alpha", the cost model, the
net-of-cost Sharpe and the largest single-name ownership (for a $100M book) are
defined identically to every other book in the project.

**`bivariate_tertile.py` — independent double sort.** Every month the
cross-section is split into three equal-count tertiles on each of two factors
(default pair: `return_stability` × `gross_profitability`); the book is long the
T3∩T3 corner and short the T1∩T1 corner, equal-weighted within each leg. A
coarser **half-intersection** rule (median split, ~1/4 of names per corner
instead of ~1/9) is reported alongside to show whether the edge survives a
milder, higher-capacity cut. Outputs land under
`output/bivariate_tertile/<slug>/` (`grid_mean_return.png` 3×3 heatmap and
`performance.png` — including alpha over each constituent's standalone book and
single-name ownership for a $100M book — plus a `half/` mirror).

**`factor_momentum.py` — rotation across the top factors.** Two pipelines over
the top five of Experiment 2's quarterly-repositioned `ranked_factors` hand-off,
each candidate book being that factor's quarterly Q5−Q1 spread:
*univariate* — every 3 months select the single factor whose own long/short book
earned the most over the trailing 12 months (all realised, look-ahead-free) and
hold it until the next selection;
*bivariate* — every 3 months take the trailing-12m **top two** factors and trade
their `bivariate_tertile` double sort, re-selecting the pair every 3 months. Outputs
land under `output/factor_momentum/{univariate,bivariate}/` (cumulative growth,
selection timeline / grid diagnostic, `performance.png`).

**`portfolio_overlay.py` — naive diversification baseline.** Hold all five
top-factor standalone books at once, 1/5 of capital each. Unlike `composite.py`
(which merges *signals* and re-sorts) the overlay merges the finished
*portfolios*; the five signed weight vectors are **netted per name** before the
cost model charges turnover, so a stock long in one book and short in another
only pays cost on the residual trade. Output:
`output/portfolio_overlay/performance.png`.

## Factor redundancy — `factor_correlation.py`

A reusable companion module that asks a different question: **how much of an
industry factor is already explained by the general market factors of
Experiment 1?** For one *target* factor it regresses the target exposure on each
market factor and reports the **R² of that linear regression** — with a single
regressor plus intercept, `R² = corr²`, so the table reads directly as the
fraction of the target's cross-sectional variation that the market factor
linearly explains. High R² ⇒ the industry factor is largely redundant with that
general factor; low R² ⇒ it carries distinct information. A final **JOINT** row
gives the multivariate R² of the target on *all* market factors together — the
headline redundancy number.

It is factor- and panel-agnostic (consumes the standard `factor_panel.csv`
schema), so the same code evaluates any other industry factor:

```bash
python factor_correlation.py                      # buyback_quality vs the 9 market factors
python -c "from factor_correlation import run; run('rd_productivity')"
```

Outputs land under `output/factor_correlation/<target>/` as a shaded `r2_table.png`
(uses the project's z-score exposures by default).

### Result — `buyback_quality` (z-score, ~128k stock-months)

| Market factor | Family | R² | corr |
|---|---|---:|---:|
| `net_issuance` | Net issuance | **0.635** | −0.797 |
| `earnings_yield` | Value | 0.146 | +0.382 |
| `asset_growth` | Investment | 0.069 | −0.263 |
| `gross_profitability` | Profitability/quality | 0.029 | +0.172 |
| `beta` | Low-risk | 0.025 | −0.157 |
| `accruals` | Accruals (Sloan) | 0.015 | +0.124 |
| `sue` | Earnings momentum/PEAD | 0.002 | +0.047 |
| `momentum_12m` | Momentum | 0.002 | +0.047 |
| `reversal_1m` | Short-term reversal | 0.002 | +0.040 |
| **JOINT (all)** | — | **0.782** | — |

`buyback_quality` is dominated by its near-mechanical inverse link to
`net_issuance` (buybacks *are* negative net issuance: R²=0.63), with a secondary
value tilt (`earnings_yield`); ~78% of its variation is jointly spanned by the
general set. (By contrast `rd_productivity` is almost orthogonal — joint R²≈0.03.)

## Changed from the regression model

Experiment 3 originally fit a t-stat-gated pooled multivariate **regression**
(`multifactor.py`): select factors with |alpha t-stat| > 5, then regress the
industry-relative next return on their exposures. That has been **replaced** by
the composite-quintile pipeline above per the current spec — the **t-stat gate is
removed** (factors are chosen explicitly, not by a significance threshold) and the
combination is a **standardised sum sorted into quintiles**, not a regression. The
reusable redundancy tool (`factor_correlation.py`) is unchanged.
