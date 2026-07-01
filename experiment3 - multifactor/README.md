# Experiment 3 — Composite multifactor model

Combine a **set of factors** into a single cross-sectional score and trade it as
a quintile long/short — the same workflow as Experiment 1, but on a *composite*
signal instead of one raw factor. For each stock each month,

```
composite_{i,t} = Σ_f  sign_f · zscore_{f,i,t}
```

is the **sum of the stock's cross-sectional z-scores** across the chosen factors.
The industry is sorted into five equal-count buckets on this composite, and the
Q5−Q1 dollar-neutral book's performance is measured (mean, t-stat, Sharpe, and
the industry-neutral alpha used throughout the project).

There is **no significance gate and no regression model**: the caller passes the
factor set explicitly, and the factors are combined by standardised aggregation —
the textbook "composite signal" construction. (This replaces the earlier
t-stat-gated pooled *regression*; see [Changed from the regression model](#changed-from-the-regression-model).)

Two ways to weight the constituents, sharing one spine:

| Module | Combination | Weights | Evaluation |
|---|---|---|---|
| [`composite.py`](composite.py) | equal-weighted (straight sum) | each factor's bullish sign `±1` | full sample + past decade |
| [`weighted_composite.py`](weighted_composite.py) | coefficient-weighted | **expanding-window** regression premia, refit every month | **walk-forward out-of-sample (2007+)** |

The weighted variant is described under
[Coefficient-weighted variant](#coefficient-weighted-variant-expanding-window-walk-forward--weighted_compositepy);
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

3. **Sort** (Experiment 1's `quintile.py`, *unmodified*). The composite is shaped
   as a one-factor panel and fed to the engine's even-quintile sort: within-month
   winsorisation, equal-count buckets, and the months × {Q1..Q5, Q5−Q1} table of
   mean next-period returns are exactly Experiment 1's code path.

4. **Measure** (Experiment 1's `regression.py` helpers). The Q5−Q1 book is scored
   over the full sample and the past decade (2016+): mean, t-stat, annualised
   Sharpe, the **industry-neutral alpha and its t-stat** (regressing the book on
   the market-cap-weighted Software & Services return), the industry beta, and the
   beta-neutralised Sharpe — the same alpha definition as every other long/short
   book in the project.

## Design — maximal reuse, zero duplication

Everything generic is reused from Experiment 1, loaded by path (its folder name
contains spaces, so it cannot be imported normally) and wired with the project's
dependency-injection convention — register the engine as `sys.modules["factors"]`
before importing the analysis modules, exactly as Experiment 2's `main.py` does:

| Concern | Source | Reuse |
|---|---|---|
| Winsorisation, equal-count quintiles, quintile-return table | `experiment1 - general factors/quintile.py` + `factors.py` | imported and run **unmodified** |
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
`plot_cumulative` / `plot_long_short` / `render_performance`); the weighted
variant reuses all of it and adds only the equal-weighted benchmark and the
expanding-window walk-forward.

## Run

```bash
python composite.py                                          # equal-weighted, default set (below)
python composite.py buyback_quality gross_profitability rd_stability
python -c "from composite import run; run(['earnings_yield','sue','beta'])"

python weighted_composite.py                                 # coefficient-weighted, expanding-window walk-forward
python weighted_composite.py buyback_quality gross_profitability rd_stability
```

Requires Experiments 1 and 2 (and, for R&D factors, the R&D extension) to have
been run first — it reads their `factor_panel.csv` and `long_short_market_alpha.csv`.

## Outputs (`output/composite/<slug>/`)

`<slug>` is the factor names joined by `__`.

```
factor_set.png              the constituents: family, source, sign, standalone alpha
exposure_correlation.csv    pairwise correlation of the oriented constituent z-scores
quintile_returns.csv        months × {Q1..Q5, Q5−Q1}, mean next-period return
quintile_cumulative.png     the five buckets as cumulative growth of $1 (log scale)
long_short.png              the Q5−Q1 book's cumulative growth of $1
performance.png             mean / t / Sharpe / industry-neutral alpha (full + 2016+)
```

## Headline result — `buyback_quality + gross_profitability + rd_stability`

The first composite: a profitability/quality + capital-discipline + R&D-commitment
blend. The three constituents are weakly correlated (pairwise z-score correlations
0.13–0.28), so they carry largely **additive** information.

α below is the industry-neutral alpha against the **market-cap-weighted** Software &
Services return; the Q5−Q1 spread, t-stat and Sharpe do not reference the benchmark
and are unchanged.

| Metric | Full sample (1999–2025, 313 mo) | Past decade (2016+, 120 mo) |
|---|---:|---:|
| Q5−Q1 mean monthly | +1.105% | +1.042% |
| t-stat | +4.08 | +2.73 |
| Sharpe (annualised) | +0.80 | +0.86 |
| **Industry-neutral α (monthly)** | **+1.260%** | **+1.213%** |
| **α t-stat** | **+4.78** | **+3.08** |
| Industry β | −0.21 | −0.11 |
| β-neutral Sharpe | +0.94 | +1.02 |

**Takeaways.**
- **The composite beats every constituent.** Its industry-neutral alpha t-stat
  (4.78) exceeds each standalone factor's — `buyback_quality` 3.66,
  `gross_profitability` 4.33, `rd_stability` 2.75 — the diversification benefit of
  combining weakly-correlated signals (plan §1). The alpha (1.26%/mo) is larger
  than any single factor's too.
- **Clean monotonic sort.** Cumulative growth is ordered Q5 > Q4 > Q3 > Q2 > Q1
  across the whole sample (`quintile_cumulative.png`); the ranking power is not a
  tail effect.
- **Defensive by construction.** The book carries a *negative* industry beta
  (−0.21) — its quality/stability tilt outperforms in down-industry months — so
  beta-hedging lifts the Sharpe from 0.80 to **0.94** (full) / **1.02** (2016+).
- **Robust across the past decade.** The alpha is essentially unchanged in the
  2016+ re-estimation (1.21%/mo, t = 3.08), not a pre-2010 artifact. (For a genuine
  walk-forward holdout with weights refit every month, see the weighted variant below.)

## Coefficient-weighted variant (expanding-window walk-forward) — `weighted_composite.py`

Instead of an equal-weighted straight sum, weight each constituent by its
**estimated return premium**, re-estimated **every month on an expanding window**
and traded strictly walk-forward. The default set here is **`revenue_stability` +
`gross_profitability`**, with an initial training period through **2006** and the
book traded from **2007 onwards** (configurable via `INITIAL_TRAIN_END`).

For each formation month `t` after the initial training period:

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
   into quintiles, long Q5 / short Q1, hold over `t+1`.
3. **Walk-forward.** The window expands one month and step 1 repeats, so **every
   traded month is out-of-sample**: the weights forming month `t`'s book never saw
   `t`'s (or any later) return. There is no single fixed train/test split — the
   whole 2007+ path is the holdout.

### Full-sample premia (reference)

`coefficients.{csv,png}` report the whole-period regression (t-stats over the
**entire** sample) for reference; the traded book uses the expanding-window
weights, not these.

| Term | Coef (= premium, ind-rel %/mo per 1σ) | t (OLS) | t (cluster) |
|---|---:|---:|---:|
| intercept | +0.043% | +1.09 | +1.02 |
| `revenue_stability` | +0.074% | +1.83 | +0.90 |
| `gross_profitability` | +0.264% | +6.41 | **+4.66** |

`gross_profitability` carries the dominant, strongly significant premium;
`revenue_stability` adds a smaller positive tilt. How each weight and its t-stat
evolve as the window grows is plotted in `beta_path.png` / `tstat_path.png` — the
`gross_profitability` weight drifts down from ~0.57%→~0.26% as more (lower-premium)
history accrues, but its clustered t-stat stays firmly above 4 throughout.

### Walk-forward Q5−Q1 book (2007+)

| Metric | **Walk-forward OOS (2007+, 228 mo)** |
|---|---:|
| Mean monthly | +0.513% |
| t-stat | +2.61 |
| Sharpe (ann.) | +0.60 |
| Industry-neutral α (monthly) | +0.644% |
| α t-stat | **+3.25** |
| Industry β | −0.11 |
| β-neutral Sharpe | +0.77 |
| Avg monthly cost (turnover) | +0.044% |

**Takeaways.**
- **The weighting holds up out-of-sample.** Refitting the premia every month on
  only prior data still yields a significant industry-neutral α of 0.64%/mo
  (t = 3.25) across the full 228-month walk-forward — no fixed-split cherry-picking.
- **Near industry-neutral outright.** The book carries only a small negative
  industry β (−0.11), so the raw mean (+0.51%/mo, t = 2.61) is itself significant;
  β-hedging lifts the Sharpe modestly from 0.60 to 0.77.
- **Stable weights.** `beta_path.png` shows both weights are smooth and never flip
  sign; the ranking is driven throughout by `gross_profitability`, with
  `revenue_stability` a steady secondary tilt.

Outputs land under `output/weighted/<slug>/`: `coefficients.{csv,png}` (full-sample
reference premia), `beta_path.{csv,png}` and `tstat_path.{csv,png}` (the
expanding-window weights and t-stats over time), `long_short.png` (the walk-forward
book's growth of $1), and `performance.png` (the walk-forward summary). No quintile
files are written — the sort is an internal step.

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

Outputs land under `output/factor_correlation/<target>/` as `r2_table.csv` and a
shaded `r2_table.png` (uses the project's z-score exposures by default).

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
