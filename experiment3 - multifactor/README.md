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
| [`weighted_composite.py`](weighted_composite.py) | coefficient-weighted | in-sample (≤2015) regression premia | **out-of-sample (2016+)** |

The weighted variant is described under
[Coefficient-weighted variant](#coefficient-weighted-variant-out-of-sample--weighted_compositepy);
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
| In-sample premium regression + train/test split | `weighted_composite.py` | **new (this experiment)** |

This experiment therefore adds only the composite construction and its reporting;
every input it consumes was already produced upstream. The pipeline is factor-set
agnostic — any factor produced by Experiments 1–2 (or the R&D extension) can be
combined by name. `composite.py` exposes the shared spine (`load_exposures`,
`scored_frame`, `as_factor_panel`, `industry_return`, `book_stats`,
`plot_cumulative` / `plot_long_short` / `render_performance`); the weighted
variant reuses all of it and adds only the regression and the split.

## Run

```bash
python composite.py                                          # equal-weighted, default set (below)
python composite.py buyback_quality gross_profitability rd_stability
python -c "from composite import run; run(['earnings_yield','sue','beta'])"

python weighted_composite.py                                 # coefficient-weighted, OOS test
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
  2016+ re-estimation (1.21%/mo, t = 3.08), not a pre-2010 artifact. (For a strict
  train-on-≤2019 / test-on-2020+ holdout, see the weighted variant below.)

## Coefficient-weighted variant (out-of-sample) — `weighted_composite.py`

Instead of an equal-weighted straight sum, weight each constituent by its
**estimated return premium**: fit the factors in-sample, use the regression
coefficients as the composite weights, and test the resulting strategy strictly
out-of-sample. The default set here is **`buyback_quality` + `rd_stability`**, with
the split at **in-sample ≤2015 / out-of-sample 2016+** (both configurable via
`IS_END` / `OOS_START`).

1. **In-sample premia (≤2015).** Pool every in-sample stock-month and regress the
   **normalised return** — the stock's month-(t+1) return minus that month's
   market-cap-weighted industry average (the within-industry "market") — on the
   formation-date factor z-scores: `(r_{i,t+1} − market_{t+1}) = a + Σ b_f·z_{f,i,t} + ε`.
   The slopes `b_f` are each factor's in-sample premium, reported with OLS and
   month-clustered t-stats.
2. **Weighted score.** For every stock-month, `score_{i,t} = Σ b_f·z_{f,i,t}` —
   the model's predicted industry-relative return (intercept dropped; it doesn't
   affect the cross-sectional ranking). Sort into quintiles, long Q5 / short Q1.
3. **Out-of-sample test (≥2016).** The fixed in-sample weights are applied to the
   2016+ cross-sections; that window's Q5−Q1 performance is the headline. The
   weights never see post-2015 data, so it is a genuine holdout.

### In-sample premia = the weights (≤2015)

| Term | Coef (= weight, ind-rel %/mo per 1σ) | t (OLS) | t (cluster) |
|---|---:|---:|---:|
| intercept | +0.787% | +11.05 | **+3.16** |
| `buyback_quality` | +0.229% | +2.82 | **+1.98** |
| `rd_stability` | +0.121% | +1.64 | +1.11 |

With `gross_profitability` dropped, `buyback_quality` carries the larger in-sample
premium (≈2× `rd_stability`); both enter with the expected positive sign. (The
intercept is larger than before because the normalised return is now relative to the
**market-cap-weighted** — and therefore lower — industry mean; it is dropped from the
score and does not affect the cross-sectional ranking.)

### In-sample vs out-of-sample (Q5−Q1 book)

| Metric | In-sample (≤2015, 193 mo) | **Out-of-sample (2016+, 120 mo)** |
|---|---:|---:|
| Mean monthly | +0.530% | +0.479% |
| t-stat | +1.23 | +1.11 |
| Sharpe (ann.) | +0.31 | +0.35 |
| Industry-neutral α (monthly) | +0.607% | +0.909% |
| α t-stat | +1.56 | **+2.13** |
| Industry β | −0.36 | −0.28 |
| β-neutral Sharpe | +0.39 | +0.70 |

**Takeaways.**
- **The weighting generalises here — OOS even beats in-sample.** The industry-
  neutral α *rises* from 0.61%/mo (t = 1.56) in-sample to 0.91%/mo (t = 2.13) on
  the 2016+ holdout. Unlike the earlier `gross_profitability`-dominated 3-factor
  fit, the two-factor weights don't overfit, so they hold up out-of-sample.
- **A defensive, hedge-then-judge book.** Both windows carry a *negative*
  industry beta (≈ −0.3): the buyback-quality + R&D-stability tilt does best when
  the industry falls. The raw mean return is therefore modest (+0.48%/mo, t = 1.11
  OOS — the negative beta drags the unhedged return down in a rising industry), but
  the **industry-neutral** α is significant out-of-sample, and β-hedging lifts the OOS
  Sharpe from 0.35 to **0.70**. This is a book to run market-neutral, not outright.
- **Robust split.** Cutting in-sample at 2015 leaves a full 120-month (2016–2025)
  holdout; the dashed line in `quintile_cumulative.png` / `long_short.png` marks
  the IS/OOS boundary.

Outputs land under `output/weighted/<slug>/`: `coefficients.{csv,png}` (the
premia/weights), `quintile_returns.csv`, `quintile_cumulative.png` and
`long_short.png` (both with the 2016 split marked), and `performance.png`
(in-sample vs out-of-sample).

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
