# Experiment 2 — Software-industry factors

Tests the **established software-tailored factors** of the project plan (§2.2) on
the GICS *Software & Services* universe — and then extends the search into three
factor-discovery libraries (R&D behaviour, second moments/stability, third
moments/skewness) — under the **same pipeline as Experiment 1**: even-quintile
sorts and cross-sectional (Fama–MacBeth) regressions, with the long/short book's
Sharpe, industry-neutral alpha, and average turnover cost, evaluated both monthly
and on a quarterly-repositioned calendar. Every subfolder's factors, formulas and
results are catalogued in [Factors](#factors) and [Headline results](#headline-results).

> **All the key comprehensive results of Experiment 2 are stored in the
> `Factor Ranking/` directory** — the all-factor comparison tables (monthly and
> quarterly × quintile/tertile/half), the quarterly cross-section regression, and
> the persisted ranked-factor CSVs that Experiments 3–5 read as their factor
> hand-off. Start there for the headline cross-factor comparison.

> The standalone realized-dilution test was dropped per the updated project
> proposal. The share-count change it measured is still used as one input to
> `buyback_quality`. Four established factors remain, later joined by the
> generic `fscore` / `zscore` composites read straight off `fundamental_master`.
>
> The *search for genuinely new* software-industry factors (§3.2) — extrapolating
> the R&D activity software firms rely on — is pursued in the **R&D-behaviour
> extension** (`rd/`), a second drop-in factor library. The two ad-hoc "novel"
> factors of the original proposal (operating-leverage realization, go-to-market
> efficiency) have been retired in favour of that directed search.

## Design — maximal reuse, zero duplication of the engine

Experiment 1's analysis modules are reused **unmodified**:

| Concern | Source | Reuse |
|---|---|---|
| Quintile sorts + cumulative plots | `experiment1 - general factors/quintile.py` | imported & run as-is |
| Trading-cost model (plan §1) & portfolio turnover cost | `experiment1 - general factors/cost.py` | imported & run as-is |
| Cross-sectional regressions, L/S books, alpha table | `experiment1 - general factors/regression.py` | imported & run as-is |
| Universe handling, winsorise, z-score, quintile assignment, OLS, `prepare_slice` | `experiment1 - general factors/factors.py` | imported as a shared engine |
| **Factor definitions + fundamentals** | `sw_factors.py` | **new (this experiment)** |

`quintile.py` / `regression.py` / `cost.py` bind to their factor library via
`import factors as F`. The driver registers `sw_factors` under that name in
`sys.modules` before importing them (a clean dependency injection), so the entire
analysis — including the cost computation — runs against the software factors
with **no change to Experiment 1**.

```
sw_factors.py          # software factor library (drop-in for the engine's interface)
monthly_position.py    # driver + monthly re-evaluation (was main.py + tertile.py):
                       #   wires sw_factors -> 'factors', runs quintile + regression, then the
                       #   software subexperiments, then the top-factor collection -- re-evaluating
                       #   EVERY factor with monthly QUINTILE (Q5-Q1), TERTILE (T3-T1) and HALF (H2-H1)
                       #   books (monthly_quintile.png / monthly_tertile.png / monthly_half.png).
                       #   Owns the shared machinery (SOURCES, evaluate_all, alpha_row, render_ranked)
driver_utils.py        # shared driver boilerplate (engine wiring, pipeline, correlation step)
correlation_matrix.py  # chosen x general correlation matrices (z-score + Q5-Q1 return) -> 2 PNGs
quarter_position.py    # re-evaluates the SAME factors with QUINTILE, TERTILE and HALF books repositioned
                       #   QUARTERLY (form new buckets only end of Feb/May/Aug/Nov, hold 3 months) -> lower
                       #   turnover cost. Reuses monthly_position's SOURCES / evaluate_all / alpha_row /
                       #   render_ranked, supplying only the quarterly book. + a QUARTERLY cross-section
                       #   regression: next-quarter return on factor z-score -> quarter_regression.png
literature/            # established §2.2 factors (outputs of sw_factors.py)
    factor_panel.csv
    quintile/   <factor>/{quintile_returns.csv, quintile_cumulative.png, long_short.png}
                + summary.csv + long_short_market_alpha.{csv,png}   # alpha table incl. avg cost
    regression/ <factor>/{regression.csv, beta.png} + summary.csv + summary_table.png
                + normalized_regression.{csv,png}
    factor_correlation/ {zscore,return}_correlation.png   # chosen x general correlation:
                       #   z-score exposures and Q5-Q1 long-short return series (see correlation_matrix.py)
rd/                    # R&D-behaviour factor library (rd_factors.py + main_rd.py)
stability/             # fundamental-consistency factor library (stability_factors.py + main_stability.py)
skew/                  # return/growth skewness factor library (skew_factors.py + main_skew.py)
cross_val/             # re-test of the top factors on Banks+Insurance+Commodity Producers
Factor Ranking/        # rendered all-factor alpha tables (monthly/quarter x quintile/tertile/half)
```

The **factor-selection hand-off** Experiments 3–5 read is a persisted CSV: running
`quarter_position.py` writes each bucketing's ranked table to
`Factor Ranking/quarter_{quintile,tertile,half}_ranked.csv`, and
`quarter_position.ranked_factors(n, "quintile"|"tertile"|"half")` reads it back
(top-`n` or all) — so a downstream process resolves the leaders with a plain
`read_csv`, without re-running the whole-universe re-evaluation (or wiring the engine)
in every process. If the CSV is missing (this module was never run standalone),
`ranked_factors` computes and persists it once, so the hand-off is self-bootstrapping.
The CSVs are regenerable build artifacts (gitignored, like every `factor_panel.csv`).
Alongside them `Factor Ranking/` holds the rendered comparison tables — one per
repositioning frequency × bucketing: `monthly_{quintile,tertile,half}.png` and
`quarter_{quintile,tertile,half}.png`, plus `quarter_regression.png` (the quarterly
cross-section regression, below).

`quarter_regression.png` is the quarterly counterpart of each subexperiment's
`regression/summary_table.png`: at every reposition date (end of Feb/May/Aug/Nov) it
regresses each stock's **next-quarter** return on its factor z-score, and reports the
mean quarterly β (the factor premium, return per 1σ) with its Fama–MacBeth t-stat,
full period and 2016+, ranking every factor by the sum of the two t-stats — the same
statistic `summary_table.png` reports, on the held-quarterly calendar rather than
monthly.

Each subexperiment folder mirrors `literature/`'s layout (`factor_panel.csv`,
`quintile/`, `regression/`, `factor_correlation/`) and is driven by its own
`main_*.py`, which injects its factor library into the shared Experiment 1
engine via `driver_utils.wire_engine` — exactly as `monthly_position.py` does for
`sw_factors.py`. The build → quintile → regression → redundancy flow itself is
`driver_utils.run_pipeline`; each driver keeps only its library, universe and
labels. (`cross_val/` opts out of the `factor_correlation/` step via
`run_pipeline(..., correlation=False)` — it is a re-test of factors whose
redundancy is already established on the software universe.)

## Subexperiments

| Folder | Library | Factors |
|---|---|---|
| `literature/` | `sw_factors.py` | `intangible_value`, `intangible_profitability`, `rd_productivity`, `buyback_quality`, `fscore`, `zscore` |
| `rd/` | `rd_factors.py` | `rd_growth`, `rd_conversion`, `rd_stability`, `innovation_mix`, `rd_intensity` — `rd_stability` is the keeper |
| `stability/` | `stability_factors.py` | second moments of quality: `revenue_growth_stability`, `cashflow_stability`, `return_stability`, `gross_profitability_stability` |
| `skew/` | `skew_factors.py` | `return_skewness`, `revenue_growth_skewness`, `eps_skewness` (all long-low: lottery/lumpiness aversion) |
| `cross_val/` | `crossval_factors.py` | re-tests the top `TOP_N` ranked factors on the **Banks + Insurance + Commodity Producers** universe (excluded from the ranking — different cross-section) |

After all subexperiments finish, `monthly_position.py` **collects** every
factor (Experiment 1's general factors on the software universe + every software
subexperiment), ranks all factors by the sum of their full-period and 2016+
net-of-cost beta-neutral Sharpe ratios, and renders the monthly-quintile
comparison table `Factor Ranking/monthly_quintile.png` plus the same factors'
coarser monthly books (`monthly_tertile.png`, `monthly_half.png`) in the identical
format (all three via its shared `run_all`). The **hand-off consumed by
Experiments 3–5** is the *quarterly-repositioned* ranking, persisted by
`quarter_position.py` to `Factor Ranking/quarter_{label}_ranked.csv` and read back via
`quarter_position.ranked_factors` (buckets formed only end of Feb/May/Aug/Nov, held
three months) — chosen because quarterly repositioning is cheaper and empirically
slightly stronger across the leaders; see the report. Each downstream
experiment slices its own top N from it.

## Run

```bash
python monthly_position.py          # Literature pipeline + software subexperiments + monthly quintile/tertile/half tables
python monthly_position.py collect  # only (re)render the monthly quintile/tertile/half tables from existing CSVs
python Cross_val/main_crossval.py   # run manually (resolves its top factors via quarter_position.ranked_factors)
python sw_factors.py      # rebuild the Literature factor panel only
python quarter_position.py  # quarterly-repositioned re-evaluation (end Feb/May/Aug/Nov), quintile + tertile + half
                            #   -> Factor Ranking/quarter_{quintile,tertile,half}.png
                            #   + quarter_{quintile,tertile,half}_ranked.csv (the persisted Exp 3-5 / Cross_val hand-off)
                            #   + quarter_regression.png (quarterly cross-section regression of next-quarter return)
                            #   ranked_factors() reads those CSVs back
```

## Factors

`K_int` is the intangible (knowledge) capital stock from past R&D by perpetual
inventory, `K_int_t = (1−δ)·K_int_{t−1} + R&D_t` (Peters & Taylor 2017), δ=0.20,
run monthly on `rd_ltm/12`. All signals are ratios / log-changes and therefore
currency-neutral. Long/short direction follows each factor's **canonical
(academic-prior) direction** (`USE_CANONICAL_LS_DIRECTION = True` in
`sw_factors.py`), not the in-sample Fama–MacBeth t-stat sign; `dir` below is
that prior.

**Established software factors (§2.2)**

| Factor | Definition | dir |
|---|---|---|
| `intangible_value` | (book_value + K_int) / market cap | long high |
| `intangible_profitability` | (operating_income_ltm + R&D) / (assets + K_int) | long high |
| `rd_productivity` | Δsales_ltm (YoY) / K_int | long high |
| `buyback_quality` | realised share reduction − gross buyback yield | long high |
| `fscore` | Piotroski F-score (from `fundamental_master`) | long high |
| `zscore` | Altman Z-score (from `fundamental_master`) | long high |

`buyback_quality` reconciles cash spent on buybacks against the *actual* fall in
share count: `(1 − shares_t/shares_{t−12m}) − (−buyback_ltm / mcap)`. It is
negative when a firm spends on buybacks that merely offset stock-based-pay grants
(share count does not fall) — software's characteristic low-quality tell.

YoY changes use a 12-month lag, matching the year-over-year convention used
throughout Experiment 1 (the monthly analogue of the plan's "4 quarters").

**Seeking new software-industry factors (§3.2).** Rather than the two ad-hoc
novel factors of the original proposal, the search for new signals now follows a
focused direction — extrapolating the **R&D activity** software firms rely on,
then generalising to the *higher moments* (consistency, skewness) of the quality
complex. The three factor-discovery libraries below are drop-ins for the same
engine; each adds only its definitions.

**R&D behaviour (`rd/` — dynamics & discipline of R&D spend)**

Where the established set uses the accumulated R&D *stock* `K_int`, these target
the *flow, output, consistency and composition* of R&D. `rd_growth` carries the
**opposite** prior to the asset-growth anomaly (R&D increases predict *high*
returns; Eberhart–Maxwell–Siddique 2004); `rd_stability` is a second moment and
`rd_intensity` a level baseline.

| Factor | Definition | dir |
|---|---|---|
| `rd_growth` | Δrd_ltm (YoY) / avg assets | long high |
| `rd_conversion` | Δgross_income_ltm (YoY) / sales_ltm_{t−12m} | long high |
| `rd_stability` | − trailing 36m coeff. of variation of rd_ltm/sales | long high |
| `innovation_mix` | rd_ltm / (rd_ltm + sga_ltm) | long high |
| `rd_intensity` | rd_ltm / sales_ltm (level baseline) | long high |

**Second moments of quality (`stability/` — consistency = inverse volatility)**

Steady, predictable fundamentals are a recognised quality dimension (Dichev–Tang
2009; Graham–Harvey–Rajgopal 2005). Each is a negated trailing dispersion, so
higher = steadier = the long leg. The CoV factors use the sign-robust `std/|mean|`
with a near-zero-mean guard (the OCF margin can turn negative); `revenue_growth_stability`
and `return_stability` use a plain std because their inputs are already scale-free
(a growth *rate*) or centred near zero (monthly returns), so mean-scaling would be
explosive and carry no signal.

| Factor | Definition | dir |
|---|---|---|
| `revenue_growth_stability` | − trailing 36m std of YoY revenue growth | long high |
| `cashflow_stability` | − trailing 12m coeff. of variation of OCF margin (ocf_ltm/sales) | long high |
| `return_stability` | − trailing 12m std of monthly total return | long high |
| `gross_profitability_stability` | − trailing 12m coeff. of variation of GP/assets | long high |

**Third moments / lottery demand (`skew/` — investors over-pay for skew)**

Investors over-pay for positive skewness, so high-skew names underperform (Boyer–
Mitton–Vorkink 2010; Bali–Cakici–Whitelaw 2011 "MAX"). The canonical trade is to
**short** the positively skewed names, so all three are long-low (`Q1−Q5`). Each
is a trailing-36m sample skewness (a scale-invariant shape statistic).

| Factor | Definition | dir |
|---|---|---|
| `return_skewness` | trailing 36m skewness of monthly total return | long low |
| `revenue_growth_skewness` | trailing 36m skewness of YoY revenue growth | long low |
| `eps_skewness` | trailing 36m skewness of diluted EPS | long low |

**Cross-validation (`cross_val/`).** No new definitions: it re-runs the *source
library* of each of the top-`TOP_N` ranked factors (`TOP_N = 5`, by quarterly
industry-neutral alpha t-stat) verbatim on the **Banks + Insurance + Commodity
Producers** universe, asking whether the software-industry leaders generalise to
structurally unrelated cross-sections. Coverage is itself a finding — `rd_stability`
in particular is thin where firms run little R&D.

## Trading cost

The plan's cost model lives in `cost.py` (Experiment 1): one-way bps
`= 3·(11/log10 Mff)^6 + 3`, with `Mff = market cap × FX × free float` floored at
$50M (so the one-way cost on any position is capped at ~0.285%). Cost is charged
on **turnover, not holdings** — `kappa` only when a position is actually traded
(opened or closed); a name carried over at the same weight costs nothing, so it
never pays more than one round-trip across its whole holding period. The book's
**average monthly turnover cost (pp)** is reported as an extra column in the
`long_short_market_alpha` table.

Because cost is charged only on the weight actually traded, **rebalancing less
often trades less**: `quarter_position.py` re-forms each factor's book (quintile
*and* tertile) only at the end of Feb/May/Aug/Nov and holds that membership fixed
for the intervening months, so `cost.turnover_cost` charges a round-trip only at
the reposition dates and ~0 in between — the lower turnover falls straight out of
the same cost machinery, at the price of letting the signal stale between
reposition dates.

## Headline results

The numbers below are the **quarterly-repositioned quintile** book (buckets formed
end of Feb/May/Aug/Nov, held three months) — the calendar the Experiment 3–5
hand-off is persisted on (`Factor Ranking/quarter_quintile_ranked.csv`). `α/mo` and
`t(α)` regress the canonically-signed long/short book on the industry return;
`combined Sharpe` is the sum of the full-period and 2016+ net-of-cost
beta-neutral Sharpe ratios (the ranking key). `dir` is `Q5−Q1` unless noted.
Sample runs 1998–2026 (per-factor month count varies with data coverage).

**Established software (`literature/`)**

| Factor | α/mo | t(α) | combined Sharpe |
|---|---:|---:|---:|
| `intangible_profitability` | +0.87% | **+3.15** | 1.38 |
| `fscore` | +0.86% | **+4.02** | 1.47 |
| `buyback_quality` | +0.75% | **+3.22** | 1.12 |
| `intangible_value` | +0.51% | **+2.02** | 0.73 |
| `rd_productivity` | −0.44% | −2.06 | −0.41 |
| `zscore` | −0.80% | −2.57 | −0.90 |

**R&D behaviour (`rd/`)**

| Factor | α/mo | t(α) | combined Sharpe |
|---|---:|---:|---:|
| `rd_stability` | +0.89% | **+3.94** | 1.40 |
| `innovation_mix` | +0.27% | +1.14 | −0.25 |
| `rd_conversion` | +0.17% | +0.74 | −0.16 |
| `rd_growth` | −0.35% | −1.35 | −0.93 |
| `rd_intensity` | −0.22% | −0.69 | −1.13 |

**Second moments (`stability/`)**

| Factor | α/mo | t(α) | combined Sharpe |
|---|---:|---:|---:|
| `revenue_growth_stability` | +0.68% | **+3.45** | 1.82 |
| `return_stability` | +1.01% | **+3.40** | 1.59 |
| `gross_profitability_stability` | +0.79% | **+3.20** | 1.05 |
| `cashflow_stability` | +0.57% | **+2.71** | 0.86 |

**Third moments (`skew/`, all `Q1−Q5`)**

| Factor | α/mo | t(α) | combined Sharpe |
|---|---:|---:|---:|
| `return_skewness` | +0.65% | **+3.11** | 0.52 |
| `revenue_growth_skewness` | +0.20% | +1.41 | 1.06 |
| `eps_skewness` | −0.08% | −0.46 | −0.51 |

**Takeaways.**
- **The stability library is the strongest discovery.** All four second-moment
  factors carry significant industry-neutral alpha, and `revenue_growth_stability` /
  `return_stability` top the whole cross-experiment ranking by combined Sharpe
  (1.82 / 1.59). `return_stability` measures dispersion with a **plain std** of
  monthly returns (not a coefficient of variation) — the low-volatility anomaly in
  its cleanest form.
- **The R&D directed search yields exactly one keeper.** `rd_stability` (a *second
  moment* of R&D commitment) is a top-tier factor (t = 3.9), while the flow/level
  members (`rd_growth`, `rd_intensity`) fail — consistency, not level, is what R&D
  rewards.
- **Skew is a mixed bag.** `return_skewness` is significant (short lottery-like
  return streams), `eps_skewness` is not; the shape signals are weaker and turnover-
  sensitive relative to the stability complex.
- **Among the established factors** `fscore`, `intangible_profitability` and
  `buyback_quality` remain the durable performers; `rd_productivity` fails as a
  level signal (its R&D content is better captured by `rd_stability`) and `zscore`
  is outright negative in this universe.
- **Implication for Experiments 3–5.** The hand-off carries the leaders across
  libraries — the stability factors, `rd_stability`, `fscore`,
  `intangible_profitability` and `buyback_quality` — chosen for significant alpha,
  low mutual correlation and (being slow, fundamental signals) low natural
  turnover, so the cost-aware position selection downstream costs little to run.
