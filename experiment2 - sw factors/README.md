# Experiment 2 — Software-industry factors

Tests the **established software-tailored factors** of the project plan (§2.2) on
the GICS *Software & Services* universe, under the **same pipeline as
Experiment 1** — even-quintile sorts and monthly cross-sectional (Fama–MacBeth)
regressions, with the long/short book's Sharpe, industry-neutral alpha, and
average turnover cost.

> The standalone realized-dilution test was dropped per the updated project
> proposal. The share-count change it measured is still used as one input to
> `buyback_quality`. Four established factors remain, later joined by the
> generic `fscore` / `zscore` composites read straight off `fundamental_master`.
>
> The *search for genuinely new* software-industry factors (§3.2) — extrapolating
> the R&D activity software firms rely on — is pursued in the **R&D-behaviour
> extension** (`RD/`), a second drop-in factor library. The two ad-hoc "novel"
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
main.py                # driver: wires sw_factors -> 'factors', runs quintile + regression,
                       #   then the software subexperiments, then the top-factor collection
driver_utils.py        # shared driver boilerplate (engine wiring, pipeline, redundancy step)
tertile.py             # re-evaluates EVERY factor with tertile (T3-T1) books instead of quintiles
quarter_position.py    # re-evaluates EVERY factor with QUINTILE and TERTILE books repositioned QUARTERLY
                       #   (form new buckets only end of Feb/May/Aug/Nov, hold 3 months) -> lower turnover cost
Standard/              # established §2.2 factors (outputs of sw_factors.py)
    factor_panel.csv
    quintile/   <factor>/{quintile_returns.csv, quintile_cumulative.png, long_short.png}
                + summary.csv + long_short_market_alpha.{csv,png}   # alpha table incl. avg cost
    regression/ <factor>/{regression.csv, beta.png} + summary.csv + summary_table.png
                + normalized_regression.{csv,png}
    factor_correlation/ <factor>/...   # redundancy of each factor vs the Exp1 general factors
RD/                    # R&D-behaviour factor library (rd_factors.py + main_rd.py)
Stability/             # fundamental-consistency factor library (stability_factors.py + main_stability.py)
Skew/                  # return/growth skewness factor library (skew_factors.py + main_skew.py)
Cross_val/             # re-test of the top factors on Banks+Insurance+Commodity Producers
factor_ranking/        # cross-subexperiment ranking + hand-off for Experiments 3-5
```

Each subexperiment folder mirrors `Standard/`'s layout (`factor_panel.csv`,
`quintile/`, `regression/`, `factor_correlation/`) and is driven by its own
`main_*.py`, which injects its factor library into the shared Experiment 1
engine via `driver_utils.wire_engine` — exactly as `main.py` does for
`sw_factors.py`. The build → quintile → regression → redundancy flow itself is
`driver_utils.run_pipeline`; each driver keeps only its library, universe and
labels.

## Subexperiments

| Folder | Library | Factors |
|---|---|---|
| `Standard/` | `sw_factors.py` | `intangible_value`, `intangible_profitability`, `rd_productivity`, `buyback_quality`, `fscore`, `zscore` |
| `RD/` | `rd_factors.py` | `rd_growth`, `rd_conversion`, `rd_stability`, `innovation_mix`, `rd_intensity` — `rd_stability` is the keeper |
| `Stability/` | `stability_factors.py` | second moments of quality: `revenue_stability`, `cashflow_stability`, `return_stability`, `gross_profitability_stability` |
| `Skew/` | `skew_factors.py` | `return_skewness`, `revenue_growth_skewness`, `eps_skewness` (all long-low: lottery/lumpiness aversion) |
| `Cross_val/` | `crossval_factors.py` | re-tests the top `TOP_N` ranked factors on the **Banks + Insurance + Commodity Producers** universe (excluded from the ranking — different cross-section) |

After all subexperiments finish, `main.py` **collects** every
`quintile/long_short_market_alpha.csv` (Experiment 1's general factors on the
software universe + every software subexperiment), ranks all factors by the sum
of their full-period and 2016+ alpha t-stats, and writes the single ranking
consumed by Experiments 3–5 to `factor_ranking/`:
`monthly_quintile_ranked.csv` (every factor, ranked — each downstream experiment
slices its own top N) and the rendered alpha tables.

## Run

```bash
python main.py            # Standard pipeline + software subexperiments + top-factor collection
python main.py collect    # only (re)collect the top factors from existing CSVs
python Cross_val/main_crossval.py   # run manually AFTER the collection (reads monthly_quintile_ranked.csv)
python sw_factors.py      # rebuild the Standard factor panel only
python tertile.py         # tertile re-evaluation -> factor_ranking/monthly_tertile.png
python quarter_position.py  # quarterly-repositioned re-evaluation (end Feb/May/Aug/Nov), quintile + tertile
                            #   -> factor_ranking/quarter_quintile.png (quintile)
                            #   -> factor_ranking/quarter_tertile.png  (tertile)
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
focused direction — extrapolating the **R&D activity** software firms rely on
(its growth, conversion into profit, consistency, and composition). That work
lives in the sibling **`RD/`** library.

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

## Headline results (1998–2026, 336 months)

Alpha and t(alpha) are from regressing the (canonically-signed) long/short book
on the industry return; avg cost is the mean monthly turnover cost of that book.

| Factor | Gross Q5−Q1/mo | Alpha/mo | t(α) | Avg cost (pp/mo) | FM t (full) | FM t (2016+) |
|---|---:|---:|---:|---:|---:|---:|
| buyback_quality | +0.53% | +1.13% | **+5.16** | 0.075 | +1.38 | +0.60 |
| intangible_profitability | +0.37% | +1.13% | **+4.07** | 0.039 | +0.71 | +0.66 |
| intangible_value | +0.66% | +0.71% | **+2.40** | 0.075 | +1.03 | +0.81 |
| rd_productivity | −0.33% | −0.28% | −1.21 | 0.062 | −0.49 | +1.81 |

**Takeaways.**
- **Signal is real and survives cost.** `buyback_quality`,
  `intangible_profitability` and `intangible_value` carry significant
  **industry-neutral alpha** (t = 2.4–5.2). Because these are fundamental,
  slow-moving signals, the quintile books turn over little (~0.04–0.08 pp/month
  round-trip cost), so the cost barely dents the gross premium — unlike fast
  signals such as short-term reversal in Experiment 1 (~0.55 pp/month). The
  earlier "costs erase everything" read was an artefact of the conservative
  full-turnover assumption; charging cost on actual turnover shows these factors
  are economically tradable.
- **`buyback_quality` is the standout**, with the highest alpha t-stat (5.2) and
  low cost — the one factor that clears Experiment 3's |alpha t|>5 bar under the
  point-in-time fundamentals.
- **`rd_productivity` does not work as a level signal** (negative full-sample
  alpha). The R&D angle is better captured by *behaviour* signals — see the
  `RD/` extension, where `rd_stability` is the keeper.
- **Implication for Experiment 3.** Carry `buyback_quality` (and the intangibles)
  into the multivariate, cost-aware composite for their significant alpha and low
  mutual correlation; their low natural turnover means the plan's cost-aware
  position selection costs little to run.
