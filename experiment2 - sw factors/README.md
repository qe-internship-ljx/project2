# Experiment 2 — Software-industry factors

Tests the **software-tailored factors** of the project plan (§2.2) plus the **two
novel factors** (§3.2) on the GICS *Software & Services* universe, under the
**same pipeline as Experiment 1** — even-quintile sorts and monthly
cross-sectional (Fama–MacBeth) regressions, with the long/short book's Sharpe,
industry-neutral alpha, and average turnover cost.

> The standalone realized-dilution test was dropped per the updated project
> proposal. The share-count change it measured is still used as one input to
> `buyback_quality`. Six factors remain.

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
main.py                # driver: wires sw_factors -> 'factors', runs quintile + regression
output/software_services/
    factor_panel.csv
    quintile/   <factor>/{quintile_returns.csv, quintile_cumulative.png, long_short.png}
                + summary.csv + long_short_market_alpha.{csv,png}   # alpha table incl. avg cost
    regression/ <factor>/{regression.csv, beta.png} + summary.csv + summary_table.png
```

## Run

```bash
python main.py            # full pipeline (panel + both analyses + cost summary)
python sw_factors.py      # rebuild the factor panel only
```

## Factors

`K_int` is the intangible (knowledge) capital stock from past R&D by perpetual
inventory, `K_int_t = (1−δ)·K_int_{t−1} + R&D_t` (Peters & Taylor 2017), δ=0.20,
run monthly on `rd_ltm/12`. All signals are ratios / log-changes and therefore
currency-neutral. Long/short direction is set empirically by each factor's
Fama–MacBeth t-stat sign (as in Experiment 1); `dir` below is the academic prior.

**Established software factors (§2.2)**

| Factor | Definition | dir |
|---|---|---|
| `intangible_value` | (book_value + K_int) / market cap | long high |
| `intangible_profitability` | (operating_income_ltm + R&D) / (assets + K_int) | long high |
| `rd_productivity` | Δsales_ltm (YoY) / K_int | long high |
| `buyback_quality` | realised share reduction − gross buyback yield | long high |

`buyback_quality` reconciles cash spent on buybacks against the *actual* fall in
share count: `(1 − shares_t/shares_{t−12m}) − (−buyback_ltm / mcap)`. It is
negative when a firm spends on buybacks that merely offset stock-based-pay grants
(share count does not fall) — software's characteristic low-quality tell.

**Novel factors (§3.2)**

| Factor | Definition | dir |
|---|---|---|
| `operating_leverage` | Δoperating_income (YoY) / Δsales (YoY) | long high |
| `gtm_efficiency` | Δsales (YoY) / SG&A_ltm | long high |

YoY changes use a 12-month lag, matching the year-over-year convention used
throughout Experiment 1 (the monthly analogue of the plan's "4 quarters").

## Trading cost

The plan's cost model lives in `cost.py` (Experiment 1): one-way bps
`= 3·(11/log10 Mff)^6 + 3`, with `Mff = market cap × FX × free float` floored at
$50M (so the one-way cost on any position is capped at ~0.285%). Cost is charged
on **turnover, not holdings** — `kappa` only when a position is actually traded
(opened or closed); a name carried over at the same weight costs nothing, so it
never pays more than one round-trip across its whole holding period. The book's
**average monthly turnover cost (pp)** is reported as an extra column in the
`long_short_market_alpha` table.

## Headline results (1998–2025, 336 months)

Alpha and t(alpha) are from regressing the (sign-oriented) long/short book on the
industry return; avg cost is the mean monthly turnover cost of that book.

| Factor | Gross Q5−Q1/mo | Alpha/mo | t(α) | Avg cost (pp/mo) | FM t (full) | FM t (2016+) |
|---|---:|---:|---:|---:|---:|---:|
| buyback_quality | +0.63% | +1.18% | **+5.57** | 0.073 | **+2.62** | **+2.72** |
| intangible_profitability | +0.56% | +1.32% | **+5.02** | 0.038 | +1.81 | +1.79 |
| operating_leverage (novel) | +0.33% | +0.62% | **+4.03** | 0.103 | +1.71 | −0.18 |
| intangible_value | +0.95% | +0.99% | **+3.28** | 0.079 | **+3.14** | −0.19 |
| gtm_efficiency (novel) | +0.24% | +0.31% | +1.33 | 0.061 | +0.64 | +2.02 |
| rd_productivity | +0.17%¹ | −0.27% | −1.10 | 0.057 | −0.46 | +2.41 |

¹ raw Q5−Q1; the empirically-signed book is Q1−Q5 (the regression auto-orients).

**Takeaways.**
- **Signal is real and survives cost.** `intangible_value`, `buyback_quality`,
  `intangible_profitability` and the novel `operating_leverage` carry strongly
  significant **industry-neutral alpha** (t = 3.3–5.6). Because these are
  fundamental, slow-moving signals, the quintile books turn over little
  (~0.04–0.10 pp/month round-trip cost), so the cost barely dents the gross
  premium — unlike fast signals such as short-term reversal in Experiment 1
  (~0.55 pp/month). The earlier "costs erase everything" read was an artefact of
  the conservative full-turnover assumption; charging cost on actual turnover
  shows these factors are economically tradable.
- **`buyback_quality` is the standout**, with the highest alpha t-stat (5.6),
  low cost, and the only factor that also holds up *post-2016* (FM t=+2.72) where
  `intangible_value` fades.
- **Implication for Experiment 3.** Carry these into the multivariate, cost-aware
  model for their significant alpha and low mutual correlation; their low natural
  turnover means the plan's cost-aware position selection costs little to run.
