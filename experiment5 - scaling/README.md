# Experiment 5 — Capacity scaling

Re-evaluates the project's factor books with **market-cap-aware position
sizing** instead of equal weighting, asking how much of each premium survives
when the book is tilted toward the names that can actually absorb capital.

The standard quintile book puts as much money into a $150M microcap as into a
$150B megacap — uninvestable at size; pure cap weighting swings to the opposite
extreme. `capacity_scaling.py` tests two monotone middle grounds, applied **within
each leg** of the **quarterly-repositioned** quintile book (buckets, formed
quarterly and held three months via `quarter_position.quarter_held_membership`,
plus orientation, benchmark and every downstream statistic are unchanged, so each
table reads directly against the standard equal-weighted quarterly book):

```
sqrt:  w_i ∝ sqrt(mcap_usd_i)      # mild tilt toward larger, more liquid names
log6:  w_i ∝ log(mcap_usd_i)^6     # steep tilt, concentrates on the largest names
```

## Pipelines

Output is grouped by pipeline under `output/capacity_scaling/`
(`univariate_scaled/`, `bivariate_scaled/`, `composite_scaled/`), plus the 0.5%
ownership-capped raw equal-weighted variant under `output/ownership_threshold/` and
the softmax-conviction half book under `output/confidence_scaling/`.

**Univariate (`run`).** Re-evaluates **exactly the same factor universe as
Experiment 2's `quarter_position.py`** (Experiment 1's general factors + every
Experiment 2 software subexperiment; `quarter_position.SOURCES`, re-exported from
the shared monthly re-evaluation, is imported by path as the single source of
truth). For each factor the even-quintile sort is
Experiment 1's `prepare_slice`, held quarterly via
`quarter_position.quarter_held_membership`; only the within-leg weights change.
Turnover cost is charged on the **actual capacity-weighted legs** (`cost.turnover_cost`),
and each factor is scored with the standard alpha-table statistics (full sample
and 2016+), ranked by combined (full + 2016+) net-of-cost beta-neutral Sharpe. Output: one alpha table per
weighting scheme —

```
output/capacity_scaling/univariate_scaled/sqrt_quintile_long_short_market_alpha.png        # sqrt
output/capacity_scaling/univariate_scaled/log6_quintile_long_short_market_alpha.png   # log6
```

**Ownership threshold (`run(_equal_weight, ..., ownership_cap=)`).** The **raw
equal-weighted** quintile book (not the capacity-weighted ones above), re-evaluated
under a **0.5% point-in-time single-name ownership ceiling**: starting from equal
within-leg weights, any name whose position (per-leg capital × its within-leg weight)
would own more than 0.5% of its own market cap is pushed down to the 0.5% weight, and
the freed weight is redistributed pro-rata across the still-uncapped names,
water-filled until nothing breaches the cap (`_cap_ownership`). Where a whole leg is
cap-bound the leg simply holds less than 100% — the honest capacity limit. Everything
else is the standard equal-weighted book's, so the table reads directly against it.
Run for both the **quintile** (top/bottom fifth) and **tertile** (top/bottom third)
sort. Output: two alpha tables —

```
output/ownership_threshold/long_short_market_alpha.png           # quintile, equal-weighted, 0.5% cap
output/ownership_threshold/tertile_long_short_market_alpha.png   # tertile,  equal-weighted, 0.5% cap
```

**Bivariate (`run_bivariate`).** Applies the sqrt weighting to Experiment 3's
double-sort corners: the T3∩T3 / T1∩T1 legs come from
`bivariate_gate.double_sorted` (imported by path), sqrt-cap-weighted within
each corner. Default pairs: `return_stability × gross_profitability` and
`revenue_stability × gross_profitability` (or pass pairs on the CLI). Each pair
is benchmarked against its constituents' standalone books and reports the
largest single-name ownership for a $100M book. Output: one performance table
per pair under `output/capacity_scaling/bivariate_scaled/`.

**Composite (`run_composite`).** Applies the sqrt weighting to Experiment 3's
composite z-score book (`composite.py`, imported by path): the top-five factors'
sign-oriented z-scores are summed into one score and the top / bottom buckets of
that composite are sqrt-cap-weighted within each leg (everything else — the
composite construction, factor selection, quarterly-held even-bucket sort,
orientation and statistics — is the composite driver's own). It is run over both
of `composite.RANKINGS`, each bucketed the way its constituents were ranked: the
`quarter_quintile` selection into **quintiles**, the `quarter_tertile` selection
into **tertiles**. Output: one performance table per ranking, in the same format
as the bivariate table —

```
output/capacity_scaling/composite_scaled/quarter_quintile_performance.png   # quintile sort
output/capacity_scaling/composite_scaled/quarter_tertile_performance.png    # tertile sort
```

**Confidence (`confidence_scaling.py`).** A conviction tilt rather than a capacity
one, applied to each factor's **quarter-half** book (long the top half, short the
bottom half — the coarsest sort). Within each leg every name is weighted by the
**softmax of its leg-oriented cross-sectional z-score** (`w_i ∝ exp(z_i)`, with `z`
flipped to `−z` in the short leg so a large positive oriented score always means
strong conviction *for that leg*), so the highest-conviction names carry the most
capital and the softmax does the work of picking out the names a finer bucket sort
would have isolated. Everything else — the same factor universe, the quarterly-held
sort, orientation, regression / cost / ranking / rendering pipeline — is
`capacity_scaling`'s, reused via its `evaluate_factor(..., legs_fn=)` override
(`confidence_scaling` adds only the leg builder). The half-book direction label is
`H2-H1` / `H1-H2`. Output: one alpha table —

```
output/confidence_scaling/softmax_half_long_short_market_alpha.png   # half, softmax(z-score)-weighted
```

## Reuse

Experiment 1's `factors` / `cost` / `regression` are imported off `sys.path`
(the generic engine, no library injection); Experiment 2's `quarter_position.py`
and Experiment 3's `bivariate_gate.py` / `composite.py` are loaded by file path
for the factor universe, the double-sort mechanics and the composite construction.
This experiment adds **only** the within-leg weighting schemes.

## Run

```bash
python capacity_scaling.py      # univariate (sqrt + log6) + default bivariate pairs + both composite rankings
python confidence_scaling.py    # softmax(z-score)-weighted quarter-half book
```

Requires Experiments 1–3 to have been run first (it reads their factor panels
and alpha CSVs).
