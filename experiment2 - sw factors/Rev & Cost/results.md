# Revenue & Cost factors — Results vs. Hypothesis

*Companion to `Theory & hypothesis.md`. Same universe (GICS Software & Services,
1998–2026, ~1,100 stocks, point-in-time), same pipeline as `RD/` (even-quintile
sorts, Fama–MacBeth regressions, canonically-signed dollar-neutral long/short books
with realistic turnover cost, and redundancy R² vs the Exp-1 general, Exp-2 standard-
software, and Exp-2 RD factors). Every L/S book is signed by the **ex-ante** `higher_is_bullish`
prior (all five: long high), so a **negative alpha t-stat falsifies the hypothesised
direction** — this is a genuine out-of-hypothesis test, not a fit.*

Outputs: `quintile/`, `regression/`, `factor_correlation/{vs_general,vs_software,vs_rd}/`,
headline table `quintile/long_short_market_alpha.png`.

> **Benchmark convention (updated).** Every industry-neutral α below is now measured
> against the **market-cap-weighted** Software & Services return (each name weighted by
> its USD market cap), not the equal-weighted mean of companies. This recalibrates the
> α's and industry βs versus a mega-cap-dominated benchmark; the raw L/S spreads,
> Fama–MacBeth slopes, turnover costs and redundancy R²/correlations are unaffected
> (they do not reference the benchmark). All qualitative verdicts are unchanged.

---

## 1. Headline results

The cleanest test is the **industry-neutral alpha** of the canonically-signed L/S book
(`ls_t = α + β·industry_t + ε`): α is the return left after hedging out industry
exposure, and its t-stat is the verdict. I report it alongside the raw quintile spread,
the Fama–MacBeth slope, the β-neutral Sharpe, turnover cost, and redundancy against
**three** existing factor sets — the Exp-1 general factors (gen), the Exp-2 "standard"
software factors (sw), **and the Exp-2 RD factors (rd)**. The `vs_rd` comparison was
added after the fact and proved decisive (§3.5): the RD-behaviour factors are the
closest cousins of these revenue/cost signals and span far more of them than the other
two sets do.

| factor | hyp. dir | raw L/S (ann) | **ind-neutral α /mo** | **α t (full)** | α t (2016+) | β-neutral Sharpe (full → 2016+) | FM t (full) | turnover cost (pp/mo) | joint R² (gen / sw / **rd**) | verdict |
|---|:--:|--:|--:|:--:|:--:|:--:|:--:|--:|--:|:--|
| **revenue_stability** | + | +3.4% | **+0.69%** | **+2.96** | +3.21 | 0.60 → 1.06 | +2.69 | 0.032 | 0.10 / 0.08 / **0.27** | ✅ strong stand-alone, **but ≈ `rd_stability`** |
| deferred_rev_intensity | + | +3.2% | −0.10% | −0.48 | −0.14 | −0.09 → −0.05 | +0.35 | 0.051 | 0.08 / 0.07 / **0.33** | ❌ not confirmed (no α) |
| cost_scalability | + | +2.4% | +0.18% | +0.93 | +1.59 | 0.18 → 0.53 | −0.50 | 0.085 | 0.06 / 0.01 / **0.29** | 🟡 partial (recent only) |
| labor_productivity | + | −0.8% | −0.22% | −1.08 | +0.62 | −0.21 → 0.20 | −1.17 | 0.102 | 0.04 / 0.02 / **0.18** | ❌ not confirmed full-sample |
| gross_margin (baseline) | + | +7.9% | +0.60% | +2.45 | +0.51 | 0.51 → 0.17 | +1.92 | 0.027 | **0.37 / 0.20 / 0.23** | 🟡 worked, then arbitraged |

Two scores per factor: the **direction** (sign of α) tests the hypothesis; the
**magnitude/significance** and **redundancy** test whether it is investable and *novel*.
The headline lesson is that the directional hypotheses fared well for one factor but the
**novelty** claims do not survive the `vs_rd` comparison — every factor is materially
more spanned by the RD set than the hypothesis anticipated (which only checked gen/sw).

---

## 2. Per-factor grading

### 2.1 `revenue_stability` — ✅ STRONG STAND-ALONE α, ❗ BUT ≈ `rd_stability`

- **Direction & significance (stand-alone).** Industry-neutral α = **+0.69%/month,
  t = +2.96** (full), **+1.26%/month, t = +3.21** in 2016+ — *strengthening* in the
  recent decade, the opposite of decay; FM slope positive and significant (t = +2.69).
  As a **stand-alone signal the directional hypothesis is confirmed on every metric.**
- **Why the raw spread (t = 1.02) looked weak but the alpha is large.** The
  stable-minus-lumpy book carries a strongly **negative industry beta (β = −0.45,
  t = −11.3)**: stable-revenue firms are structurally lower-beta, so the L/S is
  implicitly *short the industry*. Because the industry rose on average, the raw spread
  *understates* the stock-selection skill; hedging out the −0.45 beta lifts the Sharpe
  from 0.20 to **0.60 (full) / 1.06 (2016+)**. Exactly the "durability is a low-beta
  tilt with positive alpha" picture the hypothesis described — and why the
  *industry-neutral* test is the right one.
- **❗ The novelty claim fails: this is largely `rd_stability` in disguise.** The
  hypothesis claimed orthogonality from the joint R² vs the general (0.10) and software
  (0.08) sets — but it **never checked the RD set, which is where the cousin lives**.
  Against the RD factors the joint R² jumps to **0.27**, driven almost entirely by
  **`rd_stability` (corr +0.50, R² 0.25)** — unsurprising, since both are 36-month
  fundamental-*stability* second moments. The decisive test is a **bivariate
  Fama–MacBeth on the sample where both exist**: `revenue_stability`'s slope **flips
  negative (t = −1.71) once `rd_stability` is controlled for**, while `rd_stability`
  *retains* its positive slope (t = +1.70). So `revenue_stability`'s return-relevant
  content is **subsumed by — and dominated by — the pre-existing `rd_stability`**; its
  orthogonal component earns ≈ 0 (if anything negative).
- **What it *does* add: coverage, not orthogonal alpha.** `revenue_stability` is
  defined for **~32% more stock-months** than `rd_stability` (111.8k vs a 75.8k
  overlap) — the no-/low-R&D names (IT services, some infrastructure) where an
  R&D-stability signal cannot be computed but a revenue-stability one can. That breadth
  is its genuine, and only, incremental value.
- **Cost.** Second-lowest turnover (0.032 pp/mo) — a 36-month statistic barely moves, so
  the gross alpha survives net of trading.
- **Verdict vs. hypothesis.** Direction ✔ and magnitude **exceeded** target on a
  risk-adjusted basis — but the **"orthogonal to standard factors" claim is falsified
  by the `vs_rd` check**: it is a wider-coverage twin of `rd_stability`, not an
  independent source of return. A strong *confirmation of the durability thesis*, not a
  *new* factor.

### 2.2 `deferred_rev_intensity` — ❌ NOT CONFIRMED

- **Direction & significance.** Industry-neutral α = **−0.10%/mo, t = −0.48** (full),
  ≈ 0 in 2016+ (t = −0.14); FM t = +0.35. The raw +3.2%/yr spread is almost entirely
  **industry beta (β = +0.35, t = +10.8)** — once hedged, the signal vanishes
  (β-neutral Sharpe −0.09). No reliable alpha in either direction.
- **Diagnosis (anticipated).** The hypothesis flagged proxy contamination as the most
  likely failure mode, and that is what the data show: orthogonality held (joint R²
  0.08/0.07 — it *is* a distinct exposure, as designed), but the exposure does not pay.
  `operating_liabilities − accounts_payable` evidently mixes deferred revenue with
  accrued compensation, deferred taxes and other operating accruals too heavily to
  isolate the contract-liability signal. **The idea is not refuted — the *proxy* is.**
  A clean contract-liabilities (or billings = revenue + ΔDeferredRevenue) field, which
  this dataset lacks, would be needed for a fair test.
- **RD overlap.** It is also the *most* RD-spanned factor (joint R² 0.33), correlating
  **+0.57 with `rd_intensity`** — i.e. firms carrying a large deferred-revenue balance
  are the same high-R&D-intensity "real product" firms — so even its exposure is partly
  a restatement of R&D intensity, not a clean balance-sheet signal.
- **Verdict.** Direction ✘ (slightly negative, insignificant), and not as orthogonal as
  hoped (overlaps `rd_intensity`). Shelve until a cleaner deferred-revenue input exists.

### 2.3 `cost_scalability` — 🟡 PARTIALLY CONFIRMED (right sign, weak, recent-period)

- **Direction & significance.** Industry-neutral α positive but insignificant full-sample
  (+0.18%/mo, t = +0.93), **strengthening to +0.43%/mo, t = +1.59 in 2016+**
  (β-neutral Sharpe 0.18 → **0.53**). The quintile monotonicity is real (Q5 next-month
  return 1.43% vs Q1 1.23%). But the **FM slope is flat/negative (t = −0.50)** — the
  effect is **non-linear**, concentrated in the tails (extreme operating leverage), not
  in a smooth cross-sectional gradient, which is why the tail-based quintile book is
  positive while the linear FM regression is not.
- **Redundancy / the `sue` question.** As predicted, its single largest general
  correlate is `sue` (corr +0.18, R² 0.033) — *positive but small*; it is **not**
  spanned by earnings momentum (joint R² only 0.06 general / 0.01 software). But the
  `vs_rd` check again bites: joint R² 0.29, driven by **`rd_conversion` (corr +0.45)** —
  the R&D-profit-output factor — because both measure margin/profit-accretion dynamics.
  So the incremental-information claim holds against gen/sw but only **partially** against
  the RD set. The raw signal is also just weak/noisy, and the YoY-growth construction
  costs the most in turnover (0.085 pp/mo).
- **Verdict.** Direction ✔ (especially post-2016), but only marginally significant and
  only recently; novelty ✔. A real-but-faint operating-leverage premium, better in the
  modern SaaS era — consistent with the structural story, short of an investable
  stand-alone signal.

### 2.4 `labor_productivity` — ❌ NOT CONFIRMED full-sample

- **Direction & significance.** Wrong sign over the full sample: α = **−0.22%/mo,
  t = −1.08**, FM t = −1.17 (not significantly negative, but clearly not the
  hypothesised positive). It flips positive in 2016+ (α +0.18%/mo) but insignificantly
  (t = +0.62). Highest turnover cost of the five (0.102 pp/mo).
- **Diagnosis (anticipated).** The hypothesis named this the least certain factor and
  predicted substantial overlap with `cost_scalability` — confirmed: the two are
  **+0.57 correlated**, the highest pair in the library (shared `growth(sales)` term).
  The headcount channel adds noise rather than signal over `cost_scalability`, and the
  sparse, annually-updated `employee_count` (~83% coverage) makes it the noisiest input.
- **Verdict.** Direction ✘ full-sample; novelty ✔ but largely redundant with
  `cost_scalability`. Two views of operating leverage; the **financial-cost** view
  (`cost_scalability`) dominates the **headcount** view. Drop in favour of the former.

### 2.5 `gross_margin` (baseline) — 🟡 WORKED, THEN GOT ARBITRAGED (confirms the framework)

- **Result.** The strongest *raw* book (+7.9%/yr, raw t = 2.72) and a significant
  full-sample industry-neutral α (+0.60%/mo, **t = +2.45**) — but it **decays out of
  sample**: α falls to +0.21%/mo, t = +0.51 in 2016+, and the FM slope goes from
  marginally significant (t = +1.92) to **zero (t = −0.03) post-2016**.
- **Interpretation.** Exactly the efficient-markets prior the hypothesis stated: the
  single most-watched software KPI generated a premium historically but has been
  **competed away in the past decade** as the crowd's screens converged on it. It is
  also the **most redundant** factor — joint R² **0.37 (general) / 0.20 (software)**,
  dominated by `intangible_profitability` (corr +0.46) — i.e. it largely re-expresses
  existing profitability factors. As a baseline it did its job: it shows that "obvious"
  margin information is now priced away, **whereas the stability signal still pays and
  even strengthens** — though (§3.5) that stability premium is the *already-known*
  `rd_stability`, not something this library discovered.

---

## 3. Cross-factor priors (from §2 of the hypothesis) — scorecard

| prior | predicted | realised | ✓/✗ |
|---|---|---|---|
| (a) slow level/stability signals trade least | `revenue_stability`, `deferred_rev_intensity` lowest turnover | 0.032, 0.051 vs 0.085 (cost_scal), 0.102 (labor) pp/mo | ✓ |
| (b) `cost_scalability` ↔ `labor_productivity` positively correlated | yes (shared op-leverage root) | **+0.573** (highest pair) | ✓ |
| (b) `cost_scalability` most correlated with `sue` (of the *general* set) | yes, but not spanned | corr +0.18, R² 0.033 (its top general correlate); joint R² 0.06 | ✓ |
| (c) `gross_margin` most redundant vs existing factors | yes | joint R² 0.37/0.20 (gen/sw) — highest of five | ✓ |
| (d) incremental value concentrated in `revenue_stability` & `deferred_rev_intensity` | both orthogonal; expected to carry the library | orthogonal vs gen/sw, **but both heavily overlap the RD set (§3.5); only `revenue_stability` pays, and its α is spanned by `rd_stability`** | ✗ |

### 3.5 The decisive `vs_rd` redundancy check (added after the fact)

The hypothesis benchmarked novelty only against the general (Exp 1) and standard-software
(Exp 2) sets, and on those the four novel factors looked clean (joint R² ≤ 0.10). But the
**closest existing cousins are the RD-behaviour factors**, and against *that* set every
Rev & Cost factor is markedly more spanned:

| Rev & Cost factor | joint R² vs gen | vs sw | **vs rd** | dominant RD correlate |
|---|--:|--:|--:|:--|
| revenue_stability | 0.10 | 0.08 | **0.27** | `rd_stability` (corr +0.50) |
| deferred_rev_intensity | 0.08 | 0.07 | **0.33** | `rd_intensity` (corr +0.57) |
| cost_scalability | 0.06 | 0.01 | **0.29** | `rd_conversion` (corr +0.45) |
| labor_productivity | 0.04 | 0.02 | **0.18** | `rd_conversion` / `rd_intensity` |
| gross_margin | 0.37 | 0.20 | 0.23 | (already redundant everywhere) |

Linear R² of ~0.25–0.33 still leaves most variance unexplained, but the **return-spanning
test is sharper than the variance test**: in a bivariate Fama–MacBeth, `revenue_stability`
loses all its premium to `rd_stability` (§2.1). The reading is that the *durability /
stability / margin-accretion* themes this library probes were **already captured by the
RD-behaviour library on the R&D-reporting subset**; the revenue/cost framing mostly
*re-discovers* them (a strong independent corroboration of those RD signals) and extends
them to non-R&D names, rather than adding orthogonal information. Lesson for future
factor design: **benchmark novelty against the *nearest* existing library, not only the
generic ones.**

The orthogonality predictions were almost perfectly right; the **return** predictions
were right for `revenue_stability`, directionally-but-weakly right for `cost_scalability`,
and wrong for `deferred_rev_intensity` (proxy) and `labor_productivity` (redundant/noisy).

## 4. Synthesis — what we learned

1. **A strong stand-alone signal that is *not* a new one.** `revenue_stability` has a
   genuine, **investable, robust** industry-neutral alpha (t = +2.96 full, +3.21 2016+;
   β-neutral Sharpe 0.6–1.1; trivial turnover), driven exactly as theorised — the market
   under-prices recurring-revenue durability, which shows up as a low-beta tilt with large
   hedged alpha (β = −0.45). **But it is ~0.50-correlated with, and return-dominated by,
   the pre-existing `rd_stability`** (§2.1, §3.5): controlling for `rd_stability` kills its
   premium. So the *durability thesis is confirmed*, but the *factor is a wider-coverage
   twin of a signal the project already has*, not a new alpha source. Honest headline:
   **independent corroboration of `rd_stability`, plus a coverage extension to non-R&D
   names** — valuable, but not the orthogonal discovery the hypothesis hoped for.
2. **The efficient-markets logic held for the baseline.** `gross_margin` — the obvious KPI
   — paid historically and was arbitraged away post-2016, while the stability signal kept
   paying. Direct evidence that the alpha lives where the crowd is *not* looking (a
   fundamental time-series property), not in the headline ratio everyone screens — the
   central thesis of the hypothesis document, even if the winning signal turned out to be
   a known one.
3. **A near-miss with a clean story.** `cost_scalability` shows a real but faint
   operating-leverage premium, non-linear (tails only) and stronger in the SaaS era;
   distinct from `sue` but overlapping `rd_conversion` (corr +0.45). Worth revisiting as a
   *conditioning* variable, not a stand-alone signal.
4. **Two honest failures, both anticipated.** `deferred_rev_intensity` failed on **data**
   (no clean deferred-revenue field; the operating-liability proxy is too contaminated, and
   it half-restates `rd_intensity`) — the *idea* is untested, not refuted. `labor_productivity`
   failed on **redundancy + noise** (0.57 correlated with `cost_scalability`, sparse annual
   headcount). Both failure modes were called out ex-ante, which is the value of writing
   predictions down first.
5. **Methodological lesson.** The "orthogonal to standard factors" claim looked solid until
   the signals were benchmarked against the **nearest** existing library (RD), not just the
   generic ones — where 3 of 5 jumped to joint R² ≈ 0.27–0.33. Always test a candidate
   against its closest cousin, and prefer the **return-spanning** (bivariate FMB) test over
   the variance-R² test, which is more forgiving.

## 5. Caveats & next steps

- **Industry-neutral is the right lens here.** Raw L/S spreads conflate stock-selection
  with industry-beta tilts (most extreme for `revenue_stability`, β = −0.45, and
  `deferred_rev_intensity`, β = +0.35). All verdicts above use the β-hedged alpha; the
  raw `quintile/summary.csv` numbers should not be read in isolation.
- **`revenue_stability` direction nuance** (from the hypothesis): it does not distinguish
  steady growth from steady decline. It worked anyway, but a refinement — interacting
  stability with the *sign/level* of revenue growth (reward stable **growers**, not stable
  shrinkers) — could de-correlate it from `rd_stability` and is the obvious next iteration.
- **`deferred_rev_intensity` deserves a re-test** if a billings or contract-liability
  line item becomes available; the construction here is the best the dataset allows but is
  the weak link, and it overlaps `rd_intensity`.
- **For Experiment 3:** do **not** stack `revenue_stability` on top of `rd_stability` —
  the orthogonal component earns ≈ 0, so it would add turnover and correlation without
  alpha. Its only justified use is as a **coverage-extending stand-in** for `rd_stability`
  on the ~32% of stock-months (no-/low-R&D names) where `rd_stability` is undefined.
  `cost_scalability` is a possible *conditioning* overlay only; `deferred_rev_intensity`
  and `labor_productivity` are not recommended on this data.
