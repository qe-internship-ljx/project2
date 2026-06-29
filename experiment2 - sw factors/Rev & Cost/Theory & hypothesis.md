# Revenue & Cost factors for Software & Services — Theory & Hypothesis

*Experiment 2, "Rev & Cost" extension. Universe: GICS **Software & Services** industry
group (~1,100 securities, 1998–2025, point-in-time fundamentals).*

This document is written **before any back-test is run**. It reasons from the
distinctive economics of software firms' **revenue and cost** to a small set of
*non-standard* factors, states a falsifiable hypothesis for each (direction,
mechanism, counterparty, robustness), and only afterwards do we test them with the
exact Experiment-2/RD pipeline (even-quintile sorts, Fama–MacBeth cross-sectional
regressions, dollar-neutral long/short books with realistic trading cost, and a
redundancy check against the existing general and software factors). Realised results
are written separately in `results.md`.

---

## 0. What makes software revenue and cost economically different

Before proposing factors, fix the economic facts that the standard factor zoo
(value, momentum, quality, profitability, investment, accruals, low-risk) was *not*
built to capture, and that are most pronounced in software:

1. **Negligible marginal production cost / extreme operating leverage.** Once a
   product is built, serving one more customer costs almost nothing (cloud, support).
   Empirically, the median software firm here runs a **52% gross margin** but only a
   **~3% operating margin** — i.e. the cost base is dominated not by production (COGS)
   but by *semi-fixed* discretionary spend (R&D, sales & marketing, G&A). The economic
   signature of a healthy software firm is therefore **costs that grow slower than
   revenue** as it scales past that fixed base — margin expansion that is *latent* in
   the cost structure and not yet in the trailing margin a screen would read.

2. **Deferred revenue / the subscription (SaaS) model.** Customers pre-pay (annual or
   multi-year) for a service delivered over time. Cash and *billings* lead recognised
   revenue; the unearned portion sits on the **balance sheet as a contract liability
   (deferred revenue)**, not on the income statement. Healthy subscription firms are
   thus **customer-funded** (negative operating working capital) and carry a large
   contract-liability balance — a forward book of revenue that income-statement-centric
   investors systematically under-weight.

3. **Recurring, renewable revenue → predictability.** Subscriptions renew; revenue is
   smooth and high-visibility relative to license/transaction models. The *durability*
   of a revenue stream is economically valuable (lower cash-flow risk, higher terminal
   value) but is a **time-series property** invisible in any single income statement.

4. **The "factory" is people, not plant.** Software is asset-light; the binding input
   is engineering and go-to-market **talent**, expensed as incurred. The relevant
   productivity question is *revenue generated per employee* and whether that is
   **rising** (automation, product leverage) or falling (services drag, hiring ahead of
   monetisation).

5. **GAAP mismeasures software profit.** R&D and S&M build durable assets (product IP,
   customer relationships with high lifetime value) but are **expensed immediately**, so
   GAAP earnings understate the economics of firms that are investing. *(This investment
   dimension is the subject of the sibling `standard/` and `RD/` factor libraries; we do
   **not** re-tread it here. Our focus is the **flow of revenue and the structure of
   cost**, not the R&D investment stock.)*

The standard factors miss (1)–(4) by construction: they are built on **levels** (a
margin, a yield, a beta) or on **price** (momentum), whereas these software features
live in **second moments** (revenue stability), **balance-sheet contra-revenue**
(deferred revenue), **growth wedges** (cost vs revenue), and **per-head productivity**.
That gap is the opportunity.

### Method note — how each factor is signed and tested

Every factor below is signed by its **ex-ante hypothesised direction** (the
`higher_is_bullish` prior), *not* by its in-sample t-stat
(`USE_CANONICAL_LS_DIRECTION = True`, identical to Experiment 1 and the `RD/` module).
Consequently a **negative realised alpha t-stat means the factor worked *against* the
hypothesis** — this is genuine hypothesis testing, not curve-fitting. Each factor is
cross-sectionally winsorised (1%) and z-scored within the monthly software cross-section
before sorting/regression, exactly as in Experiments 1–2. All five signals are **ratios,
growth rates, or growth-differences and are therefore currency-neutral**, so no FX
conversion is needed for within-industry comparison (consistent with the rest of the
project, and verified for the one factor — revenue per employee — where a *level* would
not be: we use its within-firm growth).

---

## 1. The factors

| name | dimension | definition (raw) | dir | data inputs |
|------|-----------|------------------|-----|-------------|
| `revenue_stability` | revenue — durability (2nd moment) | − trailing-36m std of YoY revenue growth | long high | `sales_ltm` |
| `deferred_rev_intensity` | revenue — subscription/billings model | (`operating_liabilities` − `accounts_payable`) / `sales_ltm` | long high | `operating_liabilities`, `accounts_payable`, `sales_ltm` |
| `cost_scalability` | cost — operating leverage | YoY growth(`sales_ltm`) − YoY growth(`operating_expenses_ltm`) | long high | `sales_ltm`, `operating_expenses_ltm` |
| `labor_productivity` | revenue per cost-input — human capital | YoY growth of (`sales_ltm` / `employee_count`) | long high | `sales_ltm`, `employee_count` |
| `gross_margin` | cost — near-zero marginal cost (**baseline**) | `gross_income_ltm` / `sales_ltm` | long high | `gross_income_ltm`, `sales_ltm` |

The first four target genuinely *non-standard* dimensions; `gross_margin` is retained as
the **conventional anchor** — the single most-watched software KPI — against which the
four novel signals are contrasted (mirroring how `RD/` keeps `rd_intensity` as a level
baseline). Every input above is ≥92% populated for the software universe except
`employee_count` (~83%, so `labor_productivity` has thinner coverage by design).

> **Data caveat made explicit.** The dataset has **no dedicated deferred-revenue line
> item**. `deferred_rev_intensity` is therefore a *proxy*: non-trade operating
> liabilities = `operating_liabilities − accounts_payable`, which for software is
> dominated by deferred/unearned revenue plus accrued compensation. Empirically it is
> positive for 98.6% of software firms with a median of 0.36× sales — economically
> sensible for contract liabilities. The proxy is noisier than a clean
> contract-liability balance, and the hypothesis is stated with that in mind.

---

### 1.1 `revenue_stability` — recurring-revenue durability

**Definition.** For each stock, monthly YoY revenue growth `g_t = sales_ltm_t /
sales_ltm_{t−12} − 1`; the factor is the **negative** of the trailing-36-month standard
deviation of `g_t` (≥24 months required). High (near 0) = a smooth, predictable,
recurring revenue stream; low (very negative) = lumpy, license/deal-driven revenue with
renewal/air-pocket risk. We negate so that *higher = more stable = the bullish leg*.

**Financial logic / what drives the return.** This is primarily an **edge
(mispricing), with a secondary risk-premium interpretation that points the *same*
way under the quality anomaly.** Recurring revenue is worth more per dollar than
one-off revenue — it has a higher renewal probability, lower cash-flow variance, and a
larger terminal value — yet a cross-sectional screen on *level* metrics (growth rate,
margin, yield) cannot see it, because durability is a **property of the revenue
*time-series*, not of any single statement**. The market, anchored on the most recent
growth print, rewards the *level* and *acceleration* of growth and under-weights its
*consistency*. Stable compounders are "boring," under-owned relative to high-variance
high-fliers, and so are priced for less than the durability is worth. A naive
risk-based view would say "stable ⇒ low risk ⇒ low expected return," but the empirical
**quality / low-fundamental-volatility premium** (Asness–Frazzini–Pedersen, *Quality
Minus Junk*; Novy-Marx on earnings stability) is exactly the opposite sign — safe,
stable, profitable firms have *out-earned* — which is itself the behavioural anomaly we
are harvesting. We side with the quality prior: **higher stability → higher returns.**

**Why it is orthogonal to standard factors.** It is a **second moment**. By
construction it is uncorrelated with any level signal (margin, yield, growth, beta,
profitability), and it is computed on *fundamentals*, not price, so it is distinct from
the low-**return**-volatility / low-beta factor (a stock can have volatile prices but
very stable revenue, and vice versa). It is the revenue analogue of the project's
existing `rd_stability` (which measures consistency of R&D spend) but on a different and
larger quantity — the top line itself.

**Why it is not already arbitraged.** Computing it requires assembling a *multi-year,
point-in-time* revenue series and taking its dispersion — more work than a one-line
screen, and it produces a "low-octane" signal that career-incentivised, benchmark-aware
managers find unattractive to hold (it underweights the exciting names). The premium
persists because the **behavioural preference for glamour/growth and the institutional
aversion to "boring"** are structural, not informational.

**Counterparty & why they trade at a favourable price.** Our short leg is the
glamour/high-variance grower the crowd over-pays for (a **behavioural** crowd
inefficiency — extrapolation of recent rapid growth); our long leg is the steady
compounder that momentum/growth investors leave on the table. We are not betting on
private information — we are providing the patience that the marginal, short-horizon
investor will not.

**Predicted direction & magnitude.** **Positive.** Long stable / short lumpy. I expect
one of the *more* robust signals in the set: a long/short spread of roughly
**+3% to +6% annualised**, Fama–MacBeth / alpha **t ≈ 1.5–2.5**, with **low turnover**
(a 36-month stat moves slowly → low trading cost, which should make its *net* and
beta-neutral Sharpe relatively attractive). Caveat: pure stability conflates "stable up"
with "stable down"; a steadily-shrinking firm also scores high. In software steady
shrinkers are rare, but this is the most likely source of disappointment.

---

### 1.2 `deferred_rev_intensity` — the subscription / billings model

**Definition.** `(operating_liabilities − accounts_payable) / sales_ltm` — non-trade
operating liabilities (a deferred-revenue + accrued-comp proxy) as a fraction of annual
revenue. High = a heavily **pre-billed, recurring, customer-funded** business; low = a
firm that bills in arrears / recognises as it sells (license, services, usage).

**Financial logic / what drives the return.** A pure **edge**. Deferred revenue is a
**forward book of already-contracted revenue** that will be recognised in coming
quarters — a leading indicator of revenue *with the sale already closed and frequently
the cash already collected*. But it lives on the **balance sheet as a liability** and is
absent from headline EPS, revenue, and the metrics that dominate sell-side models and
quant screens. Income-statement-anchored investors therefore *systematically
under-value the visibility and stickiness* a large contract-liability balance confers.
There is also a quality dimension: a high deferred-revenue intensity means the firm is
**customer-funded** (negative operating working capital), needs less external capital to
grow, and has demonstrably high switching costs (customers willingly pre-pay) — all
return-supportive and none of them visible in a margin or a yield.

**Why it is orthogonal to standard factors.** No factor in Experiments 1–2 touches the
**operating-liability** side of the balance sheet. It is not value (no price in it), not
profitability (no earnings in it), not investment/asset-growth (a liability ratio, not
asset growth), and only loosely related to accruals — Sloan's accruals is a *change in
net working capital scaled by assets*, whereas this is a *level of contract liabilities
scaled by sales*; the sign convention also differs (here, more deferred revenue is
**good**, the opposite of the "high accruals are bad" prior, because for software the
working-capital contra-asset is a *pre-payment*, not an aggressive accrual).

**Why it is not already arbitraged.** Three frictions: (i) there is no clean
"deferred revenue" field in most standardised datasets (as here — it must be *proxied*
from operating liabilities), so it is costly to compute and noisy; (ii) it is a
balance-sheet ratio in a world that trades on income-statement surprises; (iii) it reads
as a *liability*, and liabilities intuitively feel bad — a framing the careless screen
penalises.

**Counterparty & why they trade at a favourable price.** A mix of **technical** (data
vendors don't isolate the field; many quant pipelines never build it) and **behavioural**
(liabilities are framed as negative; the visibility premium is under-appreciated)
inefficiency. The counterparty is the income-statement-only investor who cannot see the
contracted forward book.

**Predicted direction & magnitude.** **Positive**, long high deferred-revenue intensity.
Because the input is a *noisy proxy*, I expect a **weaker / less certain** result than
`revenue_stability`: spread perhaps **+2% to +5% annualised**, alpha **t ≈ 1.0–2.0**.
The main risk is proxy contamination (accrued comp, taxes payable) diluting the
deferred-revenue signal; if the realised sign is right but weak, that is the most likely
reason. Turnover is low (a slow-moving level), helping the net Sharpe.

---

### 1.3 `cost_scalability` — realised operating leverage

**Definition.** `YoY growth(sales_ltm) − YoY growth(operating_expenses_ltm)`. Because
`operating_expenses_ltm` is the **full** operating cost base (it includes COGS — verified
in-data: `sales − operating_expenses ≈ operating_income` for 95.5% of software firms),
a positive value means **revenue is outgrowing the entire cost base** — i.e. operating
margin is *expanding* and the firm is climbing its operating-leverage curve. Negative =
costs outrunning revenue (de-leveraging, land-grab, or stalling).

**Financial logic / what drives the return.** A **behavioural underreaction edge** —
the cost-structure analogue of post-earnings-announcement drift. Software's semi-fixed
cost base means that *once revenue crosses the fixed-cost hump, incremental margins are
very high and margin expansion tends to **persist** for several quarters* (cloud,
platform and core-engineering costs do not rise one-for-one with seats). The market
**anchors on the trailing (low) margin** of a still-scaling software firm and is slow to
re-rate the *trajectory*; the realised wedge between revenue and cost growth is a
forward signal of continued margin expansion and the earnings surprises that follow.
The return is compensation for supplying the re-rating the anchored crowd delays — an
**edge**, not a risk premium (there is no obvious systematic risk for which "margins are
improving" is a loading).

**Why it is orthogonal to standard factors.** It is a **growth-of-cost-vs-growth-of-
revenue wedge**, not a level. It is not `gross_profitability` or `gross_margin` (levels),
not `asset_growth` (uses costs, not assets — and carries the *opposite* sign: here more
cost growth is bad *only relative to* revenue growth). Its closest cousin is `sue`
(earnings momentum): both are "things are improving" signals, and a positive correlation
is plausible. We therefore explicitly **measure** its redundancy against `sue` and the
profitability factors in the `factor_correlation` step; the claim is that it carries
*incremental* information because it isolates the **cost-side** driver of the improvement
(operating leverage) rather than the bottom-line outcome, and excludes below-the-line
and one-off items that move `sue`.

**Why it is not already arbitraged.** It requires two YoY growth computations and a
*difference*; most "margin" screens look at the level or a one-period change in the
margin, not the revenue-vs-cost growth wedge, which behaves better when the level margin
is near zero or negative (true of ~37% of software firms with negative operating income,
where a margin-*level* signal is ill-defined but a growth-wedge is not).

**Counterparty & why they trade at a favourable price.** **Behavioural** — anchoring on
trailing margins and underreaction to inflection. The counterparty sells the
just-inflected scaler too cheaply (still "unprofitable" on trailing GAAP) and holds the
de-leveraging firm too long (trailing margin still looks fine). Public information,
slow re-rating.

**Predicted direction & magnitude.** **Positive**, long high. I expect a **moderate**
spread, **+3% to +7% annualised**, alpha **t ≈ 1.5–2.5**, but with **higher turnover**
than the level factors (YoY growth refreshes each quarter), so the *net-of-cost* Sharpe
will be meaningfully lower than the gross. The chief risk is overlap with `sue` — if the
redundancy check shows it is largely spanned by earnings momentum, the *incremental*
alpha shrinks even if the raw spread is healthy.

---

### 1.4 `labor_productivity` — human-capital leverage

**Definition.** YoY growth of **revenue per employee**: `(sales_ltm / employee_count)_t
/ (sales_ltm / employee_count)_{t−12} − 1`. We use the *growth*, not the level, for two
reasons: (i) the level (sales/head) is reported in **local currency** and so is *not*
currency-neutral across the cross-section, whereas its within-firm growth is; (ii) the
*improvement* in per-head output is the genuine alpha signal — the static level largely
reflects business mix (product vs services) and is more likely already priced. Positive =
the firm is monetising faster than it is hiring (automation, product leverage, upsell);
negative = headcount is outrunning revenue (services drag, hiring ahead of monetisation).

**Financial logic / what drives the return.** An **edge** rooted in software's
asset-light reality: the "factory" is people, expensed as incurred, so **revenue per
head is the truest unit of productivity** and its trend is the human-capital analogue of
operating leverage. A firm whose revenue-per-employee is rising is building **scalable
product leverage**; the market, which fixates on absolute revenue growth and on
headcount as a *growth* signal ("they're hiring, must be booming"), under-weights the
*efficiency* of that growth. Rising productivity is also a leading indicator of future
margin expansion (people are the dominant cost), so it predicts the same re-rating as
`cost_scalability` but through the **physical/headcount** channel rather than the
financial cost channel.

**Why it is orthogonal to standard factors.** No factor in the project uses
`employee_count`. It is not profitability (a per-head *growth*, not a margin), not asset
growth (no assets), not net issuance. It is most likely correlated with `cost_scalability`
(both share `growth(sales)`; the difference is the second term — *headcount* growth vs
*total cost-dollar* growth), and the two will diverge whenever a firm trades labour for
non-labour cost (e.g. shifts spend to cloud/marketing, or automates). We measure that
correlation rather than assert it away.

**Why it is not already arbitraged.** `employee_count` is sparsely covered (~83% here)
and updates only annually, so the signal is **noisy and low-frequency** — unattractive
to high-turnover quant strategies and easy to dismiss as a "soft" metric, which is
precisely why a patient, fundamentals-based investor can harvest it.

**Counterparty & why they trade at a favourable price.** **Behavioural** — the crowd
reads hiring as a bullish growth tell and revenue growth as the headline, while
under-pricing the *ratio*. The counterparty over-pays for the headcount-heavy "growth"
story and under-pays for the quietly-automating efficient compounder.

**Predicted direction & magnitude.** **Positive**, long rising-productivity. Given the
noisy, low-coverage, annually-updated input, I expect this to be the **least certain** of
the four novel factors: spread **+2% to +5% annualised**, alpha **t ≈ 1.0–2.0**, and a
real chance the redundancy check shows substantial overlap with `cost_scalability`. If it
adds little beyond `cost_scalability`, that is the expected failure mode and would still
be an informative result (two views of the same operating-leverage phenomenon).

---

### 1.5 `gross_margin` — near-zero marginal cost (**conventional baseline**)

**Definition.** `gross_income_ltm / sales_ltm`. The textbook software KPI: the fraction
of each revenue dollar left after the (small) cost of delivering the product. High = a
truly scalable, software-like cost structure; low = services/hardware drag or commoditised
delivery.

**Role & logic.** This is the **baseline anchor**, included precisely *because* it is the
obvious, widely-watched metric — to show how much (if anything) the four novel factors add
beyond "high gross margin = good software business." A high gross margin certainly
*describes* a good software business, but it is **the first thing every investor screens
on**, so by the efficient-markets logic of this project it should be **largely priced** and
generate **little long/short alpha** within the industry. Its theoretical return driver, to
the extent any survives, is a **risk premium** (high-margin asset-light firms may be more
exposed to growth-stock / duration risk) which could even point the spread the *other* way.

**Why it is (mostly) not orthogonal — and that is the point.** It is a level
profitability ratio and will correlate with `gross_profitability` (Exp 1; same numerator,
assets in the denominator instead of sales) and with the intangible-profitability signals.
We expect the redundancy check to confirm it is the *most* spanned of the five.

**Predicted direction & magnitude.** Nominally **positive** but **weak**: spread near
**0% to +3% annualised**, alpha **|t| < 1.5**, quite possibly insignificant or
wrong-signed. A strong, significant result here would actually be *surprising* and would
suggest the within-industry margin premium is less arbitraged than efficient-markets logic
implies. The factor's job is to be the **null/benchmark** the novel signals must beat on
an *incremental* (orthogonalised, net-of-cost) basis.

---

## 2. Summary of hypotheses

| factor | dir | return driver | primary counterparty inefficiency | predicted spread (ann.) | predicted alpha t | turnover |
|--------|-----|---------------|-----------------------------------|------------------------|-------------------|----------|
| `revenue_stability` | + (long stable) | quality/durability mispricing (edge) | behavioural: glamour-growth preference | +3% … +6% | 1.5 – 2.5 | low |
| `deferred_rev_intensity` | + (long high) | hidden forward book / customer-funded (edge) | technical + behavioural: balance-sheet blind spot | +2% … +5% | 1.0 – 2.0 | low |
| `cost_scalability` | + (long high) | operating-leverage underreaction (edge) | behavioural: anchoring on trailing margin | +3% … +7% | 1.5 – 2.5 | higher |
| `labor_productivity` | + (long rising) | human-capital leverage underreaction (edge) | behavioural: hiring read as bullish | +2% … +5% | 1.0 – 2.0 | higher |
| `gross_margin` (baseline) | + (long high) | mostly priced; residual = risk premium | — (expected ≈ efficient) | 0% … +3% | low |

**Cross-factor priors.** (a) The two *level/slow* signals (`revenue_stability`,
`deferred_rev_intensity`) should have the best **net-of-cost** profile because they barely
trade; the two *growth* signals (`cost_scalability`, `labor_productivity`) will have higher
gross spreads but pay more in turnover. (b) I expect `cost_scalability` and
`labor_productivity` to be **positively correlated** with each other (shared
operating-leverage root) and `cost_scalability` to be the one most correlated with the
existing `sue`. (c) `gross_margin` should be the most redundant vs the existing software /
general factors. (d) Net: the *incremental* value of this library lies mainly in
`revenue_stability` and `deferred_rev_intensity`, which probe dimensions
(fundamental second moment; contract-liability balance) **no existing factor in the
project measures at all**.

## 3. How these will be tested (pipeline)

Identical wiring to `RD/main_rd.py` — the `revcost_factors` library is registered as the
analysis engine's `factors` module, so Experiment 1's `quintile.py`, `regression.py` and
`cost.py` run against it unchanged. Outputs land under `Rev & Cost/`:

- **`quintile/<factor>/`** — even-quintile next-month returns, cumulative growth-of-$1 by
  quintile, and the directional (canonically-signed) dollar-neutral long/short book; plus a
  cross-factor `quintile/summary.csv` and the `long_short_market_alpha` table (industry-
  neutral alpha, annualised Sharpe, **β-neutral** Sharpe, and **average turnover cost**).
- **`regression/<factor>/`** — monthly cross-sectional (Fama–MacBeth) regressions of next
  return on the factor z-score, full-sample and 2016+, with the summary t-stat table.
- **`factor_correlation/{vs_general,vs_software}/<factor>/`** — R² of each Rev & Cost factor
  spanned by (i) the nine Experiment-1 general factors and (ii) the Experiment-2 software
  factors, the empirical backing for every "orthogonal to standard factors" claim above.

A factor **confirms** its hypothesis if the canonically-signed long/short book earns a
**positive** industry-neutral alpha (the sign test), ideally with `|t| ≳ 1.65`, a positive
net-of-cost and β-neutral Sharpe, and **low spanning** (`vs_general`/`vs_software` R²) for
the four novel signals. A **negative** alpha t-stat falsifies the hypothesised direction.
`results.md` will grade each factor against these predictions explicitly.
