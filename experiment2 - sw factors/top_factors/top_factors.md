# Top factors -- definition & intuition

The top 5 factors across the Software & Services factor experiments -- Experiment 1's general market factors plus Experiment 2's software subexperiments -- ranked by the sum of the full-period and 2016-onward industry-neutral long/short alpha t-stats (see `top_factors.csv` and `all_factors_ranked.csv`).  For each factor below: its **definition** (how the raw signal is computed from fundamentals / prices) and its **intuition** (why it is expected to predict returns).  All signals are z-scored cross-sectionally against the industry mean before sorting.

## 1. Profitability/quality

- **Factor:** `gross_profitability` &nbsp; · &nbsp; **Subexperiment:** General
- **Long/short book:** Q5-Q1 (long the top quintile (Q5), short the bottom (Q1))
- **Industry-neutral alpha:** +1.0261% / month &nbsp; · &nbsp; t(alpha) = +4.17

**Definition.** Gross profitability: gross_income_ltm / total assets.

**Intuition.** Gross profits are the cleanest measure of true economic profitability (Novy-Marx 2013); more-profitable firms earn higher returns, a quality dimension orthogonal to value.

## 2. Revenue stability (recurring-revenue durability)

- **Factor:** `revenue_stability` &nbsp; · &nbsp; **Subexperiment:** Rev & Cost
- **Long/short book:** Q5-Q1 (long the top quintile (Q5), short the bottom (Q1))
- **Industry-neutral alpha:** +0.7116% / month &nbsp; · &nbsp; t(alpha) = +3.60

**Definition.** Negative trailing-36m standard deviation of YoY revenue growth.

**Intuition.** Recurring (subscription) revenue is smooth; lumpy license/deal revenue is not.  The market rewards the level/acceleration of growth and underweights its durability, so stable top-lines (a quality dimension) outperform.

## 3. R&D stability (commitment consistency)

- **Factor:** `rd_stability` &nbsp; · &nbsp; **Subexperiment:** RD
- **Long/short book:** Q5-Q1 (long the top quintile (Q5), short the bottom (Q1))
- **Industry-neutral alpha:** +0.8980% / month &nbsp; · &nbsp; t(alpha) = +4.04

**Definition.** Negative trailing-36m coefficient of variation of R&D intensity (rd_ltm / sales).

**Intuition.** A second moment -- the consistency of R&D commitment.  Firms that hold R&D steady (rather than cutting it to manage earnings) signal durable innovation and earnings quality; orthogonal by construction to every level signal.

## 4. Piotroski F-score

- **Factor:** `fscore` &nbsp; · &nbsp; **Subexperiment:** Standard
- **Long/short book:** Q5-Q1 (long the top quintile (Q5), short the bottom (Q1))
- **Industry-neutral alpha:** +0.8150% / month &nbsp; · &nbsp; t(alpha) = +3.81

**Definition.** Piotroski (2000) F-score: the sum of 9 binary fundamental-health tests (profitability, leverage/liquidity, operating efficiency), 0-9; supplied pre-computed and used verbatim.

**Intuition.** A higher score marks a financially strengthening firm; this bundle of accounting-improvement signals predicts higher subsequent returns, classically strongest among value names.

## 5. Buyback quality

- **Factor:** `buyback_quality` &nbsp; · &nbsp; **Subexperiment:** Standard
- **Long/short book:** Q5-Q1 (long the top quintile (Q5), short the bottom (Q1))
- **Industry-neutral alpha:** +0.7724% / month &nbsp; · &nbsp; t(alpha) = +3.41

**Definition.** Realised share-count reduction minus the gross buyback yield (-buyback_ltm / market cap): genuine net shrinkage in shares vs. cash spent repurchasing.

**Intuition.** Distinguishes real, share-reducing buybacks from repurchases that merely offset stock-comp dilution.  A firm spending on buybacks whose share count does not fall scores low (low quality); genuine net repurchasers score high.
