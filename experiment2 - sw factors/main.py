"""
main.py
=======

Experiment 2 driver: run the full factor pipeline on the **Software & Services**
cross-section using the established *software-industry* factors (``sw_factors.py``,
plan section 2.2) instead of the general market factors.

Like Experiment 1's per-universe drivers (``banks_insurance.py`` /
``commodity_producers.py``) this adds no new analytics.  It reuses Experiment 1's
``quintile.py`` and ``regression.py`` **unmodified** -- the quintile sorts (now
including the average-trading-cost summary), cross-sectional (Fama-MacBeth)
regressions, long/short books and every plot are exactly the same code path as
Experiment 1.  Only the *factor library* changes.

How the reuse works
-------------------
Experiment 1's analysis modules bind to their factor library via ``import
factors as F``.  We register our software-factor library under that name in
``sys.modules`` *before* importing them, so their ``F`` resolves to
``sw_factors`` -- a clean dependency injection that leaves Experiment 1 wholly
untouched.  Results land in ``Standard/``, mirroring the
Experiment 1 layout one-for-one::

    Standard/
      factor_panel.csv
      quintile/    <factor>/...   + long_short_market_alpha.{csv,png}
      regression/  <factor>/...   + summary.csv + summary_table.png
      factor_correlation/<factor>/...   redundancy of each software factor vs
                                        the Exp1 general market factors

After the Standard run finishes, this driver also runs every sibling
subexperiment (``RD/``, ``Rev & Cost/``, ``Stability/``, ``Skew/``) by invoking
each one's ``main_*.py`` in its own subprocess.  Each subexperiment binds
``factors`` to its own library at import time, so they must not share an
interpreter -- a subprocess per driver keeps them isolated.  (``Cross_val/`` is
run manually *after* the top-factor collection below: its library resolves the
factors to re-test from ``top_factors/top_factors.csv``.)

Cross-subexperiment top-factor collection
-----------------------------------------
Once every subexperiment has written its ``quintile/long_short_market_alpha.csv``,
this driver collects them, ranks every tested factor by the sum of its
full-period and 2016-onward industry-neutral alpha t-stats, and reports the
leaders (see :func:`collect_top_factors`)::

    top_factors/
      all_factors_ranked.csv                  every factor, ranked by alpha t-stat
      top_factors.csv                          the top N (hand-off for Experiment 3)
      top_factors.md                           the top N, with each factor's definition & intuition
      top_factors_long_short_market_alpha.png  the top N in the standard alpha-table format
      quintile_long_short_market_alpha.png     every factor's quintile book (the
                                               quintile counterpart of tertile.py's table)

The ``top_factors.csv`` hand-off is what Experiment 3's ``factor_momentum.py``
consumes.  ``Cross_val/`` is excluded from the ranking: it re-tests factors on the
Banks+Insurance universe, so its alphas are not comparable to a Software &
Services ranking.

Run standalone::

    python main.py                  # full pipeline + subexperiments + collection
    python main.py collect          # only (re)collect the top factors from existing CSVs
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

import sw_factors as S
import driver_utils as D

# --- Dependency injection: make Experiment 1's analysis modules reuse our
#     software-factor library (``driver_utils.wire_engine`` binds ``factors``
#     -> ``sw_factors`` before importing them, so every reference targets the
#     software factors with zero changes to Experiment 1).
quintile, regression = D.wire_engine(S)

UNIVERSE = S.SOFTWARE_SERVICES

# --- Sibling subexperiments.  Each lives in its own subfolder with its own
#     factor library and `main_*.py` driver that does the same ``import factors
#     as F`` dependency injection this file does -- binding ``factors`` to *its*
#     library before importing quintile/regression.  Those module-level bindings
#     are cached per Python process, so the subexperiments cannot share one
#     interpreter without clobbering each other's ``factors`` (the first one
#     imported would win for all).  We therefore run each as its own subprocess,
#     exactly the documented ``python main_*.py`` standalone path.
_THIS_DIR = Path(__file__).resolve().parent
_SUBEXPERIMENTS = [
    _THIS_DIR / "RD" / "main_rd.py",
    _THIS_DIR / "Rev & Cost" / "main_revcost.py",
    _THIS_DIR / "Stability" / "main_stability.py",
    _THIS_DIR / "Skew" / "main_skew.py",
]


def run_subexperiments() -> None:
    """Run every subexperiment driver in its own subprocess, continuing past
    any that fail and reporting the roster at the end."""
    failed: list[str] = []
    for path in _SUBEXPERIMENTS:
        label = f"{path.parent.name}/{path.name}"
        print(f"\n{'#' * 72}\n# Subexperiment: {label}\n{'#' * 72}")
        result = subprocess.run([sys.executable, path.name], cwd=path.parent)
        if result.returncode != 0:
            failed.append(label)
            print(f"!! {label} exited with code {result.returncode}")

    if failed:
        print(f"\n{len(failed)} subexperiment(s) FAILED: {', '.join(failed)}")
    else:
        print(f"\nAll {len(_SUBEXPERIMENTS)} subexperiments completed.")


# --------------------------------------------------------------------------- #
# Cross-subexperiment collection
#
# Every subexperiment writes a ``quintile/long_short_market_alpha.csv`` -- the
# same table Experiment 1 produces, one row per factor, with the project's
# "alpha t-stat" in the ``alpha_tstat`` column (the t-stat of the long/short
# book's industry-neutral alpha).  We concatenate those tables across the
# software-universe subexperiments, rank by the sum of the full-period and
# 2016-onward alpha t-stats, and both (a) render the
# top few in the *same* table format as the per-subexperiment alpha plots and
# (b) write a small ``top_factors.csv`` hand-off that Experiment 3's
# ``factor_momentum.py`` consumes.
#
# Experiment 1's general market factors are *also* tested on the Software &
# Services universe (``experiment1 - general factors/output/software/``), so they
# belong in the same ranking -- they are directly comparable, same cross-section
# and same "alpha t-stat" definition.  They are included as one more source.
#
# ``Cross_val/`` is intentionally excluded: it re-tests software factors on the
# Banks+Insurance universe, so its alphas are not comparable to a Software &
# Services ranking (and its rows would collide on factor name with the software
# copies).
# --------------------------------------------------------------------------- #
_SOFTWARE_SUBEXPERIMENTS = ["Standard", "RD", "Rev & Cost",
                            "Stability", "Skew"]

# Sources contributing to the cross-experiment ranking, each a
# ``quintile/long_short_market_alpha.csv`` on the Software & Services universe:
# Experiment 1's general market factors first, then every Experiment 2 software
# subexperiment.  ``(label, alpha-csv path)`` -- the label tags each row's origin
# in the ranking (the ``subexperiment`` column).
_EXP1_SOFTWARE_ALPHA = (_THIS_DIR.parent / "experiment1 - general factors"
                        / "output" / "software" / "quintile"
                        / "long_short_market_alpha.csv")
_RANKING_SOURCES: list[tuple[str, Path]] = (
    [("General", _EXP1_SOFTWARE_ALPHA)]
    + [(lib, _THIS_DIR / lib / "quintile" / "long_short_market_alpha.csv")
       for lib in _SOFTWARE_SUBEXPERIMENTS])

TOP_FACTORS_DIR = _THIS_DIR / "top_factors"
TOP_N = 5

# --------------------------------------------------------------------------- #
# Factor documentation (for the top_factors explainer)
#
# Definition (how the raw signal is computed) + intuition (why it is expected to
# predict returns) for every factor tested across the software subexperiments,
# keyed by factor name.  Sourced from each factor library's module docstring;
# collected here because the top-factor explainer is a cross-subexperiment report
# -- exactly the role this driver already plays for the ranking.  ``higher`` is
# the bullish prior in words (matches each library's ``higher_is_bullish``).
# :func:`write_top_factors_md` renders the entries for whatever factors land in
# ``top_factors.csv``; a factor with no entry here falls back to its ``family``
# label so a newly added factor still documents (minimally) without a code change.
# --------------------------------------------------------------------------- #
FACTOR_DOC: dict[str, dict[str, str]] = {
    # --- General: Experiment 1 market factors on Software & Services (factors.py) ---
    "earnings_yield": {
        "definition": "Trailing earnings yield: earnings_ltm / market cap.",
        "intuition": "The classic value signal -- cheap stocks (high earnings "
            "yield) are under-priced and outperform richly-valued ones.",
    },
    "momentum_12m": {
        "definition": "Cumulative total return over months [t-12, t-1] "
            "(the 12 months before formation, skipping the most recent month).",
        "intuition": "Past winners keep winning over 3-12 month horizons "
            "(Jegadeesh & Titman 1993) as the market underreacts to news; the "
            "most recent month is excluded to avoid short-term reversal.",
    },
    "reversal_1m": {
        "definition": "The most recent month's total return (long Q1 / short Q5).",
        "intuition": "Last month's losers bounce back and winners give back -- "
            "short-horizon overreaction / liquidity-provision reversal "
            "(Jegadeesh 1990).",
    },
    "gross_profitability": {
        "definition": "Gross profitability: gross_income_ltm / total assets.",
        "intuition": "Gross profits are the cleanest measure of true economic "
            "profitability (Novy-Marx 2013); more-profitable firms earn higher "
            "returns, a quality dimension orthogonal to value.",
    },
    "beta": {
        "definition": "Rolling 36-month market beta = cov(r_i, r_mkt) / "
            "var(r_mkt) (long Q1 / short Q5).",
        "intuition": "The low-risk anomaly -- low-beta stocks deliver higher "
            "risk-adjusted returns than high-beta ones (Frazzini & Pedersen "
            "2014), so the book is long low-beta / short high-beta.",
    },
    "asset_growth": {
        "definition": "Year-over-year growth in total assets "
            "(long Q1 / short Q5).",
        "intuition": "The investment/asset-growth anomaly -- firms that expand "
            "their asset base aggressively subsequently underperform "
            "(Cooper, Gulen & Schill 2008), so the trade is long low-growth.",
    },
    "net_issuance": {
        "definition": "Year-over-year change in diluted shares outstanding "
            "(long Q1 / short Q5).",
        "intuition": "Net share issuance predicts low returns and net "
            "repurchases predict high returns (Pontiff & Woodgate 2008); the "
            "book is long the shrinkers / short the diluters.",
    },
    "sue": {
        "definition": "Standardised unexpected earnings: YoY change in "
            "earnings_ltm divided by its trailing standard deviation.",
        "intuition": "Post-earnings-announcement drift -- prices adjust slowly "
            "to earnings surprises, so firms with large positive surprises "
            "keep drifting up (Bernard & Thomas 1989).",
    },
    "accruals": {
        "definition": "Sloan (1996) accruals: (earnings_ltm - operating_cf_ltm) "
            "/ total assets (long Q1 / short Q5).",
        "intuition": "High accruals mark low earnings quality -- earnings far "
            "above cash flow are less persistent and the market overvalues "
            "them, so the trade is long low-accrual / short high-accrual.",
    },
    # --- Standard: established software factors (sw_factors.py) ---
    "intangible_value": {
        "definition": "Intangible-adjusted book-to-market: (book_value + K_int) / "
            "market cap, where K_int is the knowledge-capital stock built up from "
            "past R&D by perpetual inventory (Peters & Taylor 2017).",
        "intuition": "GAAP expenses R&D, so the reported book value understates the "
            "true equity of R&D-intensive software firms.  Adding the capitalised "
            "R&D stock back restores a cleaner value signal, and cheap (high "
            "book-to-market) intangible-rich firms tend to outperform.",
    },
    "intangible_profitability": {
        "definition": "R&D-adjusted profitability: (operating_income_ltm + R&D) / "
            "(assets + K_int) -- treat R&D as investment (add it back to profit) on "
            "an asset base that recognises the intangible capital it created.",
        "intuition": "Profitability (quality) predicts returns, but for software the "
            "standard ratio is distorted by expensed R&D.  The intangible adjustment "
            "measures the franchise's true economic profitability.",
    },
    "rd_productivity": {
        "definition": "Year-over-year change in LTM sales per unit of knowledge "
            "capital: d(sales_ltm, YoY) / K_int.",
        "intuition": "Measures whether research dollars convert into top-line growth; "
            "firms that efficiently turn their accumulated R&D stock into sales "
            "growth should be rewarded.",
    },
    "buyback_quality": {
        "definition": "Realised share-count reduction minus the gross buyback yield "
            "(-buyback_ltm / market cap): genuine net shrinkage in shares vs. cash "
            "spent repurchasing.",
        "intuition": "Distinguishes real, share-reducing buybacks from repurchases "
            "that merely offset stock-comp dilution.  A firm spending on buybacks "
            "whose share count does not fall scores low (low quality); genuine net "
            "repurchasers score high.",
    },
    "fscore": {
        "definition": "Piotroski (2000) F-score: the sum of 9 binary "
            "fundamental-health tests (profitability, leverage/liquidity, operating "
            "efficiency), 0-9; supplied pre-computed and used verbatim.",
        "intuition": "A higher score marks a financially strengthening firm; this "
            "bundle of accounting-improvement signals predicts higher subsequent "
            "returns, classically strongest among value names.",
    },
    "zscore": {
        "definition": "Altman (1968) Z-score: a weighted sum of five "
            "solvency/profitability ratios; higher = further from financial distress; "
            "supplied pre-computed and used verbatim.",
        "intuition": "Distress risk is mispriced -- financially robust (high-Z) firms "
            "tend to outperform distress-prone ones on a risk-adjusted basis.",
    },
    # --- RD: R&D-behaviour factors (rd_factors.py) ---
    "rd_growth": {
        "definition": "Year-over-year change in R&D spend scaled by average assets: "
            "d(rd_ltm, YoY) / avg assets.",
        "intuition": "Unlike tangible asset growth (which predicts LOW returns), R&D "
            "increases predict HIGH returns (Eberhart, Maxwell & Siddique 2004) "
            "because expensing hides the investment -- the market underreacts to "
            "stepped-up innovation.",
    },
    "rd_conversion": {
        "definition": "Year-over-year change in gross profit per dollar of "
            "prior-year sales: d(gross_income_ltm, YoY) / sales_ltm_{t-12m}.",
        "intuition": "Measures R&D output in PROFIT, not just revenue.  It rewards "
            "margin-accretive innovation rather than low-margin revenue chasing "
            "(the documented failure mode of rd_productivity).",
    },
    "rd_stability": {
        "definition": "Negative trailing-36m coefficient of variation of R&D "
            "intensity (rd_ltm / sales).",
        "intuition": "A second moment -- the consistency of R&D commitment.  Firms "
            "that hold R&D steady (rather than cutting it to manage earnings) signal "
            "durable innovation and earnings quality; orthogonal by construction to "
            "every level signal.",
    },
    "innovation_mix": {
        "definition": "R&D's share of discretionary spend: rd_ltm / (rd_ltm + "
            "sga_ltm).",
        "intuition": "The composition of discretionary spend -- product (R&D) vs. "
            "go-to-market/admin (SG&A).  A product-tilted mix marks a builder of "
            "durable competitive advantage.",
    },
    "rd_intensity": {
        "definition": "R&D / sales (a level).",
        "intuition": "The textbook R&D signal (Chan, Lakonishok & Sougiannis 2001), "
            "kept as the conventional baseline: R&D is undervalued, so high R&D "
            "intensity is associated with future returns.",
    },
    # --- Rev & Cost: revenue/cost-structure factors (revcost_factors.py) ---
    "revenue_stability": {
        "definition": "Negative trailing-36m standard deviation of YoY revenue "
            "growth.",
        "intuition": "Recurring (subscription) revenue is smooth; lumpy license/deal "
            "revenue is not.  The market rewards the level/acceleration of growth and "
            "underweights its durability, so stable top-lines (a quality dimension) "
            "outperform.",
    },
    "deferred_rev_intensity": {
        "definition": "Non-trade operating liabilities (operating_liabilities - "
            "accounts_payable) / sales -- a proxy for the deferred/unearned-revenue "
            "balance (there is no dedicated deferred-revenue field in the data).",
        "intuition": "Deferred revenue is software's hidden forward book of "
            "already-contracted revenue.  Income-statement-anchored investors "
            "undervalue this balance-sheet signal of future revenue.",
    },
    "cost_scalability": {
        "definition": "Growth wedge between revenue and the full operating cost base: "
            "YoY growth(sales) - YoY growth(operating_expenses) (opex includes COGS).",
        "intuition": "A positive wedge is realised operating leverage / margin "
            "expansion; the market anchors on the trailing margin and underreacts to "
            "the inflection.",
    },
    "labor_productivity": {
        "definition": "Year-over-year growth of revenue per employee (sales / "
            "employee_count).",
        "intuition": "The human-capital analogue of operating leverage in an "
            "asset-light, talent-driven industry; rising revenue per head signals "
            "scaling efficiency.",
    },
    "gross_margin": {
        "definition": "gross_income / sales (a level).",
        "intuition": "The single most-watched software KPI, kept as the conventional "
            "baseline; expected to be largely priced within the industry.",
    },
    # --- Stability: second moments of quality (stability_factors.py) ---
    "cashflow_stability": {
        "definition": "Negative trailing-36m coefficient of variation of the "
            "operating-cash-flow margin (operating_cf_ltm / sales_ltm).",
        "intuition": "The consistency of CASH generation -- harder to manage than "
            "accruals and a classic earnings-quality cross-check.  Steady cash "
            "conversion signals real, durable profitability; lumpy conversion flags "
            "accrual- or working-capital-driven earnings.",
    },
    "return_stability": {
        "definition": "Negative trailing-12m robust coefficient of variation "
            "(std / |mean|) of the monthly total return.",
        "intuition": "A smooth, low-volatility return stream -- the same "
            "direction as the low-volatility / high-Sharpe anomaly: erratic "
            "(high-CoV) names underperform their steadier peers risk-adjusted.",
    },
    "gross_profitability_stability": {
        "definition": "Negative trailing-12m coefficient of variation of gross "
            "profitability (gross_income_ltm / assets).",
        "intuition": "The consistency of the profitability engine itself -- a "
            "firm that earns its gross profits steadily has a more durable "
            "franchise than one whose profitability swings.",
    },
    "tangible_capital_stability": {
        "definition": "Negative trailing-12m coefficient of variation of the "
            "tangible-capital ratio net PPE / invested_capital.",
        "intuition": "How steadily the firm's capital structure leans on physical, "
            "tangible assets.  A settled asset base scores high; one being rebuilt "
            "through acquisitions, divestitures or capex bursts scores low.",
    },
    "operating_income_growth_stability": {
        "definition": "Negative trailing-36m coefficient of variation of the "
            "year-over-year change in operating income (operating_income_ltm_t - "
            "operating_income_ltm_{t-12}).",
        "intuition": "How steadily the earnings engine GROWS.  A consistent annual "
            "earnings step signals a durable franchise; year-over-year swings that "
            "lurch around signal an unpredictable one.",
    },
    "investing_cf_growth_stability": {
        "definition": "Negative trailing-36m coefficient of variation of the "
            "year-over-year change in investing cash flow, proxied by fixed capex "
            "(capex_fix_ltm_t - capex_fix_ltm_{t-12}).",
        "intuition": "The consistency of the firm's investment cadence -- steady, "
            "programmatic capex growth versus lumpy, opportunistic bursts.",
    },
    "tangible_asset_growth_stability": {
        "definition": "Negative trailing-36m coefficient of variation of the "
            "year-over-year change in tangible assets (net PPE_t - net PPE_{t-12}).",
        "intuition": "How EVENLY the tangible asset base is built up rather than its "
            "level.  The asset-growth / investment anomaly flags erratic, spiky asset "
            "growth as a low-return signal, so a steady builder is the long leg.",
    },
    # --- Skew: third moments / lottery demand (skew_factors.py) ---
    "return_skewness": {
        "definition": "Trailing-36m sample skewness of the stock's monthly total "
            "return.",
        "intuition": "Investors over-pay for positive-skew 'lottery' payoffs (Boyer, "
            "Mitton & Vorkink 2010; Bali, Cakici & Whitelaw 2011), so high positive "
            "skew predicts LOW subsequent returns -- the trade longs negatively skewed "
            "and shorts positively skewed names.",
    },
    "revenue_growth_skewness": {
        "definition": "Trailing-36m sample skewness of YoY revenue (sales_ltm) "
            "growth.",
        "intuition": "A fundamental 'lottery': positively skewed growth (occasional "
            "spectacular spurts around a modest trend) is bid up and underperforms, "
            "while collapse-prone (negative-skew) names earn a premium.",
    },
    "eps_skewness": {
        "definition": "Trailing-36m sample skewness of diluted EPS (earnings_ltm / "
            "diluted_shares).",
        "intuition": "The bottom-line lottery: a positively skewed earnings path (rare "
            "blow-out quarters around a flat base) is the signature of a lottery stock "
            "that subsequently underperforms.",
    },
}


def rank_factors() -> pd.DataFrame:
    """Concatenate every Software & Services factor source's long/short alpha
    table -- Experiment 1's general market factors plus every Experiment 2
    software subexperiment -- and return one frame ranked by the *sum* of the
    full-period and 2016-onward industry-neutral alpha t-stats (descending).

    Ranking on ``alpha_tstat + alpha_tstat_2016`` rewards factors that are
    significant over the full sample *and* hold up in the recent (2016-onward)
    regime, rather than full-period significance alone.  The combined score is
    persisted in the ``alpha_tstat_combined`` column.

    Each row keeps its source (``subexperiment`` column) and a 1-based ``rank``;
    sources whose alpha table is missing are skipped with a note (so a partial
    run still ranks what exists).
    """
    tables = []
    for label, alpha_csv in _RANKING_SOURCES:
        if not alpha_csv.exists():
            print(f"  (skip {label}: {alpha_csv.name} missing -- run its driver first)")
            continue
        tbl = pd.read_csv(alpha_csv)
        tbl.insert(0, "subexperiment", label)
        tables.append(tbl)
    if not tables:
        raise FileNotFoundError(
            "No long_short_market_alpha.csv found; run Experiment 1 and the "
            "Experiment 2 subexperiments first (python main.py).")

    ranked = pd.concat(tables, ignore_index=True)
    ranked["alpha_tstat_combined"] = (
        ranked["alpha_tstat"] + ranked["alpha_tstat_2016"])
    ranked = (ranked
                .sort_values("alpha_tstat_combined", ascending=False, kind="stable")
                .reset_index(drop=True))
    ranked.insert(0, "rank", ranked.index + 1)
    return ranked


def write_top_factors_md(top: pd.DataFrame, path: Path) -> None:
    """Write a human-readable explainer of the top factors in rank order: each
    one's **definition** (how the raw signal is computed) and **intuition** (why
    it is expected to predict returns), alongside its realised direction/alpha.

    Definition + intuition come from :data:`FACTOR_DOC`; the numbers
    (direction, alpha, t-stat) come from the ranking.  A factor with no
    ``FACTOR_DOC`` entry still gets a stub with its ``family`` label, so a newly
    added factor documents (minimally) without requiring this file to change.
    """
    lines = [
        "# Top factors -- definition & intuition",
        "",
        f"The top {len(top)} factors across the Software & Services factor "
        "experiments -- Experiment 1's general market factors plus Experiment 2's "
        "software subexperiments -- ranked by the sum of the full-period and "
        "2016-onward industry-neutral long/short alpha t-stats (see "
        "`top_factors.csv` "
        "and `all_factors_ranked.csv`).  For each factor below: its **definition** "
        "(how the raw signal is computed from fundamentals / prices) and its "
        "**intuition** (why it is expected to predict returns).  All signals are "
        "z-scored cross-sectionally against the industry mean before sorting.",
        "",
    ]
    for r in top.itertuples():
        long_leg = ("long the top quintile (Q5), short the bottom (Q1)"
                    if r.direction == "Q5-Q1"
                    else "long the bottom quintile (Q1), short the top (Q5)")
        lines += [
            f"## {r.rank}. {r.family}",
            "",
            f"- **Factor:** `{r.factor}` &nbsp; · &nbsp; **Subexperiment:** {r.subexperiment}",
            f"- **Long/short book:** {r.direction} ({long_leg})",
            f"- **Industry-neutral alpha:** {r.alpha:+.4%} / month &nbsp; · &nbsp; "
            f"t(alpha) = {r.alpha_tstat:+.2f}",
            "",
        ]
        doc = FACTOR_DOC.get(r.factor)
        if doc:
            lines += [f"**Definition.** {doc['definition']}", "",
                      f"**Intuition.** {doc['intuition']}", ""]
        else:
            lines += [f"_No detailed write-up registered for `{r.factor}` "
                      f"(family: {r.family})._", ""]

    path.write_text("\n".join(lines), encoding="utf-8")


def collect_top_factors(top_n: int = TOP_N) -> pd.DataFrame:
    """Rank every tested factor, persist the full ranking plus a top-``top_n``
    hand-off CSV, and render the top ``top_n`` long/short alpha table in the
    standard plot format.  Returns the top-``top_n`` rows.
    """
    ranked = rank_factors()
    TOP_FACTORS_DIR.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(TOP_FACTORS_DIR / "all_factors_ranked.csv", index=False)

    top = ranked.head(top_n).copy()
    handoff_cols = ["rank", "factor", "family", "subexperiment",
                    "direction", "alpha", "alpha_tstat", "alpha_tstat_2016",
                    "alpha_tstat_combined"]
    top[handoff_cols].to_csv(TOP_FACTORS_DIR / "top_factors.csv", index=False)

    # Human-readable explainer: each top factor's definition + intuition.
    write_top_factors_md(top, TOP_FACTORS_DIR / "top_factors.md")

    # Plot in the SAME format as each subexperiment's long_short_market_alpha.png
    # (Experiment 1's render_alpha_table), tagging each family with its source
    # subexperiment for provenance.
    plot_rows = top.copy()
    plot_rows["family"] = plot_rows["family"] + "  [" + plot_rows["subexperiment"] + "]"
    regression.render_alpha_table(
        plot_rows, TOP_FACTORS_DIR / "top_factors_long_short_market_alpha.png")

    # Also plot EVERY factor's quintile book in the same format -- the quintile
    # counterpart of tertile.py's all-factor table.
    plot_all = ranked.copy()
    plot_all["family"] = plot_all["family"] + "  [" + plot_all["subexperiment"] + "]"
    regression.render_alpha_table(
        plot_all, TOP_FACTORS_DIR / "quintile_long_short_market_alpha.png")

    print(f"\n=== Top {top_n} factors across subexperiments "
          f"(by full-period + 2016-onward alpha t-stat) ===")
    for r in top.itertuples():
        print(f"  {r.rank}. {r.factor:<24} [{r.subexperiment:<10}] "
              f"alpha={r.alpha:+.4%}/mo  t(alpha)={r.alpha_tstat:+.2f}  "
              f"t(alpha,2016+)={r.alpha_tstat_2016:+.2f}  "
              f"t(sum)={r.alpha_tstat_combined:+.2f}")
    print(f"Saved -> {TOP_FACTORS_DIR}")
    return top


def main() -> None:
    D.run_pipeline(S, UNIVERSE, "software factor", quintile, regression,
                   done_suffix=" (Standard)")

    # Run the remaining subexperiments (each in its own process, see above).
    run_subexperiments()

    # Collect every subexperiment's factors and report/plot the leaders.
    print(f"\n{'#' * 72}\n# Collecting top factors across subexperiments\n{'#' * 72}")
    collect_top_factors()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in {"collect", "--collect-only"}:
        collect_top_factors()
    else:
        main()
