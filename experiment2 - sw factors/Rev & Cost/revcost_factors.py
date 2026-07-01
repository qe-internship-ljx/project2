"""
revcost_factors.py
==================

Compute a set of **revenue- and cost-structure factors** for the GICS *Software &
Services* universe, standardise each cross-sectionally (z-score vs the industry
mean), and write a tidy monthly panel to ``Rev & Cost/factor_panel.csv``.

This module, its driver (``main_revcost.py``) and its documentation
(``Theory & hypothesis.md`` / ``results.md``) all live in experiment2's
``Rev & Cost/`` subfolder, and it writes its analysis outputs there too.  It is a
*third* factor library for Experiment 2, alongside ``sw_factors.py`` (``standard/``)
and ``rd_factors.py`` (``RD/``).  Like both it is a **drop-in for Experiment 1's
analysis engine**: the quintile sorts, cross-sectional (Fama-MacBeth) regressions,
long/short books, trading-cost model and every plot are reused **verbatim** from
``experiment1 - general factors/{quintile,regression,cost}.py`` via the shared
engine in ``experiment1 - general factors/factors.py``.  Only the *factor
definitions* and the fundamentals they need are new here, and the universe's
``output_dir`` is the ``Rev & Cost/`` folder itself, so these factors never collide
with the established software factors (``standard/``) or the R&D factors (``RD/``).

Motivation
----------
Where ``sw_factors.py`` and ``rd_factors.py`` target the *investment* side of
software (the R&D stock and the discipline of R&D spending), this module targets
the **flow of revenue and the structure of cost** -- the dimensions that are most
distinctive to software economics and least captured by the standard factor zoo:

    name                    definition                                           dir          dimension
    ----------------------  ---------------------------------------------------  ----------   ----------------------
    revenue_stability       - trailing 36m std of YoY revenue growth             long high    revenue durability (2nd moment)
    deferred_rev_intensity  (operating_liabilities - accounts_payable) / sales   long high    subscription / billings model
    cost_scalability        YoY growth(sales) - YoY growth(operating_expenses)   long high    realised operating leverage
    labor_productivity      YoY growth of (sales / employee_count)               long high    human-capital leverage
    gross_margin            gross_income / sales                                 long high    near-zero marginal cost (baseline)

The first four target genuinely *non-standard* revenue/cost dimensions; ``gross_margin``
(a level) is retained only as the **conventional baseline** -- the single most-watched
software KPI -- against which the four novel signals are contrasted.

* ``revenue_stability`` is a *second moment* (consistency of the top line), so it is
  orthogonal by construction to every level signal in the project.  Recurring
  (subscription) revenue is smooth; lumpy license/deal revenue is not.  The market
  rewards the *level/acceleration* of growth and under-weights its *durability*
  (Asness-Frazzini-Pedersen "Quality minus Junk"; Novy-Marx earnings stability).
* ``deferred_rev_intensity`` proxies the contract-liability (deferred/unearned revenue)
  balance, which is software's hidden forward book of already-contracted revenue.  There
  is **no dedicated deferred-revenue field** in the data, so we proxy it with non-trade
  operating liabilities (``operating_liabilities - accounts_payable``), dominated by
  deferred revenue + accrued compensation for software (positive for 98.6% of the
  universe, median 0.36x sales).  Income-statement-anchored investors under-value it.
* ``cost_scalability`` is the *growth wedge* between revenue and the full operating
  cost base (``operating_expenses_ltm`` includes COGS -- verified in-data:
  ``sales - operating_expenses == operating_income`` for 95.5% of software firms).  A
  positive wedge is realised operating leverage / margin expansion; the market anchors
  on the trailing margin and underreacts to the inflection.
* ``labor_productivity`` is the YoY growth of revenue per employee -- the human-capital
  analogue of operating leverage in an asset-light, talent-driven industry.  We use the
  *growth* (currency-neutral within a firm), not the *level* (reported in local currency
  and therefore not comparable across the cross-section).
* ``gross_margin`` (gross income / sales) is the textbook software cost-structure ratio,
  kept as a conventional anchor; it is expected to be largely priced within the industry.

All signals are ratios, growth rates or growth-differences and are therefore
currency-neutral, so no FX conversion is required for within-industry comparison --
consistent with the rest of the project.

Run standalone to (re)build the panel::

    python revcost_factors.py

To run the full pipeline (panel + quintile sorts + regressions + redundancy) use::

    python main_revcost.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Shared engine (Experiment 1) -- loaded by path under a private module name
# (its folder name contains spaces, and we must not shadow the name ``factors``,
# which the analysis modules bind to and the driver points at *this* library).
# --------------------------------------------------------------------------- #
_EXP1_DIR = Path(__file__).resolve().parent.parent.parent / "experiment1 - general factors"


def _load_engine():
    spec = importlib.util.spec_from_file_location(
        "revcost_factor_engine", _EXP1_DIR / "factors.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_engine = _load_engine()

# Re-export the generic helpers the analysis modules (and our own build) reuse,
# so this module satisfies the exact interface ``quintile.py`` / ``regression.py``
# / ``cost.py`` expect from their ``import factors as F``.
Universe = _engine.Universe
DATA_DIR = _engine.DATA_DIR
WINSOR_PCT = _engine.WINSOR_PCT
load_universe = _engine.load_universe
load_prices = _engine.load_prices
attach_pit_fundamentals = _engine.attach_pit_fundamentals   # point-in-time as-of merge on observation_date
attach_usd_market_cap = _engine.attach_usd_market_cap       # month-end USD market cap (cross-sectional weight)
cap_weighted_market_return = _engine.cap_weighted_market_return  # cap-weighted within-industry "market" return
weighted_group_mean = _engine.weighted_group_mean          # used by the injected regression.industry_monthly_return
winsorize_cross_section = _engine.winsorize_cross_section
cross_sectional_zscore = _engine.cross_sectional_zscore
add_next_return = _engine.add_next_return
apply_mcap_screen = _engine.apply_mcap_screen      # generic: point-in-time min-mcap screen
assign_quintiles = _engine.assign_quintiles
prepare_slice = _engine.prepare_slice
ols = _engine.ols


# --------------------------------------------------------------------------- #
# Paths & configuration
# --------------------------------------------------------------------------- #
# This module lives in (and writes to) experiment2's "Rev & Cost/" subfolder, so
# these factors and their outputs live in their own subtree and never collide with
# standard/ (sw_factors) or RD/ (rd_factors).  OUTPUT_DIR is the "Rev & Cost/" folder
# itself (the directory holding this file), so the analysis outputs land directly
# under it (Rev & Cost/{factor_panel.csv, quintile/, regression/, ...}).
OUTPUT_DIR = Path(__file__).resolve().parent

INDUSTRY_GROUP = "Software & Services"

# Estimation windows / parameters.
YOY_LAG = 12                # year-over-year lag (months) for growth signals
STAB_WINDOW = 36            # trailing months for the revenue-growth stability moment
STAB_MIN_PERIODS = 24       # require >=2y of history before a stability score exists

# Ordered list of the factors produced by this module.  ``higher_is_bullish`` is
# the academically expected sign of the long leg (top quintile): True => the
# canonical trade is long Q5 / short Q1.  With ``USE_CANONICAL_LS_DIRECTION``
# below set True the long/short book is signed by this prior (NOT by the
# in-sample t-stat), so a negative realised alpha t-stat means the factor worked
# *against* the hypothesis -- exactly what we want for hypothesis testing.
FACTORS: dict[str, dict] = {
    "revenue_stability":      {"family": "Revenue stability (recurring-revenue durability)", "higher_is_bullish": True},
    "deferred_rev_intensity": {"family": "Deferred-revenue intensity (subscription model)",  "higher_is_bullish": True},
    "cost_scalability":       {"family": "Cost scalability (operating leverage)",            "higher_is_bullish": True},
    "labor_productivity":     {"family": "Labor productivity growth (human capital)",        "higher_is_bullish": True},
    "gross_margin":           {"family": "Gross margin (level; baseline)",                   "higher_is_bullish": True},
}
FACTOR_NAMES = list(FACTORS)

# Sign the directional long/short book by each factor's canonical prior
# (``higher_is_bullish``) rather than the in-sample Fama-MacBeth t-stat, so the
# realised result can be compared against the hypothesised direction.
USE_CANONICAL_LS_DIRECTION = True


# --------------------------------------------------------------------------- #
# Universe -- same cross-section as Experiment 2, separate output subtree
# --------------------------------------------------------------------------- #
SOFTWARE_SERVICES = Universe(
    slug="software_services",
    price_file="price_software_services.feather",
    output_dir=OUTPUT_DIR,
    industry_group=INDUSTRY_GROUP,
    min_mcap_usd=0.1e9,   # point-in-time screen: hold only names >= $0.1B at formation
)

UNIVERSES: dict[str, Universe] = {SOFTWARE_SERVICES.slug: SOFTWARE_SERVICES}


def universe_from_argv(default: Universe = SOFTWARE_SERVICES) -> Universe:
    """Pick a universe from argv[1] (its slug); fall back to ``default``."""
    if len(sys.argv) > 1:
        return UNIVERSES[sys.argv[1]]
    return default


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_fundamentals(universe: pd.Index) -> pd.DataFrame:
    """
    Point-in-time monthly fundamentals carrying everything the revenue/cost
    factors need: revenue, gross income, total operating expense, operating
    liabilities, accounts payable and employee count -- all in the *base*
    ``fundamental_master`` table (no extended-table merge required).

    Each record carries ``observation_date`` (when the report became
    observable); we align on it in :func:`build_monthly_panel` via the shared
    :func:`attach_pit_fundamentals` to avoid look-ahead -- identical convention
    to Experiment 1 / ``sw_factors.py`` / ``rd_factors.py``.
    """
    cols = ["date_fundamental", "observation_date", "stock_id",
            "sales_ltm", "gross_income_ltm", "operating_expenses_ltm",
            "operating_liabilities", "accounts_payable", "employee_count"]
    fm = pd.read_feather(DATA_DIR / "fundamental_master.feather", columns=cols)
    fm["stock_id"] = fm["stock_id"].astype(str)
    fm = fm[fm["stock_id"].isin(universe)].copy()
    fm["date_fundamental"] = pd.to_datetime(fm["date_fundamental"])
    fm["observation_date"] = pd.to_datetime(fm["observation_date"])
    return fm


# --------------------------------------------------------------------------- #
# Monthly panel construction (mirrors the engine / sw_factors.py exactly)
# --------------------------------------------------------------------------- #
def build_monthly_panel(universe: pd.Index | None = None,
                        u: Universe = SOFTWARE_SERVICES) -> pd.DataFrame:
    """
    Assemble a (stock_id, period) monthly panel: monthly total return, month-end
    market cap (local and USD), the market-cap-weighted universe ("market")
    return, and the point-in-time revenue/cost fundamentals.  The price->monthly
    aggregation mirrors Experiment 1 so the experiments share an identical return
    definition and month-end alignment.
    """
    if universe is None:
        universe = load_universe(u)

    px = load_prices(universe, u)
    px["period"] = px["date"].dt.to_period("M")
    px["gross"] = 1.0 + px["total_return"].fillna(0.0)

    monthly = (px.groupby(["stock_id", "period"], observed=True)
                 .agg(mret=("gross", "prod"),
                      security_mcap_local=("security_mcap_local", "last"),
                      price_local=("price_local", "last"),
                      n_days=("date", "size"))
                 .reset_index())
    monthly["mret"] = monthly["mret"] - 1.0

    # USD market cap (the cross-sectional weight) and the market-cap-weighted
    # universe return = within-industry "market" proxy (prior month-end weights).
    monthly = attach_usd_market_cap(monthly)
    monthly["mkt_ret"] = monthly["period"].map(cap_weighted_market_return(monthly))

    fund = load_fundamentals(universe)
    monthly = attach_pit_fundamentals(monthly, fund)

    monthly = monthly.sort_values(["stock_id", "period"]).reset_index(drop=True)
    return monthly


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
def _yoy_growth(p: pd.DataFrame, s: pd.Series) -> pd.Series:
    """Year-over-year growth (level_t / level_{t-12m} - 1) of a series, per stock.

    The prior-year level is guarded to be strictly positive (``.where(prev > 0)``)
    so the growth rate is well-defined and not dominated by sign flips / tiny
    denominators; otherwise the observation is left missing."""
    prev = s.groupby(p["stock_id"], observed=True).shift(YOY_LAG)
    return s / prev.where(prev > 0.0) - 1.0


# --------------------------------------------------------------------------- #
# Factor definitions
# --------------------------------------------------------------------------- #
def _f_revenue_stability(p: pd.DataFrame) -> pd.Series:
    """
    Recurring-revenue durability: the NEGATIVE trailing-36m standard deviation of
    YoY revenue growth, per stock.  High (near 0) => a smooth, predictable,
    recurring (subscription-like) top line; low (very negative) => lumpy
    license/deal revenue with renewal/air-pocket risk.  A second moment, so
    orthogonal by construction to every level signal in the project.  We negate so
    that higher = more stable = the bullish (long) leg.
    """
    g = _yoy_growth(p, p["sales_ltm"].astype(float))
    g = g.replace([np.inf, -np.inf], np.nan)
    std = (g.groupby(p["stock_id"], observed=True)
            .transform(lambda s: s.rolling(STAB_WINDOW, min_periods=STAB_MIN_PERIODS).std()))
    return -std


def _f_deferred_rev_intensity(p: pd.DataFrame) -> pd.Series:
    """
    Subscription / billings model: non-trade operating liabilities
    (``operating_liabilities - accounts_payable``, a deferred/unearned-revenue +
    accrued-comp proxy) as a fraction of annual revenue.  High => a heavily
    pre-billed, recurring, customer-funded business carrying a large forward book
    of already-contracted revenue; low => billing in arrears / recognise-as-sold.
    (No dedicated deferred-revenue field exists in the data, hence the proxy.)
    """
    proxy = p["operating_liabilities"].astype(float) - p["accounts_payable"].astype(float)
    sales = p["sales_ltm"].astype(float)
    return proxy / sales.where(sales > 0.0)


def _f_cost_scalability(p: pd.DataFrame) -> pd.Series:
    """
    Realised operating leverage: YoY growth in sales minus YoY growth in the full
    operating cost base.  ``operating_expenses_ltm`` is the total operating cost
    (it includes COGS -- verified: sales - operating_expenses == operating_income
    for 95.5% of software firms), so a positive value means revenue is outgrowing
    the entire cost base and operating margin is expanding (the firm is climbing
    its operating-leverage curve); negative means costs are outrunning revenue.
    """
    g_sales = _yoy_growth(p, p["sales_ltm"].astype(float))
    g_cost = _yoy_growth(p, p["operating_expenses_ltm"].astype(float))
    return g_sales - g_cost


def _f_labor_productivity(p: pd.DataFrame) -> pd.Series:
    """
    Human-capital leverage: YoY growth of revenue per employee.  We use the
    *growth* of sales/headcount (currency-neutral within a firm), not the level
    (reported in local currency and therefore not comparable across the cross-
    section).  Positive => the firm is monetising faster than it is hiring
    (automation, product leverage, upsell); negative => headcount is outrunning
    revenue (services drag, hiring ahead of monetisation).
    """
    sales = p["sales_ltm"].astype(float)
    emp = p["employee_count"].astype(float)
    rpe = sales / emp.where(emp > 0.0)
    return _yoy_growth(p, rpe)


def _f_gross_margin(p: pd.DataFrame) -> pd.Series:
    """Gross margin = gross income / sales.  The textbook software cost-structure
    ratio (near-zero marginal cost => high gross margin), kept as the conventional
    baseline against which the four novel signals are contrasted."""
    sales = p["sales_ltm"].astype(float)
    return p["gross_income_ltm"].astype(float) / sales.where(sales > 0.0)


_FACTOR_FUNCS = {
    "revenue_stability": _f_revenue_stability,
    "deferred_rev_intensity": _f_deferred_rev_intensity,
    "cost_scalability": _f_cost_scalability,
    "labor_productivity": _f_labor_productivity,
    "gross_margin": _f_gross_margin,
}


def compute_factors(panel: pd.DataFrame) -> pd.DataFrame:
    """Add a raw value column for every factor in :data:`FACTOR_NAMES`."""
    for name in FACTOR_NAMES:
        panel[name] = _FACTOR_FUNCS[name](panel).replace([np.inf, -np.inf], np.nan)
    return panel


# --------------------------------------------------------------------------- #
# Cross-sectional standardisation & tidy output (mirror the engine's versions)
# --------------------------------------------------------------------------- #
def add_zscores(panel: pd.DataFrame) -> pd.DataFrame:
    for name in FACTOR_NAMES:
        panel[f"{name}_z"] = cross_sectional_zscore(panel[name], panel["period"])
    return panel


def to_long_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Reshape to one row per (date, stock_id, factor): value, z-score, next
    return, and the market-cap ``weight`` (month-end USD market cap).  Rows with a
    missing factor value are dropped.  Identical schema to Experiment 1 / 2."""
    panel = panel.copy()
    panel["date"] = panel["period"].dt.to_timestamp(how="end").dt.normalize()
    frames = []
    for name in FACTOR_NAMES:
        f = panel[["date", "stock_id", name, f"{name}_z",
                   "next_return", "security_mcap_usd"]].copy()
        f.columns = ["date", "stock_id", "value", "zscore", "next_return", "weight"]
        f.insert(2, "factor", name)
        frames.append(f.dropna(subset=["value"]))
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["factor", "date", "stock_id"]).reset_index(drop=True)


def build(save: bool = True, u: Universe = SOFTWARE_SERVICES) -> pd.DataFrame:
    """Full build: panel -> factors -> next return -> mcap screen -> z-scores -> tidy long."""
    panel = build_monthly_panel(u=u)
    panel = compute_factors(panel)
    panel = add_next_return(panel)
    panel = apply_mcap_screen(panel, u)
    panel = add_zscores(panel)
    long = to_long_panel(panel)
    if save:
        u.output_dir.mkdir(parents=True, exist_ok=True)
        long.to_csv(u.panel_path, index=False)
    return long


def load_panel(rebuild: bool = False,
               u: Universe = SOFTWARE_SERVICES) -> pd.DataFrame:
    """Load the tidy factor panel, building it first if needed (engine contract)."""
    if rebuild or not u.panel_path.exists():
        return build(save=True, u=u)
    df = pd.read_csv(u.panel_path, parse_dates=["date"])
    df["stock_id"] = df["stock_id"].astype(str)
    return df


if __name__ == "__main__":
    u = universe_from_argv()
    long = build(save=True, u=u)
    n_months = long["date"].nunique()
    n_stocks = long["stock_id"].nunique()
    print(f"Built Rev & Cost factor panel: {len(long):,} rows | "
          f"{n_stocks} stocks | {n_months} months "
          f"({long['date'].min():%Y-%m} .. {long['date'].max():%Y-%m})")
    print(f"Saved -> {u.panel_path}")
    counts = (long.groupby("factor")["value"].size().reindex(FACTOR_NAMES))
    print("\nObservations per factor:")
    for name, c in counts.items():
        print(f"  {name:<26} {c:>9,}")
