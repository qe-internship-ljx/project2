"""
crossval_factors.py
===================

**Cross-validation** of two stability signals -- ``rd_stability`` (from the
``RD/`` extension) and ``revenue_stability`` (from the ``Rev & Cost/``
extension) -- on a *different* universe: the **Banks + Insurance** cross-section
(``gics_industry_name in {'Banks', 'Insurance'}``).

Both factors were discovered and validated on GICS *Software & Services*, where
each earned a strongly positive industry-neutral alpha (see ``MEMORY``: rd_stability
alpha t ~+3.9 full, revenue_stability ~+4.4 full).  This module asks the
out-of-sample-universe question: **do the same two stability constructs survive
in a structurally unrelated industry** (financials), or were they software-specific?

This is a *sixth-style* factor library for Experiment 2 (alongside
``sw_factors.py`` -> ``Standard/``, ``rd_factors.py`` -> ``RD/``,
``revcost_factors.py`` -> ``Rev & Cost/``, ``stability_factors.py`` ->
``Stability/``).  Like the others it is a **drop-in for Experiment 1's analysis
engine**: the quintile sorts, cross-sectional (Fama-MacBeth) regressions,
long/short books, trading-cost model and every plot are reused **verbatim** from
``experiment1 - general factors/{quintile,regression,cost}.py`` via the shared
engine in ``experiment1 - general factors/factors.py``.  The *only* things that
change here relative to ``Stability/`` are (a) the two factor definitions, copied
**verbatim** from their source libraries so the cross-validation tests the exact
same construct, and (b) the **universe**: this library runs on
``BANKS_INSURANCE`` (the same universe Experiment 1 defines), writing to the
``Cross_val/`` folder.

Factor definitions (identical to their source libraries)
--------------------------------------------------------
    name               definition                                                source
    -----------------  --------------------------------------------------------  ------------------
    rd_stability       - trailing 36m coeff. of variation of (rd_ltm / sales)    RD/rd_factors.py
    revenue_stability  - trailing 36m std of YoY revenue growth                  Rev & Cost/revcost_factors.py

Both are negative trailing-36m second moments (high = steady = the bullish long
leg), currency-neutral (ratios / growth rates of same-currency line items), and
both carry ``higher_is_bullish = True`` with ``USE_CANONICAL_LS_DIRECTION`` so the
realised long/short book is signed by the software-discovered prior -- a negative
alpha t-stat here therefore means the signal failed to *generalise* to financials.

Coverage caveat
---------------
``rd_stability`` depends on a meaningful, sustained R&D programme (its CoV is
undefined when trailing mean R&D/sales < ``MIN_MEAN_INTENSITY``).  Banks and
insurers overwhelmingly report *no* R&D line, so ``rd_stability`` is expected to
have very thin coverage on this universe; ``revenue_stability`` (which needs only
a sales history) should cover the bulk of the cross-section.  The realised
coverage is printed by ``main_crossval.py`` and is itself part of the finding.

Run standalone to (re)build the panel::

    python crossval_factors.py

To run the full pipeline (panel + quintile sorts + regressions + redundancy)::

    python main_crossval.py
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
        "crossval_factor_engine", _EXP1_DIR / "factors.py")
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
assign_quintiles = _engine.assign_quintiles
prepare_slice = _engine.prepare_slice
ols = _engine.ols


# --------------------------------------------------------------------------- #
# Paths & configuration
# --------------------------------------------------------------------------- #
# This module lives in (and writes to) experiment2's Cross_val/ subfolder, so the
# cross-validation outputs land in their own subtree (Cross_val/{factor_panel.csv,
# quintile/, regression/, factor_correlation/}) and never collide with the
# software-universe libraries.
OUTPUT_DIR = Path(__file__).resolve().parent

# Estimation windows / parameters -- copied verbatim from the two source
# libraries so the cross-validation tests the exact same constructs.
YOY_LAG = 12                # year-over-year lag (months) for revenue growth (revenue_stability)
STAB_WINDOW = 36            # trailing months for both coefficient-of-variation moments
STAB_MIN_PERIODS = 24       # require >=2y of history before a stability score exists
MIN_MEAN_INTENSITY = 0.005  # floor on trailing mean R&D/sales below which the
                            # R&D-intensity CoV is undefined (R&D ~ 0) -- RD/rd_factors.py

# Ordered list of the factors produced by this module.  ``higher_is_bullish`` is
# the academically expected sign of the long leg (top quintile): True => the
# canonical trade is long Q5 / short Q1.  With ``USE_CANONICAL_LS_DIRECTION``
# below set True the long/short book is signed by this prior (NOT by the
# in-sample t-stat), so a negative realised alpha t-stat means the factor failed
# to generalise to this universe -- exactly what we want for cross-validation.
FACTORS: dict[str, dict] = {
    "rd_stability":      {"family": "R&D stability (commitment consistency) [from RD/]",        "higher_is_bullish": True},
    "revenue_stability": {"family": "Revenue stability (recurring-revenue durability) [from Rev & Cost/]", "higher_is_bullish": True},
}
FACTOR_NAMES = list(FACTORS)

# Sign the directional long/short book by each factor's canonical (software-
# discovered) prior rather than the in-sample Fama-MacBeth t-stat, so the realised
# result on this universe can be read directly against the hypothesised direction.
USE_CANONICAL_LS_DIRECTION = True


# --------------------------------------------------------------------------- #
# Universe -- Banks + Insurance (gics_industry_name), the cross-validation target.
# Mirrors Experiment 1's BANKS_INSURANCE universe exactly (same price feather,
# same industry list) but writes to this Cross_val/ folder.
# --------------------------------------------------------------------------- #
BANKS_INSURANCE = Universe(
    slug="banks_insurance",
    price_file="price_banks_insurance.feather",
    output_dir=OUTPUT_DIR,
    industries=("Banks", "Insurance"),
)

# Experiment 1's analysis modules (quintile.py / regression.py / cost.py) bind the
# symbol ``F.SOFTWARE_SERVICES`` as their default-universe argument, evaluated at
# import time.  We always pass ``u=BANKS_INSURANCE`` explicitly, so this alias is
# only here to satisfy that default-argument binding -- it points at this library's
# real (banks_insurance) universe so even an un-passed default would be correct.
SOFTWARE_SERVICES = BANKS_INSURANCE

UNIVERSES: dict[str, Universe] = {BANKS_INSURANCE.slug: BANKS_INSURANCE}


def universe_from_argv(default: Universe = BANKS_INSURANCE) -> Universe:
    """Pick a universe from argv[1] (its slug); fall back to ``default``."""
    if len(sys.argv) > 1:
        return UNIVERSES[sys.argv[1]]
    return default


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_fundamentals(universe: pd.Index) -> pd.DataFrame:
    """
    Point-in-time monthly fundamentals carrying everything the two stability
    factors need: R&D expense and sales (``rd_stability`` = CoV of R&D/sales) and
    sales again (``revenue_stability`` = std of YoY sales growth).  Both live in
    the *base* ``fundamental_master`` table (no extended-table merge required).

    Each record carries ``observation_date`` (when the report became
    observable); we align on it in :func:`build_monthly_panel` via the shared
    :func:`attach_pit_fundamentals` to avoid look-ahead -- identical convention
    to Experiment 1 / the other Experiment 2 libraries.
    """
    cols = ["date_fundamental", "observation_date", "stock_id",
            "rd_ltm", "sales_ltm"]
    fm = pd.read_feather(DATA_DIR / "fundamental_master.feather", columns=cols)
    fm["stock_id"] = fm["stock_id"].astype(str)
    fm = fm[fm["stock_id"].isin(universe)].copy()
    fm["date_fundamental"] = pd.to_datetime(fm["date_fundamental"])
    fm["observation_date"] = pd.to_datetime(fm["observation_date"])
    return fm


# --------------------------------------------------------------------------- #
# Monthly panel construction (mirrors the engine / the other libraries exactly)
# --------------------------------------------------------------------------- #
def build_monthly_panel(universe: pd.Index | None = None,
                        u: Universe = BANKS_INSURANCE) -> pd.DataFrame:
    """
    Assemble a (stock_id, period) monthly panel: monthly total return, month-end
    market cap (local and USD), the market-cap-weighted universe ("market")
    return, and the point-in-time fundamentals.  The price->monthly aggregation
    mirrors Experiment 1 so the experiments share an identical return definition
    and month-end alignment.
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
# Building blocks (copied verbatim from the source libraries)
# --------------------------------------------------------------------------- #
def _rd_clip(p: pd.DataFrame) -> pd.Series:
    """R&D expense floored at 0 (a handful of reported R&D values are negative --
    data artefacts of restatements; genuine R&D cannot be negative).
    Verbatim from RD/rd_factors.py."""
    return p["rd_ltm"].astype(float).clip(lower=0.0)


def _rd_intensity_series(p: pd.DataFrame) -> pd.Series:
    """R&D / sales, the operating R&D-intensity ratio (sales must be positive).
    Verbatim from RD/rd_factors.py."""
    sales = p["sales_ltm"].astype(float)
    intensity = _rd_clip(p) / sales.where(sales > 0.0)
    return intensity


def _yoy_growth(p: pd.DataFrame, s: pd.Series) -> pd.Series:
    """Year-over-year growth (level_t / level_{t-12m} - 1) of a series, per stock.

    The prior-year level is guarded to be strictly positive (``.where(prev > 0)``)
    so the growth rate is well-defined and not dominated by sign flips / tiny
    denominators; otherwise the observation is left missing.
    Verbatim from Rev & Cost/revcost_factors.py."""
    prev = s.groupby(p["stock_id"], observed=True).shift(YOY_LAG)
    return s / prev.where(prev > 0.0) - 1.0


# --------------------------------------------------------------------------- #
# Factor definitions (copied verbatim from the source libraries)
# --------------------------------------------------------------------------- #
def _f_rd_stability(p: pd.DataFrame) -> pd.Series:
    """
    R&D commitment stability: the NEGATIVE trailing-36m coefficient of variation
    (std / mean) of R&D intensity, per stock.  High (near 0) => a steady,
    committed R&D programme; low (very negative) => erratic spending, e.g. R&D
    cut to manage earnings (Graham, Harvey & Rajgopal 2005).  A second moment, so
    orthogonal by construction to every level signal in the project.  Undefined
    when the trailing mean intensity is below ``MIN_MEAN_INTENSITY`` (R&D ~ 0).
    Verbatim from RD/rd_factors.py.
    """
    intensity = _rd_intensity_series(p)
    g = intensity.groupby(p["stock_id"], observed=True)
    mean = g.transform(lambda s: s.rolling(STAB_WINDOW, min_periods=STAB_MIN_PERIODS).mean())
    std = g.transform(lambda s: s.rolling(STAB_WINDOW, min_periods=STAB_MIN_PERIODS).std())
    cov = std / mean.where(mean > MIN_MEAN_INTENSITY)
    return -cov


def _f_revenue_stability(p: pd.DataFrame) -> pd.Series:
    """
    Recurring-revenue durability: the NEGATIVE trailing-36m standard deviation of
    YoY revenue growth, per stock.  High (near 0) => a smooth, predictable,
    recurring (subscription-like) top line; low (very negative) => lumpy
    license/deal revenue with renewal/air-pocket risk.  A second moment, so
    orthogonal by construction to every level signal in the project.  We negate so
    that higher = more stable = the bullish (long) leg.
    Verbatim from Rev & Cost/revcost_factors.py.
    """
    g = _yoy_growth(p, p["sales_ltm"].astype(float))
    g = g.replace([np.inf, -np.inf], np.nan)
    std = (g.groupby(p["stock_id"], observed=True)
            .transform(lambda s: s.rolling(STAB_WINDOW, min_periods=STAB_MIN_PERIODS).std()))
    return -std


_FACTOR_FUNCS = {
    "rd_stability": _f_rd_stability,
    "revenue_stability": _f_revenue_stability,
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


def build(save: bool = True, u: Universe = BANKS_INSURANCE) -> pd.DataFrame:
    """Full build: panel -> factors -> z-scores -> next return -> tidy long."""
    panel = build_monthly_panel(u=u)
    panel = compute_factors(panel)
    panel = add_zscores(panel)
    panel = add_next_return(panel)
    long = to_long_panel(panel)
    if save:
        u.output_dir.mkdir(parents=True, exist_ok=True)
        long.to_csv(u.panel_path, index=False)
    return long


def load_panel(rebuild: bool = False,
               u: Universe = BANKS_INSURANCE) -> pd.DataFrame:
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
    print(f"Built cross-validation factor panel: {len(long):,} rows | "
          f"{n_stocks} stocks | {n_months} months "
          f"({long['date'].min():%Y-%m} .. {long['date'].max():%Y-%m})")
    print(f"Saved -> {u.panel_path}")
    counts = (long.groupby("factor")["value"].size().reindex(FACTOR_NAMES))
    print("\nObservations per factor:")
    for name, c in counts.items():
        print(f"  {name:<26} {c:>9,}")
