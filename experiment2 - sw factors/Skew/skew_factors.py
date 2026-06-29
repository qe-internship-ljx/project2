"""
skew_factors.py
===============

Compute a set of **skewness factors** for the GICS *Software & Services*
universe, standardise each cross-sectionally (z-score vs the industry mean), and
write a tidy monthly panel to ``Skew/factor_panel.csv``.

This module, its driver and its outputs all live in experiment2's ``Skew/``
subfolder.  It is a *sixth* factor library for Experiment 2 (alongside
``sw_factors.py`` -> ``Standard/``, ``rd_factors.py`` -> ``RD/``,
``revcost_factors.py`` -> ``Rev & Cost/`` and ``stability_factors.py`` ->
``Stability/``).  Like the others it is a **drop-in for Experiment 1's analysis
engine**: the quintile sorts, cross-sectional (Fama-MacBeth) regressions,
long/short books, trading-cost model and every plot are reused **verbatim** from
``experiment1 - general factors/{quintile,regression,cost}.py`` via the shared
engine in ``experiment1 - general factors/factors.py``.  Only the *factor
definitions* and the fundamentals they need are new here, and the universe's
``output_dir`` is the ``Skew/`` folder itself, so these factors never collide
with the other libraries.

Motivation -- third moments (skewness / lottery demand)
-------------------------------------------------------
Where the ``Stability/`` library captured *second* moments (consistency =
inverse volatility), this library captures the *third* moment, **skewness**.  A
large body of work shows investors over-pay for positive skewness -- the
"lottery" preference -- so assets with high expected positive skew earn *lower*
subsequent risk-adjusted returns (Boyer, Mitton & Vorkink 2010, "Expected
idiosyncratic skewness"; Bali, Cakici & Whitelaw 2011, "MAX"; Amaya et al. 2015
on realised skewness).  The canonical trade is therefore to **short** the
positively skewed names and **long** the negatively skewed ones, so all three
factors below carry ``higher_is_bullish = False``: the hypothesis is that *high*
skew predicts *low* returns.

    name                      definition                                          dir         dimension
    ------------------------  --------------------------------------------------  ----------  --------------------------
    return_skewness           trailing 36m skewness of monthly total return       short high  return lottery demand
    revenue_growth_skewness   trailing 36m skewness of YoY revenue growth         short high  top-line lottery / lumpiness
    eps_skewness              trailing 36m skewness of diluted EPS                 short high  bottom-line lottery / lumpiness

* ``return_skewness`` is the trailing-36m sample skewness of the stock's monthly
  total return -- the direct realised-skewness analogue of the expected
  idiosyncratic skewness in the lottery-demand literature.  High positive skew
  marks a lottery-like return profile (rare large up-months) that investors
  bid up and that subsequently underperforms; very negative skew marks
  crash-prone names that earn a premium.
* ``revenue_growth_skewness`` is the trailing-36m sample skewness of YoY revenue
  (``sales_ltm``) growth.  It pushes the same idea onto the top line: a firm
  whose growth is positively skewed delivers occasional spectacular growth
  spurts around an otherwise modest trend (a fundamental "lottery"), whereas
  negative skew marks names prone to sudden growth collapses.  Currency-neutral
  by construction (skewness of a unitless growth *rate*).
* ``eps_skewness`` is the trailing-36m sample skewness of diluted EPS
  (``earnings_ltm / diluted_shares_outstanding``).  The bottom-line cousin: a
  positively skewed earnings path (rare blow-out quarters around a flat base) is
  the fundamental signature of a lottery stock; a negatively skewed path flags
  episodic earnings collapses.  Skewness is **scale-invariant** (it divides the
  third central moment by ``std**3``), so although EPS is a per-share local-
  currency quantity the score is currency-neutral and no FX conversion is
  needed, consistent with the rest of the project.

All three are pure *shape* statistics -- standardised third moments -- and are
therefore orthogonal by construction to every level / valuation signal and (being
a moment one order up) largely orthogonal to the ``Stability/`` second-moment
factors.  Skewness is undefined for a constant series and noisy for short ones,
so each requires :data:`SKEW_MIN_PERIODS` of trailing history before a score
exists (same 36m window / >=2y minimum as the stability factors).

Run standalone to (re)build the panel::

    python skew_factors.py

To run the full pipeline (panel + quintile sorts + regressions + redundancy)::

    python main_skew.py
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
        "skew_factor_engine", _EXP1_DIR / "factors.py")
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
# This module lives in (and writes to) experiment2's Skew/ subfolder, so these
# factors and their outputs live in their own subtree.  OUTPUT_DIR is the Skew/
# folder itself (the directory holding this file), so the analysis outputs land
# directly under Skew/ (Skew/{factor_panel.csv, quintile/, regression/,
# factor_correlation/}).
OUTPUT_DIR = Path(__file__).resolve().parent

INDUSTRY_GROUP = "Software & Services"

# Estimation windows / parameters.
SKEW_WINDOW = 36            # trailing months for the skewness (third-moment) statistic
SKEW_MIN_PERIODS = 24       # require >=2y of history before a skewness score exists
YOY_LAG = 12                # year-over-year lag (months), for the revenue-growth series

# Ordered list of the factors produced by this module.  ``higher_is_bullish`` is
# the academically expected sign of the long leg (top quintile): for all three
# skewness factors the lottery-demand hypothesis predicts that *high* positive
# skew earns *low* subsequent returns, so the canonical trade is long Q1 (most
# negatively skewed) / short Q5 -- i.e. ``higher_is_bullish = False``.  With
# ``USE_CANONICAL_LS_DIRECTION`` below set True the long/short book is signed by
# this prior (NOT by the in-sample t-stat), so a negative realised alpha t-stat
# means the factor worked *against* the hypothesis -- exactly what we want for
# hypothesis testing.
FACTORS: dict[str, dict] = {
    "return_skewness":         {"family": "Return skewness (monthly-return lottery demand)",   "higher_is_bullish": False},
    "revenue_growth_skewness": {"family": "Revenue-growth skewness (top-line lottery/lumpiness)", "higher_is_bullish": False},
    "eps_skewness":            {"family": "EPS skewness (bottom-line lottery/lumpiness)",       "higher_is_bullish": False},
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
    Point-in-time monthly fundamentals carrying everything the skewness factors
    need: LTM earnings (net income), LTM sales (for the YoY revenue-growth
    series), and the diluted share count used to turn earnings into EPS.

    Each record carries ``observation_date`` (when the report became
    observable); we align on it in :func:`build_monthly_panel` via the shared
    :func:`attach_pit_fundamentals` to avoid look-ahead -- identical convention
    to Experiment 1 / ``sw_factors.py`` / ``rd_factors.py`` / ``stability_factors.py``.
    """
    base_cols = ["date_fundamental", "observation_date", "stock_id",
                 "earnings_ltm", "sales_ltm"]
    fm = pd.read_feather(DATA_DIR / "fundamental_master.feather", columns=base_cols)
    fm["stock_id"] = fm["stock_id"].astype(str)
    fm = fm[fm["stock_id"].isin(universe)].copy()

    # Diluted share count lives in the extended fundamentals table.  It has no
    # observation_date but shares the date_fundamental grid, so it merges on that
    # key and inherits the observation_date above (same convention as
    # sw_factors.py / stability_factors.py / the engine).
    ext = pd.read_feather(
        DATA_DIR / "Industry Fundamentals Data" / "fundamental_master_extended.feather",
        columns=["date_fundamental", "stock_id", "diluted_shares_outstanding"])
    ext["stock_id"] = ext["stock_id"].astype(str)
    ext = ext[ext["stock_id"].isin(universe)]

    fm = fm.merge(ext, on=["stock_id", "date_fundamental"], how="left")
    fm["date_fundamental"] = pd.to_datetime(fm["date_fundamental"])
    fm["observation_date"] = pd.to_datetime(fm["observation_date"])
    return fm


# --------------------------------------------------------------------------- #
# Monthly panel construction (mirrors the engine / stability_factors.py exactly)
# --------------------------------------------------------------------------- #
def build_monthly_panel(universe: pd.Index | None = None,
                        u: Universe = SOFTWARE_SERVICES) -> pd.DataFrame:
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
# Building blocks
# --------------------------------------------------------------------------- #
def _eps_series(p: pd.DataFrame) -> pd.Series:
    """Diluted EPS = LTM earnings / diluted shares outstanding (may be negative).

    Local-currency per-share earnings; only consumed inside the scale-invariant
    skewness statistic below, so no FX conversion is required."""
    earnings = p["earnings_ltm"].astype(float)
    shares = p["diluted_shares_outstanding"].astype(float)
    return earnings / shares.where(shares > 0.0)


def _yoy_growth(p: pd.DataFrame, s: pd.Series) -> pd.Series:
    """Year-over-year growth (level_t / level_{t-12m} - 1) of a series, per stock.

    The prior-year level is guarded strictly positive so the growth rate is well
    defined; same convention as ``Rev & Cost/revcost_factors.py`` /
    ``stability_factors.py``."""
    prev = s.groupby(p["stock_id"], observed=True).shift(YOY_LAG)
    return s / prev.where(prev > 0.0) - 1.0


def _trailing_skewness(p: pd.DataFrame, x: pd.Series) -> pd.Series:
    """
    Trailing-:data:`SKEW_WINDOW`-month sample skewness of ``x`` per stock.

    Positive => a right-skewed (lottery-like) profile with rare large upside;
    negative => a left-skewed (crash-prone) profile.  Skewness is the third
    central moment divided by ``std**3``, hence **scale- and location-invariant**
    (so currency-neutral for the per-share / level series).  Requires
    :data:`SKEW_MIN_PERIODS` of history; ``pandas`` returns NaN for a degenerate
    (zero-variance / too-short) window, which propagates as an undefined score.
    """
    return (x.groupby(p["stock_id"], observed=True)
             .transform(lambda s: s.rolling(SKEW_WINDOW,
                                             min_periods=SKEW_MIN_PERIODS).skew()))


# --------------------------------------------------------------------------- #
# Factor definitions
# --------------------------------------------------------------------------- #
def _f_return_skewness(p: pd.DataFrame) -> pd.Series:
    """
    Return skewness: the trailing-36m sample skewness of the stock's monthly
    total return -- the realised analogue of expected idiosyncratic skewness.
    High positive skew marks a lottery-like return profile that investors bid up
    and that subsequently underperforms (Boyer, Mitton & Vorkink 2010; Bali,
    Cakici & Whitelaw 2011); very negative skew marks crash-prone names that earn
    a premium.  Hypothesised to predict returns *negatively*
    (``higher_is_bullish = False``).
    """
    return _trailing_skewness(p, p["mret"].astype(float))


def _f_revenue_growth_skewness(p: pd.DataFrame) -> pd.Series:
    """
    Revenue-growth skewness: the trailing-36m sample skewness of YoY revenue
    (``sales_ltm``) growth.  Positive skew marks a top line that delivers
    occasional spectacular growth spurts around an otherwise modest trend (a
    fundamental "lottery"); negative skew marks names prone to sudden growth
    collapses.  The top-line cousin of ``return_skewness`` -- same lottery
    hypothesis applied to fundamentals.  Currency-neutral (skewness of a unitless
    growth rate).
    """
    growth = _yoy_growth(p, p["sales_ltm"].astype(float)).replace([np.inf, -np.inf], np.nan)
    return _trailing_skewness(p, growth)


def _f_eps_skewness(p: pd.DataFrame) -> pd.Series:
    """
    EPS skewness: the trailing-36m sample skewness of diluted EPS
    (``earnings_ltm / diluted_shares_outstanding``).  A positively skewed
    earnings path (rare blow-out quarters around a flat base) is the fundamental
    signature of a lottery stock; a negatively skewed path flags episodic
    earnings collapses.  The bottom-line cousin of ``return_skewness``; skewness
    is scale-invariant so the per-share local-currency input needs no FX
    conversion.
    """
    return _trailing_skewness(p, _eps_series(p))


_FACTOR_FUNCS = {
    "return_skewness": _f_return_skewness,
    "revenue_growth_skewness": _f_revenue_growth_skewness,
    "eps_skewness": _f_eps_skewness,
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
    print(f"Built skewness factor panel: {len(long):,} rows | "
          f"{n_stocks} stocks | {n_months} months "
          f"({long['date'].min():%Y-%m} .. {long['date'].max():%Y-%m})")
    print(f"Saved -> {u.panel_path}")
    counts = (long.groupby("factor")["value"].size().reindex(FACTOR_NAMES))
    print("\nObservations per factor:")
    for name, c in counts.items():
        print(f"  {name:<26} {c:>9,}")
