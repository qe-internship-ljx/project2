"""
growth_factors.py
=================

Compute a set of **growth factors** for the GICS *Software & Services* universe,
standardise each cross-sectionally (z-score vs the industry mean), and write a
tidy monthly panel to ``Growth/factor_panel.csv``.

This module, its driver and its outputs all live in experiment2's ``Growth/``
subfolder.  It is a *sixth* factor library for Experiment 2 (alongside
``sw_factors.py`` -> ``standard/``, ``rd_factors.py`` -> ``RD/``, ``Rev & Cost/``
and ``Stability/``).  Like the others it is a **drop-in for Experiment 1's
analysis engine**: the quintile sorts, cross-sectional (Fama-MacBeth)
regressions, long/short books, trading-cost model and every plot are reused
**verbatim** from ``experiment1 - general factors/{quintile,regression,cost}.py``
via the shared engine in ``experiment1 - general factors/factors.py``.  Only the
*factor definitions* and the fundamentals they need are new here, and the
universe's ``output_dir`` is the ``Growth/`` folder itself, so these factors
never collide with the other libraries.

Motivation -- margin trajectory, not margin level
-------------------------------------------------
The ``Rev & Cost`` extension found that the gross-margin *level* (``gross_margin``)
earned industry-neutral alpha early in the sample but **decayed to roughly zero
post-2016** -- an obvious, well-followed KPI that the market arbitraged away.  The
``Stability`` extension instead used the gross-margin *second moment*
(``gross_margin_stability``, the consistency of the margin).  This module targets
the **first difference**: whether the margin is *improving*.

    name                    definition                          dir          dimension
    ----------------------  ----------------------------------  ----------   -----------
    gross_margin_expansion  d(gross_margin, YoY)                long high    margin trajectory

* ``gross_margin_expansion`` is the year-over-year change in gross margin
  (``gross_income_ltm / sales_ltm``): ``gm_t - gm_{t-12m}``.  A widening gross
  margin signals strengthening pricing power, an improving revenue mix (e.g. a
  shift toward higher-margin subscription/software revenue) or operating
  efficiencies converting to the gross line -- a forward-looking quality signal
  that the *static* margin level does not capture.  A contracting margin (very
  negative) flags pricing pressure, mix deterioration or a rising cost base.
  Higher (expanding) is the bullish prior.

  Where the level signal asks "is this a high-margin business?" (a fact the
  market already prices), the *change* asks "is this business getting better?",
  a fresher, less-arbitraged dimension and the canonical "fundamental momentum"
  framing of margin improvement.  It is currency-neutral by construction (a
  difference of two same-currency ratios), so no FX conversion is required for
  within-industry comparison -- consistent with the rest of the project.

Run standalone to (re)build the panel::

    python growth_factors.py

To run the full pipeline (panel + quintile sorts + regressions + redundancy)::

    python main_growth.py
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
        "growth_factor_engine", _EXP1_DIR / "factors.py")
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
# This module lives in (and writes to) experiment2's Growth/ subfolder, so these
# factors and their outputs live in their own subtree.  OUTPUT_DIR is the
# Growth/ folder itself (the directory holding this file), so the analysis
# outputs land directly under Growth/ (Growth/{factor_panel.csv, quintile/,
# regression/, factor_correlation/}).
OUTPUT_DIR = Path(__file__).resolve().parent

INDUSTRY_GROUP = "Software & Services"

# Estimation windows / parameters.
YOY_LAG = 12                # year-over-year lag (months) for the margin-change signal

# Ordered list of the factors produced by this module.  ``higher_is_bullish`` is
# the academically expected sign of the long leg (top quintile): True => the
# canonical trade is long Q5 / short Q1.  With ``USE_CANONICAL_LS_DIRECTION``
# below set True the long/short book is signed by this prior (NOT by the
# in-sample t-stat), so a negative realised alpha t-stat means the factor worked
# *against* the hypothesis -- exactly what we want for hypothesis testing.
FACTORS: dict[str, dict] = {
    "gross_margin_expansion": {"family": "Gross-margin expansion (margin trajectory)", "higher_is_bullish": True},
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
    Point-in-time monthly fundamentals carrying everything the growth factors
    need: LTM sales and LTM gross income (for gross margin).

    Each record carries ``observation_date`` (when the report became
    observable); we align on it in :func:`build_monthly_panel` via the shared
    :func:`attach_pit_fundamentals` to avoid look-ahead -- identical convention
    to Experiment 1 / ``sw_factors.py`` / ``rd_factors.py`` / ``stability_factors.py``.
    """
    cols = ["date_fundamental", "observation_date", "stock_id",
            "sales_ltm", "gross_income_ltm"]
    fm = pd.read_feather(DATA_DIR / "fundamental_master.feather", columns=cols)
    fm["stock_id"] = fm["stock_id"].astype(str)
    fm = fm[fm["stock_id"].isin(universe)].copy()
    fm["date_fundamental"] = pd.to_datetime(fm["date_fundamental"])
    fm["observation_date"] = pd.to_datetime(fm["observation_date"])
    return fm


# --------------------------------------------------------------------------- #
# Monthly panel construction (mirrors the engine / rd_factors.py exactly)
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
def _gross_margin_series(p: pd.DataFrame) -> pd.Series:
    """Gross margin = LTM gross income / LTM sales (sales must be positive).

    Currency-neutral (a ratio of same-currency line items) and almost always
    positive in software."""
    sales = p["sales_ltm"].astype(float)
    gross = p["gross_income_ltm"].astype(float)
    return gross / sales.where(sales > 0.0)


def _yoy_change(p: pd.DataFrame, s: pd.Series) -> pd.Series:
    """Year-over-year change (level_t - level_{t-12m}) of a series, per stock."""
    prev = s.groupby(p["stock_id"], observed=True).shift(YOY_LAG)
    return s - prev


# --------------------------------------------------------------------------- #
# Factor definitions
# --------------------------------------------------------------------------- #
def _f_gross_margin_expansion(p: pd.DataFrame) -> pd.Series:
    """
    Gross-margin expansion: the year-over-year change in gross margin
    (``gross_income_ltm / sales_ltm``), ``gm_t - gm_{t-12m}``.  A widening margin
    signals strengthening pricing power, an improving (higher-margin) revenue mix
    or operating efficiencies reaching the gross line -- a forward-looking quality
    signal the static margin *level* does not capture; a contracting margin flags
    pricing pressure or mix/cost deterioration.  Higher (expanding) is bullish.

    Where the level signal (Rev & Cost ``gross_margin``) decayed to ~zero alpha
    post-2016 as an arbitraged KPI, the *change* is the fresher "is this business
    getting better?" dimension.  Currency-neutral (a difference of two
    same-currency ratios).
    """
    return _yoy_change(p, _gross_margin_series(p))


_FACTOR_FUNCS = {
    "gross_margin_expansion": _f_gross_margin_expansion,
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
    print(f"Built growth factor panel: {len(long):,} rows | "
          f"{n_stocks} stocks | {n_months} months "
          f"({long['date'].min():%Y-%m} .. {long['date'].max():%Y-%m})")
    print(f"Saved -> {u.panel_path}")
    counts = (long.groupby("factor")["value"].size().reindex(FACTOR_NAMES))
    print("\nObservations per factor:")
    for name, c in counts.items():
        print(f"  {name:<26} {c:>9,}")
