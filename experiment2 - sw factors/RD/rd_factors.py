"""
rd_factors.py
=============

Compute a set of **R&D-behavior factors** for the GICS *Software & Services*
universe, standardise each cross-sectionally (z-score vs the industry mean), and
write a tidy monthly panel to ``RD/factor_panel.csv``.

This module, its driver and its documentation all live in experiment2's ``RD/``
subfolder, and it writes its analysis outputs there too.  It is a *second*
factor library for Experiment 2, alongside ``sw_factors.py``.  Like that module
it is a **drop-in for Experiment 1's analysis engine**: the quintile sorts,
cross-sectional (Fama-MacBeth) regressions, long/short books, trading-cost model
and every plot are reused **verbatim** from
``experiment1 - general factors/{quintile,regression,cost}.py`` via the shared
engine in ``experiment1 - general factors/factors.py``.  Only the *factor
definitions* and the fundamentals they need are new here, and the universe's
``output_dir`` is the ``RD/`` folder itself, so these factors never collide with
the established software factors of ``sw_factors.py`` (which write to ``standard/``).

Motivation
----------
``sw_factors.py`` already captures three R&D-*adjusted* characteristics:
``intangible_value`` (R&D capital relative to price), ``intangible_profitability``
(R&D-adjusted profit on an R&D-adjusted asset base) and ``rd_productivity``
(sales growth per unit of R&D capital).  All three are *level / valuation*
signals built on the accumulated R&D **stock** ``K_int``.

This module instead targets the **dynamics and discipline of R&D spending** --
behavioral dimensions the existing set does not measure:

    name              definition                                              dir          dimension
    ----------------  ------------------------------------------------------  ----------   -----------
    rd_growth         d(rd_ltm, YoY) / avg assets                             long high    flow
    rd_conversion     d(gross_income_ltm, YoY) / sales_ltm_{t-12m}            long high    profit output
    rd_stability      - trailing 36m coeff. of variation of (rd_ltm/sales)    long high    consistency
    innovation_mix    rd_ltm / (rd_ltm + sga_ltm)                             long high    composition
    rd_intensity      rd_ltm / sales_ltm                                      long high    level (baseline)

The first four target genuinely *behavioral* dimensions the existing factors do
not measure; ``rd_intensity`` (a level) is retained only as the **conventional
baseline** against which the four behavior signals are contrasted.

* ``rd_growth`` is the *flow change* in R&D (the intangible-investment analogue
  of ``asset_growth``).  Crucially it carries the OPPOSITE academic prior to the
  asset-growth / investment anomaly: tangible asset growth predicts *low*
  returns, but R&D *increases* predict *high* returns (Eberhart, Maxwell &
  Siddique 2004), because expensing hides the investment.  This sign flip is the
  cleanest reason it is not redundant with ``asset_growth``.
* ``rd_conversion`` measures R&D *output in PROFIT*: the YoY change in gross
  profit per dollar of prior-year sales.  Distinct from ``rd_productivity``
  (Δsales / K_int) on two axes -- numerator (gross *profit* vs sales) and
  denominator (a safe sales base vs the unstable K_int) -- and designed to fix
  ``rd_productivity``'s documented failure (it rewards low-margin revenue
  chasing; this rewards margin-accretive innovation).  Lev & Sougiannis (1996),
  Novy-Marx (2013).
* ``rd_stability`` is a *second moment* (consistency of R&D commitment) -- by
  construction orthogonal to every level signal in the project.  Firms that cut
  R&D to manage earnings (Graham, Harvey & Rajgopal 2005) score low.
* ``innovation_mix`` is the *composition* of discretionary spend -- product
  (R&D) vs go-to-market/admin (SG&A) -- distinct from ``gtm_efficiency``, which
  is an output ratio (sales growth per SG&A), not an input mix.
* ``rd_intensity`` is the operating R&D *level* (R&D/sales) -- the textbook R&D
  signal (Chan, Lakonishok & Sougiannis 2001), kept as a conventional anchor.

All signals are ratios / scaled changes and therefore currency-neutral, so no FX
conversion is required for within-industry comparison -- consistent with the
rest of the project.

Run standalone to (re)build the panel::

    python rd_factors.py

To run the full pipeline (panel + quintile sorts + regressions) use::

    python main_rd.py
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
        "rd_factor_engine", _EXP1_DIR / "factors.py")
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
# This module lives in (and writes to) experiment2's RD/ subfolder, so these
# factors and their outputs live in their own subtree and never collide with
# sw_factors.py's output/{quintile,regression}.  OUTPUT_DIR is the RD/ folder
# itself (the directory holding this file), so the analysis outputs land
# directly under RD/ (RD/{factor_panel.csv, quintile/, regression/, ...}).
OUTPUT_DIR = Path(__file__).resolve().parent

INDUSTRY_GROUP = "Software & Services"

# Estimation windows / parameters.
YOY_LAG = 12                # year-over-year lag (months) for flow-change signals
STAB_WINDOW = 36            # trailing months for the R&D-intensity stability moment
STAB_MIN_PERIODS = 24       # require >=2y of history before a stability score exists
MIN_MEAN_INTENSITY = 0.005  # floor on trailing mean R&D/sales below which the
                            # coefficient of variation is undefined (R&D ~ 0)

# Ordered list of the factors produced by this module.  ``higher_is_bullish`` is
# the academically expected sign of the long leg (top quintile): True => the
# canonical trade is long Q5 / short Q1.  With ``USE_CANONICAL_LS_DIRECTION``
# below set True the long/short book is signed by this prior (NOT by the
# in-sample t-stat), so a negative realised alpha t-stat means the factor worked
# *against* the hypothesis -- exactly what we want for hypothesis testing.
FACTORS: dict[str, dict] = {
    "rd_growth":      {"family": "R&D growth (flow change)",            "higher_is_bullish": True},
    "rd_conversion":  {"family": "R&D profit conversion (output)",      "higher_is_bullish": True},
    "rd_stability":   {"family": "R&D stability (commitment consistency)", "higher_is_bullish": True},
    "innovation_mix": {"family": "Innovation mix (build vs sell)",      "higher_is_bullish": True},
    "rd_intensity":   {"family": "R&D intensity (level; baseline)",     "higher_is_bullish": True},
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
    Point-in-time monthly fundamentals carrying everything the R&D-behavior
    factors need: R&D expense, sales, SG&A, total assets and gross income.

    Each record carries ``observation_date`` (when the report became
    observable); we align on it in :func:`build_monthly_panel` via the shared
    :func:`attach_pit_fundamentals` to avoid look-ahead -- identical convention
    to Experiment 1 / ``sw_factors.py``.
    """
    cols = ["date_fundamental", "observation_date", "stock_id",
            "rd_ltm", "sales_ltm", "sga_ltm", "assets", "gross_income_ltm"]
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
    return, and the point-in-time R&D fundamentals.  The price->monthly
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
def _rd_clip(p: pd.DataFrame) -> pd.Series:
    """R&D expense floored at 0 (a handful of reported R&D values are negative --
    data artefacts of restatements; genuine R&D cannot be negative)."""
    return p["rd_ltm"].astype(float).clip(lower=0.0)


def _rd_intensity_series(p: pd.DataFrame) -> pd.Series:
    """R&D / sales, the operating R&D-intensity ratio (sales must be positive)."""
    sales = p["sales_ltm"].astype(float)
    intensity = _rd_clip(p) / sales.where(sales > 0.0)
    return intensity


def _yoy_change(p: pd.DataFrame, s: pd.Series) -> pd.Series:
    """Year-over-year change (level_t - level_{t-12m}) of a series, per stock."""
    prev = s.groupby(p["stock_id"], observed=True).shift(YOY_LAG)
    return s - prev


# --------------------------------------------------------------------------- #
# Factor definitions
# --------------------------------------------------------------------------- #
def _f_rd_growth(p: pd.DataFrame) -> pd.Series:
    """
    Abnormal R&D growth: the year-over-year change in R&D expense, scaled by the
    average total assets over the year.  This is the intangible-investment
    analogue of ``asset_growth`` (Exp1), but with the OPPOSITE prior: tangible
    asset/capex growth predicts low returns, whereas R&D increases predict *high*
    returns (Eberhart, Maxwell & Siddique 2004) because expensed R&D hides the
    investment from earnings.  Scaling by average assets keeps it bounded and
    directly comparable to ``asset_growth``.
    """
    rd = _rd_clip(p)
    drd = _yoy_change(p, rd)
    assets = p["assets"].astype(float)
    prev_assets = assets.groupby(p["stock_id"], observed=True).shift(YOY_LAG)
    avg_assets = 0.5 * (assets + prev_assets)
    return drd / avg_assets.where(avg_assets > 0.0)


def _f_rd_conversion(p: pd.DataFrame) -> pd.Series:
    """
    R&D output measured in PROFIT: the year-over-year change in gross profit,
    scaled by prior-year sales (a stable, always-positive base -- deliberately
    NOT K_int).  Where ``rd_productivity`` (Exp2) asks whether R&D produces
    *sales* (which can be bought via discounting or unprofitable land-grab),
    this asks whether it produces *gross profit* -- durable, high-margin,
    pricing-power output (gross income nets out COGS / hosting).  It is distinct
    from ``rd_productivity`` on both numerator (gross profit vs sales) and
    denominator (a safe sales base vs the unstable K_int), and is built to repair
    ``rd_productivity``'s documented negative alpha by rewarding margin-accretive
    rather than low-margin innovation.  Lev & Sougiannis (1996); Novy-Marx (2013).
    """
    dgp = _yoy_change(p, p["gross_income_ltm"].astype(float))
    sales = p["sales_ltm"].astype(float)
    prev_sales = sales.groupby(p["stock_id"], observed=True).shift(YOY_LAG)
    return dgp / prev_sales.where(prev_sales > 0.0)


def _f_rd_intensity(p: pd.DataFrame) -> pd.Series:
    """R&D intensity = R&D / sales.  The conventional/textbook R&D signal, kept as
    a baseline: the cleanest operating measure of how much of each revenue dollar
    is reinvested into product; expensing it understates the GAAP earnings of
    intensive innovators (Chan, Lakonishok & Sougiannis 2001)."""
    return _rd_intensity_series(p)


def _f_rd_stability(p: pd.DataFrame) -> pd.Series:
    """
    R&D commitment stability: the NEGATIVE trailing-36m coefficient of variation
    (std / mean) of R&D intensity, per stock.  High (near 0) => a steady,
    committed R&D programme; low (very negative) => erratic spending, e.g. R&D
    cut to manage earnings (Graham, Harvey & Rajgopal 2005).  A second moment, so
    orthogonal by construction to every level signal in the project.  Undefined
    when the trailing mean intensity is below ``MIN_MEAN_INTENSITY`` (R&D ~ 0).
    """
    intensity = _rd_intensity_series(p)
    g = intensity.groupby(p["stock_id"], observed=True)
    mean = g.transform(lambda s: s.rolling(STAB_WINDOW, min_periods=STAB_MIN_PERIODS).mean())
    std = g.transform(lambda s: s.rolling(STAB_WINDOW, min_periods=STAB_MIN_PERIODS).std())
    cov = std / mean.where(mean > MIN_MEAN_INTENSITY)
    return -cov


def _f_innovation_mix(p: pd.DataFrame) -> pd.Series:
    """
    Innovation mix: R&D as a share of total discretionary spend, R&D / (R&D +
    SG&A).  High => a product-led firm building a durable, scalable intangible
    asset; low => a go-to-market/admin-heavy firm buying revenue that must be
    re-bought.  An input-MIX ratio, distinct from ``gtm_efficiency`` (an output
    ratio: sales growth per SG&A dollar).
    """
    rd = _rd_clip(p)
    sga = p["sga_ltm"].astype(float).clip(lower=0.0)
    denom = rd + sga
    return rd / denom.where(denom > 0.0)


_FACTOR_FUNCS = {
    "rd_growth": _f_rd_growth,
    "rd_conversion": _f_rd_conversion,
    "rd_stability": _f_rd_stability,
    "innovation_mix": _f_innovation_mix,
    "rd_intensity": _f_rd_intensity,
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
    print(f"Built R&D factor panel: {len(long):,} rows | "
          f"{n_stocks} stocks | {n_months} months "
          f"({long['date'].min():%Y-%m} .. {long['date'].max():%Y-%m})")
    print(f"Saved -> {u.panel_path}")
    counts = (long.groupby("factor")["value"].size().reindex(FACTOR_NAMES))
    print("\nObservations per factor:")
    for name, c in counts.items():
        print(f"  {name:<26} {c:>9,}")
