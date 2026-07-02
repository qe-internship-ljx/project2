"""
crossval_factors.py
===================

**Cross-validation** of Experiment 2's **top factors** on the **Banks + Insurance +
Commodity Producers** universe -- the union of Experiment 1's two non-software
cross-sections (financials + resources), a structurally unrelated test bed that is
broad enough to probe generalisation without the cost of the whole market.  The
same market-cap screen the software libraries apply is used.

The candidate set is no longer a hand-picked pair -- it is the **top ``TOP_N``
factors** of the cross-experiment top-factor hand-off
(``top_factors/top_factors.csv``), read the *same way* Experiment 3's
``factor_momentum.py`` / ``composite.py`` read it (the file is pre-sorted by
industry-neutral alpha t-stat, so the first ``TOP_N`` rows are the leaders).  Each
was discovered and validated on GICS *Software & Services*; this module asks the
out-of-sample-universe question: **do the software-industry leaders survive in
structurally unrelated industries** (financials + resources), or were they
software-specific?

Reuse -- test the *exact* construct that earned the ranking
-----------------------------------------------------------
The factors in the hand-off live in several source libraries (Experiment 1's
general factors plus Experiment 2's software subexperiments), each of which is a
**universe-parameterised** drop-in for Experiment 1's engine.  Rather than
re-implement (or copy) each definition here, this module **reuses the source
library verbatim**: for every requested factor it looks up its source
subexperiment (the ``subexperiment`` column of ``top_factors.csv``), runs that
library's own :func:`build` on the **Banks + Insurance** universe, and keeps just
that factor's rows.  So the construct cross-validated here is byte-for-byte the
one that produced the software-industry ranking -- if a source definition changes,
this test tracks it with no edit.

Like every factor library in the project it is also a **drop-in for Experiment
1's analysis engine**: the quintile sorts, cross-sectional (Fama-MacBeth)
regressions, long/short books, trading-cost model and every plot are reused
**verbatim** from ``experiment1 - general factors/{quintile,regression,cost}.py``
via the shared engine in ``experiment1 - general factors/factors.py``.  The only
things that change relative to the software subexperiments are (a) the factor set
(the top-``TOP_N`` hand-off, resolved to their source libraries) and (b) the
**universe**: this library runs on the ``BANKS_COMMODITY`` universe (Banks +
Insurance + Commodity Producers), writing to the ``Cross_val/`` folder.

Direction / sign
----------------
Every factor is signed by its software-discovered bullish prior
(``higher_is_bullish``, read from the source library) with
``USE_CANONICAL_LS_DIRECTION`` -- so the realised long/short book on this universe
can be read directly against the hypothesised direction, and a negative alpha
t-stat means the signal failed to *generalise* to financials + resources.

Coverage caveat
---------------
The top factors span several data footprints; some depend on line items banks,
insurers and resource firms rarely report.  ``rd_stability`` in particular needs a
meaningful R&D programme (its coefficient of variation is undefined when trailing
mean R&D/sales is tiny), which these industries largely lack, so it is expected to
have thin coverage; revenue/quality signals built only on a sales/earnings history
should cover more of the cross-section.  The realised coverage per factor is
printed by ``main_crossval.py`` and is itself part of the finding.

Run standalone to (re)build the panel::

    python crossval_factors.py

To run the full pipeline (panel + quintile sorts + regressions + redundancy)::

    python main_crossval.py
"""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- #
# Shared engine (Experiment 1) -- loaded by path under a private module name
# (its folder name contains spaces, and we must not shadow the name ``factors``,
# which the analysis modules bind to and the driver points at *this* library).
# This engine is *also* the source library for any "General" (Experiment 1)
# top factor, so ``_source_library("General")`` returns it directly.
# --------------------------------------------------------------------------- #
_EXP1_DIR = Path(__file__).resolve().parent.parent.parent / "experiment1 - general factors"
_EXP2_DIR = Path(__file__).resolve().parent.parent


def _load_module(name: str, path: Path):
    """Import a module by file path under ``name`` and register it in sys.modules."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_engine = _load_module("crossval_factor_engine", _EXP1_DIR / "factors.py")

# Re-export the generic helpers the analysis modules (quintile.py / regression.py
# / cost.py) reuse, so this module satisfies the exact interface they expect from
# their ``import factors as F``.
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

# The cross-experiment top-factor hand-off (written by
# ``experiment2 - sw factors/main.py``, ranking Experiment 1's general factors
# together with every Experiment 2 software subexperiment by industry-neutral
# alpha t-stat).  We test its leaders -- read exactly as Experiment 3 reads it.
TOP_FACTORS_CSV = _EXP2_DIR / "top_factors" / "top_factors.csv"
TOP_N = 5

# Sign the directional long/short book by each factor's canonical (software-
# discovered) prior rather than the in-sample Fama-MacBeth t-stat, so the realised
# result on this universe can be read directly against the hypothesised direction.
USE_CANONICAL_LS_DIRECTION = True


# --------------------------------------------------------------------------- #
# Top-factor hand-off -> the factor set (read the same way Experiment 3 does)
# --------------------------------------------------------------------------- #
def load_top_factors(csv_path: Path = TOP_FACTORS_CSV, n: int = TOP_N) -> pd.DataFrame:
    """The top-``n`` rows of the cross-experiment top-factor hand-off
    (``top_factors.csv``), pre-sorted by industry-neutral alpha t-stat -- one row
    per factor carrying its ``factor`` name, source ``subexperiment``, bullish
    ``direction`` and ``family``.

    Read afresh each call, so the cross-validation always tracks whatever factors
    rank highest after the latest Experiment 1/2 run.  Raises a clear error if the
    hand-off is missing (identical contract to ``composite.top_factors`` /
    ``factor_momentum.load_top_factors``)."""
    if not Path(csv_path).exists():
        raise FileNotFoundError(
            f"{csv_path} not found.  Run Experiment 2 first -- `python main.py` "
            "(or `python main.py collect`) in 'experiment2 - sw factors' writes "
            "the top-factor hand-off.")
    return pd.read_csv(csv_path).head(n).reset_index(drop=True)


# The active top-factor table and the module-level constants the analysis engine
# expects (``FACTOR_NAMES`` / ``FACTORS[name]{family, higher_is_bullish}``), built
# from the hand-off.  ``direction == 'Q5-Q1'`` => the bullish leg is the top
# z-score quintile, i.e. higher_is_bullish.
TOP_FACTORS = load_top_factors()
FACTOR_NAMES = TOP_FACTORS["factor"].tolist()
FACTORS: dict[str, dict] = {
    r.factor: {"family": r.family,
               "higher_is_bullish": str(r.direction).strip() == "Q5-Q1"}
    for r in TOP_FACTORS.itertuples()
}


# --------------------------------------------------------------------------- #
# Universe -- Banks + Insurance + Commodity Producers, the cross-validation target.
# A structurally unrelated union of Experiment 1's two non-software universes
# (financials + resources), broad enough to test generalisation without the cost
# of the whole market.  Its two constituents already have small, pre-filtered daily
# price feathers; we concatenate those once into a combined feather (see
# :func:`_ensure_combined_price_file`) so each source library's ``build`` reads only
# the cheap per-universe data and never touches the 150M-row whole-market feed.
# It applies the **same market-cap screen as the Experiment 2 software libraries**
# (``sw_factors.py``): no flat USD floor (0.0 only drops names with no established
# cap) plus a relative per-month floor that drops the lowest 20% of active names by
# USD cap.  The screen is enforced by each source library's own ``build``
# (``apply_mcap_screen(panel, u)``), so setting the thresholds here is all that is
# required.
# --------------------------------------------------------------------------- #
_CONSTITUENT_PRICE_FILES = ("price_banks_insurance.feather",
                            "price_commodity_producers.feather")
_COMBINED_PRICE_FILE = "price_banks_insurance_commodity.feather"
_INDUSTRIES = ("Banks", "Insurance", "Metals & Mining", "Oil, Gas & Consumable Fuels")


def _ensure_combined_price_file() -> str:
    """Concatenate the Banks+Insurance and Commodity-Producers daily price feathers
    into one combined feather in ``data/`` (built once, cached on disk) and return
    its filename.  Cheap -- both inputs are the small, pre-filtered per-universe
    feathers (~21.5M rows combined), so this never reads the whole-market
    ``price.feather``.  The two industries are disjoint, so no de-duplication is
    needed."""
    out = DATA_DIR / _COMBINED_PRICE_FILE
    if not out.exists():
        frames = [pd.read_feather(DATA_DIR / f) for f in _CONSTITUENT_PRICE_FILES]
        pd.concat(frames, ignore_index=True).to_feather(out)
    return _COMBINED_PRICE_FILE


BANKS_COMMODITY = Universe(
    slug="banks_insurance_commodity",
    price_file=_ensure_combined_price_file(),
    output_dir=OUTPUT_DIR,
    industries=_INDUSTRIES,
    min_mcap_usd=0.0,
    min_mcap_pct=0.20,
)

# Experiment 1's analysis modules (quintile.py / regression.py / cost.py) bind the
# symbol ``F.SOFTWARE_SERVICES`` as their default-universe argument, evaluated at
# import time.  We always pass ``u=BANKS_COMMODITY`` explicitly, so this alias is
# only here to satisfy that default-argument binding -- it points at this library's
# real (banks+insurance+commodity) universe so even an un-passed default is correct.
SOFTWARE_SERVICES = BANKS_COMMODITY

UNIVERSES: dict[str, Universe] = {BANKS_COMMODITY.slug: BANKS_COMMODITY}


def universe_from_argv(default: Universe = BANKS_COMMODITY) -> Universe:
    """Pick a universe from argv[1] (its slug); fall back to ``default``."""
    if len(sys.argv) > 1:
        return UNIVERSES[sys.argv[1]]
    return default


# Redundancy benchmark: Experiment 1's *full* general market factor set, built on
# this SAME (banks+insurance+commodity, cap-screened) universe -- the only valid
# same-universe comparison for the factor_correlation step.  It writes to a
# dedicated subfolder so it never collides with this library's own factor_panel.csv.
GENERAL_MARKET = dataclasses.replace(
    BANKS_COMMODITY, slug="cross_universe_general", output_dir=OUTPUT_DIR / "general_market")


def build_general_market_panel(rebuild: bool = False) -> Path:
    """Build (or reuse) Experiment 1's full general market factor panel on the SAME
    (banks+insurance+commodity, cap-screened) universe -- the redundancy benchmark
    for the factor_correlation step -- by reusing Experiment 1's engine verbatim.
    Returns the panel path."""
    if rebuild or not GENERAL_MARKET.panel_path.exists():
        _engine.build(save=True, u=GENERAL_MARKET)
    return GENERAL_MARKET.panel_path


# --------------------------------------------------------------------------- #
# Source libraries -- resolve each top factor to the library that computes it
# --------------------------------------------------------------------------- #
# Every factor in the hand-off is produced by exactly one source library.  We map
# the ``subexperiment`` label recorded in ``top_factors.csv`` to that library's
# module file; "General" is Experiment 1's engine (already loaded above).  Each
# library is a universe-parameterised drop-in whose ``build(save, u)`` computes
# all its factors on universe ``u`` and returns the tidy (date, stock_id, factor,
# value, zscore, next_return, weight) panel -- so we reuse the exact construct.
_SOURCE_LIB_PATHS: dict[str, Path] = {
    "Standard":   _EXP2_DIR / "sw_factors.py",
    "RD":         _EXP2_DIR / "RD" / "rd_factors.py",
    "Rev & Cost": _EXP2_DIR / "Rev & Cost" / "revcost_factors.py",
    "Stability":  _EXP2_DIR / "Stability" / "stability_factors.py",
    "Skew":       _EXP2_DIR / "Skew" / "skew_factors.py",
}
_LIB_CACHE: dict[str, object] = {}


def _source_library(subexperiment: str):
    """The factor library that computes a given subexperiment's factors, loaded by
    path and cached.  "General" resolves to Experiment 1's engine; every other
    label resolves through :data:`_SOURCE_LIB_PATHS`.  Raises a clear error if the
    subexperiment is unknown (e.g. a hand-off from a subexperiment not yet wired
    here)."""
    if subexperiment == "General":
        return _engine
    if subexperiment in _LIB_CACHE:
        return _LIB_CACHE[subexperiment]
    if subexperiment not in _SOURCE_LIB_PATHS:
        known = ", ".join(["General", *_SOURCE_LIB_PATHS])
        raise KeyError(
            f"top factor from subexperiment {subexperiment!r} has no source "
            f"library wired in crossval_factors (known: {known}).")
    path = _SOURCE_LIB_PATHS[subexperiment]
    module = _load_module(f"crossval_src_{path.stem}", path)
    _LIB_CACHE[subexperiment] = module
    return module


# --------------------------------------------------------------------------- #
# Build -- reuse each source library on the Banks + Insurance universe
# --------------------------------------------------------------------------- #
def build(save: bool = True, u: Universe = BANKS_COMMODITY) -> pd.DataFrame:
    """
    Build the cross-validation factor panel: for every top factor, run its source
    library's own ``build`` on the cross-validation universe ``u`` and keep just
    that factor's rows, then stack into one tidy long panel (date, stock_id, factor,
    value, zscore, next_return, weight) -- the exact schema Experiment 1 / 2 use.

    Each source library is built at most once (its ``build`` computes *all* its
    factors on ``u``; we slice out only the requested ones), so the returned panel
    carries only the top-factor rows.  ``next_return`` / ``weight`` are identical
    across libraries for a given (date, stock_id) since all are built from the same
    universe and prices.
    """
    frames = []
    # Group by source subexperiment so each library is built once; preserve the
    # hand-off's (rank) order of appearance.
    for subexp in TOP_FACTORS["subexperiment"].drop_duplicates():
        wanted = TOP_FACTORS.loc[TOP_FACTORS["subexperiment"] == subexp, "factor"].tolist()
        lib = _source_library(subexp)
        long = lib.build(save=False, u=u)                 # all of this library's factors on banks_insurance
        frames.append(long[long["factor"].isin(wanted)])

    out = (pd.concat(frames, ignore_index=True)
             .sort_values(["factor", "date", "stock_id"])
             .reset_index(drop=True))
    if save:
        u.output_dir.mkdir(parents=True, exist_ok=True)
        out.to_csv(u.panel_path, index=False)
    return out


def load_panel(rebuild: bool = False,
               u: Universe = BANKS_COMMODITY) -> pd.DataFrame:
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
    print(f"Top factors ({len(FACTOR_NAMES)}): " + ", ".join(
        f"{r.factor} [{r.subexperiment}]" for r in TOP_FACTORS.itertuples()))
    print(f"Saved -> {u.panel_path}")
    counts = (long.groupby("factor")["value"].size().reindex(FACTOR_NAMES))
    print("\nObservations per factor:")
    for name, c in counts.items():
        print(f"  {name:<26} {c:>9,}")
