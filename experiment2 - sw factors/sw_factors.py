"""
sw_factors.py
=============

Compute the **established software-industry factors** of the project plan
(section 2.2) on the GICS *Software & Services* universe, standardise each cross-
sectionally (z-score relative to the industry mean), and write a tidy monthly
panel to ``literature/factor_panel.csv``.

This is Experiment 2.  It is a *drop-in factor library* for Experiment 1's
analysis machinery: the heavy lifting -- quintile sorts, cross-sectional
(Fama-MacBeth) regressions, long/short books and all plotting -- is reused
verbatim from ``experiment1 - general factors/{quintile,regression}.py`` via the
shared engine in ``experiment1 - general factors/factors.py``.  Everything that
is *generic* (universe handling, price->monthly aggregation primitives,
winsorisation, z-scoring, quintile assignment, OLS, ``prepare_slice``,
``load_panel`` machinery contract) is imported from that engine; only the
factor *definitions* and the fundamentals they need are new here.

Factors implemented
-------------------
Established software factors (plan section 2.2):

    name                     definition                                         dir
    -----------------------  -------------------------------------------------  ---
    intangible_value         (book_value + K_int) / market cap                  long high
    intangible_profitability (operating_income_ltm + R&D) / (assets + K_int)    long high
    rd_productivity          d(sales_ltm, YoY) / K_int                          long high
    buyback_quality          realized share reduction - gross buyback yield     long high

(The standalone realized-dilution test was dropped per the updated project
proposal; buyback_quality retains the share-count change as one of its inputs.)

All four signals are software-specific restatements of value, quality, R&D output
and capital discipline; the textbook composite scores (Piotroski F-score, Altman
Z-score) that once anchored this library have been removed, as F-score merely
re-expressed the general quality premium and Z-score inverted inside software.

The search for genuinely new software-industry factors (plan section 3.2) is
pursued separately in the R&D-behaviour extension (``rd/``), which extrapolates
the R&D activity software firms rely on rather than the two ad-hoc novel factors
of the original proposal.

``K_int`` is the intangible (knowledge) capital stock accumulated from past R&D
by perpetual inventory, ``K_int_t = (1-delta) K_int_{t-1} + R&D_t``
(Peters & Taylor 2017).  All signals are ratios or log-changes and therefore
currency-neutral, so no FX conversion is required for within-industry
comparison -- consistent with Experiment 1.

Run standalone to (re)build the panel::

    python sw_factors.py

To run the full pipeline (panel + quintile sorts + regressions) use the driver::

    python monthly_position.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Shared engine (Experiment 1)
# --------------------------------------------------------------------------- #
# Experiment 1's ``factors.py`` is a universe-parameterised engine whose generic
# pieces are factor-agnostic and fully reusable.  We load it *by path* under a
# private module name (its folder name contains spaces, so it cannot be imported
# normally, and we must not shadow the name ``factors`` -- the analysis modules
# bind to that name, and the driver points it at *this* library instead).
_EXP1_DIR = Path(__file__).resolve().parent.parent / "experiment1 - general factors"


def _load_engine():
    spec = importlib.util.spec_from_file_location(
        "sw_factor_engine", _EXP1_DIR / "factors.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_engine = _load_engine()

# Re-export the generic helpers the analysis modules (and our own build) reuse,
# so this module satisfies the exact interface ``quintile.py`` / ``regression.py``
# expect from their ``import factors as F``.
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
add_next_return = _engine.add_next_return          # generic: shift(-1) of mret
apply_mcap_screen = _engine.apply_mcap_screen      # generic: point-in-time min-mcap screen
assign_quintiles = _engine.assign_quintiles
prepare_slice = _engine.prepare_slice              # generic: operates per factor name
ols = _engine.ols


# --------------------------------------------------------------------------- #
# Paths & configuration
# --------------------------------------------------------------------------- #
OUTPUT_DIR = Path(__file__).resolve().parent / "literature"

INDUSTRY_GROUP = "Software & Services"

# Estimation windows / parameters.
YOY_LAG = 12                # year-over-year lag (months) for growth / change signals
RD_DEPRECIATION = 0.20      # delta in the K_int perpetual inventory (plan: 0.15-0.30)

# Ordered list of the factors produced by this module.  ``higher_is_bullish``
# is the academically expected sign of the long leg (top quintile): True means
# the canonical trade is long Q5 / short Q1, False means long Q1 / short Q5.  It
# fixes the long/short book's direction (see ``USE_CANONICAL_LS_DIRECTION``
# below) and does not affect any factor *value*.
FACTORS: dict[str, dict] = {
    "intangible_value":         {"family": "Intangible value",            "higher_is_bullish": True},
    "intangible_profitability": {"family": "Intangible quality",          "higher_is_bullish": True},
    "rd_productivity":          {"family": "R&D productivity",            "higher_is_bullish": True},
    "buyback_quality":          {"family": "Buyback quality",             "higher_is_bullish": True},
}
FACTOR_NAMES = list(FACTORS)

# Sign the directional long/short book by each factor's canonical literature
# direction (``higher_is_bullish`` above) rather than inferring it from the
# in-sample Fama-MacBeth t-stat.  The shared engine (Experiment 1's
# ``regression.py``) reads this flag off the injected factor library; setting it
# here applies the expected-direction convention to Experiment 2 as well.
USE_CANONICAL_LS_DIRECTION = True


# --------------------------------------------------------------------------- #
# Universe -- SINGLE SOURCE OF TRUTH for Experiment 2
# --------------------------------------------------------------------------- #
# Every Experiment 2 software factor library (this module plus the rd/, skew/
# and stability/ subexperiments) trades the *same* Software & Services
# cross-section and applies the *same* market-cap screen -- they differ only in
# where they write.  So the cross-section definition and the screen live here once
# and the subexperiments build their universe from ``software_universe`` (loading
# this module by path) instead of redefining it.

# Market-cap screen configuration (see ``factors.apply_mcap_screen``):
#   * flat point-in-time size floor -- TEMPORARILY DISABLED (0 only drops names
#     with no established USD cap); restore to 0.1e9 to re-enable the $0.1B floor.
#   * relative per-month floor -- drop the lowest 20% of active names by USD cap.
MIN_MCAP_USD: float = 0.0
MIN_MCAP_PCT: float = 0.20


def software_universe(output_dir: Path) -> Universe:
    """The shared Experiment 2 Software & Services universe, parameterised only by
    where it writes.  Bundles the cross-section (GICS industry *group*) and the
    market-cap screen so both are defined in exactly one place."""
    return Universe(
        slug="software_services",
        price_file="price_software_services.feather",
        output_dir=output_dir,
        industry_group=INDUSTRY_GROUP,
        min_mcap_usd=MIN_MCAP_USD,
        min_mcap_pct=MIN_MCAP_PCT,
    )


# This module writes to literature/; outputs mirror the Experiment 1 layout
# one-for-one: literature/{factor_panel.csv, quintile/..., regression/...}.
SOFTWARE_SERVICES = software_universe(OUTPUT_DIR)

UNIVERSES: dict[str, Universe] = {SOFTWARE_SERVICES.slug: SOFTWARE_SERVICES}


def universe_from_argv(default: Universe = SOFTWARE_SERVICES) -> Universe:
    """Pick a universe from argv[1] (its slug); fall back to ``default``.

    Experiment 2 has a single universe, but we expose this for drop-in parity
    with the engine's interface used by the analysis modules' CLI blocks.
    """
    if len(sys.argv) > 1:
        return UNIVERSES[sys.argv[1]]
    return default


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_fundamentals(universe: pd.Index) -> pd.DataFrame:
    """
    Point-in-time monthly fundamentals for the universe, carrying everything the
    software factors need.

    Each record carries ``observation_date`` -- when the underlying report became
    observable.  ``date_fundamental`` is the fiscal-period stamp and can precede
    publication, so we align on ``observation_date`` in :func:`build_monthly_panel`
    (via the shared :func:`attach_pit_fundamentals`) to avoid look-ahead --
    identical convention to Experiment 1.
    """
    base_cols = ["date_fundamental", "observation_date", "stock_id",
                 "assets", "book_value", "sales_ltm", "operating_income_ltm",
                 "sga_ltm", "rd_ltm", "buyback_ltm"]
    fm = pd.read_feather(DATA_DIR / "fundamental_master.feather", columns=base_cols)
    fm["stock_id"] = fm["stock_id"].astype(str)
    fm = fm[fm["stock_id"].isin(universe)].copy()

    # Diluted share count lives in the extended fundamentals table.  It has no
    # observation_date but shares the date_fundamental grid, so it merges on that
    # key and inherits the observation_date above.
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
# Monthly panel construction
# --------------------------------------------------------------------------- #
def build_monthly_panel(universe: pd.Index | None = None,
                        u: Universe = SOFTWARE_SERVICES) -> pd.DataFrame:
    """
    Assemble a (stock_id, period) monthly panel: monthly total return, month-end
    market cap (local and USD), the market-cap-weighted universe ("market")
    return, and the point-in-time software fundamentals.

    The price->monthly aggregation mirrors Experiment 1's engine so the two
    experiments share an identical return definition and month-end alignment.
    """
    if universe is None:
        universe = load_universe(u)

    px = load_prices(universe, u)
    px["period"] = px["date"].dt.to_period("M")
    px["gross"] = 1.0 + px["total_return"].fillna(0.0)

    # Compound daily returns to a monthly total return; carry month-end levels.
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

    # Attach point-in-time fundamentals, aligned on observation_date so a report
    # only enters a month-end once it was actually observable (no look-ahead).
    fund = load_fundamentals(universe)
    monthly = attach_pit_fundamentals(monthly, fund)

    monthly = monthly.sort_values(["stock_id", "period"]).reset_index(drop=True)
    return monthly


# --------------------------------------------------------------------------- #
# Intangible (knowledge) capital -- perpetual inventory of past R&D
# --------------------------------------------------------------------------- #
def knowledge_capital(panel: pd.DataFrame, delta: float = RD_DEPRECIATION) -> pd.Series:
    """
    Intangible capital stock ``K_int`` accumulated from past R&D by perpetual
    inventory (Peters & Taylor 2017):  ``K_int_t = (1-delta) K_int_{t-1} + R&D_t``.

    The recursion is annual in the source paper; our panel is monthly, so we run
    it at monthly frequency with a monthly retention ``rho = (1-delta)^(1/12)``
    and a monthly R&D flow ``rd_ltm / 12`` (so 12 months of a stable LTM figure
    accumulate ~one year of R&D and depreciate by ``delta`` -- a smooth monthly
    analogue of the annual stock).  Each stock is seeded at its steady-state
    level ``flow_0 / (1 - rho)`` (== annual R&D / delta), the standard initial
    stock.  Firms with no R&D (much of IT Services) get ``K_int = 0``.
    """
    rho = (1.0 - delta) ** (1.0 / 12.0)
    flow = (panel["rd_ltm"].astype(float) / 12.0).clip(lower=0.0).fillna(0.0)

    def _accumulate(f: pd.Series) -> pd.Series:
        vals = f.to_numpy(dtype=float)
        k = np.empty(len(vals), dtype=float)
        if len(vals):
            k[0] = vals[0] / (1.0 - rho)          # steady-state seed
            for i in range(1, len(vals)):
                k[i] = rho * k[i - 1] + vals[i]
        return pd.Series(k, index=f.index)

    return flow.groupby(panel["stock_id"], observed=True, group_keys=False).apply(_accumulate)


# --------------------------------------------------------------------------- #
# Factor definitions
# --------------------------------------------------------------------------- #
# Each function takes the sorted monthly panel and returns a Series aligned to
# its index.  Time-series operations (YoY changes) are done per stock; cross-
# sectional standardisation is handled later in :func:`add_zscores`.
def _yoy_change(p: pd.DataFrame, col: str) -> pd.Series:
    """Year-over-year change in ``col`` (level_t - level_{t-12m}), per stock."""
    prev = p.groupby("stock_id", observed=True)[col].shift(YOY_LAG)
    return p[col] - prev


def _f_intangible_value(p: pd.DataFrame) -> pd.Series:
    # Intangible-adjusted book-to-market: add the capitalised R&D stock back into
    # book equity (which expensing R&D understates), then scale by market cap.
    return (p["book_value"] + p["k_int"]) / p["security_mcap_local"]


def _f_intangible_profitability(p: pd.DataFrame) -> pd.Series:
    # Intangible-adjusted profitability: treat R&D as investment (add it back to
    # operating income rather than expensing it) and recognise the intangible
    # capital it created in the asset base.
    profit = p["operating_income_ltm"] + p["rd_ltm"].fillna(0.0)
    return profit / (p["assets"] + p["k_int"])


def _f_rd_productivity(p: pd.DataFrame) -> pd.Series:
    # Does research convert into growth?  YoY sales change per unit of R&D capital.
    return _yoy_change(p, "sales_ltm") / p["k_int"]


def _f_buyback_quality(p: pd.DataFrame) -> pd.Series:
    # Buyback reconciliation: realised reduction in share count minus the gross
    # buyback yield.  ``buyback_ltm`` is a cash-flow figure (negative = cash spent
    # repurchasing), so gross buyback yield = -buyback_ltm / market cap.  A firm
    # that spends on buybacks (positive yield) but whose share count does not fall
    # (reduction ~ 0, because grants offset the repurchases) scores negative -- a
    # low-quality tell.  Genuine, share-reducing buybacks score near zero/positive.
    sh = p["diluted_shares_outstanding"]
    prev = sh.groupby(p["stock_id"], observed=True).shift(YOY_LAG)
    realized_reduction = 1.0 - sh / prev                       # >0 if shares shrank
    buyback_yield = -p["buyback_ltm"] / p["security_mcap_local"]  # >0 if repurchasing
    return realized_reduction - buyback_yield


_FACTOR_FUNCS = {
    "intangible_value": _f_intangible_value,
    "intangible_profitability": _f_intangible_profitability,
    "rd_productivity": _f_rd_productivity,
    "buyback_quality": _f_buyback_quality,
}


def compute_factors(panel: pd.DataFrame) -> pd.DataFrame:
    """Add the intangible-capital stock and a raw value column for every factor."""
    panel["k_int"] = knowledge_capital(panel)
    for name in FACTOR_NAMES:
        panel[name] = _FACTOR_FUNCS[name](panel).replace([np.inf, -np.inf], np.nan)
    return panel


# --------------------------------------------------------------------------- #
# Cross-sectional standardisation & tidy output
# --------------------------------------------------------------------------- #
# These mirror the engine's versions but iterate *this* module's FACTOR_NAMES;
# the per-column maths (winsorise + z-score) is reused from the engine.
def add_zscores(panel: pd.DataFrame) -> pd.DataFrame:
    for name in FACTOR_NAMES:
        panel[f"{name}_z"] = cross_sectional_zscore(panel[name], panel["period"])
    return panel


def to_long_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape to one row per (date, stock_id, factor) with the factor value, its
    z-score, the next-period return, and the market-cap ``weight`` (month-end USD
    market cap).  Rows with a missing factor value are dropped to keep the file
    lean.  Identical schema to Experiment 1.
    """
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
    """Full build: panel -> factors -> next return -> mcap screen -> z-scores -> tidy long.

    Screen order matches the engine's ``build``: the point-in-time min-mcap screen
    runs after the time-series factors and ``next_return`` (built on full history)
    but before the cross-sectional z-scores.
    """
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
    print(f"Built software factor panel: {len(long):,} rows | "
          f"{n_stocks} stocks | {n_months} months "
          f"({long['date'].min():%Y-%m} .. {long['date'].max():%Y-%m})")
    print(f"Saved -> {u.panel_path}")
    counts = (long.groupby("factor")["value"].size().reindex(FACTOR_NAMES))
    print("\nObservations per factor:")
    for name, c in counts.items():
        print(f"  {name:<26} {c:>9,}")
