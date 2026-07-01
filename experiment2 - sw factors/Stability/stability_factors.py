"""
stability_factors.py
====================

Compute a set of **stability factors** for the GICS *Software &
Services* universe, standardise each cross-sectionally (z-score vs the industry
mean), and write a tidy monthly panel to ``Stability/factor_panel.csv``.

This module, its driver and its outputs all live in experiment2's
``Stability/`` subfolder.  It is a *fifth* factor library for Experiment 2
(alongside ``sw_factors.py`` -> ``standard/``, ``rd_factors.py`` -> ``RD/``,
``cash_conversion`` and ``Rev & Cost/``).  Like the others it is a **drop-in for
Experiment 1's analysis engine**: the quintile sorts, cross-sectional
(Fama-MacBeth) regressions, long/short books, trading-cost model and every plot
are reused **verbatim** from
``experiment1 - general factors/{quintile,regression,cost}.py`` via the shared
engine in ``experiment1 - general factors/factors.py``.  Only the *factor
definitions* and the fundamentals they need are new here, and the universe's
``output_dir`` is the ``Stability/`` folder itself, so these factors never
collide with the other libraries.

Motivation -- second moments of earnings
----------------------------------------
The R&D extension (``rd_factors.py``) found that the one keeper among the R&D
behaviour signals was ``rd_stability`` -- the trailing coefficient of variation
of R&D intensity -- i.e. a *second moment* (consistency), not a level.  This
module pushes that idea onto the bottom line, where earnings *quality* lives:
stable, predictable earnings are a recognised quality dimension (Dichev & Tang
2009, "earnings volatility and future earnings"; Graham, Harvey & Rajgopal 2005
on smoothing).  Low-volatility earners are rewarded with lower cost of capital
and tend to outperform on a risk-adjusted basis.

    name                          definition                                                       dir          dimension
    ----------------------------  ---------------------------------------------------------------  ----------   -----------
    earning_stability             - trailing 12m coeff. of variation of EPS                        long high    earnings consistency
    rd_earning_stability          - trailing 12m coeff. of variation of (EPS / trailing-36m        long high    R&D-vs-earnings consistency
                                    avg R&D intensity)
    cashflow_stability            - trailing 12m coeff. of variation of OCF margin                 long high    cash-generation consistency
    return_stability              - trailing 12m coeff. of variation of monthly return             long high    return consistency (low-vol)
    gross_profitability_stability - trailing 12m coeff. of variation of GP/assets                  long high    gross-profitability consistency

All factors use a **trailing 12-month** window (down from 36m): a shorter window
makes each stability score a more *current* read on consistency and roughly
triples the number of scorable stock-months at the front of each name's history.

* ``earning_stability`` is the negative trailing-12m coefficient of variation
  (std / |mean|) of diluted EPS (``earnings_ltm / diluted_shares_outstanding``).
  High (near 0) => steady, predictable earnings; very negative => erratic
  earnings.  A pure second moment, orthogonal by construction to every earnings
  *level* / valuation signal in the project.
* ``rd_earning_stability`` is the negative trailing-12m coefficient of variation
  of the ratio ``EPS / avg_rd_intensity``, where ``avg_rd_intensity`` is the
  firm's **trailing-36m average R&D intensity** (``rd_ltm / sales_ltm``).  It
  measures how steadily earnings are delivered per unit of the firm's baseline
  reinvestment posture: normalising EPS by a slow-moving (36m) R&D-intensity base
  puts high- and low-R&D firms on a comparable footing, then the 12m coefficient
  of variation scores the consistency of that earnings-per-reinvestment stream.
  A firm that earns steadily against its established R&D level scores high; one
  whose earnings swing relative to that base scores low.
* ``cashflow_stability`` is the negative trailing-12m coefficient of variation of
  the operating cash-flow margin (``operating_cf_ltm / sales_ltm``).  It watches
  the consistency of *cash* generation -- harder to manage and a
  classic earnings-quality cross-check (steady cash conversion signals real,
  durable profitability; lumpy cash conversion flags accrual-driven or
  working-capital-driven earnings).  Operating cash flow is positive for ~72% of
  software stock-months (less reliably so than gross margin but far more than
  EPS), so the robust |mean| denominator and near-zero-mean guard do real work
  here.  Currency-neutral by construction (a ratio of same-currency line items).
* ``return_stability`` is the negative trailing-12m coefficient of variation of
  the monthly total return.  A high score (return std small relative to its mean)
  marks a smooth, low-volatility return stream -- the same direction as the
  low-volatility / high-Sharpe anomaly; a very negative score marks an erratic,
  lottery-like stream.  Built straight off prices, so it needs no fundamentals
  and is the most data-complete factor here; monthly returns swing sign with a
  small trailing mean, so the sign-robust |mean| denominator and near-zero-mean
  guard do the most work for this factor.
* ``gross_profitability_stability`` is the negative trailing-12m coefficient of
  variation of gross profitability (``gross_income_ltm / assets``; Novy-Marx
  2013).  It watches the consistency of gross profit *per dollar of
  assets* -- how steadily the asset base is turned into gross profit.  Like gross
  margin it is almost always positive in software and currency-neutral (a ratio
  of same-currency line items), so the near-zero-mean guard rarely binds.

EPS sign instability (important)
--------------------------------
Roughly a third of software stock-months carry *negative* earnings, and EPS can
cross zero within a trailing window, so a raw std/mean coefficient of variation
is sign-unstable and explosive.  We therefore (a) use the **absolute** mean in
the denominator (``std / |mean|`` -- the standard robust CoV for series that may
be negative) and (b) treat the score as **undefined when the window mean is
small relative to the window's average magnitude** (``|mean| < MIN_MEAN_REL *
mean|x|``), i.e. when earnings oscillate around zero and the CoV carries no
information.  Both the ratio std/|mean| and the guard |mean|/mean|x| are ratios
of same-currency quantities, so the factors are **currency-neutral** -- no FX
conversion is needed for within-industry comparison, consistent with the rest of
the project.  (A fixed absolute floor like ``rd_stability``'s would be
currency-dependent and is therefore inappropriate for a per-share quantity.)

Run standalone to (re)build the panel::

    python stability_factors.py

To run the full pipeline (panel + quintile sorts + regressions + redundancy)::

    python main_stability.py
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
        "stability_factor_engine", _EXP1_DIR / "factors.py")
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
# This module lives in (and writes to) experiment2's Stability/ subfolder, so
# these factors and their outputs live in their own subtree.  OUTPUT_DIR is the
# Stability/ folder itself (the directory holding this file), so the analysis
# outputs land directly under Stability/ (Stability/{factor_panel.csv,
# quintile/, regression/, factor_correlation/}).
OUTPUT_DIR = Path(__file__).resolve().parent

INDUSTRY_GROUP = "Software & Services"

# Estimation windows / parameters.
STAB_WINDOW = 12            # trailing months for the coefficient-of-variation moment
STAB_MIN_PERIODS = 9        # require >=9 of the trailing 12 months before a score exists
                            # (the same ~2/3-of-window coverage the 36m window used)
RD_AVG_WINDOW = 36          # trailing months for rd_earning_stability's R&D-intensity base
RD_AVG_MIN_PERIODS = 24     # require >=24 of the trailing 36 months (~2/3 coverage)
                            # before the R&D-intensity base is defined
MIN_MEAN_REL = 0.10         # the trailing |mean| must be at least this fraction of
                            # the trailing mean magnitude (mean|x|); below it the
                            # series oscillates around zero and the CoV is undefined.
                            # Currency-neutral (a ratio of same-currency means), so
                            # it replaces rd_stability's currency-bound absolute floor.

# Ordered list of the factors produced by this module.  ``higher_is_bullish`` is
# the academically expected sign of the long leg (top quintile): True => the
# canonical trade is long Q5 / short Q1.  With ``USE_CANONICAL_LS_DIRECTION``
# below set True the long/short book is signed by this prior (NOT by the
# in-sample t-stat), so a negative realised alpha t-stat means the factor worked
# *against* the hypothesis -- exactly what we want for hypothesis testing.
FACTORS: dict[str, dict] = {
    "earning_stability":    {"family": "Earnings stability (EPS consistency)",            "higher_is_bullish": True},
    "rd_earning_stability": {"family": "R&D-earnings stability (EPS-per-R&D-intensity consistency)", "higher_is_bullish": True},
    "cashflow_stability":    {"family": "Cash-flow stability (OCF-margin consistency)",      "higher_is_bullish": True},
    "return_stability":      {"family": "Return stability (monthly-return consistency)",     "higher_is_bullish": True},
    "gross_profitability_stability": {"family": "Gross-profitability stability (GP/assets consistency)", "higher_is_bullish": True},
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
    Point-in-time monthly fundamentals carrying everything the stability factors
    need: LTM earnings (net income), R&D expense, sales, LTM gross income (for
    gross margin and gross profitability), total assets (for gross
    profitability), LTM operating cash flow (for the OCF margin), and the diluted
    share count used to turn earnings into EPS.

    Each record carries ``observation_date`` (when the report became
    observable); we align on it in :func:`build_monthly_panel` via the shared
    :func:`attach_pit_fundamentals` to avoid look-ahead -- identical convention
    to Experiment 1 / ``sw_factors.py`` / ``rd_factors.py``.
    """
    base_cols = ["date_fundamental", "observation_date", "stock_id",
                 "earnings_ltm", "rd_ltm", "sales_ltm", "gross_income_ltm",
                 "operating_cf_ltm", "assets"]
    fm = pd.read_feather(DATA_DIR / "fundamental_master.feather", columns=base_cols)
    fm["stock_id"] = fm["stock_id"].astype(str)
    fm = fm[fm["stock_id"].isin(universe)].copy()

    # Diluted share count lives in the extended fundamentals table.  It has no
    # observation_date but shares the date_fundamental grid, so it merges on that
    # key and inherits the observation_date above (same convention as
    # sw_factors.py / the engine).
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
def _eps_series(p: pd.DataFrame) -> pd.Series:
    """Diluted EPS = LTM earnings / diluted shares outstanding (may be negative).

    Local-currency per-share earnings; only used inside currency-neutral
    coefficient-of-variation ratios, so no FX conversion is required."""
    earnings = p["earnings_ltm"].astype(float)
    shares = p["diluted_shares_outstanding"].astype(float)
    return earnings / shares.where(shares > 0.0)


def _rd_intensity_series(p: pd.DataFrame) -> pd.Series:
    """R&D / sales, the operating R&D-intensity ratio (sales must be positive).
    R&D is floored at 0 (a few reported R&D values are negative restatement
    artefacts; genuine R&D cannot be negative)."""
    sales = p["sales_ltm"].astype(float)
    rd = p["rd_ltm"].astype(float).clip(lower=0.0)
    return rd / sales.where(sales > 0.0)


def _ocf_margin_series(p: pd.DataFrame) -> pd.Series:
    """Operating cash-flow margin = LTM operating cash flow / LTM sales (sales
    must be positive).

    Currency-neutral (a ratio of same-currency line items).  Positive for the
    majority of software firms but can be negative, so it is only consumed inside
    the sign-robust coefficient-of-variation ratio below."""
    sales = p["sales_ltm"].astype(float)
    ocf = p["operating_cf_ltm"].astype(float)
    return ocf / sales.where(sales > 0.0)


def _return_series(p: pd.DataFrame) -> pd.Series:
    """Monthly total return (already on the panel as ``mret``).

    Returns are unitless and currency-neutral, so they feed the coefficient of
    variation directly.  The series swings sign month to month and its trailing
    mean is small relative to its dispersion, so the sign-robust |mean|
    denominator and the near-zero-mean guard in :func:`_neg_coeff_of_variation`
    do real work here."""
    return p["mret"].astype(float)


def _gross_profitability_series(p: pd.DataFrame) -> pd.Series:
    """Gross profitability = LTM gross income / total assets (Novy-Marx 2013;
    assets must be positive).

    The cleanest level measure of productivity/quality.  Currency-neutral (a
    ratio of same-currency line items) and almost always positive in software,
    so the |mean| denominator of the CoV is robust -- like gross margin, the
    near-zero-mean guard rarely binds."""
    assets = p["assets"].astype(float)
    gross = p["gross_income_ltm"].astype(float)
    return gross / assets.where(assets > 0.0)


def _neg_coeff_of_variation(p: pd.DataFrame, x: pd.Series) -> pd.Series:
    """
    Negative trailing-:data:`STAB_WINDOW`-month robust coefficient of variation
    of ``x`` per stock: ``-std / |mean|``.  High (near 0) => steady; very
    negative => erratic.

    Uses the **absolute** mean so a sign flip in ``x`` (earnings can be negative)
    does not flip the score, and marks the score undefined when ``|mean|`` is
    small relative to the window's average magnitude (the series oscillates
    around zero -> CoV uninformative).  Requires :data:`STAB_MIN_PERIODS` of
    history.
    """
    g = x.groupby(p["stock_id"], observed=True)
    mean = g.transform(lambda s: s.rolling(STAB_WINDOW, min_periods=STAB_MIN_PERIODS).mean())
    std = g.transform(lambda s: s.rolling(STAB_WINDOW, min_periods=STAB_MIN_PERIODS).std())
    abs_mean = g.transform(lambda s: s.abs().rolling(STAB_WINDOW, min_periods=STAB_MIN_PERIODS).mean())

    denom = mean.abs()
    # Undefined where the mean is dominated by sign cancellation (near-zero net
    # earnings around which the series oscillates) -- the CoV carries no signal.
    denom = denom.where(denom >= MIN_MEAN_REL * abs_mean)
    cov = std / denom
    return -cov


# --------------------------------------------------------------------------- #
# Factor definitions
# --------------------------------------------------------------------------- #
def _f_earning_stability(p: pd.DataFrame) -> pd.Series:
    """
    Earnings stability: the negative trailing-12m robust coefficient of
    variation of diluted EPS.  High => steady, predictable earnings (a quality
    hallmark -- low earnings volatility commands a lower cost of capital and
    predicts better risk-adjusted returns; Dichev & Tang 2009); very negative =>
    erratic earnings.  A second moment, orthogonal by construction to every
    earnings-level / valuation signal in the project.
    """
    return _neg_coeff_of_variation(p, _eps_series(p))


def _f_rd_earning_stability(p: pd.DataFrame) -> pd.Series:
    """
    R&D-earnings stability: the negative trailing-12m robust coefficient of
    variation of ``EPS / avg_rd_intensity``, where ``avg_rd_intensity`` is the
    firm's trailing-36m average R&D intensity (``rd_ltm / sales_ltm``).
    Normalising EPS by a slow-moving (36m) R&D-intensity base puts firms with
    different reinvestment levels on a comparable footing, then the 12m
    coefficient of variation scores the consistency of the resulting
    earnings-per-reinvestment stream: a firm that earns steadily against its
    established R&D level scores high; one whose earnings swing relative to that
    base scores low.  Combines the R&D-discipline idea of ``rd_stability`` with
    the earnings dimension of ``earning_stability``.
    """
    eps = _eps_series(p)
    rd_intensity = _rd_intensity_series(p)
    avg_rd_intensity = (rd_intensity.groupby(p["stock_id"], observed=True)
                        .transform(lambda s: s.rolling(
                            RD_AVG_WINDOW, min_periods=RD_AVG_MIN_PERIODS).mean()))
    ratio = eps / avg_rd_intensity.where(avg_rd_intensity > 0.0)
    return _neg_coeff_of_variation(p, ratio)


def _f_cashflow_stability(p: pd.DataFrame) -> pd.Series:
    """
    Cash-flow stability: the negative trailing-12m robust coefficient of
    variation of the operating cash-flow margin (``operating_cf_ltm /
    sales_ltm``).  A steady cash-flow margin signals real, durable profitability
    -- cash conversion is harder to manage than accrual earnings, so its
    consistency is a classic earnings-quality cross-check (high score); a lumpy
    cash margin flags accrual- or working-capital-driven earnings (low score).
    A margin-consistency signal one line further down the conversion chain from
    revenue to cash.
    """
    return _neg_coeff_of_variation(p, _ocf_margin_series(p))


def _f_return_stability(p: pd.DataFrame) -> pd.Series:
    """
    Return stability: the negative trailing-12m robust coefficient of variation
    of the monthly total return.  A high score (return std small relative to its
    mean) marks a smooth, low-volatility return stream -- the same direction as
    the low-volatility / high-Sharpe anomaly (steady compounders outperform on a
    risk-adjusted basis); a very negative score marks an erratic, lottery-like
    return stream.  Built straight off prices, so unlike the other members it
    needs no fundamentals and is the most data-complete factor here.
    """
    return _neg_coeff_of_variation(p, _return_series(p))


def _f_gross_profitability_stability(p: pd.DataFrame) -> pd.Series:
    """
    Gross-profitability stability: the negative trailing-12m robust coefficient
    of variation of gross profitability (``gross_income_ltm / assets``;
    Novy-Marx 2013).  It watches the consistency of gross profit
    *per dollar of assets* -- how productively the asset base
    is turned into gross profit.  A steady, high gross-profitability path is a
    durable-quality hallmark (high score); a path that swings around flags an
    unstable franchise or a churning asset base (low score).  Like gross margin
    it is almost always positive in software, so the near-zero-mean guard rarely
    binds.
    """
    return _neg_coeff_of_variation(p, _gross_profitability_series(p))


_FACTOR_FUNCS = {
    "earning_stability": _f_earning_stability,
    "rd_earning_stability": _f_rd_earning_stability,
    "cashflow_stability": _f_cashflow_stability,
    "return_stability": _f_return_stability,
    "gross_profitability_stability": _f_gross_profitability_stability,
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
    print(f"Built stability factor panel: {len(long):,} rows | "
          f"{n_stocks} stocks | {n_months} months "
          f"({long['date'].min():%Y-%m} .. {long['date'].max():%Y-%m})")
    print(f"Saved -> {u.panel_path}")
    counts = (long.groupby("factor")["value"].size().reindex(FACTOR_NAMES))
    print("\nObservations per factor:")
    for name, c in counts.items():
        print(f"  {name:<26} {c:>9,}")
