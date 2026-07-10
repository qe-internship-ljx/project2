"""
final_portfolio.py
==================

Final Portfolio -- the deliverable long/short book.

A single quarterly-repositioned composite that longs / shorts the industry on the
**weighted sum of five factor z-scores**::

    score_{i,t} = sum_f  sign_f * w_f(t) * zscore_{f,i,t}

over the factors

    gross_profitability, rd_stability, revenue_growth_stability,
    revenue_growth_skewness, beta

with **every factor weighted 1 except ``rd_stability``**, which carries a
*state-dependent* weight:

    w_rd(t) = 5.0   when the current rd_stability top-minus-bottom factor-value
                    spread magnitude exceeds its trailing-12-month average
              0.5   otherwise (the normal weight)

The spread signal is Experiment 4's factor-spread-timing pipeline, reused verbatim
(``spread_timing.value_spread`` / ``spread_timing.timing_flags``): the raw factor
value of the top bucket minus the bottom bucket of the *same* n-bucket sort the book
trades, compared to its own trailing-12m average (both known at ``t``, so the boost
is strictly look-ahead free).  Because the book is repositioned quarterly, the weight
that decides a held quarter is the one at that quarter's reposition month; the score
is nonetheless formed every month (the quarterly primitive holds the reposition-date
membership), exactly as ``composite.py`` does.

0.5% single-name ownership cap (enforced at each reposition)
-----------------------------------------------------------
No position may own more than **0.5% of a name's own market cap** at formation.  At
each quarterly reposition the book starts from equal weights inside each leg; any name
whose $50M-leg position would breach 0.5% of its cap is pinned at its cap weight and the
freed weight is redistributed pro-rata across the still-uncapped names, water-filled
until nothing breaches the cap (``cap_ownership``, reused verbatim from Experiment 5's
``capacity_scaling._cap_ownership``).  The cap is binding (largest formation ownership
sits at exactly 0.5000%), and the book is genuinely traded at the capped weights: the
long/short spread is the capped-weight top-minus-bottom return (``capped_spread``), and
the turnover cost, the volume drag and the ownership diagnostic all consume the same
capped legs (``capped_legs``).  The capped weights are computed once per reposition and
**held fixed across the quarter** (``quarter_position.quarter_hold``), so a name carried
across a held month keeps its reposition weight (``Δw = 0``) and is charged no cost or
drag for holding -- only the reposition rebalance trades.  (A name's mark-to-market
ownership can drift above 0.5% intra-quarter as its cap moves; the cap is a formation
constraint, and ``formation_ownership`` reports the enforced figure.)

Every generic mechanism is reused, nothing is re-implemented:

* the constituent z-score matrix, the score -> tidy-panel shaping, the
  quarterly-held bucket membership, the industry-neutral alpha / windowed performance,
  the turnover cost, the largest-single-name ownership diagnostic and the
  performance-table rendering are all Experiment 3's ``composite.py`` (which itself
  loads Experiment 1's engine + cost + regression and Experiment 2's quarterly
  primitives by path -- the project's dependency-injection convention);
* the rd_stability spread-timing signal is Experiment 4's ``spread_timing.py``;
* the 0.5% ownership-cap water-fill is Experiment 5's ``capacity_scaling._cap_ownership``.

This module adds only (a) the state-dependent rd_stability weighting, (b) the
ownership-capped weighted book, and (c) the **average-daily-volume trading-drag**
described next.

Trading-volume drag (the capacity constraint)
---------------------------------------------
Under a **$100M dollar-neutral** book ($50M per leg) we assume we may trade at most
**10% of a name's trailing-12-month average daily USD volume (ADV) per day**.  The
drag is charged **only on the weight actually traded** -- the reposition-date
rebalance.  Because the ownership-capped weights are held fixed across the quarter
(see :func:`capped_legs`), a name carried across a held month has ``|dw| = 0`` and is
charged nothing for simply holding; the drag falls entirely on the quarterly
reposition, when a name is opened, closed or re-weighted.  For each such trade, if it
cannot clear in one day the excess tail is **delayed** and we suffer the *opportunity
cost* of the return foregone on the delayed notional over the days it takes to
complete (an "untradable-tail" drag, not a fixed slippage rate):

    trade$_{i,t}  = |pos$_{i,t} - pos$_{i,t-1}|          (leg-capital * |dw|)
    cap$_{i,t}    = 0.10 * ADV_usd_{i,t}                 (one day's tradable notional)
    days_{i,t}    = trade$_{i,t} / cap$_{i,t}

If ``days <= 1`` the trade clears on the reposition day and costs nothing.  If
``days > 1`` the position is legged in linearly over ``days`` trading days, so on
average half the delayed notional is out of the market for the ramp.  The ramp is
**capped at the quarterly holding horizon** (``RAMP_HORIZON_DAYS`` = 63 trading days
= 3 months): the book repositions every quarter, so a name can never be legged in over
more than the ~63 days it is actually held -- a name whose trade needs longer is
treated as never fully established during the hold.  The drag on a name that reposition
is the fraction of **its own realised next-period return** the delayed notional
foregoes -- *not* a fraction of the whole long/short spread (a position being built
misses the return of the name it is building):

    ramp_frac       = min(days, RAMP_HORIZON_DAYS) / TRADING_DAYS_PER_MONTH  (months of ramp)
    delayed_weight  = |dw|_{i,t}                       (leg-weight actually traded)
    drag_i,t (ret)  = 0.5 * delayed_weight * ramp_frac * next_return_{i,t}

The 0.5 linear-ramp factor accounts for the entry and exit ramps together (each side
delays the same ``|dw|`` tail by half a ramp window).  The book-level drag is the
**long-leg foregone return minus the short-leg foregone return** (a long-leg name's
missed return reduces the spread; a short-leg name's -- whose return we are short --
adds to it), so it is directly comparable to, and subtracted from, the gross spread --
the same footing Experiment 4 nets its turnover cost on.  Reported both as an average
monthly drag (pp) and folded into a net-of-drag mean / Sharpe.

The ADV panel is the trailing-252-trading-day mean of daily
``volume * price_local * fx_to_usd`` (USD dollar volume) from the same price feather
and FX table the cost model uses, sampled at each formation month-end -- so no new
data source is introduced.

Outputs (``output/quarter_quintile/`` and ``output/quarter_tertile/``)
----------------------------------------------------------------------
    <sort>_cumulative.png     the buckets as cumulative growth of $1 (log scale)
    <sort>_performance.png    the L/S book's performance, using the SAME compact row
                              set as Experiment 6's liquidity-capped table: industry-
                              neutral alpha (+ t-stat), beta-neutral Sharpe, the
                              beta-neutral net-of-cost/drag Sharpe (bold -- nets BOTH
                              the turnover cost and the volume drag), the average
                              turnover cost, the largest single-name ownership and the
                              trading-volume drag (monthly), full sample + 2016+.  The
                              cost and the drag are also shown on their own rows for
                              transparency and are never netted from the gross alpha.
                              (``<sort>`` = ``quintile`` / ``tertile``)
    <sort>_factor_contribution.png
                              the **per-factor return contribution**: the realised L/S
                              spread regressed on the five constituents' standalone
                              univariate quarterly L/S spreads simultaneously
                              (``port_t = alpha + sum_f b_f * factor_f_t + eps_t``), so
                              each coefficient is the portfolio's loading on that
                              factor's return holding the others fixed, with the
                              regression R^2 / month count.  (Same attribution as
                              Experiment 6's weighted composite; nothing is imported
                              from that gitignored experiment.)

Run standalone::

    python final_portfolio.py          # both the quintile and tertile books
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths & dependency injection -- reuse Experiment 3's composite plumbing and
# Experiment 4's spread-timing signal.
# --------------------------------------------------------------------------- #
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
_EXP3_DIR = _PROJECT_ROOT / "experiment3 - multifactor"
_EXP4_DIR = _PROJECT_ROOT / "experiment4 - timing"
sys.path.insert(0, str(_EXP3_DIR))          # composite.py loads the engine (factors/cost/regression)
sys.path.insert(0, str(_EXP4_DIR))          # spread_timing.py (rd-stability spread signal)

import composite as C          # noqa: E402  the composite spine (+ engine, cost, regression, QP)
import spread_timing as ST     # noqa: E402  Exp4 factor-spread-timing signal

F = C.F                        # Experiment 1 engine, wired for the software universe
COST = C.COST                  # the turnover-cost model
QP = C.QP                      # Experiment 2 quarterly primitives

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
# The five constituents, and the state-dependent rd_stability weighting.  Every
# factor is weighted 1 except rd_stability (RD_WEIGHT_NORMAL, boosted to
# RD_WEIGHT_BOOST when its factor-value spread exceeds its trailing average).
FACTORS = ["gross_profitability", "rd_stability", "revenue_growth_stability",
           "revenue_growth_skewness", "beta"]
BASE_WEIGHTS = {f: 1.0 for f in FACTORS}
RD_FACTOR = "rd_stability"
RD_WEIGHT_NORMAL = 0.5
RD_WEIGHT_BOOST = 5.0
LOOKBACK = ST.LOOKBACK          # 12 trailing months for the spread average (Exp4's default)

# Trading-volume drag: $100M dollar-neutral book, 10% of ADV/day participation cap.
PORTFOLIO_CAPITAL = C.PORTFOLIO_CAPITAL       # $100M total (shared with the ownership diagnostic)
LEG_CAPITAL = C.LEG_CAPITAL                   # $50M per leg

# Single-name ownership ceiling held at ALL times: no position may own more than
# 0.5% of a name's own market cap.  Any within-leg weight that would breach it is
# pushed down to the cap and the freed weight redistributed across the rest of the
# leg (water-filled by :func:`cap_ownership`).  Same ceiling as Experiment 5's
# ownership-threshold variant.
OWNERSHIP_CAP = 0.005                          # 0.5% of market cap
MAX_PARTICIPATION = 0.10                      # trade at most 10% of ADV per day
TRADING_DAYS_PER_MONTH = 21.0                 # to express ramp days as a fraction of a month
# Cap a single ramp at the QUARTERLY HOLDING HORIZON (~63 trading days = 3 months):
# the book repositions every quarter, so a name can never be legged in over more than
# the ~63 trading days it is actually held.  A name whose 10%-of-ADV/day trade needs
# more than a quarter is treated as never fully established during the hold, and its
# drag is the opportunity cost over the full holding horizon (not silently clipped to
# one month).
RAMP_HORIZON_DAYS = 3.0 * TRADING_DAYS_PER_MONTH   # 63 trading days (one quarter)

DECADE_START = C.DECADE_START                 # 2016-01-01 "past decade" cut-off
N_QUINTILES = C.N_QUINTILES
N_TERTILES = C.N_TERTILES
OUTPUT_DIR = _THIS_DIR / "output"
UNIVERSE = F.SOFTWARE_SERVICES


# --------------------------------------------------------------------------- #
# Step 1 -- the state-dependent rd_stability weight
# --------------------------------------------------------------------------- #
def rd_boost_flag(n: int) -> pd.Series:
    """Per-formation-month boolean: is the rd_stability factor-value spread above
    its trailing-12m average this month?  ``True`` months take the boosted weight.

    Reuses Experiment 4's factor-spread-timing pipeline verbatim -- the top-minus-
    bottom raw factor-value spread of the same ``n``-bucket sort the book trades
    (``spread_timing.value_spread``) and its trailing-average gate
    (``spread_timing.timing_flags``) -- so the "magnitude of the spread exceeds the
    trailing 12-month average" rule is defined identically to the timing experiment.
    Months without a full trailing window are ``False`` (undefined signal -> normal
    weight)."""
    panel = _rd_panel()
    spread = ST.value_spread(panel, RD_FACTOR, n)
    trailing, _ = ST.timing_flags(spread, LOOKBACK)
    flag = (spread > trailing).where(trailing.notna(), other=False)
    return flag.astype(bool).rename("rd_boost")


_RD_PANEL: pd.DataFrame | None = None


def _rd_panel() -> pd.DataFrame:
    """The rd_stability source panel (date, stock_id, factor, value, zscore,
    next_return), read once.  ``value`` is needed for the factor-value spread."""
    global _RD_PANEL
    if _RD_PANEL is None:
        meta = C.resolve_factors([RD_FACTOR]).iloc[0]
        _RD_PANEL = pd.read_csv(meta["panel_path"], parse_dates=["date"],
                                usecols=["date", "stock_id", "factor", "value",
                                         "zscore", "next_return"])
        _RD_PANEL["stock_id"] = _RD_PANEL["stock_id"].astype(str)
    return _RD_PANEL


def weighted_composite(resolved: pd.DataFrame, n: int) -> pd.DataFrame:
    """The weighted composite score per ``(date, stock_id)`` with the realised
    ``next_return`` attached (``date, stock_id, composite, next_return``).

    Orients each constituent z-score to its bullish ``sign`` and forms the weighted
    sum ``sum_f sign_f * w_f(t) * z``.  Every factor carries weight 1 except
    ``rd_stability``: its column is scaled per-month by :func:`rd_boost_flag`
    (0.5 normally, 5.0 in the boosted months).  Reuses ``composite.load_exposures``
    for the raw z-score matrix and ``composite.scored_frame`` for the return join --
    only the column weighting is new."""
    exposures, next_ret = C.load_exposures(resolved)          # wide (date,stock_id) x factor z-scores
    signs = pd.Series(dict(zip(resolved["factor"], resolved["sign"])))
    oriented = exposures.mul(signs, axis=1)                   # +/-1 per factor

    # Per-month rd_stability weight: 5.0 in boosted months, else 0.5; every other
    # factor is a flat 1.0.
    boost = rd_boost_flag(n)
    dates = oriented.index.get_level_values("date")
    boost_aligned = boost.reindex(dates).astype("boolean").fillna(False).to_numpy(dtype=bool)
    rd_w = pd.Series(np.where(boost_aligned, RD_WEIGHT_BOOST, RD_WEIGHT_NORMAL),
                     index=oriented.index)

    weighted = oriented.copy()
    for f in resolved["factor"]:
        w = rd_w if f == RD_FACTOR else BASE_WEIGHTS[f]
        weighted[f] = weighted[f].mul(w)
    score = weighted.sum(axis=1)
    return C.scored_frame(score, next_ret)


# --------------------------------------------------------------------------- #
# Step 2 -- the ownership-capped, weighted long/short book
#
# No single-name position may own more than 0.5% of that name's market cap at
# formation.  Membership is the quarterly-held top/bottom bucket of the composite
# (the same as ``composite.quintile_legs``); at each reposition, starting from equal
# weights inside each leg, any name whose $50M-leg position would own more than 0.5%
# of its cap is water-filled down and the freed weight redistributed across the leg.
# Those capped weights are then held fixed across the quarter, so the book only trades
# at the repositions and holding a position costs nothing.  The book is genuinely
# traded at the capped weights (spread, cost, drag and the ownership diagnostic all
# consume them).
# --------------------------------------------------------------------------- #
def cap_ownership(legs: pd.DataFrame, cap: float, leg_capital: float) -> pd.Series:
    """Water-fill within-leg weights ``w`` so no name's point-in-time ownership share
    (``leg_capital * w / mcap``) exceeds ``cap``: any name over the cap is pinned at
    its ownership-cap weight (``cap * mcap / leg_capital``) and the freed weight is
    redistributed proportionally to the still-uncapped names, iterating within each
    ``(date, leg)`` until no name breaches the cap (or every name is pinned).  Returns
    the capped ``w`` (summing to one per leg where feasible), index-aligned to
    ``legs``.  Reused verbatim from Experiment 5's ``capacity_scaling._cap_ownership``
    (the project's one implementation of the 0.5% ownership cap)."""
    w = legs["w"].copy()
    ceiling = cap * legs["mcap"] / leg_capital          # ownership-cap weight per name
    key = [legs["date"], legs["leg"]]
    over = w > ceiling
    while over.any():
        w = w.mask(over, ceiling)
        pinned = over | (w >= ceiling)                  # names already at their ceiling
        free_w = w.mask(pinned, 0.0)
        # Redistribute the weight freed by pinning across the unpinned names, pro-rata.
        headroom = (1.0 - w.where(pinned, 0.0).groupby(key).transform("sum"))
        free_sum = free_w.groupby(key).transform("sum")
        scale = (headroom / free_sum).where(free_sum > 0, 1.0)
        w = w.where(pinned, free_w * scale)
        over = w > ceiling + 1e-15
    return w


def capped_legs(panel: pd.DataFrame, n: int,
                cap: float = OWNERSHIP_CAP) -> pd.DataFrame:
    """Tidy ``date, stock_id, leg, w, mcap, next_return`` for the composite's top and
    bottom buckets, repositioned quarterly, with each name's within-leg weight ``w``
    equal-weighted then **water-filled to the 0.5% ownership cap** (:func:`cap_ownership`).

    The book is **only traded at the quarterly reposition months** (end of
    Feb/May/Aug/Nov), so the ownership-capped weights are computed **once at each
    reposition month** and then **held fixed across the two following months** via the
    shared ``quarter_position.quarter_hold`` primitive.  A name carried across a held
    month therefore keeps exactly its reposition-date weight (``Δw = 0``), so the
    turnover cost and the volume drag charge it nothing for simply holding -- only the
    reposition-date rebalance trades.  (Recomputing the cap every month would let the
    weights drift with market cap and spuriously charge a "trade" every held month.)

    Membership is the shared quarterly-held bucket assignment
    (``quarter_position.quarter_held_membership`` on ``factors.prepare_slice``); the
    formation-date USD market cap is ``composite.market_cap_panel``.  ``leg`` is
    relabelled ``top`` / ``bottom``.  This is the single book behind the spread, the
    turnover cost, the volume drag and the ownership diagnostic, so every quantity is
    measured on the exact capped positions actually held."""
    held = QP.quarter_held_membership(F.prepare_slice(panel, C.COMPOSITE_FACTOR, n))
    legs = held.loc[held["leg"].isin([1.0, float(n)]),
                    ["date", "stock_id", "leg", "next_return"]].copy()
    legs = legs.merge(C.market_cap_panel(), on=["date", "stock_id"], how="left")
    legs = legs.dropna(subset=["mcap"])
    legs["leg"] = np.where(legs["leg"] == float(n), "top", "bottom")

    # Compute the equal-weight -> ownership-capped weights at the REPOSITION months
    # only, using the reposition-date market cap; then hold ``w`` (and its formation
    # ``mcap``) fixed across the quarter with the shared ``quarter_hold`` primitive.
    # Each held month carries the weight the name was given at its quarter's reposition
    # date, so a carried position has Δw = 0 and is charged no cost / drag for holding
    # (only the reposition-date rebalance trades).
    period = legs["date"].dt.to_period("M")
    is_reposition = (period.dt.month - 2) % 3 == 0
    rp = legs[is_reposition].copy()
    rp["w"] = 1.0 / rp.groupby(["date", "leg"], observed=True)["stock_id"].transform("size")
    rp["w"] = cap_ownership(rp, cap, LEG_CAPITAL)

    # Attach the reposition-date w / mcap to every month, then quarter_hold them so
    # each held month carries its quarter's reposition-date value (Δw = 0 when held).
    with_rp = legs.drop(columns=["mcap"]).merge(
        rp[["date", "stock_id", "leg", "w", "mcap"]],
        on=["date", "stock_id", "leg"], how="left")
    held_w = QP.quarter_hold(with_rp, ["w", "mcap"])
    return held_w[["date", "stock_id", "leg", "w", "mcap", "next_return"]]


def capped_spread(legs: pd.DataFrame) -> pd.Series:
    """Monthly dollar-neutral long/short return of the ownership-capped book: the
    capped-weight mean next-period return of the top leg minus the bottom leg (the
    composite is always long the high-score names, so sign is +1).  Because ``w`` sums
    to one within each ``(date, leg)``, ``Σ_i w_i · next_return_i`` per leg is the
    weighted mean -- the ownership-capped counterpart of ``composite.bucket_returns``'
    equal-weighted spread."""
    leg_ret = ((legs["w"] * legs["next_return"])
               .groupby([legs["date"], legs["leg"]], observed=True).sum()
               .unstack("leg").sort_index())
    return (leg_ret["top"] - leg_ret["bottom"]).rename("ls")


def formation_ownership(legs: pd.DataFrame) -> pd.Series:
    """Monthly largest single-name **formation** ownership share of the capped book:
    ``LEG_CAPITAL * w / mcap`` where ``mcap`` is the position's *formation-date* market
    cap held across the quarter (the same cap the 0.5% water-fill was applied to).  So
    this is the ownership the cap actually enforces -- bounded at :data:`OWNERSHIP_CAP`
    (0.5%) by construction -- rather than ``composite.leg_ownership``'s drifting
    intra-quarter mark-to-market cap (which can rise above the cap between repositions
    without any trading, and is not what the cap constrains)."""
    share = LEG_CAPITAL * legs["w"].abs() / legs["mcap"]
    return share.groupby(legs["date"]).max().sort_index()


# --------------------------------------------------------------------------- #
# Step 3 -- average-daily-volume (ADV) panel + the untradable-tail drag
# --------------------------------------------------------------------------- #
_ADV_PANEL: pd.DataFrame | None = None


def adv_panel() -> pd.DataFrame:
    """Trailing-12-month **average daily USD volume** per ``(date, stock_id)``,
    sampled at each formation month-end (tidy ``date, stock_id, adv_usd``).

    Daily USD dollar volume is ``volume * price_local * fx_to_usd`` -- the daily
    share volume times the local price times the trading currency's daily FX-to-USD
    (the same FX table ``cost.py`` uses).  It is meaned over the trailing 252 trading
    days at each month-end, so ``adv_usd`` is the average notional we could trade in a
    day.  Built once and cached; missing names get NaN and drop out of the drag."""
    global _ADV_PANEL
    if _ADV_PANEL is not None:
        return _ADV_PANEL

    universe = F.load_universe(UNIVERSE)
    px = pd.read_feather(F.DATA_DIR / UNIVERSE.price_file,
                         columns=["stock_id", "date", "volume", "price_local"])
    px["stock_id"] = px["stock_id"].astype(str)
    px = px[px["stock_id"].isin(universe)].copy()
    px["date"] = pd.to_datetime(px["date"])

    # Daily local dollar volume, converted to USD via the trading currency's daily FX.
    sm = pd.read_feather(F.DATA_DIR / "security_master.feather",
                         columns=["stock_id", "currency_code"])
    sm["stock_id"] = sm["stock_id"].astype(str)
    sm["currency_code"] = sm["currency_code"].astype(str)
    px = px.merge(sm, on="stock_id", how="left")

    fx = pd.read_feather(F.DATA_DIR / "fx_rates.feather")
    fx["currency_code"] = fx["currency_code"].astype(str)
    fx["date"] = pd.to_datetime(fx["date"]).astype("datetime64[ns]")
    px["date"] = px["date"].astype("datetime64[ns]")
    px = px.sort_values("date")
    fx = fx.sort_values("date")
    px = pd.merge_asof(px, fx, on="date", by="currency_code", direction="backward")
    px["dollar_vol_usd"] = px["volume"] * px["price_local"] * px["fx_to_usd"]

    # Trailing-252-trading-day mean per stock, then sample at each month-end.
    px = px.dropna(subset=["dollar_vol_usd"]).sort_values(["stock_id", "date"])
    px["adv_usd"] = (px.groupby("stock_id", observed=True)["dollar_vol_usd"]
                       .transform(lambda s: s.rolling(252, min_periods=63).mean()))
    px["period"] = px["date"].dt.to_period("M")
    monthly = (px.dropna(subset=["adv_usd"])
                 .groupby(["stock_id", "period"], observed=True)
                 .agg(adv_usd=("adv_usd", "last"))
                 .reset_index())
    monthly["date"] = monthly["period"].dt.to_timestamp(how="end").dt.normalize()
    _ADV_PANEL = monthly[["date", "stock_id", "adv_usd"]].reset_index(drop=True)
    return _ADV_PANEL


def volume_drag(legs: pd.DataFrame) -> pd.Series:
    """Monthly **untradable-tail volume drag** of the book, as a return subtracted
    from the gross long/short spread, indexed by formation month.

    ``legs`` is the tidy ``date, stock_id, leg, w, next_return`` capped membership
    (``leg`` = ``top`` / ``bottom``).  For each name and month the traded weight
    ``|dw|`` maps to a dollar trade ``LEG_CAPITAL * |dw|``; the days to clear it at
    10% of the name's ADV give a ramp length (capped at the quarterly holding horizon),
    over which the freshly-traded notional is only half in the market on average.

    The drag on a name is the fraction of **its own realised next-period return** that
    the delayed notional foregoes -- ``ramp_frac * |dw| * next_return`` -- **not** a
    fraction of the whole L/S spread: a position being built misses the return of the
    name it is building, so the base is that name's return.  A drag on a *long*-leg
    name reduces the book's spread; a drag on a *short*-leg name (whose return we are
    short) has the opposite sign on the spread.  The book-level drag is therefore the
    long-leg foregone return minus the short-leg foregone return -- directly comparable
    to the spread it reduces.  Names with no ADV contribute no drag."""
    adv = adv_panel()
    dates = pd.Index(sorted(legs["date"].unique()), name="date")

    # Per-name traded weight |dw| each month, per leg (the same differencing the cost
    # model uses: establish the book in month 0, then charge only the weight changed).
    # Because the capped weights are held across the quarter, |dw| is ~0 in held
    # months -- the drag falls on the reposition trades only.
    per_leg_drag = {}
    for leg_name, g in legs.groupby("leg", observed=True):
        w = (g.pivot_table(index="date", columns="stock_id", values="w", fill_value=0.0)
              .reindex(dates, fill_value=0.0))
        dw = w.diff().abs()
        dw.iloc[0] = w.iloc[0]                              # month 0 establishes the book
        traded = (dw.reset_index().melt(id_vars="date", var_name="stock_id", value_name="dw"))
        traded = traded[traded["dw"] > 0.0]
        # Attach the name's ADV (to size the ramp) and its realised return (the base
        # of the foregone return).
        traded = traded.merge(adv, on=["date", "stock_id"], how="left")
        traded = traded.merge(g[["date", "stock_id", "next_return"]],
                              on=["date", "stock_id"], how="left")
        traded = traded.dropna(subset=["adv_usd", "next_return"])

        days = (LEG_CAPITAL * traded["dw"]) / (MAX_PARTICIPATION * traded["adv_usd"])
        ramp_days = np.minimum(days, RAMP_HORIZON_DAYS)
        # Drag only on the tail that cannot clear in a single day (days > 1); linear
        # ramp (0.5) over ramp_days as a fraction of the trading month.
        ramp_frac = np.where(days > 1.0,
                             0.5 * ramp_days / TRADING_DAYS_PER_MONTH, 0.0)
        # Foregone return on the delayed notional = ramp_frac * traded weight * the
        # name's own realised return.
        traded["leg_drag"] = ramp_frac * traded["dw"] * traded["next_return"]
        per_leg_drag[leg_name] = (traded.groupby("date")["leg_drag"].sum()
                                        .reindex(dates, fill_value=0.0))

    top = per_leg_drag.get("top", pd.Series(0.0, index=dates))
    bottom = per_leg_drag.get("bottom", pd.Series(0.0, index=dates))
    return (top - bottom).rename("volume_drag")


# --------------------------------------------------------------------------- #
# Step 4 -- drag metrics for the performance table
# --------------------------------------------------------------------------- #
def _win(s: pd.Series, start, end) -> pd.Series:
    if start is not None:
        s = s[s.index >= start]
    if end is not None:
        s = s[s.index <= end]
    return s


def attach_drag(stats: dict, drag: pd.Series, start=None, end=None) -> dict:
    """Add the average monthly **trading-volume drag** (``adv_drag``, mean of the drag
    series over the window, a fraction of notional) to a :func:`composite.book_stats`
    window dict, in place.  Reported for information -- like the turnover cost, it is
    not netted from the gross alpha / Sharpe -- matching Experiment 6's
    ``liquidity_capped_composite`` table."""
    stats["adv_drag"] = float(_win(drag, start, end).mean())
    return stats


def drag_metric() -> tuple[str, str, str, bool]:
    """The ``render_performance`` ``extra_metrics`` row for the average monthly
    trading-volume drag (matching Experiment 6's ``liquidity_capped`` table row)."""
    return ("Trading-volume drag (monthly)", "adv_drag", "pct", False)


def net_cost_neutral_metric() -> tuple[str, str, str, bool]:
    """The ``render_performance`` row for the **beta-neutral** net-of-cost/drag Sharpe
    (keyed ``sharpe_cost_neutral``, populated by ``composite.attach_net_cost_sharpe``
    fed the combined turnover-cost + volume-drag series), rather than the renderer's
    default combined "raw / β-neut" row.  Same row as Experiment 6's ``liquidity_capped``
    table, but the netted friction here is turnover cost **plus** the trading-volume
    drag."""
    return ("Sharpe net of cost/drag (β-neut)", "sharpe_cost_neutral", "num", False)


def cost_metric() -> tuple[str, str, str, bool]:
    """The turnover-cost row as an explicit ``extra_metrics`` entry (keyed
    ``avg_cost_row``) so it is ordered by the caller -- above the ownership and drag
    rows -- rather than auto-appended last by the renderer, which only auto-appends its
    own ``avg_cost`` row when every window carries that key.  We deliberately store the
    value under a different key here (matching Experiment 6's convention)."""
    return ("Avg monthly cost (turnover)", "avg_cost_row", "pct", False)


# --------------------------------------------------------------------------- #
# Step 5 -- per-factor return contribution (multivariate attribution)
#
# Regress the Final Portfolio's realised long/short spread on the standalone
# long/short spreads of each of the five constituent factors simultaneously:
#
#     port_t = alpha + sum_f b_f * factor_f_t + eps_t
#
# Each regressor is that factor's own bullish-oriented, quarterly-held univariate
# L/S spread at the portfolio's SAME sort (:func:`benchmark_spread`), so the
# coefficient ``b_f`` is the portfolio's loading on that constituent's return holding
# the other four fixed -- an attribution of the portfolio's realised return to its
# building blocks.  ``alpha`` is the portfolio return not spanned by any of the five.
# (Ported from Experiment 6's ``weighted_quality_composite.multivariate_attribution``;
# nothing is imported from that gitignored experiment.)
# --------------------------------------------------------------------------- #
def benchmark_spread(factor: str, n_buckets: int) -> pd.Series:
    """A univariate constituent ``factor``'s quarterly-held long/short spread at
    ``n_buckets`` buckets, bullish-oriented -- the project's single shared
    quarterly-book primitive (``quarter_position.quarter_held_spread``) at the
    portfolio's own sort, so it is the natural same-sort regressor to attribute the
    portfolio's return to.  Equal-weighted legs (the standalone factor book), read from
    the factor's catalog source panel."""
    meta = C.resolve_factors([factor]).iloc[0]
    panel = C._read_factor_panel(meta["panel_path"])
    return QP.quarter_held_spread(panel, factor, int(meta["sign"]),
                                  n=n_buckets).rename(factor)


def _ols(y: pd.Series, X: pd.DataFrame) -> pd.DataFrame:
    """OLS of ``y`` on ``X`` (an intercept is added), returning a tidy coefficient
    table (``term, coef, std_err, tstat``) plus an ``R^2`` / ``N`` attr, using the
    same closed-form estimator and t-stat convention as
    ``regression.market_regression`` generalised to multiple regressors."""
    df = pd.concat([y.rename("__y__"), X], axis=1).dropna()
    yv = df["__y__"].to_numpy(dtype=float)
    Xv = df.drop(columns="__y__").to_numpy(dtype=float)
    terms = ["const"] + list(X.columns)
    n, k = Xv.shape[0], Xv.shape[1] + 1
    Xd = np.column_stack([np.ones(Xv.shape[0]), Xv])
    coef, *_ = np.linalg.lstsq(Xd, yv, rcond=None)
    resid = yv - Xd @ coef
    dof = n - k
    sigma2 = (resid @ resid) / dof if dof > 0 else np.nan
    cov = sigma2 * np.linalg.inv(Xd.T @ Xd)
    se = np.sqrt(np.diag(cov))
    ss_tot = ((yv - yv.mean()) ** 2).sum()
    r2 = 1.0 - (resid @ resid) / ss_tot if ss_tot > 0 else np.nan
    tbl = pd.DataFrame({"term": terms, "coef": coef, "std_err": se,
                        "tstat": coef / se})
    tbl.attrs["r2"] = r2
    tbl.attrs["n"] = n
    return tbl


def _render_attribution_png(tbl: pd.DataFrame, spread_col: str, word: str,
                            path: Path) -> None:
    """Render the per-factor return-contribution table (``coef`` and ``t-stat`` only)
    to a PNG at ``path``: one row per regressor (intercept + the five factors), t-stats
    with |t|>=2.0 shaded darkest and |t|>=1.65 lightly shaded, plus an R^2 / N footer --
    a compact standalone figure matching the project's table style."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [(r.term, f"{r.coef:+.4f}", f"{r.tstat:+.2f}", abs(r.tstat))
            for r in tbl.itertuples()]
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * (len(rows) + 3)))
    ax.axis("off")
    ax.set_title(f"Final Portfolio ({spread_col}) on univariate factor books\n"
                 f"per-factor return contribution (multivariate OLS, {word} sort)",
                 fontsize=11, pad=10)

    table = ax.table(cellText=[[t, c, ts] for t, c, ts, _ in rows],
                     colLabels=["term", "coef", "t-stat"],
                     colWidths=[0.5, 0.25, 0.25],
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    for j in range(3):                                   # header row
        table[0, j].set_facecolor("#40466e")
        table[0, j].set_text_props(color="w", fontweight="bold")
    for i, (_t, _c, _ts, at) in enumerate(rows, start=1):
        table[i, 0].set_text_props(ha="left")
        shade = "#3b7dd8" if at >= 2.0 else "#bcd4f0" if at >= 1.65 else None
        if shade:
            table[i, 2].set_facecolor(shade)

    fig.text(0.5, 0.04, f"R² = {tbl.attrs['r2']:.3f}   |   "
             f"N = {tbl.attrs['n']} months   |   "
             "t-stat shading: |t| >= 1.65 (10%), 2.0 (5%)",
             ha="center", fontsize=8, color="#555")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def per_factor_contribution(spread: pd.Series, n_buckets: int,
                            out_dir: Path) -> pd.DataFrame:
    """Attribute the Final Portfolio's realised long/short ``spread`` to its five
    constituents: regress it on their standalone univariate quarterly L/S spreads
    simultaneously and print (and save) the coefficient summary table.

    Each regressor is a constituent's bullish-oriented univariate quarterly L/S spread
    at the portfolio's ``n_buckets`` sort (:func:`benchmark_spread`).  Prints a
    ``term, coef, std_err, t-stat`` table (intercept + the five factors) with the
    regression R^2 and month count, writes it to ``<sort>_factor_contribution.png``
    and returns the table."""
    word, spread_col = C.sort_word(n_buckets), f"Q{n_buckets}-Q1"
    out_dir.mkdir(parents=True, exist_ok=True)

    regressors = pd.DataFrame(
        {f: benchmark_spread(f, n_buckets) for f in FACTORS})
    tbl = _ols(spread.rename("portfolio"), regressors)

    print(f"=== Final Portfolio: per-factor return contribution ({spread_col}) "
          f"on univariate factor books -- multivariate OLS, {word} sort ===")
    print(f"    port_t = alpha + sum_f b_f * factor_f_t + eps_t   "
          f"(N={tbl.attrs['n']} months, R^2={tbl.attrs['r2']:.3f})")
    print(f"    {'term':<26}{'coef':>10}{'std err':>10}{'t-stat':>9}")
    print("    " + "-" * 55)
    for r in tbl.itertuples():
        print(f"    {r.term:<26}{r.coef:>10.4f}{r.std_err:>10.4f}{r.tstat:>9.2f}")

    png_path = out_dir / f"{word}_factor_contribution.png"
    _render_attribution_png(tbl, spread_col, word, png_path)
    print(f"Saved -> {png_path}")
    return tbl


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run(n_buckets: int = N_QUINTILES, out_root: Path = OUTPUT_DIR) -> dict:
    """Build the ownership-capped weighted-composite book, apply the ADV trading drag,
    and write the two outputs (``<sort>_cumulative.png`` / ``<sort>_performance.png``)
    under ``out_root / "quarter_<sort>"`` (``<sort>`` = quintile / tertile).  Returns
    the per-window performance dict."""
    resolved = C.resolve_factors(FACTORS)
    word, spread_col = C.sort_word(n_buckets), f"Q{n_buckets}-Q1"
    out_dir = out_root / f"quarter_{word}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Final Portfolio: weighted-composite {word} L/S (ADV-drag) ===")
    print("Factors + weights: " + ", ".join(
        f"{r.factor} [{'+' if r.sign > 0 else '-'}]"
        f"(w={'0.5/5.0' if r.factor == RD_FACTOR else '1'})"
        for r in resolved.itertuples()))

    # 1. Weighted composite (rd_stability weight state-dependent) -> tidy panel.
    composite = weighted_composite(resolved, n_buckets)
    panel = C.as_factor_panel(composite)

    # 2. Ownership-capped, weighted long/short book: quarterly-held top/bottom buckets
    #    of the composite, each name equal-weighted then water-filled so no position
    #    owns > 0.5% of its market cap.  The traded spread is the capped-weight top-
    #    minus-bottom return (not the equal-weighted bucket mean).  The equal-weighted
    #    per-bucket ``wide`` table is kept only for the cumulative-growth chart.
    wide = C.bucket_returns(panel, n_buckets)
    legs = capped_legs(panel, n_buckets)
    spread = capped_spread(legs)

    # 3. Long/short performance, full sample + past decade.
    industry = C.industry_return()
    windows = [("Full sample", C.book_stats(spread, industry)),
               ("Past decade (2016+)", C.book_stats(spread, industry, start=DECADE_START))]
    full, decade = windows[0][1], windows[1][1]

    # 4a. Turnover cost and trading-volume drag -- both per-month frictional return
    #     costs, reported separately for information (never netted from the gross alpha).
    #     The turnover cost is stored under ``avg_cost_row`` and the drag under
    #     ``adv_drag`` (explicit extra_metrics rows, so they are ordered where the caller
    #     wants them rather than auto-appended last).
    cost_series = COST.turnover_cost(legs, C.cost_panel())
    drag = volume_drag(legs)
    full["avg_cost_row"] = C.window_cost(cost_series)
    decade["avg_cost_row"] = C.window_cost(cost_series, start=DECADE_START)
    attach_drag(full, drag)
    attach_drag(decade, drag, start=DECADE_START)

    # 4b. Net-of-cost/drag Sharpe: net BOTH the turnover cost and the volume drag from
    #     the gross spread before the (beta-neutral) Sharpe.  Both are per-formation-
    #     month return costs on the same monthly index, so their sum is the total
    #     friction; ``attach_net_cost_sharpe`` then subtracts it and hedges as usual.
    cost_and_drag = cost_series.add(drag.reindex(cost_series.index).fillna(0.0),
                                    fill_value=0.0).rename("cost_drag")
    C.attach_net_cost_sharpe(full, spread, cost_and_drag, industry)
    C.attach_net_cost_sharpe(decade, spread, cost_and_drag, industry, start=DECADE_START)

    ownership = formation_ownership(legs)
    C.attach_ownership(full, ownership)
    C.attach_ownership(decade, ownership, start=DECADE_START)

    meta = {"n_stocks": composite["stock_id"].nunique(),
            "n_months": int(full["n_months"]),
            "start": composite["date"].min(), "end": composite["date"].max()}

    # --- Persist outputs --------------------------------------------------- #
    C.plot_cumulative(wide, FACTORS, out_dir / f"{word}_cumulative.png",
                      score_desc="weighted sum of z-scores", n_buckets=n_buckets)
    # Same compact row set as Experiment 6's liquidity-capped table: industry-neutral
    # alpha (+ t-stat), beta-neutral Sharpe, the beta-neutral net-of-cost Sharpe (bold),
    # the turnover cost, the largest single-name ownership and the trading-volume drag.
    C.render_performance(
        windows,
        f"Final Portfolio: weighted-composite {word} long-short ({spread_col}) "
        f"with {OWNERSHIP_CAP:.1%} ownership cap + trading-volume drag",
        f"{' + '.join(FACTORS)} (rd_stability w=0.5, ->5 when its spread exceeds its "
        f"trailing-{LOOKBACK}m avg)   |   {meta['n_stocks']} stocks over "
        f"{meta['n_months']} months ({meta['start']:%Y-%m} .. {meta['end']:%Y-%m})   |   "
        f"each position <= {OWNERSHIP_CAP:.1%} of a name's market cap "
        f"(excess redistributed) on a ${PORTFOLIO_CAPITAL / 1e6:.0f}M book;   "
        "drag: 10% of trailing-12m ADV/day, charged on the reposition trades.   "
        "Shading: |t| >= 1.65 (10%), 2.0 (5%).",
        out_dir / f"{word}_performance.png",
        extra_metrics=[net_cost_neutral_metric(), cost_metric(),
                       C.ownership_metric(), drag_metric()],
        keep_keys=["alpha", "alpha_tstat", "sharpe_neutral", "sharpe_cost_neutral",
                   "avg_cost_row", "max_ownership", "adv_drag"],
        bold_keys=["sharpe_cost_neutral"])

    # Per-factor return contribution: attribute the realised L/S spread to the five
    # constituents' standalone univariate books (multivariate OLS), written as
    # <sort>_factor_contribution.png.
    per_factor_contribution(spread, n_buckets, out_dir)

    # --- Console summary --------------------------------------------------- #
    print(f"  {spread_col} L/S: {full['mean_monthly']:+.4%}/mo "
          f"(t={full['tstat']:+.2f}, Sharpe={full['sharpe']:+.2f})")
    print(f"  industry-neutral alpha: {full['alpha']:+.4%}/mo "
          f"(t={full['alpha_tstat']:+.2f})")
    print(f"  trading-volume drag: {full['adv_drag']:+.4%}/mo "
          f"(2016+: {decade['adv_drag']:+.4%}/mo)")
    print(f"  net-of-cost/drag beta-neutral Sharpe: {full['sharpe_cost_neutral']:+.2f}")
    print(f"  largest single-name ownership: {full['max_ownership']:.4%} "
          f"(cap {OWNERSHIP_CAP:.1%})")
    print(f"  rd-boost active in {rd_boost_flag(n_buckets).mean():.0%} of months")
    print(f"Saved -> {out_dir}")
    return {lbl: s for lbl, s in windows}


def main() -> None:
    run(N_QUINTILES)        # quintile version
    run(N_TERTILES)         # tertile version


if __name__ == "__main__":
    main()
