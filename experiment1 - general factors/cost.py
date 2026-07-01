"""
cost.py
=======

Trading-cost model from the project plan, and the **average cost incurred by a
long/short quintile portfolio over time**.

The one-way cost in basis points for a name is a function of its freely-tradable
market value in USD::

    c     = 3 * (11 / log10(Mff))^6 + 3      (basis points, one way)
    Mff   = max(market cap x FX x free float, 5e7)     (USD)
    kappa = c / 1e4                          (one-way cost, fraction of notional)

(roughly 6 bps at $100B, 8 at $10B, 13 at $1B, 23 at $100M, 29 at the $50M floor;
so the one-way cost on any single position is capped at ~0.285% at the floor).

**Cost is charged on turnover, not on holdings.**  A position is only charged
``kappa`` when it is actually traded -- once when it is opened (a buy) and once
when it is closed (a sell).  A name carried over to the next month at the same
weight trades nothing and is charged nothing; over its whole holding period it
therefore pays a single round-trip (open + close), *not* a round-trip every
month.  Concretely, with equal weights ``w_{i,t}`` within each leg, the cost
charged in month *t* is::

    cost_t = sum_i kappa_{i,t} * | w_{i,t} - w_{i,t-1} |        (summed over both legs)

i.e. one-way cost times the weight actually traded.  Entering a name (0 -> 1/N)
and exiting it (1/N -> 0) are the only large terms; a name held across the
rebalance contributes only the small ``|1/N_t - 1/N_{t-1}|`` re-equalising trade.
The net spread is ``gross_(Q5-Q1) - cost_t`` and the "average cost incurred by
the portfolio over time" is the time-series mean of ``cost_t``.

This module is universe-parameterised exactly like the rest of the pipeline: it
reuses the engine's universe handling and price loading, and is shared unchanged
by every driver (Software, Banks/Insurance, Commodity Producers) and by
Experiment 2.

Run standalone to print the average L/S cost per factor::

    python cost.py [universe_slug]
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import factors as F

MIN_MFF = 5e7          # USD floor on freely-tradable market value
N_QUINTILES = 5


# --------------------------------------------------------------------------- #
# Cost formula
# --------------------------------------------------------------------------- #
def one_way_cost_bps(mff: pd.Series | np.ndarray) -> np.ndarray:
    """One-way trading cost in basis points for a freely-tradable value ``Mff`` (USD)."""
    mff = np.asarray(mff, dtype=float)
    return 3.0 * (11.0 / np.log10(mff)) ** 6 + 3.0


# --------------------------------------------------------------------------- #
# Per-(stock, month) round-trip cost panel
# --------------------------------------------------------------------------- #
def build_cost_panel(u: "F.Universe" = F.SOFTWARE_SERVICES) -> pd.DataFrame:
    """
    Monthly round-trip trading cost per security, keyed by ``(stock_id, date)``
    on the same month-end dates as the factor panel, so it merges directly onto
    a factor slice.

    Mff = month-end security market cap (local) x month-end FX-to-USD x free float
    (point-in-time, forward-filled), floored at ``MIN_MFF``.  Names missing any
    input get a NaN cost and are simply skipped in the portfolio averages.
    """
    universe = F.load_universe(u)

    # Month-end market cap (local), mirroring the factor panel's aggregation.
    px = F.load_prices(universe, u)
    px["period"] = px["date"].dt.to_period("M")
    mcap = (px.sort_values("date")
              .groupby(["stock_id", "period"], observed=True)
              .agg(mcap_local=("security_mcap_local", "last"))
              .reset_index())

    # Trading currency, then month-end FX-to-USD for that currency.
    sm = pd.read_feather(F.DATA_DIR / "security_master.feather",
                         columns=["stock_id", "currency_code"])
    sm["stock_id"] = sm["stock_id"].astype(str)
    mcap = mcap.merge(sm, on="stock_id", how="left")

    fx = pd.read_feather(F.DATA_DIR / "fx_rates.feather")
    fx["period"] = pd.to_datetime(fx["date"]).dt.to_period("M")
    fx = (fx.sort_values("date")
            .groupby(["currency_code", "period"], observed=True)
            .agg(fx_to_usd=("fx_to_usd", "last"))
            .reset_index())
    mcap = mcap.merge(fx, on=["currency_code", "period"], how="left")

    # Free float (point-in-time snapshot): attach on observation_date so a value
    # only enters a month-end once it was observable.  The backward as-of join
    # forward-fills the last observed float, so no separate ffill is needed.
    ff = pd.read_feather(F.DATA_DIR / "fundamental_master.feather",
                         columns=["stock_id", "date_fundamental",
                                  "observation_date", "free_float_percentage"])
    ff["stock_id"] = ff["stock_id"].astype(str)
    ff = ff[ff["stock_id"].isin(universe)].copy()
    ff["date_fundamental"] = pd.to_datetime(ff["date_fundamental"])
    ff["observation_date"] = pd.to_datetime(ff["observation_date"])
    mcap = F.attach_pit_fundamentals(mcap, ff)
    mcap = mcap.sort_values(["stock_id", "period"])

    mff = (mcap["mcap_local"] * mcap["fx_to_usd"]
           * mcap["free_float_percentage"]).clip(lower=MIN_MFF)
    kappa = one_way_cost_bps(mff) / 1e4          # one-way, as a fraction
    kappa = pd.Series(kappa, index=mcap.index).where(mff.notna())

    out = pd.DataFrame({
        "stock_id": mcap["stock_id"].to_numpy(),
        "date": mcap["period"].dt.to_timestamp(how="end").dt.normalize().to_numpy(),
        "kappa_oneway": kappa.to_numpy(),
        "roundtrip": (2.0 * kappa).to_numpy(),
    })
    return out.dropna(subset=["roundtrip"])


# --------------------------------------------------------------------------- #
# Portfolio cost over time
# --------------------------------------------------------------------------- #
def turnover_cost(legs: pd.DataFrame, cost_panel: pd.DataFrame,
                  dates: pd.Index | None = None,
                  active: pd.Series | None = None) -> pd.Series:
    """
    Monthly *turnover* cost of an arbitrary long/short book, indexed by formation
    month.  ``legs`` is a tidy ``date, stock_id, leg, w`` frame: ``leg`` labels
    which side a name is on (e.g. the two quintiles, the two tertile-intersection
    corners, or "long"/"short"), and ``w`` is its weight within that leg that
    month.  The cost in month *t* is ``sum_i kappa_{i,t} * |w_{i,t} - w_{i,t-1}|``
    -- one-way cost charged only on the weight actually traded -- accumulated per
    leg and summed.  A name carried into the next month at the same weight (and on
    the same leg) costs nothing, so a position pays a single round-trip over its
    holding period, never per month.

    ``dates`` is the full month axis the weight matrices are reindexed onto
    (defaults to the distinct months present in ``legs``); supply it explicitly
    when the book spans months in which *both* legs happen to be empty so those
    months still appear (cost 0).

    ``active`` (optional) is a boolean per-formation-month flag for an overlay
    that only holds the book in selected months (e.g. a timing rule): in any month
    where it is False/missing the whole book is flat, so every name's weight is
    forced to zero *that* month.  The turnover differencing then charges a one-way
    cost to liquidate the entire book on exit and to re-establish it on re-entry --
    the genuine extra trading a timing overlay incurs.  When ``active`` is ``None``
    (the default) the book is always-on.
    """
    if dates is None:
        dates = pd.Index(sorted(legs["date"].unique()), name="date")
    kappa = cost_panel[["date", "stock_id", "kappa_oneway"]]
    held = (None if active is None
            else active.reindex(dates, fill_value=False).astype(float))

    leg_costs = []
    for _, g in legs.groupby("leg", observed=True):
        # Weight matrix over the full month axis (0 when not held).
        w = (g.pivot_table(index="date", columns="stock_id", values="w", fill_value=0.0)
              .reindex(dates, fill_value=0.0))
        if held is not None:                       # zero the book in out-of-market months
            w = w.mul(held, axis=0)
        traded = w.diff().abs()
        traded.iloc[0] = w.iloc[0]                 # month 0: establish the initial book
        # cost = one-way kappa on the weight traded; only |dw| > 0 entries matter.
        tl = (traded.reset_index()
                    .melt(id_vars="date", var_name="stock_id", value_name="dw"))
        tl = tl[tl["dw"] > 0.0].merge(kappa, on=["date", "stock_id"], how="left")
        c = (tl["dw"] * tl["kappa_oneway"].fillna(0.0)).groupby(tl["date"]).sum()
        leg_costs.append(c.reindex(dates, fill_value=0.0))

    total = sum(leg_costs) if leg_costs else pd.Series(0.0, index=dates)
    return total.rename_axis("date")


def equal_weight_legs(membership: pd.DataFrame) -> pd.DataFrame:
    """Attach an equal weight ``w = 1/N`` within each ``(date, leg)`` to a
    ``date, stock_id, leg`` membership frame (the standard equal-weighted leg)."""
    out = membership.copy()
    out["w"] = 1.0 / out.groupby(["date", "leg"], observed=True)["stock_id"].transform("size")
    return out


def long_short_cost(panel: pd.DataFrame, factor: str, cost_panel: pd.DataFrame,
                    n_quintiles: int = N_QUINTILES,
                    active: pd.Series | None = None) -> pd.Series:
    """
    Monthly turnover cost of the Q5-Q1 long/short book for ``factor``, indexed by
    formation month.  Each leg is equal-weighted (``1/N`` per name) using the
    *same* monthly quintile membership as the gross spread (via the shared
    :func:`factors.prepare_slice`), so the cost lines up with ``quintile_returns``'
    ``Q5-Q1`` column; the per-leg turnover summation is :func:`turnover_cost`.
    See it for the ``active`` overlay semantics.
    """
    sub = F.prepare_slice(panel, factor, n_quintiles)
    legs = sub.loc[sub["quintile"].isin([1.0, float(n_quintiles)]),
                   ["date", "stock_id", "quintile"]].rename(columns={"quintile": "leg"})
    legs = equal_weight_legs(legs)
    dates = pd.Index(sorted(sub["date"].unique()), name="date")
    return turnover_cost(legs, cost_panel, dates, active).rename(f"{factor}_cost")


def average_cost(cost_series: pd.Series) -> float:
    """Time-series mean monthly turnover cost of a portfolio (fraction of notional)."""
    return float(cost_series.dropna().mean())


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    u = F.universe_from_argv()
    panel = F.load_panel(u=u)
    cp = build_cost_panel(u)
    print(f"=== Average L/S turnover cost per factor: {u.slug} ===")
    for factor in F.FACTOR_NAMES:
        cs = long_short_cost(panel, factor, cp)
        print(f"  {factor:<26} avg cost = {average_cost(cs) * 100:.4f} pp/month "
              f"({cs.dropna().size} months)")
