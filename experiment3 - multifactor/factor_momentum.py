"""
factor_momentum.py
==================

Experiment 3 -- factor-momentum rotation across Experiment 2's top factors.

Take the five software-industry factors Experiment 2 ranks highest by
industry-neutral alpha t-stat (the top five of the quarterly-repositioned hand-off
``quarter_position.ranked_factors``) and run a **factor-momentum** strategy on them:

    re-select every 3 months the single factor whose own long/short book earned
    the most over the *trailing 12 months*, and hold that pick until the next
    3-monthly selection.

Every candidate factor's standalone dollar-neutral long/short book is its
**quarterly-repositioned** bullish-oriented Q5-Q1 spread (long the high-z quintile
when the factor's ``direction`` is ``Q5-Q1``, else long the low-z quintile),
computed by the shared ``quarter_position.quarter_held_spread``.  At each formation
month ``t`` (every 3rd month) we look at which candidate earned the highest
compounded spread over the *trailing 12 months* ``t-12 .. t-1`` -- all *realised*
returns, known before ``t`` -- and hold that one factor's quarterly book until the
next 3-monthly selection.  Selecting on already-realised returns makes the rule
strictly look-ahead free; because the selection cadence (3 months) matches the
constituents' quarterly repositioning, the whole strategy repositions quarterly.

The rotation is evaluated exactly like every other long/short book in the
project: its monthly return is regressed on the market-cap-weighted Software &
Services industry return for an industry-neutral alpha (and t-stat), via
Experiment 1's ``regression.market_regression`` reused through ``composite.py``.
For the alternative of *combining* the same factors into one signal rather than
rotating between them, see ``composite.py top`` (the equal-weighted z-score
composite of Experiment 2's top factors).

Design -- reuses the engine and composite.py plumbing
-----------------------------------------------------
Nothing generic is re-implemented:

* the per-factor quarterly long/short returns come from the shared
  ``quarter_position.quarter_held_spread`` (oriented by each factor's bullish
  ``direction``), so "the factor's book" is defined identically everywhere;
* the within-industry "market" return, the industry-neutral alpha / Sharpe
  statistics (``industry_return`` / ``book_stats``) and the performance-table
  renderer (``render_performance``) are imported from ``composite.py``, so
  "alpha" is defined identically to every other book in the project.

This module adds **only** the monthly rotation rule on top of returns and
statistics produced upstream.

A second pipeline -- ``bivariate``
----------------------------------
Alongside the one-factor-at-a-time rotation, this module also tests *combining*
the momentum winners.  **Every 3 months** it ranks the candidates by their
**trailing-12-month univariate long/short return** (the same realised,
look-ahead-free window the rotation selects on), takes the **top two**, and
jointly generates a portfolio from them via ``bivariate_tertile``'s
independent double sort (long the top-tertile-of-both / short the
bottom-tertile-of-both).  The chosen pair is re-selected every 3 months and its
double sort rebalanced monthly in between, so the book is the corner spread of
whichever pair momentum favours -- the double-sort mechanics (tertiles, corner legs, grid, turnover cost)
are reused wholesale from ``bivariate_tertile``; this module adds only the 3-monthly
pair selection.  The 3x3 grid diagnostic averages each cell over all months regardless
of which pair produced it (rows = tertile on the block's #1 momentum factor,
columns = tertile on its #2).

Outputs (``output/factor_momentum/``)
-------------------------------------
``univariate/`` -- the 3-monthly single-factor rotation:
    factor_momentum_cumulative.png   growth of $1 in the rotation
    selection_timeline.png           which factor is held each month + how often
    performance.png                  rotation performance (full / 2016+), with the alpha
                                     and largest single-name ownership for a $100M book
``bivariate/`` -- the 3-monthly-reselected double sort of the trailing-12m top two:
    grid_mean_return.png             3x3 tertile grid, cell returns averaged over all months
    performance.png                  book performance (full / 2016+), with the alpha
                                     and largest single-name ownership for a $100M book

Run standalone::

    python factor_momentum.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import composite as C                 # engine + analysis plumbing, loaded by path
import bivariate_tertile as BT         # double-sort pipeline reused for the top-two pair

# --------------------------------------------------------------------------- #
# Paths / configuration
# --------------------------------------------------------------------------- #
OUTPUT_DIR = C.OUTPUT_DIR / "factor_momentum"
UNIVARIATE_DIR = OUTPUT_DIR / "univariate"   # the 3-monthly one-factor rotation
BIVARIATE_DIR = OUTPUT_DIR / "bivariate"     # double sort of the trailing-12m top two
DECADE_START = C.DECADE_START          # 2016-01-01, the project "past decade" cut-off

# The candidate set is the top-N leaders of Experiment 2's quarterly-repositioned
# ranking, read through the single shared hand-off re-exported by ``composite``.
ranked_factors = C.ranked_factors


def signed_spread(panel: pd.DataFrame, factor: str, direction: str) -> pd.Series:
    """
    A factor's bullish-oriented **quarterly-repositioned** standalone long/short
    return, computed from its source ``panel`` and signed by ``direction``.

    Delegates to the shared ``quarter_position.quarter_held_spread`` (buckets formed
    quarterly, held three months): ``+`` the top-minus-bottom spread when
    ``direction == 'Q5-Q1'`` (long the high-z leg), ``-`` it (i.e. Q1-Q5) otherwise.
    Indexed by formation month -- the exact quarterly book every other Experiment 2-5
    module trades, so no return is recomputed differently here.
    """
    sign = 1 if str(direction).strip() == "Q5-Q1" else -1
    return C.QP.quarter_held_spread(panel, factor, sign).rename(factor)


def build_spread_matrix(top: pd.DataFrame) -> pd.DataFrame:
    """Wide ``month x factor`` matrix of the candidates' signed quarterly long/short
    returns, columns ordered as in the top-factor table.  Each library panel is read
    once (candidates grouped by source panel) and the shared quarterly spread taken
    per factor."""
    resolved = C.resolve_factors(top["factor"].tolist())
    signed: dict[str, pd.Series] = {}
    for panel_path, grp in resolved.groupby("panel_path", sort=False):
        panel = C._read_factor_panel(panel_path)
        for r in grp.itertuples():
            signed[r.factor] = signed_spread(panel, r.factor, r.direction)
    return pd.concat([signed[f] for f in top["factor"]], axis=1).sort_index()


# --------------------------------------------------------------------------- #
# Step 2 -- the 3-monthly rotation
# --------------------------------------------------------------------------- #
LOOKBACK = 12                                    # formation window (months) for selection
HOLD = 3                                          # re-select every HOLD months (book still rebalances monthly)


def _hold_selection(per_month: pd.Series) -> pd.Series:
    """
    Collapse a *per-month* selection into one that is re-formed only every
    :data:`HOLD` months and held constant in between.

    The pick at each formation month -- positions ``0, HOLD, 2*HOLD, ...`` of the
    (monthly, gap-free) index -- is kept and forward-filled across the intervening
    months, so the *choice* is refreshed every ``HOLD`` months and held until the
    next formation.  (The held book itself still rebalances monthly, since each
    candidate's return is a monthly-rebalanced spread.)  Works for both the
    single-factor rotation (string picks) and the top-two pair selector (tuple picks).
    """
    formation = per_month.iloc[::HOLD]
    return formation.reindex(per_month.index, method="ffill")


def trailing_returns(spreads: pd.DataFrame, lookback: int = LOOKBACK) -> pd.DataFrame:
    """
    Each candidate's *trailing* ``lookback``-month compounded long/short return, as
    known at formation.  Restricted to months where *every* candidate has a return
    (so ranks are always over the full set), it is the rolling product of
    ``(1 + spread)`` shifted one month, so row ``t`` spans the realised months
    ``t-lookback .. t-1`` -- strictly prior to ``t``, keeping any selection built on
    it look-ahead free.  The first ``lookback`` months have no full window and are
    dropped.  Shared by the 3-monthly rotation and the top-two pair selector.
    """
    panel = spreads.dropna()                     # months with all candidates present
    trailing = ((1.0 + panel).rolling(lookback).apply(np.prod, raw=True) - 1.0)
    return trailing.shift(1).dropna()            # drops the first `lookback` months


def factor_momentum(spreads: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Run the factor-momentum rotation over the months where *every* candidate has
    a return (so the pick is always among the full set).

    Returns ``(rotated, chosen)``:

      * ``rotated`` -- the rotation's monthly return: every :data:`HOLD` months
        select the factor with the highest *trailing 12-month* compounded return
        (see :func:`trailing_returns`), i.e. over the ``LOOKBACK`` months ending in
        the *previous* month, then hold that factor's monthly-rebalanced book until
        the next selection.  Each month earns that held factor's monthly spread.
      * ``chosen``  -- the factor held each month (constant within each ``HOLD``-month
        block, re-formed at the block boundaries).
    """
    panel = spreads.dropna()                     # months with all candidates present
    prior = trailing_returns(spreads)            # trailing-12m return known at formation
    per_month = prior.idxmax(axis=1)             # best trailing-12m factor as of last month
    chosen = _hold_selection(per_month)          # refresh the pick every HOLD months (book rebalances monthly)
    rotated = (pd.Series({t: panel.at[t, chosen[t]] for t in chosen.index},
                         name="factor_momentum")
                 .sort_index())
    return rotated, chosen.loc[rotated.index].rename("chosen_factor")


def held_top_pairs(spreads: pd.DataFrame, lookback: int = LOOKBACK) -> pd.Series:
    """
    The top-two momentum factors, re-selected every :data:`HOLD` months and held
    (with monthly rebalancing) until the next selection.

    At each formation month a candidate is ranked by its *trailing* ``lookback``-month
    univariate long/short return (:func:`trailing_returns`, realised over
    ``t-lookback .. t-1`` so the pick is look-ahead free), and the two strongest are
    taken as an ordered ``(best, second)`` tuple.  The pair chosen at each
    ``HOLD``-month boundary is held constant over the block (see
    :func:`_hold_selection`); the double sort of that pair still rebalances monthly.
    Returns a Series of these tuples indexed by month -- the pair the bivariate
    double sort combines that month.
    """
    trailing = trailing_returns(spreads, lookback)
    per_month = trailing.apply(
        lambda row: tuple(row.sort_values(ascending=False).head(2).index), axis=1)
    return _hold_selection(per_month)


# --------------------------------------------------------------------------- #
# Step 3 -- rotation turnover cost
# --------------------------------------------------------------------------- #
def _leg_membership(resolved_row) -> pd.DataFrame:
    """A candidate's bullish-oriented **quarterly-held** Q5/Q1 leg membership
    (``date, stock_id, leg``), reconstructed from its source panel via the shared
    ``quarter_position.quarter_held_membership`` -- the same quarterly membership its
    standalone book trades.  ``leg`` is "long"/"short" per the factor's bullish
    direction."""
    panel = C._read_factor_panel(resolved_row.panel_path)
    held = C.QP.quarter_held_membership(C.F.prepare_slice(panel, resolved_row.factor, 5))
    bull = 5.0 if resolved_row.sign > 0 else 1.0        # bullish leg
    mem = held.loc[held["leg"].isin([1.0, 5.0]), ["date", "stock_id", "leg"]].copy()
    mem["leg"] = np.where(mem["leg"] == bull, "long", "short")
    return mem


def rotation_legs(factor_names: list[str], chosen: pd.Series) -> pd.DataFrame:
    """Equal-weighted Q5/Q1 leg membership (``date, stock_id, leg, w``) the rotation
    actually holds: each month the *chosen* factor's full bullish-oriented Q5/Q1
    book.  Shared by the cost and ownership diagnostics so both price exactly the
    names the rotation trades."""
    resolved = C.resolve_factors(factor_names)
    membership = {r.factor: _leg_membership(r) for r in resolved.itertuples()}
    held = [membership[f][membership[f]["date"] == t] for t, f in chosen.items()]
    return C.COST.equal_weight_legs(pd.concat(held, ignore_index=True))


def rotation_cost(factor_names: list[str], chosen: pd.Series) -> pd.Series:
    """
    Monthly turnover cost of the rotation, indexed by formation month.  Each month
    the book is the *chosen* factor's full Q5/Q1 leg membership (:func:`rotation_legs`);
    ``cost.turnover_cost`` then charges the genuine trading -- a near-full
    liquidation/re-establishment whenever the rotation switches factor, only
    membership drift when it holds the same factor (a name that happens to sit on the
    same leg of both factors is not traded).  Reported alongside performance, never
    netted from the gross alpha.
    """
    dates = pd.Index(sorted(chosen.index), name="date")
    return C.COST.turnover_cost(rotation_legs(factor_names, chosen), C.cost_panel(), dates)


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_cumulative(rotated: pd.Series, factor_names: list[str], sharpe: float,
                    alpha: float, alpha_tstat: float, path: Path) -> None:
    """Cumulative growth of $1 in the factor-momentum rotation."""
    cum_rot = (1.0 + rotated.fillna(0.0)).cumprod()
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(cum_rot.index, cum_rot, color="C2", linewidth=1.5,
            label="Factor-momentum rotation")
    ax.axhline(1.0, color="black", linewidth=0.6)
    ax.set_yscale("log")
    ax.set_title(
        "Factor-momentum long-short: every 3 months hold the trailing-12m best factor\n"
        f"candidates: {', '.join(factor_names)}\n"
        f"[Sharpe={sharpe:+.2f}, alpha={alpha:+.4%}/mo, t(alpha)={alpha_tstat:+.2f}]")
    ax.set_xlabel("Month")
    ax.set_ylabel("Cumulative value of $1 (log scale)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_selection(chosen: pd.Series, factor_names: list[str], path: Path) -> None:
    """Which factor the rotation holds each month (timeline) and how often (bars)."""
    order = {f: i for i, f in enumerate(factor_names)}
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, 10))

    fig, (ax_t, ax_c) = plt.subplots(
        2, 1, figsize=(11, 6.5), gridspec_kw={"height_ratios": [2, 1.1]})

    y = chosen.map(order)
    ax_t.scatter(chosen.index, y, c=[colors[order[f]] for f in chosen],
                 s=14, marker="s")
    ax_t.set_yticks(range(len(factor_names)))
    ax_t.set_yticklabels(factor_names, fontsize=8)
    ax_t.set_ylim(-0.5, len(factor_names) - 0.5)
    ax_t.set_title("Factor held each month by the momentum rotation")
    ax_t.set_xlabel("Month")
    ax_t.grid(True, axis="x", alpha=0.3)

    counts = chosen.value_counts().reindex(factor_names).fillna(0).astype(int)
    ax_c.bar(range(len(factor_names)),
             [counts[f] / counts.sum() for f in factor_names],
             color=[colors[order[f]] for f in factor_names])
    ax_c.set_xticks(range(len(factor_names)))
    ax_c.set_xticklabels(factor_names, rotation=20, ha="right", fontsize=8)
    ax_c.set_ylabel("Share of months held")
    ax_c.set_title("How often each factor is selected")
    ax_c.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run_univariate(top: pd.DataFrame, spreads: pd.DataFrame,
                   out_dir: Path = UNIVARIATE_DIR) -> dict:
    """The 3-monthly factor-momentum rotation: every 3 months select the single
    trailing-12m best factor and hold it (rebalanced monthly) until the next
    selection.  Writes every rotation output under ``out_dir`` and returns the
    per-window performance dict."""
    factor_names = top["factor"].tolist()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("--- univariate rotation (every 3 months hold the trailing-12m best single factor) ---")

    # 3-monthly rotation over the candidates' signed L/S books.
    rotated, chosen = factor_momentum(spreads)

    # Performance vs the industry, exactly as every project book is measured.
    industry = C.industry_return()
    windows = [
        ("Factor momentum (full)", C.book_stats(rotated, industry)),
        ("Factor momentum (2016+)", C.book_stats(rotated, industry, start=DECADE_START)),
    ]
    full, decade = (w[1] for w in windows)
    start, end = rotated.index.min(), rotated.index.max()

    # Average monthly turnover cost of the rotation (reported, not netted) plus the
    # cost-incorporated Sharpe (raw + β-neutral) from the same cost series.
    legs = rotation_legs(factor_names, chosen)
    cost_series = C.COST.turnover_cost(legs, C.cost_panel(),
                                       pd.Index(sorted(chosen.index), name="date"))
    full["avg_cost"] = C.window_cost(cost_series)
    decade["avg_cost"] = C.window_cost(cost_series, start=DECADE_START)
    C.attach_net_cost_sharpe(full, rotated, cost_series, industry)
    C.attach_net_cost_sharpe(decade, rotated, cost_series, industry, start=DECADE_START)

    # Largest single-name ownership for a $100M dollar-neutral book (worst case per
    # window), from the same chosen-factor Q5/Q1 leg membership the cost uses.
    ownership = C.leg_ownership(legs)
    C.attach_ownership(full, ownership)
    C.attach_ownership(decade, ownership, start=DECADE_START)

    # --- Persist outputs --------------------------------------------------- #
    counts = (chosen.value_counts().rename_axis("factor").rename("months")
                    .reindex(factor_names).fillna(0).astype(int).to_frame())

    plot_cumulative(rotated, factor_names,
                    full["sharpe"], full["alpha"], full["alpha_tstat"],
                    out_dir / "factor_momentum_cumulative.png")
    plot_selection(chosen, factor_names, out_dir / "selection_timeline.png")
    C.render_performance(
        windows, "Factor-momentum rotation: long-short performance",
        f"hold the trailing-12m best of: {', '.join(factor_names)}   |   "
        f"{len(rotated)} months ({start:%Y-%m} .. {end:%Y-%m})   |   "
        "alpha from regressing the book on the market-cap-weighted industry return.   "
        "Shading: |t| >= 1.65 (10%), 2.0 (5%).",
        out_dir / "performance.png",
        extra_metrics=[C.ownership_metric()])

    # --- Console summary (ASCII only -- Windows cp1252 stdout) ------------- #
    print(f"  rotation: mean {full['mean_monthly']:+.4%}/mo "
          f"(t={full['tstat']:+.2f}, Sharpe={full['sharpe']:+.2f})")
    print(f"  industry-neutral alpha: {full['alpha']:+.4%}/mo "
          f"(t={full['alpha_tstat']:+.2f}, ind beta={full['ind_beta']:+.2f})")
    print(f"  2016+: alpha={decade['alpha']:+.4%}/mo (t={decade['alpha_tstat']:+.2f})")
    sel = ", ".join(f"{f}={counts.loc[f, 'months']}" for f in factor_names)
    print(f"  months held: {sel}")
    print(f"Saved -> {out_dir}")
    return {lbl: s for lbl, s in windows}


# Grid / plot axis labels for the rotating double sort: the pair changes every 3
# months, so the axes are the momentum *ranks* rather than fixed factor names.
BIVARIATE_AXES = ["trailing-12m best factor", "trailing-12m 2nd-best factor"]


def rotating_double_sort(pairs: pd.Series) -> pd.DataFrame:
    """
    One combined double-sorted cross-section in which each formation month uses
    that month's chosen top-two pair.

    ``bivariate_tertile.double_sorted`` is run once per distinct pair (giving that
    pair's per-month tertile labels ``tile_a`` on the #1 momentum factor / ``tile_b``
    on the #2), and only the months that pair is selected are kept.  Stacking these
    slices yields a frame with the standard ``date, stock_id, next_return, mcap,
    tile_a, tile_b`` columns whose tiles are always oriented to the month's momentum
    ranks -- so ``bivariate_tertile``'s :func:`bivariate_book`, :func:`grid_stats`
    and :func:`bivariate_cost` all apply unchanged.
    """
    keep = ["date", "stock_id", "next_return", "mcap", "tile_a", "tile_b"]
    months_of_pair: dict[tuple, list] = {}
    for month, pair in pairs.items():
        months_of_pair.setdefault(pair, []).append(month)

    slices = []
    for pair, months in months_of_pair.items():
        frame = BT.double_sorted(C.resolve_factors(list(pair)))
        slices.append(frame.loc[frame["date"].isin(months), keep])
    return pd.concat(slices, ignore_index=True).sort_values(["date", "stock_id"])


def run_bivariate(spreads: pd.DataFrame, out_dir: Path = BIVARIATE_DIR) -> dict:
    """Re-select the two strongest trailing-12m univariate factors *every 3 months*
    and combine them with ``bivariate_tertile``'s independent double sort (rebalanced
    monthly between selections), writing the rotating book's outputs under ``out_dir``.
    Only the 3-monthly pair selection lives here; the double-sort mechanics and
    diagnostics are reused from ``bivariate_tertile``.  Returns the per-window
    performance dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print("--- bivariate double sort (3-monthly top two by trailing-12m univariate return) ---")

    # 1. 3-monthly pair selection, then the rotating double-sorted cross-section.
    pairs = held_top_pairs(spreads)
    frame = rotating_double_sort(pairs)
    book = BT.bivariate_book(frame)                  # rotating corner spread per month
    grids = BT.grid_stats(frame)                     # cells averaged across all months/pairs
    spread = book["long_short"]

    # 2. Performance vs the industry, exactly as every project book is measured.
    industry = C.industry_return()
    windows = [("Bivariate momentum (full)", C.book_stats(spread, industry)),
               ("Bivariate momentum (2016+)", C.book_stats(spread, industry, start=DECADE_START))]
    full, decade = windows[0][1], windows[1][1]
    start, end = spread.index.min(), spread.index.max()

    # Average monthly turnover cost of the rotating book (reported, not netted); the
    # pair switching is genuine turnover the double-sort cost routine already charges.
    # Same series also nets the spread for the cost-incorporated Sharpe.
    cost_series = BT.bivariate_cost(frame)
    full["avg_cost"] = C.window_cost(cost_series)
    decade["avg_cost"] = C.window_cost(cost_series, start=DECADE_START)
    C.attach_net_cost_sharpe(full, spread, cost_series, industry)
    C.attach_net_cost_sharpe(decade, spread, cost_series, industry, start=DECADE_START)

    # Largest single-name ownership for a $100M dollar-neutral book (worst case per
    # window), from the rotating double sort's corner-cell legs.
    ownership = BT.bivariate_ownership(frame)
    C.attach_ownership(full, ownership)
    C.attach_ownership(decade, ownership, start=DECADE_START)

    # --- Persist outputs --------------------------------------------------- #
    held = pairs.loc[book.index]                     # pair traded each booked month
    pair_counts = (held.apply(lambda p: f"{p[0]} x {p[1]}")
                       .value_counts().rename_axis("pair").rename("months").to_frame())

    BT.plot_grid(grids, BIVARIATE_AXES, out_dir / "grid_mean_return.png")
    C.render_performance(
        windows, "Bivariate factor-momentum double sort: long-short performance",
        "every 3 months double-sort the trailing-12m top two factors (long top-tertile-of-both / "
        "short bottom-tertile-of-both)   |   "
        f"{len(spread)} months ({start:%Y-%m} .. {end:%Y-%m})   |   "
        "industry-neutral alpha from the market-cap-weighted industry return.   "
        "Shading: |t| >= 1.65 (10%), 2.0 (5%).",
        out_dir / "performance.png",
        extra_metrics=[C.ownership_metric()])

    # --- Console summary (ASCII only -- Windows cp1252 stdout) ------------- #
    print(f"  long/short: {full['mean_monthly']:+.4%}/mo "
          f"(t={full['tstat']:+.2f}, Sharpe={full['sharpe']:+.2f})")
    print(f"  industry-neutral alpha: {full['alpha']:+.4%}/mo "
          f"(t={full['alpha_tstat']:+.2f}, ind beta={full['ind_beta']:+.2f})")
    print(f"  2016+: alpha={decade['alpha']:+.4%}/mo (t={decade['alpha_tstat']:+.2f})")
    top_pairs = ", ".join(f"{p}={pair_counts.loc[p, 'months']}"
                          for p in pair_counts.head(3).index)
    print(f"  most-held pairs: {top_pairs}")
    print(f"Saved -> {out_dir}")
    return {lbl: s for lbl, s in windows}


def run(out_dir: Path = OUTPUT_DIR) -> dict:
    """Build both Experiment-3 factor-momentum pipelines from the quarterly top-factor
    hand-off -- the 3-monthly one-factor rotation (``univariate/``) and the double
    sort of the trailing-12m top two (``bivariate/``) -- and write every output
    under ``out_dir``.  Returns ``{"univariate": ..., "bivariate": ...}`` per-window
    performance dicts."""
    top = ranked_factors(C.TOP_N)
    factor_names = top["factor"].tolist()

    print("=== Experiment 3: factor momentum ===")
    print(f"Candidates ({len(factor_names)}): " + ", ".join(
        f"{r.factor} [{r.subexperiment}]" for r in top.itertuples()))

    # Each candidate's signed standalone L/S book -- shared by both pipelines.
    spreads = build_spread_matrix(top)

    univariate = run_univariate(top, spreads, out_dir / "univariate")
    bivariate = run_bivariate(spreads, out_dir / "bivariate")
    print(f"Saved -> {out_dir}")
    return {"univariate": univariate, "bivariate": bivariate}


def main() -> None:
    run()


if __name__ == "__main__":
    main()
