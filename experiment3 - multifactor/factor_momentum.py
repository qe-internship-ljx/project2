"""
factor_momentum.py
==================

Experiment 3 -- factor-momentum rotation across Experiment 2's top factors.

Take the five software-industry factors Experiment 2 ranks highest by
industry-neutral alpha t-stat (the hand-off written by
``experiment2 - sw factors/main.py`` to ``top_factors/top_factors.csv``) and run
a **factor-momentum** strategy on them:

    each month, hold the single factor whose own long/short book earned the most
    over the *trailing 12 months*.

Every candidate factor's standalone dollar-neutral long/short book is the
bullish-oriented Q5-Q1 spread Experiments 1 & 2 already computed (long the
high-z quintile when the factor's ``direction`` is ``Q5-Q1``, else long the low-z
quintile -- the same orientation used in its published ``long_short.png``).  At
each formation month ``t`` we look at which candidate earned the highest
compounded spread over the *trailing 12 months* ``t-12 .. t-1`` -- all
*realised* returns, known before ``t`` -- and hold that one factor's book over
month ``t``.  Selecting on already-realised returns makes the rule strictly
look-ahead free.

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

* the per-factor long/short returns are read straight from each subexperiment's
  ``quintile/<factor>/quintile_returns.csv`` (the ``Q5-Q1`` column) and oriented
  by the ``direction`` recorded in ``top_factors.csv`` -- no return is recomputed;
* the within-industry "market" return, the industry-neutral alpha / Sharpe
  statistics (``industry_return`` / ``book_stats``) and the performance-table
  renderer (``render_performance``) are imported from ``composite.py``, so
  "alpha" is defined identically to every other book in the project.

This module adds **only** the monthly rotation rule on top of returns and
statistics produced upstream.

A second pipeline -- ``bivariate``
----------------------------------
Alongside the one-factor-at-a-time rotation, this module also tests *combining*
the momentum winners.  **Every month** it ranks the candidates by their
**trailing-12-month univariate long/short return** (the same realised,
look-ahead-free window the rotation selects on), takes the **top two**, and that
month jointly generates a portfolio from them via ``bivariate_tertile``'s
independent double sort (long the top-tertile-of-both / short the
bottom-tertile-of-both).  The chosen pair is re-selected each month, so the book
is the month-by-month corner spread of whichever pair momentum favours -- the
double-sort mechanics (tertiles, corner legs, grid, turnover cost) are reused
wholesale from ``bivariate_tertile``; this module adds only the monthly pair
selection.  The 3x3 grid diagnostic averages each cell over all months regardless
of which pair produced it (rows = tertile on the month's #1 momentum factor,
columns = tertile on its #2).

Outputs (``output/factor_momentum/``)
-------------------------------------
``univariate/`` -- the monthly single-factor rotation:
    factor_momentum_cumulative.png   growth of $1 in the rotation
    selection_timeline.png           which factor is held each month + how often
    performance.png                  rotation performance (full / 2016+), with the alpha
    factor_momentum_returns.csv      monthly chosen factor + rotation/constituent returns
    selection_counts.csv             how many months each factor was selected
``bivariate/`` -- the monthly-reselected double sort of the trailing-12m top two:
    long_short.png                   growth of $1 in the rotating double-sort book
    grid_mean_return.png             3x3 tertile grid, cell returns averaged over all months
    performance.png                  book performance (full / 2016+), with the alpha
    bivariate_returns.csv            monthly chosen pair + long/short/spread + leg sizes
    pair_counts.csv                  how many months each pair was selected

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
TOP_FACTORS_CSV = C.EXP2_DIR / "top_factors" / "top_factors.csv"
OUTPUT_DIR = C.OUTPUT_DIR / "factor_momentum"
UNIVARIATE_DIR = OUTPUT_DIR / "univariate"   # the monthly one-factor rotation
BIVARIATE_DIR = OUTPUT_DIR / "bivariate"     # double sort of the trailing-12m top two
DECADE_START = C.DECADE_START          # 2016-01-01, the project "past decade" cut-off
SPREAD_COL = "Q5-Q1"                   # long/short column in each quintile_returns.csv


# --------------------------------------------------------------------------- #
# Step 1 -- the candidate set and each candidate's standalone long/short book
# --------------------------------------------------------------------------- #
def load_top_factors(csv_path: Path = TOP_FACTORS_CSV) -> pd.DataFrame:
    """Load Experiment 2's top-factor hand-off (factor, source subexperiment and
    bullish ``direction`` per row).  Raises a clear error if it is missing."""
    if not Path(csv_path).exists():
        raise FileNotFoundError(
            f"{csv_path} not found.  Run Experiment 2 first -- "
            "`python main.py` (or `python main.py collect`) in "
            "'experiment2 - sw factors' writes the top-factor hand-off.")
    return pd.read_csv(csv_path)


def signed_spread(subexperiment: str, factor: str, direction: str) -> pd.Series:
    """
    A factor's bullish-oriented standalone monthly long/short return, read from
    its ``quintile/<factor>/quintile_returns.csv`` and signed by ``direction``.

    The stored ``Q5-Q1`` column is the raw top-minus-bottom spread; the published
    book is long the *bullish* leg, so it is ``+Q5-Q1`` when ``direction == 'Q5-Q1'``
    and ``-(Q5-Q1)`` (i.e. Q1-Q5) otherwise.  Indexed by formation month.

    The file is located through ``composite.factor_quintile_dir`` (``subexperiment``
    is retained for the caller's display only), so a factor living in Experiment 1
    rather than an Experiment 2 subexperiment still resolves.
    """
    qr_path = C.factor_quintile_dir(factor) / factor / "quintile_returns.csv"
    qr = (pd.read_csv(qr_path, parse_dates=["date"])
            .set_index("date").sort_index())
    sign = 1 if str(direction).strip() == "Q5-Q1" else -1
    return (sign * qr[SPREAD_COL]).rename(factor)


def build_spread_matrix(top: pd.DataFrame) -> pd.DataFrame:
    """Wide ``month x factor`` matrix of the candidates' signed long/short returns,
    columns ordered as in the top-factor table."""
    cols = [signed_spread(r.subexperiment, r.factor, r.direction)
            for r in top.itertuples()]
    return pd.concat(cols, axis=1).sort_index()


# --------------------------------------------------------------------------- #
# Step 2 -- the monthly rotation
# --------------------------------------------------------------------------- #
LOOKBACK = 12                                    # formation window (months) for selection


def trailing_returns(spreads: pd.DataFrame, lookback: int = LOOKBACK) -> pd.DataFrame:
    """
    Each candidate's *trailing* ``lookback``-month compounded long/short return, as
    known at formation.  Restricted to months where *every* candidate has a return
    (so ranks are always over the full set), it is the rolling product of
    ``(1 + spread)`` shifted one month, so row ``t`` spans the realised months
    ``t-lookback .. t-1`` -- strictly prior to ``t``, keeping any selection built on
    it look-ahead free.  The first ``lookback`` months have no full window and are
    dropped.  Shared by the monthly rotation and the top-two pair selector.
    """
    panel = spreads.dropna()                     # months with all candidates present
    trailing = ((1.0 + panel).rolling(lookback).apply(np.prod, raw=True) - 1.0)
    return trailing.shift(1).dropna()            # drops the first `lookback` months


def factor_momentum(spreads: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Run the factor-momentum rotation over the months where *every* candidate has
    a return (so the pick is always among the full set).

    Returns ``(rotated, chosen)``:

      * ``rotated`` -- the rotation's monthly return: each month hold the factor
        with the highest *trailing 12-month* compounded return (see
        :func:`trailing_returns`), i.e. over the ``LOOKBACK`` months ending in the
        *previous* month.
      * ``chosen``  -- the factor held each month (the rotation's choice).
    """
    panel = spreads.dropna()                     # months with all candidates present
    prior = trailing_returns(spreads)            # trailing-12m return known at formation
    chosen = prior.idxmax(axis=1)                # best trailing-12m factor as of last month
    rotated = (pd.Series({t: panel.at[t, chosen[t]] for t in chosen.index},
                         name="factor_momentum")
                 .sort_index())
    return rotated, chosen.loc[rotated.index].rename("chosen_factor")


def monthly_top_pairs(spreads: pd.DataFrame, lookback: int = LOOKBACK) -> pd.Series:
    """
    Each formation month's top-two momentum factors, re-selected every month.

    For each month a candidate is ranked by its *trailing* ``lookback``-month
    univariate long/short return (:func:`trailing_returns`, realised over
    ``t-lookback .. t-1`` so the pick is look-ahead free), and the two strongest are
    taken as an ordered ``(best, second)`` tuple.  Returns a Series of these tuples
    indexed by formation month -- the pair the bivariate double sort combines that
    month.
    """
    trailing = trailing_returns(spreads, lookback)
    return trailing.apply(
        lambda row: tuple(row.sort_values(ascending=False).head(2).index), axis=1)


# --------------------------------------------------------------------------- #
# Step 3 -- rotation turnover cost
# --------------------------------------------------------------------------- #
def _leg_membership(resolved_row) -> pd.DataFrame:
    """A candidate's bullish-oriented Q5/Q1 leg membership (``date, stock_id,
    leg``), reconstructed from its source panel via Experiment 1's
    ``prepare_slice`` -- the same quintile membership its standalone book trades.
    ``leg`` is "long"/"short" per the factor's bullish direction."""
    panel = pd.read_csv(resolved_row.panel_path, parse_dates=["date"],
                        usecols=["date", "stock_id", "factor", "zscore", "next_return"])
    panel["stock_id"] = panel["stock_id"].astype(str)
    sub = C.F.prepare_slice(panel, resolved_row.factor, 5)
    long_q = 5.0 if resolved_row.sign > 0 else 1.0      # bullish leg
    mem = sub.loc[sub["quintile"].isin([1.0, 5.0]), ["date", "stock_id", "quintile"]].copy()
    mem["leg"] = np.where(mem["quintile"] == long_q, "long", "short")
    return mem.drop(columns="quintile")


def rotation_cost(factor_names: list[str], chosen: pd.Series) -> pd.Series:
    """
    Monthly turnover cost of the rotation, indexed by formation month.  Each month
    the book is the *chosen* factor's full Q5/Q1 leg membership; ``cost.turnover_cost``
    then charges the genuine trading -- a near-full liquidation/re-establishment
    whenever the rotation switches factor, only membership drift when it holds the
    same factor (a name that happens to sit on the same leg of both factors is not
    traded).  Reported alongside performance, never netted from the gross alpha.
    """
    resolved = C.resolve_factors(factor_names)
    membership = {r.factor: _leg_membership(r) for r in resolved.itertuples()}
    held = [membership[f][membership[f]["date"] == t] for t, f in chosen.items()]
    legs = C.COST.equal_weight_legs(pd.concat(held, ignore_index=True))
    dates = pd.Index(sorted(chosen.index), name="date")
    return C.COST.turnover_cost(legs, C.cost_panel(), dates)


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
        "Factor-momentum long-short: each month hold the trailing-12m best factor\n"
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
    """The monthly factor-momentum rotation: hold the single trailing-12m best
    factor each month.  Writes every rotation output under ``out_dir`` and returns
    the per-window performance dict."""
    factor_names = top["factor"].tolist()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("--- univariate rotation (hold the trailing-12m best single factor) ---")

    # Monthly rotation over the candidates' signed L/S books.
    rotated, chosen = factor_momentum(spreads)

    # Performance vs the industry, exactly as every project book is measured.
    industry = C.industry_return()
    windows = [
        ("Factor momentum (full)", C.book_stats(rotated, industry)),
        ("Factor momentum (2016+)", C.book_stats(rotated, industry, start=DECADE_START)),
    ]
    full, decade = (w[1] for w in windows)
    start, end = rotated.index.min(), rotated.index.max()

    # Average monthly turnover cost of the rotation (reported, not netted).
    cost_series = rotation_cost(factor_names, chosen)
    full["avg_cost"] = C.window_cost(cost_series)
    decade["avg_cost"] = C.window_cost(cost_series, start=DECADE_START)

    # --- Persist outputs --------------------------------------------------- #
    panel = spreads.loc[rotated.index].copy()
    panel["chosen_factor"] = chosen
    panel["factor_momentum"] = rotated
    panel.to_csv(out_dir / "factor_momentum_returns.csv")

    counts = (chosen.value_counts().rename_axis("factor").rename("months")
                    .reindex(factor_names).fillna(0).astype(int).to_frame())
    counts["share"] = counts["months"] / counts["months"].sum()
    counts.to_csv(out_dir / "selection_counts.csv")

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
        out_dir / "performance.png")

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


# Grid / plot axis labels for the rotating double sort: the pair changes monthly,
# so the axes are the momentum *ranks* rather than fixed factor names.
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
        frame, _ = BT.double_sorted(C.resolve_factors(list(pair)))
        slices.append(frame.loc[frame["date"].isin(months), keep])
    return pd.concat(slices, ignore_index=True).sort_values(["date", "stock_id"])


def run_bivariate(spreads: pd.DataFrame, out_dir: Path = BIVARIATE_DIR) -> dict:
    """Re-select the two strongest trailing-12m univariate factors *each month* and
    combine them with ``bivariate_tertile``'s independent double sort, writing the
    rotating book's outputs under ``out_dir``.  Only the monthly pair selection lives
    here; the double-sort mechanics and diagnostics are reused from
    ``bivariate_tertile``.  Returns the per-window performance dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print("--- bivariate double sort (monthly top two by trailing-12m univariate return) ---")

    # 1. Monthly pair selection, then the rotating double-sorted cross-section.
    pairs = monthly_top_pairs(spreads)
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
    cost_series = BT.bivariate_cost(frame)
    full["avg_cost"] = C.window_cost(cost_series)
    decade["avg_cost"] = C.window_cost(cost_series, start=DECADE_START)

    # --- Persist outputs --------------------------------------------------- #
    held = pairs.loc[book.index]                     # pair traded each booked month
    out = book.copy()
    out["factor_1"] = held.apply(lambda p: p[0])
    out["factor_2"] = held.apply(lambda p: p[1])
    out.to_csv(out_dir / "bivariate_returns.csv")

    pair_counts = (held.apply(lambda p: f"{p[0]} x {p[1]}")
                       .value_counts().rename_axis("pair").rename("months").to_frame())
    pair_counts["share"] = pair_counts["months"] / pair_counts["months"].sum()
    pair_counts.to_csv(out_dir / "pair_counts.csv")

    BT.plot_grid(grids, BIVARIATE_AXES, out_dir / "grid_mean_return.png")
    BT.plot_long_short(spread, BIVARIATE_AXES, full["sharpe"], full["alpha"],
                       full["alpha_tstat"], out_dir / "long_short.png")
    C.render_performance(
        windows, "Bivariate factor-momentum double sort: long-short performance",
        "each month double-sort the trailing-12m top two factors (long top-tertile-of-both / "
        "short bottom-tertile-of-both)   |   "
        f"{len(spread)} months ({start:%Y-%m} .. {end:%Y-%m})   |   "
        "industry-neutral alpha from the market-cap-weighted industry return.   "
        "Shading: |t| >= 1.65 (10%), 2.0 (5%).",
        out_dir / "performance.png")

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


def run(csv_path: Path = TOP_FACTORS_CSV, out_dir: Path = OUTPUT_DIR) -> dict:
    """Build both Experiment-3 factor-momentum pipelines from the top-factor
    hand-off -- the monthly one-factor rotation (``univariate/``) and the double
    sort of the trailing-12m top two (``bivariate/``) -- and write every output
    under ``out_dir``.  Returns ``{"univariate": ..., "bivariate": ...}`` per-window
    performance dicts."""
    top = load_top_factors(csv_path)
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
