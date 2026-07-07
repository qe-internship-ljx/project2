"""
bivariate_gate.py
=================

Experiment 3 -- bivariate (independent double-sort) gate long/short.

Combine **two** factors not by aggregating them into one score (the
``composite.py`` route) but by an *independent double sort*: the industry
cross-section is split into three equal-count tertiles on the first factor and,
**separately**, into three tertiles on the second.  This overlays a 3x3 grid on the
cross-section.  Buckets are formed **quarterly** (the project-wide convention -- at
each end-Feb/May/Aug/Nov reposition date, held three months, via
``quarter_position.quarter_hold``) rather than re-sorted monthly.  The book is

    long   stocks in the **top** tertile of *both* factors   (T3 x T3 cell),
    short  stocks in the **bottom** tertile of *both* factors (T1 x T1 cell),

equal-weighted within each leg, and the dollar-neutral long-short spread is the
month-(t+1) long-leg mean return minus the short-leg mean return.  This is the
classic Fama-French-style conditional double sort: a stock must rank high (low)
on *both* signals to enter the long (short) book, so the two factors act as a
joint filter rather than a weighted average -- the corners of the grid are the
names the two factors *agree* are most / least attractive.

A second, coarser selection rule is tested alongside it -- the **half
intersection** (median double sort).  Each factor is split at its monthly median
into a top and bottom half, and the book is

    long   stocks in the **top half** of *both* factors,
    short  stocks in the **bottom half** of *both* factors,

again equal-weighted and dollar-neutral.  This is the same "both signals agree"
filter, only with 2 buckets per factor instead of 3, so each leg holds a much
larger, less extreme slice of the cross-section (~1/4 of names per corner rather
than ~1/9).  Reporting both lets us see whether the double sort's edge comes from
the extreme tertile corners or survives the milder, higher-capacity median split.

Default factors -- ``return_stability`` (Experiment 2 Stability library, monthly-
return consistency) and ``gross_profitability`` (Experiment 1 general factors,
profitability/quality).  Both are ``Q5-Q1`` factors (long the high-z names), so
each is oriented +1; the orientation is read from each factor's standalone
``long_short_market_alpha.csv`` ``direction`` exactly as ``composite.py`` does,
so any pair of factors from the project's libraries can be double-sorted.

Design -- reuses composite.py wholesale
---------------------------------------
Nothing generic is re-implemented.  The constituent loading and bullish
orientation (``resolve_factors`` / ``load_exposures``), the within-industry
"market" return and the window-split long-short performance (``industry_return``
/ ``book_stats``, Experiment 1's ``regression.py``), the constituents table and
the performance table (``render_factor_set`` / ``render_performance``) are all
imported from ``composite.py`` -- the same spine ``weighted_composite.py`` and
``factor_momentum.py`` build on.  The tertile labels are Experiment 1's own
``factors.assign_quintiles`` (with ``n=3``); the within-month return
winsorisation is ``factors.winsorize_cross_section`` -- the identical tail
treatment every other long-short leg and the industry benchmark receive.  This
module adds **only** the double-sort intersection and its grid diagnostic.

Outputs (``output/bivariate_gate/<slug>/``)
-------------------------------------------
The two selection rules each get their own subfolder of parallel files:
    tertile/                    the tertile-corner (3x3) rule's outputs:
        tertile/grid_mean_return.png    3x3 heatmap: mean next-month return of each
                                tertile cell, each cell also labelled with its
                                time-average market cap and number of names
        tertile/performance.png     the book's mean / t / Sharpe / industry-neutral
                                alpha / alpha above the gross_profitability &
                                revenue_stability books (with t-stats) / largest
                                single-name ownership for a $100M dollar-neutral book
                                / avg cost
    half/                       the half-intersection (median double-sort) rule's own
                                outputs, mirroring the tertile files above:
        half/grid_mean_return.png   2x2 counterpart of tertile/grid_mean_return.png
        half/performance.png        the book's performance table

Run standalone::

    python bivariate_gate.py                                   # default pair
    python bivariate_gate.py return_stability gross_profitability
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import composite as C                  # engine + analysis plumbing, loaded by path

# Reuse Experiment 1's engine / regression helpers through composite's handles.
F, R = C.F, C.R

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DEFAULT_FACTORS = ["return_stability", "gross_profitability"]
N_TILES = 3                                  # tertiles
TOP, BOTTOM = N_TILES, 1                      # top / bottom tertile labels
TCOLS = [f"T{i}" for i in range(1, N_TILES + 1)]
N_HALVES = 2                                  # median split (top / bottom half)
HALF_TOP, HALF_BOTTOM = N_HALVES, 1           # top / bottom half labels
OUTPUT_DIR = C.OUTPUT_DIR
DECADE_START = C.DECADE_START                # "past decade" cut-off (2016+), project convention

# Standalone long/short books the double-sort return is benchmarked against: the
# strategy is regressed on each one to measure the alpha it earns *above* that
# book (does the double sort add return beyond simply holding these factors?).
# Both resolve through composite's library catalog (gross_profitability lives in
# Experiment 1, revenue_stability in Experiment 2's Stability library).
BENCHMARK_FACTORS = ["gross_profitability", "revenue_stability"]

# The capital assumption behind the largest-single-name ownership row lives in
# composite.py (``PORTFOLIO_CAPITAL`` / ``LEG_CAPITAL``); the ownership itself is
# computed by ``composite.leg_ownership`` from each book's leg membership.


# --------------------------------------------------------------------------- #
# Step 1 -- assemble the oriented two-factor cross-section
# --------------------------------------------------------------------------- #
def double_sorted(resolved: pd.DataFrame) -> pd.DataFrame:
    """
    Tidy ``date, stock_id, <fa>, <fb>, next_return, mcap, tile_a, tile_b`` frame.

    Both factors' cross-sectional z-scores are read and inner-joined on
    ``(date, stock_id)`` via ``composite.load_exposures`` (so a stock is sorted
    only where *both* signals exist), oriented to their bullish sign, then each
    month independently bucketed into ``N_TILES`` equal-count tertiles with
    Experiment 1's ``assign_quintiles``.  ``next_return`` is winsorised within
    each month -- the same tail treatment the quintile legs and the industry
    benchmark get -- before any leg mean is taken.  Each stock-month's formation
    USD market cap (``mcap``) is attached from ``composite.market_cap_panel`` so
    the grid diagnostic can report the size of the names in each cell.
    """
    exposures, next_ret = C.load_exposures(resolved)        # raw z-scores, both present
    signs = pd.Series(dict(zip(resolved["factor"], resolved["sign"])))
    oriented = exposures.mul(signs, axis=1)                 # higher = bullish per factor
    fa, fb = resolved["factor"].tolist()

    frame = oriented.join(next_ret.rename("next_return"), how="inner").reset_index()
    frame = frame.dropna(subset=["next_return"]).sort_values(["date", "stock_id"])
    frame["next_return"] = F.winsorize_cross_section(frame["next_return"], frame["date"])
    frame = frame.merge(C.market_cap_panel(), on=["date", "stock_id"], how="left")

    frame["tile_a"] = F.assign_quintiles(frame, fa, "date", N_TILES)
    frame["tile_b"] = F.assign_quintiles(frame, fb, "date", N_TILES)
    # Median split of each factor for the coarser half-intersection rule tested
    # alongside the tertile corners (same cross-section, 2 buckets instead of 3).
    frame["half_a"] = F.assign_quintiles(frame, fa, "date", N_HALVES)
    frame["half_b"] = F.assign_quintiles(frame, fb, "date", N_HALVES)
    frame = frame.dropna(subset=["tile_a", "tile_b", "half_a", "half_b"])
    # Reposition quarterly (the project-wide convention): hold each stock's four
    # bucket assignments fixed at their quarter's reposition date (end Feb / May /
    # Aug / Nov) via the shared ``quarter_position.quarter_hold``, so the double sort
    # (and its median-split counterpart) re-forms only quarterly instead of monthly.
    frame = (C.QP.quarter_hold(frame, ["tile_a", "tile_b", "half_a", "half_b"])
              .drop(columns="form_period").sort_values(["date", "stock_id"]))
    return frame


def _intersection_book(frame: pd.DataFrame, col_a: str, col_b: str,
                       top, bottom) -> pd.DataFrame:
    """
    Monthly long / short / long-short returns of an intersection double-sort book.

    Long  = ``top`` bucket on *both* factors (columns ``col_a``/``col_b``, equal-
    weighted mean next-period return); short = ``bottom`` bucket on *both*.  Only
    months where both corner cells are populated contribute a spread.  Returns a
    frame indexed by month with ``long, short, long_short, n_long, n_short``.
    """
    long_cell = frame[(frame[col_a] == top) & (frame[col_b] == top)]
    short_cell = frame[(frame[col_a] == bottom) & (frame[col_b] == bottom)]

    long_ret = long_cell.groupby("date")["next_return"].mean()
    short_ret = short_cell.groupby("date")["next_return"].mean()
    book = pd.concat(
        {"long": long_ret, "short": short_ret,
         "n_long": long_cell.groupby("date").size(),
         "n_short": short_cell.groupby("date").size()}, axis=1)
    book["long_short"] = book["long"] - book["short"]
    return book.dropna(subset=["long_short"]).sort_index()


def _intersection_legs(frame: pd.DataFrame, col_a: str, col_b: str,
                       top, bottom) -> pd.DataFrame:
    """Equal-weighted corner-cell leg membership (``date, stock_id, leg, w``) of an
    intersection double-sort book: ``top`` bucket of both factors = long, ``bottom``
    bucket of both = short.  Shared by the cost and ownership diagnostics so both
    size exactly the names the book trades."""
    long_cell = frame.loc[(frame[col_a] == top) & (frame[col_b] == top),
                          ["date", "stock_id"]].assign(leg="long")
    short_cell = frame.loc[(frame[col_a] == bottom) & (frame[col_b] == bottom),
                           ["date", "stock_id"]].assign(leg="short")
    return C.COST.equal_weight_legs(pd.concat([long_cell, short_cell], ignore_index=True))


def _intersection_cost(frame: pd.DataFrame, col_a: str, col_b: str,
                       top, bottom) -> pd.Series:
    """
    Monthly turnover cost of an intersection double-sort book, using the project's
    one cost model.  The two legs are the corner cells (:func:`_intersection_legs`);
    the per-leg turnover summation is ``cost.turnover_cost`` (the routine the
    quintile books use).
    """
    legs = _intersection_legs(frame, col_a, col_b, top, bottom)
    dates = pd.Index(sorted(frame["date"].unique()), name="date")
    return C.COST.turnover_cost(legs, C.cost_panel(), dates)


def _intersection_ownership(frame: pd.DataFrame, col_a: str, col_b: str,
                            top, bottom) -> pd.Series:
    """Monthly largest single-name ownership share of an intersection double-sort
    book -- the corner-cell leg membership (:func:`_intersection_legs`) priced by
    the shared :func:`composite.leg_ownership` (dollar position ``LEG_CAPITAL / (leg
    headcount)`` over each name's formation market cap, per-month max across legs)."""
    return C.leg_ownership(_intersection_legs(frame, col_a, col_b, top, bottom))


def bivariate_book(frame: pd.DataFrame) -> pd.DataFrame:
    """Tertile double-sort book: long top tertile of both, short bottom of both."""
    return _intersection_book(frame, "tile_a", "tile_b", TOP, BOTTOM)


def bivariate_cost(frame: pd.DataFrame) -> pd.Series:
    """Turnover cost of the tertile double-sort book (corner cells T3xT3 / T1xT1)."""
    return _intersection_cost(frame, "tile_a", "tile_b", TOP, BOTTOM)


def bivariate_ownership(frame: pd.DataFrame) -> pd.Series:
    """Largest single-name ownership share of the tertile book (corner cells)."""
    return _intersection_ownership(frame, "tile_a", "tile_b", TOP, BOTTOM)


def half_book(frame: pd.DataFrame) -> pd.DataFrame:
    """Half-intersection book: long top half of both factors, short bottom half of
    both -- the coarser median-split counterpart of :func:`bivariate_book`."""
    return _intersection_book(frame, "half_a", "half_b", HALF_TOP, HALF_BOTTOM)


def half_cost(frame: pd.DataFrame) -> pd.Series:
    """Turnover cost of the half-intersection book (top-half-of-both / bottom-half-of-both)."""
    return _intersection_cost(frame, "half_a", "half_b", HALF_TOP, HALF_BOTTOM)


def half_ownership(frame: pd.DataFrame) -> pd.Series:
    """Largest single-name ownership share of the half-intersection book."""
    return _intersection_ownership(frame, "half_a", "half_b", HALF_TOP, HALF_BOTTOM)


def grid_stats(frame: pd.DataFrame, col_a: str = "tile_a", col_b: str = "tile_b",
               n: int = N_TILES) -> dict[str, pd.DataFrame]:
    """
    Three ``n`` x ``n`` grids over the bucket cells (rows = factor-A bucket
    T1..Tn, columns = factor-B bucket T1..Tn), each time-averaged so every month
    counts equally regardless of cell size -- consistent with the long-short book:

        ``return``  mean next-period return of the cell,
        ``mcap``    mean formation USD market cap of the cell's names,
        ``count``   number of names in the cell.

    Each month a cell's mean return, mean market cap and headcount are taken, then
    averaged across months.  The diagonal corners are the short (T1,T1) and long
    (Tn,Tn) legs of the double-sort book.  Defaults sort on the tertile columns;
    pass ``half_a``/``half_b`` with ``n=N_HALVES`` for the median-split 2x2 grid.
    """
    monthly = (frame.groupby(["date", col_a, col_b])
                    .agg(ret=("next_return", "mean"),
                         mcap=("mcap", "mean"),
                         count=("stock_id", "size"))
                    .reset_index())

    def _grid(col: str) -> pd.DataFrame:
        grid = monthly.pivot_table(index=col_a, columns=col_b,
                                   values=col, aggfunc="mean")
        grid = grid.reindex(index=range(1, n + 1), columns=range(1, n + 1))
        grid.index = [f"T{i}" for i in grid.index]
        grid.columns = [f"T{i}" for i in grid.columns]
        return grid

    return {name: _grid(name) for name in ("ret", "mcap", "count")}


def benchmark_alphas(spread: pd.Series, benchmarks: dict[str, pd.Series],
                     start: pd.Timestamp | None = None,
                     end: pd.Timestamp | None = None) -> dict:
    """
    Alpha of the double-sort book *above* each benchmark long/short book, over an
    optional ``[start, end]`` window.

    For each ``name -> benchmark return`` the strategy is regressed on that book,
    ``spread_t = alpha + beta * benchmark_t + eps_t`` (Experiment 1's
    ``market_regression``, the same estimator used for industry-neutral alpha),
    so ``alpha`` is the mean return the double sort earns that the benchmark book
    does not explain.  Returns a flat dict with ``alpha_vs_<name>`` and
    ``alpha_tstat_vs_<name>`` for every benchmark -- ready to merge into a
    :func:`composite.book_stats` window and render as extra performance rows.
    """
    out: dict[str, float] = {}
    for name, bench in benchmarks.items():
        s, b = spread, bench
        if start is not None:
            s, b = s[s.index >= start], b[b.index >= start]
        if end is not None:
            s, b = s[s.index <= end], b[b.index <= end]
        reg = R.market_regression(s, b)
        out[f"alpha_vs_{name}"] = reg["alpha"]
        out[f"alpha_tstat_vs_{name}"] = reg["alpha_tstat"]
    return out


def evaluate_book(spread: pd.Series, cost_series: pd.Series, industry: pd.Series,
                  benchmarks: dict[str, pd.Series],
                  ownership: pd.Series) -> tuple[list, dict, dict]:
    """
    Full-sample and past-decade performance windows for a long-short ``spread``.

    Each window is a :func:`composite.book_stats` dict augmented with the average
    turnover cost over that window, the largest single-name ownership share within
    it (:func:`composite.attach_ownership`, windowed with ``max``), and the alpha the
    book earns *above* each standalone benchmark book (:func:`benchmark_alphas`).
    Shared by both the tertile-corner and the half-intersection selection rules so
    they are measured identically.  Returns ``(windows, full_stats, decade_stats)``.
    """
    windows = [("Full sample", C.book_stats(spread, industry)),
               ("Past decade (2016+)", C.book_stats(spread, industry, start=DECADE_START))]
    full, decade = windows[0][1], windows[1][1]
    full["avg_cost"] = C.window_cost(cost_series)
    decade["avg_cost"] = C.window_cost(cost_series, start=DECADE_START)
    C.attach_net_cost_sharpe(full, spread, cost_series, industry)
    C.attach_net_cost_sharpe(decade, spread, cost_series, industry, start=DECADE_START)
    C.attach_ownership(full, ownership)
    C.attach_ownership(decade, ownership, start=DECADE_START)
    full.update(benchmark_alphas(spread, benchmarks))
    decade.update(benchmark_alphas(spread, benchmarks, start=DECADE_START))
    return windows, full, decade


def _benchmark_extra_metrics() -> list[tuple[str, str, str, bool]]:
    """``render_performance`` rows for the per-benchmark alpha / alpha t-stat."""
    extra: list[tuple[str, str, str, bool]] = []
    for f in BENCHMARK_FACTORS:
        extra.append((f"α vs {f} book (monthly)", f"alpha_vs_{f}", "pct", False))
        extra.append((f"    α t-stat vs {f}", f"alpha_tstat_vs_{f}", "num", True))
    return extra


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _pair_label(factor_names: list[str]) -> str:
    return f"{factor_names[0]}  x  {factor_names[1]}"


def _fmt_mcap(v: float) -> str:
    """Compact USD market cap: $B above a billion, else $M."""
    if not np.isfinite(v):
        return ""
    return f"${v / 1e9:.1f}B" if v >= 1e9 else f"${v / 1e6:.0f}M"


def plot_grid(grids: dict[str, pd.DataFrame], factor_names: list[str],
              path: Path, n: int = N_TILES, split_word: str = "tertile") -> None:
    """Heatmap of mean next-month return per bucket cell (colour = return), each
    cell also labelled with its time-average market cap and headcount; the long /
    short corner cells are outlined.  ``n``/``split_word`` describe the sort
    granularity (``N_TILES``/"tertile" by default, or ``N_HALVES``/"half")."""
    fa, fb = factor_names
    ret, mcap, count = grids["ret"], grids["mcap"], grids["count"]
    vals = ret.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    vmax = np.nanmax(np.abs(vals))
    im = ax.imshow(vals, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")

    for i in range(n):
        for j in range(n):
            v = vals[i, j]
            if not np.isfinite(v):
                continue
            # Deep cmap cells (|return| near the scale extreme) are dark, so switch
            # the annotations to light text there for legibility.
            dark = abs(v) > 0.55 * vmax
            main_c = "white" if dark else "#111111"
            sub_c = "#eeeeee" if dark else "#333333"
            ax.text(j, i - 0.22, f"{v:+.3%}", ha="center", va="center",
                    fontsize=11, weight="bold", color=main_c)
            ax.text(j, i + 0.06, f"avg cap {_fmt_mcap(mcap.iat[i, j])}",
                    ha="center", va="center", fontsize=8, color=sub_c)
            ax.text(j, i + 0.24, f"avg n {count.iat[i, j]:.0f}",
                    ha="center", va="center", fontsize=8, color=sub_c)
    # Outline the long (top,top) and short (bottom,bottom) corner cells.
    top = n - 1
    for (i, j, edge, lbl) in [(top, top, "#0b3d0b", "LONG"), (0, 0, "#7f1d1d", "SHORT")]:
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor=edge, linewidth=3))
        lbl_c = "white" if abs(vals[i, j]) > 0.55 * vmax else edge
        ax.text(j, i - 0.40, lbl, ha="center", va="center", fontsize=8,
                color=lbl_c, weight="bold")

    ax.set_xticks(range(n)); ax.set_xticklabels(ret.columns)
    ax.set_yticks(range(n)); ax.set_yticklabels(ret.index)
    ax.set_xlabel(f"{fb} {split_word}  (T1 = low ... T{n} = high)")
    ax.set_ylabel(f"{fa} {split_word}  (T1 = low ... T{n} = high)")
    ax.set_title(f"Mean next-month return by {split_word} cell (independent double sort)\n"
                 "cell also shows time-average market cap and number of names\n"
                 f"{_pair_label(factor_names)}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Mean monthly return")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run(factor_names: list[str] = DEFAULT_FACTORS,
        label: str | None = None,
        out_root: Path = OUTPUT_DIR) -> dict:
    """
    Run the bivariate-gate double sort for the two ``factor_names`` and write
    every output under ``out_root / "bivariate_gate" / <slug>``.  ``label``
    overrides the slug (default: the factor names joined by ``__``).  Returns the
    per-window performance dict.
    """
    if len(factor_names) != 2:
        raise ValueError("bivariate_gate takes exactly two factors; got "
                         f"{len(factor_names)}: {factor_names}")

    resolved = C.resolve_factors(factor_names)
    slug = label or "__".join(factor_names)
    out_dir = out_root / "bivariate_gate" / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Experiment 3: bivariate gate (independent double-sort) L/S ===")
    print(f"Factors: " + ", ".join(f"{r.factor} [{'+' if r.sign > 0 else '-'}]"
                                    for r in resolved.itertuples()))

    # 1. Oriented two-factor cross-section, independently bucketed into tertiles
    #    (tile_a/tile_b) and into halves (half_a/half_b) for the two rules.
    frame = double_sorted(resolved)
    industry = C.industry_return()
    benchmarks = {f: C.factor_long_short(f) for f in BENCHMARK_FACTORS}
    extra_metrics = _benchmark_extra_metrics() + [C.ownership_metric()]

    # 2a. Tertile-corner rule: long top tertile of both / short bottom of both.
    book = bivariate_book(frame)
    grids = grid_stats(frame)
    spread = book["long_short"]
    windows, full, decade = evaluate_book(spread, bivariate_cost(frame),
                                          industry, benchmarks,
                                          bivariate_ownership(frame))
    meta = {"n_stocks": frame["stock_id"].nunique(),
            "n_months": int(full["n_months"]),
            "start": spread.index.min(), "end": spread.index.max(),
            "avg_long": book["n_long"].mean(), "avg_short": book["n_short"].mean()}

    # 2b. Half-intersection rule: long top half of both / short bottom half of both.
    hbook = half_book(frame)
    hgrids = grid_stats(frame, "half_a", "half_b", N_HALVES)
    hspread = hbook["long_short"]
    hwindows, hfull, hdecade = evaluate_book(hspread, half_cost(frame),
                                             industry, benchmarks,
                                             half_ownership(frame))
    hmeta = {"avg_long": hbook["n_long"].mean(), "avg_short": hbook["n_short"].mean(),
             "n_months": int(hfull["n_months"]),
             "start": hspread.index.min(), "end": hspread.index.max()}

    # --- Persist outputs (tertile corners, grouped in its own subfolder) ---- #
    tertile_dir = out_dir / "tertile"
    tertile_dir.mkdir(parents=True, exist_ok=True)
    plot_grid(grids, factor_names, tertile_dir / "grid_mean_return.png")
    C.render_performance(
        windows, "Bivariate tertile long-short (top-of-both - bottom-of-both) performance",
        f"{_pair_label(factor_names)}   |   {meta['n_stocks']} stocks over "
        f"{meta['n_months']} months ({meta['start']:%Y-%m} .. {meta['end']:%Y-%m}); "
        f"avg leg ~{meta['avg_long']:.0f} long / {meta['avg_short']:.0f} short   |   "
        "independent 3x3 sort; industry-neutral alpha from the market-cap-weighted "
        "industry return   |   "
        "benchmark alphas from regressing the book on each standalone factor book   |   "
        "Shading: |t| >= 1.65 (10%), 2.0 (5%).",
        tertile_dir / "performance.png",
        extra_metrics=extra_metrics)

    # --- Persist outputs (half intersection, grouped in its own subfolder) -- #
    half_dir = out_dir / "half"
    half_dir.mkdir(parents=True, exist_ok=True)
    plot_grid(hgrids, factor_names, half_dir / "grid_mean_return.png",
              n=N_HALVES, split_word="half")
    C.render_performance(
        hwindows,
        "Bivariate half-intersection long-short (top-half-of-both - bottom-half-of-both) performance",
        f"{_pair_label(factor_names)}   |   {meta['n_stocks']} stocks over "
        f"{hmeta['n_months']} months ({hmeta['start']:%Y-%m} .. {hmeta['end']:%Y-%m}); "
        f"avg leg ~{hmeta['avg_long']:.0f} long / {hmeta['avg_short']:.0f} short   |   "
        "median (2x2) double sort; industry-neutral alpha from the market-cap-weighted "
        "industry return   |   "
        "benchmark alphas from regressing the book on each standalone factor book   |   "
        "Shading: |t| >= 1.65 (10%), 2.0 (5%).",
        half_dir / "performance.png",
        extra_metrics=extra_metrics)

    # --- Console summary (ASCII only -- Windows cp1252 stdout) ------------- #
    ret_grid = grids["ret"]
    print(f"  grid mean next-month return (rows={factor_names[0]} tertile, "
          f"cols={factor_names[1]} tertile):")
    for ti in ret_grid.index:
        print("    " + ti + "  " +
              "  ".join(f"{tj}={ret_grid.loc[ti, tj]:+.3%}" for tj in ret_grid.columns))
    print("  [tertile corners] "
          f"long leg (T3,T3) ~{meta['avg_long']:.0f} names / "
          f"short leg (T1,T1) ~{meta['avg_short']:.0f} names")
    print(f"  long/short: {full['mean_monthly']:+.4%}/mo "
          f"(t={full['tstat']:+.2f}, Sharpe={full['sharpe']:+.2f})")
    print(f"  industry-neutral alpha: {full['alpha']:+.4%}/mo "
          f"(t={full['alpha_tstat']:+.2f}, ind beta={full['ind_beta']:+.2f})")
    for f in BENCHMARK_FACTORS:
        print(f"  alpha vs {f} book: {full[f'alpha_vs_{f}']:+.4%}/mo "
              f"(t={full[f'alpha_tstat_vs_{f}']:+.2f})")
    print(f"  2016+: alpha={decade['alpha']:+.4%}/mo (t={decade['alpha_tstat']:+.2f})")
    print("  [half intersection] "
          f"long leg (top half x top half) ~{hmeta['avg_long']:.0f} names / "
          f"short leg (bottom x bottom) ~{hmeta['avg_short']:.0f} names")
    print(f"  long/short: {hfull['mean_monthly']:+.4%}/mo "
          f"(t={hfull['tstat']:+.2f}, Sharpe={hfull['sharpe']:+.2f})")
    print(f"  industry-neutral alpha: {hfull['alpha']:+.4%}/mo "
          f"(t={hfull['alpha_tstat']:+.2f}, ind beta={hfull['ind_beta']:+.2f})")
    for f in BENCHMARK_FACTORS:
        print(f"  alpha vs {f} book: {hfull[f'alpha_vs_{f}']:+.4%}/mo "
              f"(t={hfull[f'alpha_tstat_vs_{f}']:+.2f})")
    print(f"  2016+: alpha={hdecade['alpha']:+.4%}/mo (t={hdecade['alpha_tstat']:+.2f})")
    print(f"Saved -> {out_dir}")
    return {**{lbl: s for lbl, s in windows},
            **{f"half: {lbl}": s for lbl, s in hwindows}}


def main() -> None:
    args = sys.argv[1:]
    run(args or DEFAULT_FACTORS)


if __name__ == "__main__":
    main()
