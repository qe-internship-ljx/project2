"""
composite.py
============

Experiment 3 -- composite z-score long/short, repositioned quarterly.

Combine Experiment 2's **top five factors** -- the top five of the quarterly-
repositioned cross-experiment ranking (``quarter_position.ranked_factors``, the
single factor-selection hand-off; the monthly re-evaluation renders comparison
PNGs only, no ranking CSV) -- into one cross-sectional
score, the sum of each stock's sign-oriented factor z-scores, and run Experiment 1's
quintile workflow on that composite::

    composite_{i,t} = sum_f  sign_f * zscore_{f,i,t}

The composite score is formed from each stock's *monthly* cross-sectional z-scores,
but the book is **repositioned quarterly** (the project-wide convention): buckets are
formed only at the end of Feb / May / Aug / Nov and held fixed for the three
following months, via ``quarter_position.quarter_held_membership`` -- the same
quarterly-holding primitive every Experiment 2-5 book now uses.  The next-period
(month t+1) return of each held bucket is tracked, and the top-minus-bottom long/short
book's performance is measured: mean, t-stat, annualised Sharpe, the industry-neutral
alpha and its t-stat, and the cumulative growth path.

``sign_f`` orients every factor to its *bullish* direction before summing, read
from the ``direction`` column of that factor's standalone long/short alpha table
(``Q5-Q1`` -> +1, long the high-z names;  ``Q1-Q5`` -> -1).  A high composite is
therefore "attractive across the whole set", so the book is always long Q5 /
short Q1 and no per-factor sign bookkeeping is left to the caller.

Design -- maximal reuse, zero duplication of the engine
-------------------------------------------------------
Nothing generic is re-implemented:

* the cross-section mechanics -- within-month winsorisation, equal-count
  quintile labels, the quintile-return table -- are Experiment 1's:
  ``quintile.py`` and the ``factors.py`` engine are reused *unmodified* through
  the project's dependency-injection convention (register the engine as
  ``sys.modules["factors"]`` before importing the analysis modules, exactly as
  Experiment 2's ``monthly_position.py`` does);
* the long/short book's industry-neutral alpha is measured by Experiment 1's own
  ``regression.market_regression`` / ``industry_monthly_return``, so "alpha" is
  defined identically to every other long/short book in the project;
* the factor exposures and their bullish orientation are read straight from the
  ``factor_panel.csv`` / ``long_short_market_alpha.csv`` files Experiments 1 & 2
  already wrote.

This module therefore adds only the *composite construction* and the per-set
reporting; every input it consumes was produced upstream.

The pipeline is run over **three quarterly-repositioned factor selections** -- the
``quarter_quintile`` ranking (each factor's quarterly QUINTILE book, the project's
headline hand-off), the ``quarter_tertile`` ranking (its quarterly TERTILE book) and
the ``quarter_half`` ranking (its quarterly HALF book).  All three slice the top five
of ``quarter_position.ranked_factors`` for that bucketing; the composite is built
identically each time -- from the factors' monthly cross-sectional z-scores -- so this
isolates the effect of *which* five factors a ranking picks (quintile-conviction
leaders vs tertile-diversity vs half-breadth leaders), not how the composite is scored.

The ``quarter_quintile`` selection's composite is sorted into **quintiles**, the
``quarter_tertile`` selection's into **tertiles**, and the ``quarter_half``
selection's into **halves** (top-half-minus-bottom-half of the same z-score sum), so
the composite is always bucketed the same way its constituents were ranked.

Outputs (``output/composite/<ranking>/``, one subdir per ranking -- ``quarter_quintile`` /
``quarter_tertile`` / ``quarter_half``)
-----------------------------------------------------------------------------------------
    <sort>_cumulative.png       the buckets as cumulative growth of $1 (log scale)
    <sort>_performance.png      the L/S book's mean / t / Sharpe / industry-neutral alpha /
                                largest single-name ownership for a $100M book / avg cost
                                (``<sort>`` = ``quintile`` for ``quarter_quintile``,
                                ``tertile`` for ``quarter_tertile``, ``half`` for
                                ``quarter_half``)

Run standalone::

    python composite.py         # all three quarterly rankings' top-5 composites
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
EXP3_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXP3_DIR.parent
EXP1_DIR = PROJECT_ROOT / "experiment1 - general factors"
EXP2_DIR = PROJECT_ROOT / "experiment2 - sw factors"
OUTPUT_DIR = EXP3_DIR / "output"

# The single factor-selection hand-off is Experiment 2's *quarterly-repositioned*
# ranking, which ``quarter_position.ranked_factors`` reads from the CSV Experiment 2
# persists (``Factor Ranking/quarter_{label}_ranked.csv``): the top-N consumers slice
# its leaders, while modules that sweep the whole candidate set (e.g. Experiment 4's
# timing overlays) read every row.  Re-exported here as :func:`ranked_factors` so
# Experiment 3-5 modules that already import ``composite`` reach it without importing
# ``quarter_position`` directly.
TOP_N = 5

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
COMPOSITE_FACTOR = "composite"          # synthetic factor name fed to the reused sort
N_QUINTILES = 5
N_TERTILES = 3
N_HALVES = 2
# Word for an even ``n``-bucket sort of the composite score, used in filenames and
# captions (5 -> "quintile", 3 -> "tertile", 2 -> "half"; any other count falls back
# to "n-bucket").
_SORT_WORD = {N_QUINTILES: "quintile", N_TERTILES: "tertile", N_HALVES: "half"}


def sort_word(n: int) -> str:
    return _SORT_WORD.get(n, f"{n}-bucket")


MONTHS_PER_YEAR = 12
DECADE_START = pd.Timestamp("2016-01-01")   # "past decade" cut-off (project convention)

# Capital assumption for the largest-single-name ownership / capacity diagnostic
# reported on every Experiment 3 book.  Each book is dollar-neutral, so a $100M
# *total* portfolio funds $50M in each leg; a name's dollar position is
# ``LEG_CAPITAL * (its weight within the leg)`` and its ownership share is that over
# the name's own formation market cap.
PORTFOLIO_CAPITAL = 100_000_000.0           # $100M total (dollar-neutral book)
LEG_CAPITAL = PORTFOLIO_CAPITAL / 2         # $50M invested in each leg


# --------------------------------------------------------------------------- #
# Experiment 1 engine + analysis modules, loaded by path and wired for reuse
# --------------------------------------------------------------------------- #
def _load(name: str, path: Path):
    """Import a module by file path under ``name`` and register it in sys.modules."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_exp1():
    """
    Load Experiment 1's engine and analysis modules by path (their folder name
    has spaces, so they cannot be imported normally), wired for reuse.

    ``quintile`` / ``regression`` (and ``cost``) bind their factor library via
    ``import factors as F`` / ``import cost``; registering the engine and cost
    under those names *first* makes those imports resolve here -- the project's
    standard dependency-injection convention (cf. Experiment 2's ``monthly_position.py``).
    """
    engine = _load("factors", EXP1_DIR / "factors.py")
    _load("cost", EXP1_DIR / "cost.py")              # imported by regression.py
    quintile = _load("quintile", EXP1_DIR / "quintile.py")
    regression = _load("regression", EXP1_DIR / "regression.py")
    return engine, quintile, regression


F, Q, R = _load_exp1()
import cost as COST                       # registered by _load_exp1; the turnover-cost model

# Experiment 2's quarterly-repositioning driver, loaded by path.  It owns the shared
# quarterly-book primitives (``quarter_held_membership`` / ``quarter_held_spread`` /
# ``quarter_held_legs``) every Experiment 3-5 book reuses, and the ``ranked_factors``
# hand-off.  Its own ``import factors / cost / regression`` resolve to the engine
# just registered above, so the generic engine is shared (no re-load).
QP = _load("exp2_quarter_position", EXP2_DIR / "quarter_position.py")

# Factor libraries that can supply a constituent.  Each contributes a tidy panel
# (date, stock_id, factor, value, zscore, next_return) carrying the exposures and
# an alpha table whose ``direction`` column records each factor's bullish sign.
# A requested factor is resolved against these in order (first match wins), so its
# source -- panel and orientation -- is always unambiguous.  The roster is exactly
# Experiment 2's own factor-universe source-of-truth (``quarter_position.SOURCES``,
# spanning Experiment 1's general factors plus every Experiment 2 software factor
# library), reused wholesale so Experiment 3 never reaches into Experiment 2's
# internal subfolders itself and any factor in the top-factor hand-off resolves.
# The Banks+Insurance cross-validation library is excluded there: a different
# cross-section.
LIBRARIES: list[dict] = [
    {"label": src["label"], "panel": src["panel"], "alpha": src["alpha_csv"]}
    for src in QP.SOURCES
]


def ranked_factors(n: int | None = None, label: str = "quintile") -> pd.DataFrame:
    """The quarterly-repositioned factor-selection hand-off -- a thin re-export of
    :func:`quarter_position.ranked_factors` (see it).  ``label`` picks the bucketing
    (``quintile`` book, the headline; ``tertile`` for the tertile-ranked composite);
    ``n`` keeps the top-``n`` leaders or, when ``None``, every ranked factor."""
    return QP.ranked_factors(n, label)


def _read_factor_panel(path: str | Path) -> pd.DataFrame:
    """Read a factor library's tidy panel (the columns the quarterly sort needs)."""
    panel = pd.read_csv(path, parse_dates=["date"],
                        usecols=["date", "stock_id", "factor", "zscore", "next_return"])
    panel["stock_id"] = panel["stock_id"].astype(str)
    return panel


# --------------------------------------------------------------------------- #
# Step 1 -- resolve the requested factors to (source panel, bullish sign)
# --------------------------------------------------------------------------- #
def resolve_factors(factor_names: list[str]) -> pd.DataFrame:
    """
    Map each requested factor to its source library, family, bullish sign and
    standalone long/short alpha, read from the libraries' alpha tables.

    The sign orients the factor so a higher score is bullish: ``+1`` when the
    factor's standalone book is long the top z-score quintile (``direction`` ==
    ``Q5-Q1``), ``-1`` otherwise.  Returns one row per factor in the requested
    order; raises if a factor is in none of the libraries.
    """
    catalog: dict[str, dict] = {}
    for lib in LIBRARIES:
        if not Path(lib["alpha"]).exists():
            continue
        tbl = pd.read_csv(lib["alpha"])
        for _, r in tbl.iterrows():
            catalog.setdefault(r["factor"], {           # first library wins
                "factor": r["factor"], "family": r["family"],
                "direction": r["direction"], "alpha": r["alpha"],
                "alpha_tstat": r["alpha_tstat"],
                "library": lib["label"], "panel_path": str(lib["panel"])})

    rows = []
    for name in factor_names:
        if name not in catalog:
            libs = ", ".join(lib["label"] for lib in LIBRARIES)
            raise KeyError(f"factor {name!r} not found in any library ({libs}). "
                           "Run Experiments 1-2 (and the R&D extension) first.")
        meta = dict(catalog[name])
        meta["sign"] = 1 if str(meta["direction"]).strip() == "Q5-Q1" else -1
        rows.append(meta)

    return pd.DataFrame(rows, columns=["factor", "family", "library", "direction",
                                       "sign", "alpha", "alpha_tstat", "panel_path"])


def factor_long_short(factor: str) -> pd.Series:
    """
    A single factor's bullish-oriented **quarterly-repositioned** standalone
    long/short return, indexed by formation month.

    Computed from the factor's source ``factor_panel.csv`` via the shared
    ``quarter_position.quarter_held_spread`` (buckets formed quarterly, held three
    months), oriented by its bullish ``sign`` -- the exact quarterly dollar-neutral
    book every other Experiment 2-5 book now trades, so it is the natural benchmark
    to regress another strategy against.  Shared by ``factor_momentum.py`` (rotation
    candidates) and ``bivariate_gate.py`` / ``weighted_composite.py``
    (benchmark-relative alpha).
    """
    meta = resolve_factors([factor]).iloc[0]
    panel = _read_factor_panel(meta["panel_path"])
    return QP.quarter_held_spread(panel, factor, int(meta["sign"])).rename(factor)


# --------------------------------------------------------------------------- #
# Step 2 -- assemble the composite score
#
# These three helpers are the reusable spine of every composite variant:
# ``load_exposures`` reads the raw constituent matrix, ``scored_frame`` attaches
# the realised return to any per-stock-month score, and ``load_composite`` is the
# straight (sign-oriented, equal-weighted) sum built from them.  A weighted
# variant just supplies a different score over the same exposures (see
# ``weighted_composite.py``).
# --------------------------------------------------------------------------- #
def load_exposures(resolved: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Wide ``(date, stock_id) x factor`` matrix of the constituents' **raw**
    cross-sectional z-scores (in the requested order), plus the per-stock-month
    ``next_return`` series, assembled across the constituents' source panels.

    The per-library blocks are aligned on the shared ``(date, stock_id)`` key and
    rows are kept only where **every** factor is present, so any downstream
    combination (straight sum, sign-oriented sum, coefficient-weighted sum) scores
    each stock on the complete set.  Orientation/weighting is *not* applied here --
    callers weight the columns themselves (by sign, by regression coefficient...).
    """
    blocks: dict[str, pd.Series] = {}        # factor -> z-score, indexed by key
    next_ret: pd.Series | None = None        # next_return, indexed by key

    for panel_path, grp in resolved.groupby("panel_path", sort=False):
        names = grp["factor"].tolist()
        panel = pd.read_csv(panel_path, parse_dates=["date"],
                            usecols=["date", "stock_id", "factor", "zscore", "next_return"])
        panel["stock_id"] = panel["stock_id"].astype(str)
        panel = panel[panel["factor"].isin(names)]

        wide = panel.pivot_table(index=["date", "stock_id"], columns="factor",
                                 values="zscore")
        for name in names:
            blocks[name] = wide[name]

        # next_return is identical across factors for a given stock-month; take
        # it once per (date, stock_id) and union across libraries.
        nr = (panel.dropna(subset=["next_return"])
                   .drop_duplicates(["date", "stock_id"])
                   .set_index(["date", "stock_id"])["next_return"])
        next_ret = nr if next_ret is None else next_ret.combine_first(nr)

    exposures = pd.concat(blocks, axis=1)[resolved["factor"].tolist()].dropna()
    return exposures, next_ret


def scored_frame(score: pd.Series, next_ret: pd.Series) -> pd.DataFrame:
    """
    Attach the realised ``next_return`` to a per-``(date, stock_id)`` composite
    ``score`` and return a tidy ``date, stock_id, composite, next_return`` frame
    (dropping the final month, which has no t+1 return).  Shared by every variant.
    """
    return (pd.concat([score.rename("composite"),
                       next_ret.rename("next_return")], axis=1)
              .loc[score.index]                  # scored rows only
              .dropna(subset=["next_return"])    # drop the last month (no t+1 return)
              .reset_index()
              .sort_values(["date", "stock_id"]))


def load_composite(resolved: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Straight (equal-weighted) composite: orient each constituent z-score to its
    bullish sign and sum across factors.

    Returns ``(composite, exposures)`` where ``composite`` has columns
    ``date, stock_id, composite, next_return`` and ``exposures`` is the wide
    matrix of oriented constituent z-scores (for the correlation diagnostic).
    """
    exposures, next_ret = load_exposures(resolved)
    signs = pd.Series(dict(zip(resolved["factor"], resolved["sign"])))
    oriented = exposures.mul(signs, axis=1)              # +/-1 per factor
    score = oriented.sum(axis=1)
    return scored_frame(score, next_ret), oriented


def as_factor_panel(composite: pd.DataFrame) -> pd.DataFrame:
    """
    Shape the composite as a one-factor tidy panel so Experiment 1's quintile
    code (which keys off ``factor`` and reads ``zscore`` / ``next_return``) runs
    on it unmodified -- the composite plays the role of the factor z-score.
    """
    panel = composite.copy()
    panel["factor"] = COMPOSITE_FACTOR
    panel["value"] = panel["composite"]
    panel["zscore"] = panel["composite"]
    return panel


def bucket_returns(panel: pd.DataFrame, n: int = N_QUINTILES,
                   factor: str = COMPOSITE_FACTOR) -> pd.DataFrame:
    """Wide monthly table of each composite-score bucket's mean next-period return
    plus the top-minus-bottom (``Q{n}-Q1``) long/short spread, for an even
    ``n``-bucket sort (``n=5`` quintiles, ``n=3`` tertiles), **repositioned quarterly**.

    Even-count buckets are formed at each reposition date via
    ``factors.prepare_slice`` and held fixed for the quarter with
    ``quarter_position.quarter_held_membership`` (the shared quarterly primitive),
    then the held bucket's mean next-period return is tracked -- the quarterly,
    bucket-count-generic counterpart of Experiment 1's ``quintile.quintile_returns``,
    so the composite is tracked in tertiles as well as quintiles with no new code."""
    cols = [f"Q{i}" for i in range(1, n + 1)]
    held = QP.quarter_held_membership(F.prepare_slice(panel, factor, n))
    wide = (held.pivot_table(index="date", columns="leg",
                             values="next_return", aggfunc="mean")
                .rename(columns={i: f"Q{i}" for i in range(1, n + 1)})
                .sort_index()
                .reindex(columns=cols))
    wide[f"Q{n}-Q1"] = wide[f"Q{n}"] - wide["Q1"]
    return wide


# --------------------------------------------------------------------------- #
# Step 3 -- long/short performance (reusing the engine's alpha definition)
# --------------------------------------------------------------------------- #
def industry_return() -> pd.Series:
    """
    Market-cap-weighted Software & Services next-period return -- the within-
    industry "market" -- from Experiment 1's saved panel.  This is the exact series
    every project long/short book is regressed on for its industry-neutral alpha.
    """
    return R.industry_monthly_return(F.load_panel(u=F.SOFTWARE_SERVICES))


_COST_PANEL: pd.DataFrame | None = None


def cost_panel() -> pd.DataFrame:
    """The Software & Services per-(stock, month) round-trip cost panel, built once
    and cached.  Shared by every Experiment 3 book so the reported turnover cost
    uses the project's one cost model (``cost.py``)."""
    global _COST_PANEL
    if _COST_PANEL is None:
        _COST_PANEL = COST.build_cost_panel(F.SOFTWARE_SERVICES)
    return _COST_PANEL


_MCAP_PANEL: pd.DataFrame | None = None


def market_cap_panel() -> pd.DataFrame:
    """Per-``(date, stock_id)`` formation-date USD market cap, built once and
    cached (tidy ``date, stock_id, mcap``).

    ``mcap`` is the ``weight`` column of the saved Software & Services panel --
    the month-end USD cap that weights each stock's next-period return, i.e. its
    market cap at the formation of the return every book earns.  Shared by any
    Experiment 3 book that needs to describe or size the names it holds."""
    global _MCAP_PANEL
    if _MCAP_PANEL is None:
        panel = F.load_panel(u=F.SOFTWARE_SERVICES)
        _MCAP_PANEL = (panel.dropna(subset=["weight"])
                            .drop_duplicates(["date", "stock_id"])
                            [["date", "stock_id", "weight"]]
                            .rename(columns={"weight": "mcap"})
                            .reset_index(drop=True))
    return _MCAP_PANEL


def window_cost(cost_series: pd.Series,
                start: pd.Timestamp | None = None,
                end: pd.Timestamp | None = None) -> float:
    """Average monthly turnover cost over an optional ``[start, end]`` window
    (fraction of notional).  Reported alongside performance, **not** netted from
    the alpha or Sharpe -- those stay gross, as in every other project book."""
    s = cost_series.dropna()
    if start is not None:
        s = s[s.index >= start]
    if end is not None:
        s = s[s.index <= end]
    return float(s.mean()) if len(s) else float("nan")


# --------------------------------------------------------------------------- #
# Largest single-name ownership -- shared capacity diagnostic
#
# The one reusable ownership computation every Experiment 3 book reports.  A book
# is described to the cost model as a tidy ``date, stock_id, leg, w`` frame (``w``
# = a name's weight within its leg that month); the same frame prices the biggest
# single position we would hold.  ``quintile_legs`` builds that frame for the Q5/Q1
# quintile books; the tertile double-sort and the netted overlay build their own,
# then all feed :func:`leg_ownership`.
# --------------------------------------------------------------------------- #
def quintile_legs(panel: pd.DataFrame, factor: str = COMPOSITE_FACTOR,
                  n_quintiles: int = N_QUINTILES) -> pd.DataFrame:
    """Equal-weighted top (long) / bottom (short) bucket leg membership of a scored
    ``panel`` as a ``date, stock_id, leg, w`` frame, **repositioned quarterly** -- a
    thin call to ``quarter_position.quarter_held_legs``, the same quarterly-held
    membership :func:`bucket_returns` trades, so the cost and ownership diagnostics
    size exactly the names the book holds."""
    return QP.quarter_held_legs(panel, factor, n_quintiles)


def leg_ownership(legs: pd.DataFrame) -> pd.Series:
    """
    Monthly largest single-name ownership share of a weighted long/short book.

    ``legs`` is a tidy ``date, stock_id, leg, w`` frame -- the same shape the cost
    model consumes -- where ``w`` is a name's weight within its leg that month.
    Each leg is funded with :data:`LEG_CAPITAL`, so a name's dollar position is
    ``LEG_CAPITAL * w`` and its ownership share is that over the name's formation
    market cap (:func:`market_cap_panel`).  Returns, per month, the max such share
    across the book -- the most concentrated single position we would hold.
    """
    df = legs.merge(market_cap_panel(), on=["date", "stock_id"], how="left")
    share = LEG_CAPITAL * df["w"].abs() / df["mcap"]
    return share.groupby(df["date"]).max().sort_index()


def window_max(series: pd.Series, start: pd.Timestamp | None = None,
               end: pd.Timestamp | None = None) -> float:
    """Worst-case (max) value of a per-month series over an optional ``[start, end]``
    window -- the ownership counterpart of :func:`window_cost`'s mean."""
    s = series.dropna()
    if start is not None:
        s = s[s.index >= start]
    if end is not None:
        s = s[s.index <= end]
    return float(s.max()) if len(s) else float("nan")


def attach_ownership(stats: dict, ownership: pd.Series,
                     start: pd.Timestamp | None = None,
                     end: pd.Timestamp | None = None) -> dict:
    """Add the largest single-name ownership over the window (``max_ownership``,
    :func:`window_max`) to a :func:`book_stats` dict, in place, so
    :func:`render_performance` can show it via :func:`ownership_metric`."""
    stats["max_ownership"] = window_max(ownership, start, end)
    return stats


def ownership_metric() -> tuple[str, str, str, bool]:
    """The :func:`render_performance` ``extra_metrics`` row for the largest
    single-name ownership share (populated by :func:`attach_ownership`)."""
    return (f"Largest single-name ownership (${PORTFOLIO_CAPITAL / 1e6:.0f}M total)",
            "max_ownership", "pct", False)


def book_stats(spread: pd.Series, industry: pd.Series,
               start: pd.Timestamp | None = None,
               end: pd.Timestamp | None = None) -> dict:
    """
    Performance of a long/short spread over an optional ``[start, end]`` window:
    mean, t-stat, annualised Sharpe, the industry-neutral alpha (and its t-stat),
    the industry beta and the beta-neutralised Sharpe.

    Window-parameterised so the same helper serves the full / past-decade split
    here and the walk-forward out-of-sample window in ``weighted_composite.py``.
    All quantities use Experiment 1's own helpers, so "alpha" is defined identically
    to every other long/short book in the project.  ``sharpe_neutral`` is the
    walk-forward beta-neutral Sharpe (:func:`regression.beta_neutral_sharpe`): the
    *full* spread/industry are handed to it with the window bounds, so each month's
    hedge beta is fit on an expanding, look-ahead-free window of all prior data even
    when the window starts mid-sample.
    """
    s, mkt = spread, industry
    if start is not None:
        s, mkt = s[s.index >= start], mkt[mkt.index >= start]
    if end is not None:
        s, mkt = s[s.index <= end], mkt[mkt.index <= end]
    stats = Q.long_short_stats(s)
    mreg = R.market_regression(s, mkt)
    return {**stats,
            "alpha": mreg["alpha"], "alpha_tstat": mreg["alpha_tstat"],
            "ind_beta": mreg["beta"], "ind_beta_tstat": mreg["beta_tstat"],
            "sharpe_neutral": R.beta_neutral_sharpe(spread, industry, start=start, end=end)}


def attach_net_cost_sharpe(stats: dict, spread: pd.Series, cost_series: pd.Series,
                           industry: pd.Series,
                           start: pd.Timestamp | None = None,
                           end: pd.Timestamp | None = None) -> dict:
    """
    Add the **cost-incorporated Sharpe** to a :func:`book_stats` window dict, in
    place, so :func:`render_performance` can show it beside the gross Sharpe.

    Sets two keys: ``sharpe_cost`` (Experiment 1's :func:`regression.net_of_cost_sharpe`
    -- the annualised Sharpe of the book's gross spread less its per-month turnover
    cost) and ``sharpe_cost_neutral`` (:func:`regression.net_of_cost_neutral_sharpe`
    -- the same net series hedged with the walk-forward ``-beta*industry`` overlay
    behind ``sharpe_neutral``, betas re-estimated on an expanding, look-ahead-free
    window).  ``start`` / ``end`` window it exactly as ``book_stats``.  The cost
    model / alignment is Experiment 1's, so this cost-incorporated Sharpe is defined
    identically to every other book's.
    """
    stats["sharpe_cost"] = R.net_of_cost_sharpe(spread, cost_series, start=start, end=end)
    stats["sharpe_cost_neutral"] = R.net_of_cost_neutral_sharpe(
        spread, cost_series, industry, start=start, end=end)
    return stats


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _set_label(factor_names: list[str]) -> str:
    return " + ".join(factor_names)


def plot_cumulative(wide: pd.DataFrame, factor_names: list[str], path: Path,
                    score_desc: str = "sum of z-scores",
                    boundary: pd.Timestamp | None = None,
                    n_buckets: int = N_QUINTILES) -> None:
    """Cumulative growth of $1 in each composite bucket (log scale).

    ``n_buckets`` sets the even sort (5 = quintiles, 3 = tertiles); ``score_desc``
    labels how the composite was built; ``boundary``, if given, draws a vertical
    rule (e.g. an in-sample / out-of-sample split).
    """
    cols = [f"Q{i}" for i in range(1, n_buckets + 1)]
    word = sort_word(n_buckets)
    cum = (1.0 + wide[cols].fillna(0.0)).cumprod()
    fig, ax = plt.subplots(figsize=(11, 5))
    for q in cols:
        ax.plot(cum.index, cum[q], label=q, linewidth=1.3)
    if boundary is not None:
        ax.axvline(boundary, color="black", linestyle="--", linewidth=0.9, alpha=0.7)
    ax.set_yscale("log")
    ax.set_title(f"Cumulative growth of $1 by composite-score {word}\n"
                 f"composite = {score_desc}: {_set_label(factor_names)}")
    ax.set_xlabel("Month")
    ax.set_ylabel("Cumulative value (log scale)")
    ax.legend(title=word.capitalize(), ncol=len(cols), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# Metric row = (label, stats-key, format, shade).  ``format`` is int/pct/num;
# ``shade`` True tints the cell by the |t| significance of its own value (used for
# t-stat rows).  ``book_stats`` keys are shared by every variant so this table
# renders any window set; callers may append further rows via ``extra_metrics``.
PERF_METRICS: list[tuple[str, str, str, bool]] = [
    ("Months (n)",                   "n_months",       "int", False),
    ("Mean monthly return",          "mean_monthly",   "pct", False),
    ("t-stat (mean ≠ 0)",            "tstat",          "num", False),
    ("Sharpe (annualised)",          "sharpe",         "num", False),
    ("Industry-neutral α (monthly)", "alpha",          "pct", False),
    ("α t-stat",                     "alpha_tstat",    "num", True),
    ("β-neutral Sharpe",             "sharpe_neutral", "num", False),
]


def _fmt_cell(value, kind: str) -> str:
    if kind == "int":
        return f"{int(value)}"
    return R._fmt_pct(value) if kind == "pct" else R._fmt_num(value)


def render_performance(windows: list[tuple[str, dict]], title: str,
                       subtitle: str, path: Path,
                       extra_metrics: list[tuple[str, str, str, bool]] | None = None,
                       keep_keys: list[str] | None = None,
                       bold_keys: list[str] | None = None
                       ) -> None:
    """
    Render long/short performance across one or more named windows as a PNG.

    ``windows`` is a list of ``(column_label, stats)`` where ``stats`` is a
    :func:`book_stats` result -- e.g. full vs past-decade, or in-sample vs
    out-of-sample.  Rows flagged ``shade`` (the α t-stat, and any shaded
    ``extra_metrics``) are tinted by significance per column.

    ``extra_metrics`` appends caller-specific rows (same
    ``(label, key, format, shade)`` shape as :data:`PERF_METRICS`) -- e.g. a
    strategy's alpha above a benchmark book -- but only those whose key every
    window carries, so a partially-populated stat is silently skipped rather than
    raising.  When every window also carries an ``avg_cost`` key an extra trailing
    row reports it, for information only -- never netted from the gross alpha /
    Sharpe above.  When every window carries the cost-incorporated Sharpe keys
    (``sharpe_cost`` / ``sharpe_cost_neutral``, e.g. via
    :func:`attach_net_cost_sharpe`) a combined "Sharpe net of cost" row is inserted
    beside the gross Sharpe -- the raw and beta-neutral net-of-cost Sharpe in one
    cell (``raw / β-neut``).

    ``keep_keys`` optionally restricts the table to a subset of metric keys (the
    second element of each row tuple), rendered **in the given key order** rather
    than the default assembly order; any listed key not present is silently
    skipped.  ``None`` (the default) keeps every assembled row, so existing callers
    are unaffected.

    ``bold_keys`` optionally renders the whole row (metric label + every window
    cell) of the listed metric keys in **bold**, to draw the eye to a headline
    figure -- e.g. the net-of-cost Sharpe.  ``None`` (the default) bolds nothing, so
    existing callers are unaffected.  The combined net-of-cost Sharpe row is keyed
    ``sharpe_cost`` (matching the row inserted above).
    """
    metrics = list(PERF_METRICS)
    # Cost-incorporated Sharpe sits next to the gross β-neutral Sharpe when every
    # window carries both net-of-cost variants (raw and β-neutral).
    if windows and all("sharpe_cost" in s and "sharpe_cost_neutral" in s
                       for _, s in windows):
        net_row = ("Sharpe net of cost (raw / β-neut)", "sharpe_cost", "net_sharpe", False)
        anchor = next((i for i, m in enumerate(metrics) if m[1] == "sharpe_neutral"),
                      next((i for i, m in enumerate(metrics) if m[1] == "sharpe"),
                           len(metrics) - 1))
        metrics.insert(anchor + 1, net_row)
    if extra_metrics:
        metrics += [m for m in extra_metrics if all(m[1] in s for _, s in windows)]
    if windows and all("avg_cost" in s for _, s in windows):
        metrics.append(("Avg monthly cost (turnover)", "avg_cost", "pct", False))

    # Optional row subset, rendered in the caller's requested key order.
    if keep_keys is not None:
        by_key = {m[1]: m for m in metrics}
        metrics = [by_key[k] for k in keep_keys if k in by_key]

    headers = ["Metric"] + [label for label, _ in windows]
    cell_text, cell_colors = [], []
    for name, key, kind, shade in metrics:
        if kind == "net_sharpe":       # combined raw / β-neutral net-of-cost Sharpe
            cell_text.append([name] + [
                f"{R._fmt_num(s['sharpe_cost'])} / {R._fmt_num(s['sharpe_cost_neutral'])}"
                for _, s in windows])
        else:
            cell_text.append([name] + [_fmt_cell(s[key], kind) for _, s in windows])
        colors = ["white"] * (len(windows) + 1)
        if shade:
            for j, (_, s) in enumerate(windows, start=1):
                colors[j] = R._tstat_color(s[key])
        cell_colors.append(colors)

    nrows, ncols = len(metrics), len(windows) + 1
    metric_w = 0.40
    col_w = [metric_w] + [(1 - metric_w) / len(windows)] * len(windows)
    fig, ax = plt.subplots(figsize=(4.2 + 2.8 * len(windows), 0.5 * (nrows + 1) + 1.6))
    ax.axis("off")
    tbl = ax.table(cellText=cell_text, colLabels=headers, cellColours=cell_colors,
                   colWidths=col_w, cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for j in range(ncols):
        tbl[0, j].set_text_props(weight="bold", color="white")
        tbl[0, j].set_facecolor("#404040")
    for i in range(1, nrows + 1):
        tbl[i, 0].set_text_props(ha="left")

    # Bold every cell of the requested metric rows (label + each window value).
    if bold_keys:
        bold_set = set(bold_keys)
        for i, (_name, key, _kind, _shade) in enumerate(metrics, start=1):
            if key in bold_set:
                for j in range(ncols):
                    tbl[i, j].set_text_props(fontweight="bold")

    fig.suptitle(title, fontsize=11, y=0.99)
    # Wrap the "   |   "-separated clauses onto their own lines so the caption
    # never overflows the figure width regardless of the number of windows.
    fig.text(0.5, 0.03, subtitle.replace("   |   ", "\n"),
             ha="center", va="bottom", fontsize=8, color="#555555")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.80, bottom=0.16)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Factor selections -- the three quarterly rankings the composite is run over
#
# The composite is always built from each constituent's *monthly* cross-sectional
# z-scores (:func:`load_exposures`); a ranking only decides *which* five factors go
# in.  All three selections come from Experiment 2's quarterly-repositioned hand-off
# (:func:`ranked_factors`): the ``quarter_quintile`` ranking (each factor's quarterly
# QUINTILE book), the ``quarter_tertile`` ranking (its quarterly TERTILE book) and the
# ``quarter_half`` ranking (its quarterly HALF book).  Running the composite over each
# answers "which five factors does the quintile-vs-tertile-vs-half book pick, and does
# the composite still lead under any selection?".
# --------------------------------------------------------------------------- #
# ranking name -> (hand-off bucketing label, zero-arg provider of its top-``TOP_N``
# factor names in rank order).
RANKINGS: dict = {
    "quarter_quintile": lambda: ranked_factors(TOP_N, "quintile")["factor"].tolist(),
    "quarter_tertile": lambda: ranked_factors(TOP_N, "tertile")["factor"].tolist(),
    "quarter_half": lambda: ranked_factors(TOP_N, "half")["factor"].tolist(),
}


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run(ranking: str = "quarter_quintile",
        factor_names: list[str] | None = None,
        n_buckets: int = N_QUINTILES,
        out_root: Path = OUTPUT_DIR) -> dict:
    """
    Run the straight-sum composite pipeline for one ranking's top factors and write
    its two outputs -- ``<sort>_cumulative.png`` and ``<sort>_performance.png``
    (``<sort>`` = ``quintile`` / ``tertile`` / ``half``) -- under
    ``out_root / "composite" / ranking``.  The book is repositioned quarterly.
    Returns the per-window performance dict.

    ``ranking`` names the factor selection (a key of :data:`RANKINGS`, default the
    headline ``quarter_quintile`` hand-off); ``factor_names`` overrides the list
    (default: that ranking's top ``TOP_N``).  ``n_buckets`` sets the composite sort
    (5 = quintile of the z-score sum, 3 = tertile, 2 = half); each ranking is run with
    the one sort it was ranked by (see :func:`_sort_for`).
    """
    if factor_names is None:
        factor_names = RANKINGS[ranking]()
    resolved = resolve_factors(factor_names)
    word, spread_col = sort_word(n_buckets), f"Q{n_buckets}-Q1"
    out_dir = out_root / "composite" / ranking
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Experiment 3: composite z-score {word} L/S [{ranking}] ===")
    print(f"Factors ({len(factor_names)}): " +
          ", ".join(f"{r.factor} [{'+' if r.sign > 0 else '-'}]"
                    for r in resolved.itertuples()))

    # 1-2. Assemble the (sign-oriented, equal-weighted) composite and shape it.
    composite, _ = load_composite(resolved)
    panel = as_factor_panel(composite)

    # 3. Even n-bucket sort of the composite (Experiment 1's slice, unmodified).
    wide = bucket_returns(panel, n_buckets)
    spread = wide[spread_col]

    # 4. Long/short performance over the full sample and the past decade.
    industry = industry_return()
    windows = [("Full sample", book_stats(spread, industry)),
               ("Past decade (2016+)", book_stats(spread, industry, start=DECADE_START))]
    full, decade = windows[0][1], windows[1][1]

    # Average monthly turnover cost of the top-minus-bottom book (reported, not
    # netted) plus the cost-incorporated Sharpe (raw + β-neutral) from the same series.
    legs = quintile_legs(panel, n_quintiles=n_buckets)
    cost_series = COST.turnover_cost(legs, cost_panel())
    full["avg_cost"] = window_cost(cost_series)
    decade["avg_cost"] = window_cost(cost_series, start=DECADE_START)
    attach_net_cost_sharpe(full, spread, cost_series, industry)
    attach_net_cost_sharpe(decade, spread, cost_series, industry, start=DECADE_START)

    # Largest single-name ownership for a $100M dollar-neutral book (worst-case per
    # window), sized from the same top/bottom leg membership the cost uses.
    ownership = leg_ownership(legs)
    attach_ownership(full, ownership)
    attach_ownership(decade, ownership, start=DECADE_START)
    meta = {"n_stocks": composite["stock_id"].nunique(),
            "n_months": int(full["n_months"]),
            "start": composite["date"].min(), "end": composite["date"].max()}

    # --- Persist outputs --------------------------------------------------- #
    plot_cumulative(wide, factor_names, out_dir / f"{word}_cumulative.png",
                    n_buckets=n_buckets)
    render_performance(
        windows,
        f"Composite {word} long-short ({spread_col}) performance -- {ranking} top-{TOP_N}",
        f"{_set_label(factor_names)}   |   {meta['n_stocks']} stocks over "
        f"{meta['n_months']} months ({meta['start']:%Y-%m} .. {meta['end']:%Y-%m})   |   "
        "α from regressing the book on the market-cap-weighted industry return.   "
        "Shading: |t| ≥ 1.65 (10%), 2.0 (5%).",
        out_dir / f"{word}_performance.png",
        extra_metrics=[ownership_metric()])

    # --- Console summary --------------------------------------------------- #
    cols = [f"Q{i}" for i in range(1, n_buckets + 1)]
    qmeans = {q: wide[q].mean() for q in cols}
    print(f"  {word} mean next-month return: " +
          "  ".join(f"{q}={qmeans[q]:+.3%}" for q in cols))
    print(f"  {spread_col} long/short: {full['mean_monthly']:+.4%}/mo "
          f"(t={full['tstat']:+.2f}, Sharpe={full['sharpe']:+.2f})")
    print(f"  industry-neutral alpha: {full['alpha']:+.4%}/mo "
          f"(t={full['alpha_tstat']:+.2f}, ind beta={full['ind_beta']:+.2f})")
    print(f"  2016+: alpha={decade['alpha']:+.4%}/mo (t={decade['alpha_tstat']:+.2f})")
    print(f"Saved -> {out_dir}")
    return {lbl: s for lbl, s in windows}

def _sort_for(ranking: str) -> int:
    if "tertile" in ranking:
        return N_TERTILES
    if "half" in ranking:
        return N_HALVES
    return N_QUINTILES


def run_all(out_root: Path = OUTPUT_DIR) -> dict:
    """Run the composite pipeline over all rankings in :data:`RANKINGS` -- the
    ``quarter_quintile``, ``quarter_tertile`` and ``quarter_half``
    quarterly-repositioned hand-offs.

    Each ranking's top factors are sorted the one way they were ranked
    (:func:`_sort_for`: quintile / tertile / half).  Returns ``{ranking: per-window
    performance dict}``."""
    return {ranking: run(ranking, RANKINGS[ranking](), _sort_for(ranking), out_root)
            for ranking in RANKINGS}


def main() -> None:
    run_all()


if __name__ == "__main__":
    main()
