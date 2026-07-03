"""
capacity_scaling.py
===================

Re-evaluate **exactly the same set of factors** as
``experiment2 - sw factors/tertile.py`` -- Experiment 1's general market factors
plus every Experiment 2 software subexperiment (Standard, RD, Stability, Skew)
-- but change **how the two legs are weighted**.

Where the standard quintile book (and ``tertile.py``) equal-weights every name
inside the top and bottom bucket, here each name is weighted by a monotone
function of its USD market cap, normalised to sum to one within its leg.  Two
capacity tilts are tested (see :data:`WEIGHTINGS`):

    sqrt:  w_{i,t}  =  sqrt(mcap_usd_{i,t})       / Σ_j sqrt(mcap_usd_{j,t})
    log6:  w_{i,t}  =  log(mcap_usd_{i,t})**6     / Σ_j log(mcap_usd_{j,t})**6

Both are *capacity* / liquidity tilts.  Equal weighting puts as much money into
a $150M microcap as into a $150B megacap, which is uninvestable at size; pure
cap weighting swings to the opposite extreme and lets a few megacaps dominate.
The square root sits between the two -- a mild tilt toward the larger, more
liquid names in each leg (so more of the book can actually be traded) while
still diversifying across the bucket.  ``log(mcap)**6`` is a far steeper tilt:
because ``log(mcap)`` grows slowly, its 6th power still increases monotonically
with size but concentrates the book much more aggressively on the largest names.
Only the **within-leg weighting** changes; the buckets, orientation, benchmark
and every downstream statistic are identical to the quintile pipeline, so each
alpha table reads directly against the standard (equal-weighted) quintile book.

The output is one alpha table per weighting scheme, in the *same* format as
``experiment2 - sw factors/factor_ranking/monthly_tertile.png``.

Bivariate extension (Experiment 3's double sort)
------------------------------------------------
The single-factor books above sqrt-cap-weight the top / bottom **quintile** legs.
:func:`run_bivariate` applies the *same* sqrt(market-cap) within-leg tilt to
**Experiment 3's bivariate tertile double sort** (``bivariate_tertile.py``), which
longs the T3xT3 corner (top tertile of *both* factors) and shorts the T1xT1
corner.  Only the within-corner weighting changes from equal to sqrt-cap; the
double sort, orientation, benchmark books, industry-neutral alpha and every
downstream statistic are the Experiment 3 driver's own (reused by path), so the
rendered table reads directly against that experiment's equal-weighted
``bivariate_tertile/<slug>/performance.png``.  It writes a single performance
table per factor pair, grouped under ``output/bivariate_tertile/``.

Bucketing
---------
The user's baseline is the **quintile** book ("weighting each stock in top and
bottom quintile equally"), so we sort into quintiles (:data:`N_QUINTILES` = 5),
long the top quintile / short the bottom quintile -- the sqrt-cap-weighted analog
of Experiment 1's equal-weighted quintile book.  Each factor's long/short
*orientation* is a property of the factor, not of the weighting, and was already
recorded in each source's ``quintile/long_short_market_alpha.csv`` (``direction``
= ``Q5-Q1`` for long-top/short-bottom, ``Q1-Q5`` for the reverse).  We reuse that
verbatim and keep the ``Q5-Q1`` / ``Q1-Q5`` labels.

Reuse (the project's standard conventions)
------------------------------------------
* The **factor universe is not redefined here** -- we import ``tertile.py`` by
  path and reuse its ``SOURCES`` list and ``_load_panel`` helper, so "the same
  set of factors as tertile.py" stays literally true even if that list changes.
* Experiment 1's engine is imported by path (``factors`` / ``cost`` /
  ``regression`` off ``sys.path`` -- the *generic* engine, driven purely by the
  panel columns; we do **not** override ``sys.modules['factors']``, exactly like
  ``tertile.py``).
* ``factors.prepare_slice`` gives the even quintile sort; the ``weight`` column
  the panel already carries (formation-date USD market cap) supplies the sqrt-cap
  weights, so no market-cap plumbing is duplicated.
* ``regression.industry_monthly_return`` / ``market_regression`` /
  ``long_short_stats`` / ``beta_neutral_sharpe`` compute the industry-neutral
  alpha, its t-stat and the (beta-neutral) Sharpe, full sample and 2016+, exactly
  as the quintile pipeline does.
* ``cost.turnover_cost`` charges the turnover of the **actual sqrt-cap-weighted**
  legs (not the equal-weighted ``long_short_cost``), so the reported cost matches
  the book being measured.
* ``regression.render_alpha_table`` renders the PNG (its ``title`` argument gives
  the capacity-weighted caption).

Nothing here mutates the shared modules or the Experiment 2 folder.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Load Experiment 1's engine (generic, by path) + reuse Experiment 2's tertile
# driver for the factor universe.  Putting the Experiment 1 directory on
# sys.path lets ``factors`` / ``cost`` / ``regression`` -- and their own
# ``import factors as F`` -- resolve to the shared engine.
# --------------------------------------------------------------------------- #
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
_EXP1_DIR = _PROJECT_ROOT / "experiment1 - general factors"
_EXP2_DIR = _PROJECT_ROOT / "experiment2 - sw factors"
sys.path.insert(0, str(_EXP1_DIR))

import factors as F        # noqa: E402  (import after sys.path wiring)
import cost                # noqa: E402
import regression          # noqa: E402


def _load_by_path(name: str, path: Path):
    """Import a module from an explicit file path (the folder name has spaces, so
    the usual ``import`` won't find it).  Used to reuse ``tertile.py`` without
    copying its source list."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)                       # type: ignore[union-attr]
    return mod


# Reuse the tertile driver's factor universe + panel loader -- single source of
# truth for "the same set of factors as tertile.py".  Importing it is side-effect
# free (its ``run()`` is guarded by ``__main__``).
_tertile = _load_by_path("exp2_tertile", _EXP2_DIR / "tertile.py")
SOURCES = _tertile.SOURCES
_load_panel = _tertile._load_panel

# Reuse Experiment 3's bivariate double-sort driver by path for the capacity-
# scaled bivariate books (:func:`run_bivariate`).  It does ``import composite``
# for its engine handle, so put Experiment 3 on sys.path first; loading it is
# side-effect free (its own ``run()`` is guarded by ``__main__``).
_EXP3_DIR = _PROJECT_ROOT / "experiment3 - multifactor"
sys.path.insert(0, str(_EXP3_DIR))
_bivariate = _load_by_path("exp3_bivariate_tertile", _EXP3_DIR / "bivariate_tertile.py")

N_QUINTILES = 5
DECADE_START = regression.DECADE_START     # 2016+ ("past decade") window, shared


# --------------------------------------------------------------------------- #
# Within-leg weighting schemes.  Each is the whole difference from the
# equal-weighted quintile book: it maps a name's USD market cap to an
# unnormalised within-leg weight (the leg is then renormalised to sum to one).
# Swap in ``pd.Series(1.0, ...)`` for equal weights, or drop the transform for
# pure cap weighting.  ``run`` iterates over :data:`WEIGHTINGS`, rendering one
# alpha table per scheme.
# --------------------------------------------------------------------------- #
def _sqrt_cap_weight(mcap_usd: pd.Series) -> pd.Series:
    """Unnormalised within-leg weight = sqrt(USD market cap) -- a mild tilt toward
    the larger, more liquid names in each leg (a capacity / liquidity tilt that
    sits between equal and pure cap weighting)."""
    return np.sqrt(mcap_usd)


def _log6_cap_weight(mcap_usd: pd.Series) -> pd.Series:
    """Unnormalised within-leg weight = log(USD market cap)**6 -- a far steeper
    tilt than sqrt: because log(mcap) grows slowly, its 6th power still increases
    monotonically with size but concentrates the book much more aggressively on
    the largest (most tradable) names in each leg."""
    return np.log(mcap_usd) ** 6


WEIGHTINGS = [
    {
        "label": "sqrt",
        "raw_weight": _sqrt_cap_weight,
        "out_png": _THIS_DIR / "output" / "capacity_scaling" / "long_short_market_alpha.png",
        "title": (
            "Long-short QUINTILE strategy with sqrt(market-cap)-weighted legs, "
            "regressed on the industry return\n(long top quintile / short bottom "
            "quintile; within each leg w_i ∝ √(USD market cap);  "
            "ls_t = α + β·industry_t + ε,  α = industry-neutral monthly return, "
            "t-stat tests α ≠ 0)"),
    },
    {
        "label": "log6",
        "raw_weight": _log6_cap_weight,
        "out_png": _THIS_DIR / "output" / "capacity_scaling" / "log6_long_short_market_alpha.png",
        "title": (
            "Long-short QUINTILE strategy with log(market-cap)**6-weighted legs, "
            "regressed on the industry return\n(long top quintile / short bottom "
            "quintile; within each leg w_i ∝ log(USD market cap)⁶;  "
            "ls_t = α + β·industry_t + ε,  α = industry-neutral monthly return, "
            "t-stat tests α ≠ 0)"),
    },
]


# --------------------------------------------------------------------------- #
# Capacity-scaled (market-cap-weighted) long/short book
# --------------------------------------------------------------------------- #
def scaled_legs(panel: pd.DataFrame, factor: str, raw_weight, n: int = N_QUINTILES) -> pd.DataFrame:
    """
    Tidy ``date, stock_id, leg, w, next_return`` for the top and bottom buckets of
    ``factor``, with each name's within-leg weight ``w`` proportional to
    ``raw_weight`` (a scheme from :data:`WEIGHTINGS`, mapping USD market cap to an
    unnormalised weight) and normalised to sum to one inside its leg each month.

    The even quintile sort and the within-month return winsorisation come from the
    shared :func:`factors.prepare_slice`; the market cap is the ``weight`` column
    the panel already carries (formation-date USD cap).  Names whose weight is not
    finite and positive cannot be sized and drop out (the panel is market-cap
    screened, so this is rare) -- the surviving names' weights are renormalised to
    sum to one.
    """
    sub = F.prepare_slice(panel, factor, n)
    caps = (panel.loc[panel["factor"] == factor, ["date", "stock_id", "weight"]]
                 .drop_duplicates(["date", "stock_id"]))
    legs = (sub.loc[sub["quintile"].isin([1.0, float(n)])]
               .merge(caps, on=["date", "stock_id"], how="left"))
    legs["leg"] = np.where(legs["quintile"] == float(n), "top", "bottom")

    legs["raw_w"] = raw_weight(legs["weight"])
    legs = legs.loc[np.isfinite(legs["raw_w"]) & (legs["raw_w"] > 0)].copy()
    denom = legs.groupby(["date", "leg"], observed=True)["raw_w"].transform("sum")
    legs["w"] = legs["raw_w"] / denom
    return legs[["date", "stock_id", "leg", "w", "next_return"]]


def scaled_spread(legs: pd.DataFrame, sign: int) -> pd.Series:
    """
    Monthly dollar-neutral long/short return of the capacity-scaled book: the
    sqrt-cap-weighted mean next-period return of the top leg minus that of the
    bottom leg, oriented by ``sign`` (+1 = long top / short bottom, -1 = reverse).

    Because ``w`` already sums to one within each ``(date, leg)`` (see
    :func:`scaled_legs`), ``Σ_i w_i · next_return_i`` per leg *is* the weighted
    mean.  Returns a month-indexed Series (empty if a leg is absent).
    """
    leg_ret = ((legs["w"] * legs["next_return"])
               .groupby([legs["date"], legs["leg"]], observed=True).sum()
               .unstack("leg").sort_index())
    top, bottom = leg_ret.get("top"), leg_ret.get("bottom")
    if top is None or bottom is None:
        return pd.Series(dtype=float, name="ls")
    spread = (top - bottom) if sign > 0 else (bottom - top)
    return spread.rename("ls")


def evaluate_factor(panel: pd.DataFrame, industry_ret: pd.Series,
                    cost_panel: pd.DataFrame, factor: str, family: str,
                    direction: str, raw_weight) -> dict:
    """One :func:`regression.render_alpha_table` row for ``factor``: build its
    cap-weighted quintile book (via ``raw_weight``) with the recorded orientation
    and reuse Experiment 1's regression / cost helpers to measure the
    industry-neutral alpha, Sharpe, beta-neutral Sharpe and turnover cost, full
    sample and 2016+.
    """
    sign = 1 if direction == "Q5-Q1" else -1
    legs = scaled_legs(panel, factor, raw_weight)
    spread = scaled_spread(legs, sign)
    recent = spread.index >= DECADE_START
    ind_recent = industry_ret[industry_ret.index >= DECADE_START]

    ls = regression.long_short_stats(spread)
    ls_2016 = regression.long_short_stats(spread[recent])
    mreg = regression.market_regression(spread, industry_ret)
    mreg_2016 = regression.market_regression(spread[recent], ind_recent)
    # Beta-neutral Sharpe: hedge with a walk-forward -beta*industry overlay whose
    # beta is re-estimated on an expanding, look-ahead-free window (2016+ windows the
    # same hedged series, so its betas still use all prior history).
    sr_neutral = regression.beta_neutral_sharpe(spread, industry_ret)
    sr_neutral_2016 = regression.beta_neutral_sharpe(spread, industry_ret, start=DECADE_START)

    # Turnover cost of the ACTUAL sqrt-cap-weighted legs (not equal-weighted), so
    # the reported cost matches the book measured above.  Its mean (pp/month) is
    # reported and it nets the gross spread for the cost-incorporated Sharpe
    # (raw + beta-neutral, full & 2016+).
    dates = pd.Index(sorted(legs["date"].unique()), name="date")
    cost_series = cost.turnover_cost(legs[["date", "stock_id", "leg", "w"]], cost_panel, dates)
    avg_cost_pp = cost.average_cost(cost_series) * 100.0
    sharpe_cost = regression.net_of_cost_sharpe(spread, cost_series)
    sharpe_cost_neutral = regression.net_of_cost_neutral_sharpe(
        spread, cost_series, industry_ret)
    sharpe_cost_2016 = regression.net_of_cost_sharpe(spread, cost_series, start=DECADE_START)
    sharpe_cost_neutral_2016 = regression.net_of_cost_neutral_sharpe(
        spread, cost_series, industry_ret, start=DECADE_START)

    return {
        "factor": factor, "family": family,
        "direction": "Q5-Q1" if sign > 0 else "Q1-Q5",
        "alpha": mreg["alpha"], "alpha_tstat": mreg["alpha_tstat"],
        "sharpe": ls["sharpe"], "sharpe_neutral": sr_neutral,
        "sharpe_cost": sharpe_cost, "sharpe_cost_neutral": sharpe_cost_neutral,
        "avg_cost_pp": avg_cost_pp, "n": mreg["n"],
        "alpha_2016": mreg_2016["alpha"], "alpha_tstat_2016": mreg_2016["alpha_tstat"],
        "sharpe_2016": ls_2016["sharpe"], "sharpe_neutral_2016": sr_neutral_2016,
        "sharpe_cost_2016": sharpe_cost_2016,
        "sharpe_cost_neutral_2016": sharpe_cost_neutral_2016,
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run(raw_weight, title: str, out_png: Path, label: str) -> pd.DataFrame:
    """Re-evaluate every factor across all sources with cap-weighted quintile books
    (within-leg weighting given by ``raw_weight``), rank by the sum of the
    full-period and 2016+ alpha t-stats (as ``main.py`` ranks the top factors), and
    render the whole table to ``out_png``.  Returns the ranked table."""
    # One cost panel for the whole run: every software library shares the same
    # Software & Services universe, so the per-(stock, month) costs are identical.
    cost_panel = cost.build_cost_panel(F.SOFTWARE_SERVICES)

    rows: list[dict] = []
    for src in SOURCES:
        if not src["alpha_csv"].exists() or not src["panel"].exists():
            print(f"  (skip {src['label']}: panel or alpha csv missing -- "
                  f"run its driver first)")
            continue
        directions = pd.read_csv(src["alpha_csv"])[["factor", "family", "direction"]]
        panel = _load_panel(src["panel"])
        industry_ret = regression.industry_monthly_return(panel)
        print(f"--- {src['label']}: {len(directions)} factors ---")
        for r in directions.itertuples(index=False):
            row = evaluate_factor(panel, industry_ret, cost_panel,
                                  r.factor, r.family, r.direction, raw_weight)
            row["subexperiment"] = src["label"]
            rows.append(row)
            print(f"  {row['factor']:<26} [{src['label']:<10}] {row['direction']}  "
                  f"alpha={row['alpha']:+.4%}/mo  t={row['alpha_tstat']:+.2f}  "
                  f"(2016+ t={row['alpha_tstat_2016']:+.2f})")

    if not rows:
        raise FileNotFoundError(
            "No source alpha tables found; run Experiment 1 and the Experiment 2 "
            "subexperiments first (python main.py).")

    table = pd.DataFrame(rows)
    table["alpha_tstat_combined"] = table["alpha_tstat"] + table["alpha_tstat_2016"]
    table = (table.sort_values("alpha_tstat_combined", ascending=False, kind="stable")
                  .reset_index(drop=True))

    # Render in the standard alpha-table format, tagging each family with its
    # source subexperiment for provenance -- exactly like the tertile / factor_ranking PNG.
    plot_rows = table.copy()
    plot_rows["family"] = plot_rows["family"] + "  [" + plot_rows["subexperiment"] + "]"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    regression.render_alpha_table(plot_rows, out_png, title=title)

    print(f"\nSaved capacity-scaled ({label}) long-short alpha table "
          f"({len(table)} factors) -> {out_png}")
    return table


# --------------------------------------------------------------------------- #
# Capacity scaling applied to Experiment 3's bivariate tertile double sort
#
# Experiment 3's ``bivariate_tertile.py`` longs the T3xT3 corner (top tertile of
# *both* factors) and shorts the T1xT1 corner, equal-weighted within each corner.
# Here we apply the SAME sqrt(market-cap) within-leg tilt used above for the
# quintile books to those two corner legs, changing nothing else -- the double
# sort, orientation, benchmark books, industry-neutral alpha and every downstream
# statistic are the bivariate driver's own (reused by path).  ``_C`` is that
# driver's Experiment 1/3 engine + analysis handle.
# --------------------------------------------------------------------------- #
_C = _bivariate.C

# The bivariate tertile pairs re-evaluated under sqrt-cap scaling -- the same two
# pairs Experiment 3 double-sorts (both against gross_profitability): the stability
# of monthly returns and the stability of revenue.  Overridden by a CLI-supplied
# two-factor pair.
BIVARIATE_PAIRS = [
    ["return_stability", "gross_profitability"],
    ["revenue_stability", "gross_profitability"],
]


def corner_legs(frame: pd.DataFrame, col_a: str, col_b: str, top, bottom,
                raw_weight) -> pd.DataFrame:
    """Tidy ``date, stock_id, leg, w, next_return, mcap`` for the long (``top``
    bucket of *both* factors) and short (``bottom`` bucket of both) corner cells of
    a double sort, with each name's within-leg weight ``w`` proportional to
    ``raw_weight(mcap)`` (a scheme from :data:`WEIGHTINGS`) and renormalised to sum
    to one inside its corner each month.

    The oriented tertile buckets (``col_a``/``col_b``) and the formation USD market
    cap (``mcap``) both come from ``bivariate_tertile.double_sorted``; only the
    within-corner weighting differs from Experiment 3's equal-weighted corner book.
    Names whose weight is not finite and positive cannot be sized and drop out."""
    long_cell = frame.loc[(frame[col_a] == top) & (frame[col_b] == top)].assign(leg="long")
    short_cell = frame.loc[(frame[col_a] == bottom) & (frame[col_b] == bottom)].assign(leg="short")
    legs = pd.concat([long_cell, short_cell], ignore_index=True)
    legs["raw_w"] = raw_weight(legs["mcap"])
    legs = legs.loc[np.isfinite(legs["raw_w"]) & (legs["raw_w"] > 0)].copy()
    denom = legs.groupby(["date", "leg"], observed=True)["raw_w"].transform("sum")
    legs["w"] = legs["raw_w"] / denom
    return legs[["date", "stock_id", "leg", "w", "next_return", "mcap"]]


def corner_spread(legs: pd.DataFrame) -> pd.Series:
    """Monthly dollar-neutral long/short return of the capacity-scaled corner book:
    the cap-weighted mean next-period return of the long corner minus that of the
    short corner.  ``w`` already sums to one within each ``(date, leg)`` (see
    :func:`corner_legs`), so ``Σ_i w_i · next_return_i`` per corner *is* the
    weighted mean.  Returns a month-indexed Series (empty if a corner is absent)."""
    leg_ret = ((legs["w"] * legs["next_return"])
               .groupby([legs["date"], legs["leg"]], observed=True).sum()
               .unstack("leg").sort_index())
    long_leg, short_leg = leg_ret.get("long"), leg_ret.get("short")
    if long_leg is None or short_leg is None:
        return pd.Series(dtype=float, name="long_short")
    return (long_leg - short_leg).rename("long_short").dropna()


def corner_ownership(legs: pd.DataFrame, leg_capital: float | None = None) -> pd.Series:
    """Monthly largest single-name ownership share of the capacity-scaled corner
    book: a name's dollar position is ``leg_capital * w`` (its sqrt-cap weight times
    the per-leg capital) and its ownership share is that over its formation market
    cap (``mcap``).  Returns the per-month max across both corners -- the most
    concentrated single position held -- the sqrt-cap analog of
    ``bivariate_tertile._intersection_ownership``'s equal-weighted share."""
    leg_capital = _C.LEG_CAPITAL if leg_capital is None else leg_capital
    share = (leg_capital * legs["w"]) / legs["mcap"]
    return share.groupby(legs["date"]).max().sort_index()


def run_bivariate(factor_names: list[str] | None = None,
                  raw_weight=_sqrt_cap_weight,
                  label: str | None = None,
                  out_root: Path | None = None) -> dict:
    """Apply the sqrt(market-cap) within-leg tilt to Experiment 3's bivariate
    tertile double sort for the two ``factor_names`` (default: the driver's own
    ``return_stability x gross_profitability`` pair) and render a single
    performance table -- the *same* format and metrics as
    ``experiment3 .../bivariate_tertile/<slug>/performance.png`` -- as
    ``output/bivariate_tertile/<slug>_performance.png``.  Returns the per-window
    stats dict."""
    factor_names = list(factor_names) if factor_names else list(_bivariate.DEFAULT_FACTORS)
    if len(factor_names) != 2:
        raise ValueError("bivariate capacity scaling takes exactly two factors; "
                         f"got {len(factor_names)}: {factor_names}")
    out_root = out_root or (_THIS_DIR / "output" / "capacity_scaling")
    slug = label or "__".join(factor_names)
    out_root.mkdir(parents=True, exist_ok=True)
    out_png = out_root / f"{slug}_performance.png"

    print("=== Experiment 5: capacity-scaled bivariate tertile (sqrt market cap) L/S ===")
    resolved = _C.resolve_factors(factor_names)
    print("Factors: " + ", ".join(f"{r.factor} [{'+' if r.sign > 0 else '-'}]"
                                   for r in resolved.itertuples()))

    # Oriented two-factor cross-section + tertile corners, straight from Exp 3.
    frame, _oriented = _bivariate.double_sorted(resolved)
    industry = _C.industry_return()
    # Benchmark the double sort against *its own two constituents* (the bivariate
    # legs), not Experiment 3's fixed global list -- the question is whether the
    # double sort adds alpha above simply holding either standalone factor it is
    # built from.  For return_stability x gross_profitability this measures alpha
    # above univariate return_stability, not the unrelated revenue_stability book.
    benchmark_factors = list(factor_names)
    benchmarks = {f: _C.factor_long_short(f) for f in benchmark_factors}

    # sqrt-cap-weighted corner book, its turnover cost and its ownership share.
    legs = corner_legs(frame, "tile_a", "tile_b",
                       _bivariate.TOP, _bivariate.BOTTOM, raw_weight)
    spread = corner_spread(legs)
    dates = pd.Index(sorted(frame["date"].unique()), name="date")
    cost_series = _C.COST.turnover_cost(legs[["date", "stock_id", "leg", "w"]],
                                        _C.cost_panel(), dates)
    ownership = corner_ownership(legs)

    # Reuse Experiment 3's window evaluator (book_stats + avg cost + max ownership
    # + benchmark-relative alpha) and performance renderer verbatim, so the table
    # is identical in format and metrics to the equal-weighted bivariate one.
    windows, full, decade = _bivariate.evaluate_book(
        spread, cost_series, industry, benchmarks, ownership)
    # Per-benchmark alpha rows keyed on this pair's own constituents (mirrors
    # _bivariate._benchmark_extra_metrics, but over benchmark_factors rather than
    # the global BENCHMARK_FACTORS).
    benchmark_metrics: list[tuple[str, str, str, bool]] = []
    for f in benchmark_factors:
        benchmark_metrics.append((f"α vs {f} book (monthly)", f"alpha_vs_{f}", "pct", False))
        benchmark_metrics.append((f"    α t-stat vs {f}", f"alpha_tstat_vs_{f}", "num", True))
    extra_metrics = benchmark_metrics + [
        (f"Largest single-name ownership (${_C.PORTFOLIO_CAPITAL / 1e6:.0f}M total)",
         "max_ownership", "pct", False)]

    n_stocks = frame["stock_id"].nunique()
    avg_long = legs.loc[legs["leg"] == "long"].groupby("date").size().mean()
    avg_short = legs.loc[legs["leg"] == "short"].groupby("date").size().mean()
    _C.render_performance(
        windows,
        "Bivariate tertile long-short with sqrt(market-cap)-weighted corner legs "
        "(top-of-both - bottom-of-both) performance",
        f"{_bivariate._pair_label(factor_names)}   |   {n_stocks} stocks over "
        f"{int(full['n_months'])} months ({spread.index.min():%Y-%m} .. "
        f"{spread.index.max():%Y-%m}); avg leg ~{avg_long:.0f} long / {avg_short:.0f} short   |   "
        "independent 3x3 sort; within each corner leg w_i ∝ √(USD market cap); "
        "industry-neutral alpha from the market-cap-weighted industry return   |   "
        "benchmark alphas from regressing the book on each standalone factor book   |   "
        "Shading: |t| >= 1.65 (10%), 2.0 (5%).",
        out_png,
        extra_metrics=extra_metrics)

    print(f"  long/short: {full['mean_monthly']:+.4%}/mo "
          f"(t={full['tstat']:+.2f}, Sharpe={full['sharpe']:+.2f})")
    print(f"  industry-neutral alpha: {full['alpha']:+.4%}/mo "
          f"(t={full['alpha_tstat']:+.2f}, ind beta={full['ind_beta']:+.2f})")
    for f in benchmark_factors:
        print(f"  alpha vs {f} book: {full[f'alpha_vs_{f}']:+.4%}/mo "
              f"(t={full[f'alpha_tstat_vs_{f}']:+.2f})")
    print(f"  2016+: alpha={decade['alpha']:+.4%}/mo (t={decade['alpha_tstat']:+.2f})")
    print(f"Saved capacity-scaled bivariate performance -> {out_png}")
    return {lbl: s for lbl, s in windows}


if __name__ == "__main__":
    for _w in WEIGHTINGS:
        print(f"\n===== weighting: {_w['label']} =====")
        run(_w["raw_weight"], _w["title"], _w["out_png"], _w["label"])

    # Sqrt-cap capacity scaling applied to Experiment 3's bivariate tertile double
    # sorts -- both :data:`BIVARIATE_PAIRS` (or a CLI-supplied two-factor pair),
    print("\n===== bivariate tertile: sqrt market-cap capacity scaling =====")
    _argv = sys.argv[1:]
    _pairs = [_argv] if len(_argv) == 2 else BIVARIATE_PAIRS
    for _pair in _pairs:
        run_bivariate(_pair)
