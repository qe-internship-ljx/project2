"""
vol_timing.py
=============

Experiment 4 -- **volatility-regime timing** of a single factor book.

This is the first module of Experiment 4 ("timing"): instead of proposing a new
factor, it asks whether an *existing* factor's long/short book can be improved by
only holding it in a chosen market regime.  Concretely it tests the
``revenue_stability`` factor (Experiment 2's Rev & Cost library -- the negative
trailing-36m std of YoY revenue growth, long high) under a **vol-of-vol timing
rule**:

    On each monthly rebalance day (formation month-end ``t``), enter the
    dollar-neutral Q5-Q1 ``revenue_stability`` book for month ``t+1`` *only if*
    the 10-trading-day moving average of the CBOE VVIX (vol-of-vol) index as of
    that day is above 95; otherwise sit in cash (0 return) for that month.

The economic prior: ``revenue_stability`` is a defensive, negative-industry-beta
"durability" signal (see the project memory) -- the durability premium should pay
best when vol-of-vol is elevated (stressed, risk-off regimes), and add little
churn/risk in calm regimes.  VVIX (the vol of VIX) is a forward-looking gauge of
tail-risk demand, so a high 10-day VVIX average flags exactly those regimes.  We
therefore compare the **always-on** book against the **timed** book over the
common sample for which the timing signal exists (VVIX history starts 2006-03).

Engine reuse (the project's standard dependency-injection convention)
---------------------------------------------------------------------
The factor panel, the directionally-signed Q5-Q1 book, the cap-weighted industry
("market") return, and every performance statistic (industry-neutral alpha,
beta-neutral Sharpe, annualised Sharpe) are reused **verbatim** from Experiment
1's ``regression.py`` driven by Experiment 2's ``Rev & Cost/revcost_factors.py``
factor library: we load ``revcost_factors`` by path, register it as
``sys.modules["factors"]`` so ``regression.py`` binds to it, and then build the
``revenue_stability`` book exactly as ``main_revcost.py`` would.  Only the timing
overlay and its evaluation are new here.

Run standalone::

    python vol_timing.py

Outputs land under ``experiment4 - timing/output/vol_timing/``:

    timing_comparison.csv   always-on vs timed: mean, t-stat, Sharpe, industry-
                            neutral alpha (+t) & beta, beta-neutral Sharpe, months
                            in-market -- full signal sample and 2016+.
    cumulative.png          cumulative growth of $1, always-on vs timed, with the
                            in-market (VVIX-MA>95) months shaded.
    signal_timeline.png     the 10-day VVIX moving average vs the threshold.
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
# Paths & dependency injection (mirror main_revcost.py's wiring)
# --------------------------------------------------------------------------- #
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
_EXP1_DIR = _PROJECT_ROOT / "experiment1 - general factors"
_REVCOST_DIR = _PROJECT_ROOT / "experiment2 - sw factors" / "Rev & Cost"


def _load_revcost():
    """Load Experiment 2's Rev & Cost factor library by path (its folder name
    contains spaces, so it cannot be imported normally)."""
    spec = importlib.util.spec_from_file_location(
        "revcost_factors", _REVCOST_DIR / "revcost_factors.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


R = _load_revcost()

# Make Experiment 1's analysis modules bind to the Rev & Cost library, then
# import the regression engine (whose helpers we reuse for every statistic).
sys.modules["factors"] = R
sys.path.insert(0, str(_EXP1_DIR))

import regression  # noqa: E402  (import after sys.modules / sys.path wiring)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
FACTOR = "revenue_stability"
VVIX_SECURITY = "VVIX Index"
VVIX_MA_WINDOW = 10          # trading-day moving-average window of VVIX
VVIX_THRESHOLD = 95.0        # enter the book only when the 10d VVIX MA exceeds this
DECADE_START = pd.Timestamp("2016-01-01")   # "past decade" cut-off (engine convention)
MONTHS_PER_YEAR = 12

OUTPUT_DIR = _THIS_DIR / "output" / "vol_timing"


# --------------------------------------------------------------------------- #
# Timing signal: 10-day moving average of VVIX as of each rebalance day
# --------------------------------------------------------------------------- #
def vvix_ma_series() -> pd.DataFrame:
    """Daily 10-trading-day moving average of the VVIX close.

    Returns a frame ``[date, vvix_ma]`` (sorted, no NaNs) ready for an as-of
    merge onto the monthly rebalance dates.  The moving average requires a full
    ``VVIX_MA_WINDOW`` window, so the first few daily observations are dropped.
    """
    raw = pd.read_csv(R.DATA_DIR / "VolatilityIndexData.csv")
    vv = raw[raw["SECURITY"] == VVIX_SECURITY].copy()
    vv["date"] = pd.to_datetime(vv["DATE"])
    vv["vvix"] = pd.to_numeric(vv["INDEX_VALUE"], errors="coerce")
    vv = (vv.dropna(subset=["vvix"])
            .sort_values("date")
            .drop_duplicates("date", keep="last"))
    vv["vvix_ma"] = vv["vvix"].rolling(VVIX_MA_WINDOW, min_periods=VVIX_MA_WINDOW).mean()
    return vv.dropna(subset=["vvix_ma"])[["date", "vvix_ma"]].reset_index(drop=True)


def timing_signal(rebalance_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """For each monthly rebalance date, the 10d VVIX MA *as of that day* and the
    resulting in-market flag (``vvix_ma > VVIX_THRESHOLD``).

    The MA is read with a backward as-of merge -- the last VVIX observation on or
    before the rebalance day -- so a month-end that is not itself a trading day
    still reads the most recent value (no look-ahead).  Rebalance dates earlier
    than the first available MA carry NaN and are excluded by the caller.
    """
    ma = vvix_ma_series()
    dates = (pd.DataFrame({"date": pd.to_datetime(rebalance_dates)})
               .sort_values("date").reset_index(drop=True))
    asof = pd.merge_asof(dates, ma, on="date", direction="backward")
    asof["in_market"] = asof["vvix_ma"] > VVIX_THRESHOLD
    return asof.set_index("date")


# --------------------------------------------------------------------------- #
# Book construction & evaluation (all statistics reused from regression.py)
# --------------------------------------------------------------------------- #
def revenue_stability_book(panel: pd.DataFrame) -> tuple[pd.Series, int]:
    """The directionally-signed Q5-Q1 ``revenue_stability`` long/short spread,
    indexed by formation month-end.  Direction follows the factor library's
    canonical prior (``higher_is_bullish`` -> long Q5 / short Q1)."""
    return regression.long_short_portfolio(panel, FACTOR, fm_tstat=np.nan)


def evaluate(book: pd.Series, industry_ret: pd.Series,
             start: pd.Timestamp | None = None) -> dict:
    """Performance of a monthly book over an optional ``>= start`` window, using
    the engine's own statistics: annualised Sharpe & mean/t-stat of the book, its
    industry-neutral alpha (and t-stat) and industry beta from a regression on the
    cap-weighted industry return, and the industry-beta-neutralised Sharpe."""
    b = book if start is None else book[book.index >= start]
    ind = industry_ret if start is None else industry_ret[industry_ret.index >= start]
    ls = regression.long_short_stats(b)
    mreg = regression.market_regression(b, ind)
    sr_neutral = regression.beta_neutral_sharpe(b, ind, mreg["beta"])
    return {
        "mean_monthly": ls["mean_monthly"], "tstat": ls["tstat"],
        "sharpe": ls["sharpe"], "ann_return": ls["ann_return"],
        "alpha": mreg["alpha"], "alpha_tstat": mreg["alpha_tstat"],
        "beta": mreg["beta"], "beta_tstat": mreg["beta_tstat"],
        "sharpe_neutral": sr_neutral, "n_months": ls["n_months"],
    }


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def _shade_in_market(ax, signal: pd.Series) -> None:
    """Shade the spans of consecutive in-market (VVIX-MA>95) months."""
    s = signal.sort_index()
    dates = s.index
    in_run = False
    start = None
    for i, d in enumerate(dates):
        on = bool(s.iloc[i])
        if on and not in_run:
            start, in_run = d, True
        if in_run and (not on or i == len(dates) - 1):
            end = d if not on else d
            ax.axvspan(start, end, color="C1", alpha=0.12, lw=0)
            in_run = False


def plot_cumulative(base: pd.Series, timed: pd.Series, signal: pd.Series,
                    stats_base: dict, stats_timed: dict, path: Path) -> None:
    cum_base = (1.0 + base.fillna(0.0)).cumprod()
    cum_timed = (1.0 + timed.fillna(0.0)).cumprod()
    fig, ax = plt.subplots(figsize=(11, 5))
    _shade_in_market(ax, signal)
    ax.plot(cum_base.index, cum_base, color="C0", linewidth=1.3,
            label=f"always-on  [Sharpe={stats_base['sharpe']:+.2f}, "
                  f"alpha={stats_base['alpha']:+.4%}/mo, t={stats_base['alpha_tstat']:+.2f}]")
    ax.plot(cum_timed.index, cum_timed, color="C3", linewidth=1.3,
            label=f"VVIX-timed  [Sharpe={stats_timed['sharpe']:+.2f}, "
                  f"alpha={stats_timed['alpha']:+.4%}/mo, t={stats_timed['alpha_tstat']:+.2f}]")
    ax.axhline(1.0, color="black", linewidth=0.6)
    ax.set_title("revenue_stability Q5-Q1: always-on vs VVIX-timed\n"
                 f"(enter only when {VVIX_MA_WINDOW}d VVIX MA > {VVIX_THRESHOLD:g}; "
                 "shaded = in-market months)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Cumulative value of $1")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_signal(asof: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(asof.index, asof["vvix_ma"], color="C4", linewidth=1.0,
            label=f"{VVIX_MA_WINDOW}d VVIX moving average")
    ax.axhline(VVIX_THRESHOLD, color="C3", linewidth=1.0, linestyle="--",
               label=f"threshold = {VVIX_THRESHOLD:g}")
    ax.fill_between(asof.index, VVIX_THRESHOLD, asof["vvix_ma"],
                    where=asof["vvix_ma"] > VVIX_THRESHOLD,
                    color="C1", alpha=0.25, interpolate=True)
    ax.set_title("Timing signal: 10-day moving average of CBOE VVIX at each "
                 "monthly rebalance day")
    ax.set_xlabel("Month")
    ax.set_ylabel("VVIX (10d MA)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run() -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== Experiment 4: VVIX-regime timing of '{FACTOR}' ===")
    panel = R.load_panel(u=R.SOFTWARE_SERVICES)

    # The directionally-signed Q5-Q1 book and the cap-weighted industry return,
    # both built exactly as the Rev & Cost pipeline does.
    spread, sign = revenue_stability_book(panel)
    industry_ret = regression.industry_monthly_return(panel)
    direction = "Q5-Q1" if sign > 0 else "Q1-Q5"
    print(f"  book direction: {direction} (canonical long high-stability)")
    print(f"  book months: {spread.dropna().size} "
          f"({spread.index.min():%Y-%m} .. {spread.index.max():%Y-%m})")

    # Timing signal on the book's own rebalance (formation month-end) dates, then
    # restrict to the common sample where the signal is defined (VVIX from 2006).
    asof = timing_signal(spread.index)
    sig = asof.dropna(subset=["vvix_ma"])["in_market"]
    eval_index = spread.index.intersection(sig.index).sort_values()

    base = spread.reindex(eval_index)
    in_market = sig.reindex(eval_index).astype(bool)
    timed = base.where(in_market, 0.0)   # cash (0 return) when out of market

    pct_in = float(in_market.mean())
    pct_in_2016 = float(in_market[in_market.index >= DECADE_START].mean())
    print(f"  signal sample: {eval_index.min():%Y-%m} .. {eval_index.max():%Y-%m} "
          f"({len(eval_index)} months); in-market {in_market.sum()}/{len(eval_index)} "
          f"= {pct_in:.1%} of months (VVIX {VVIX_MA_WINDOW}d MA > {VVIX_THRESHOLD:g})")

    # Evaluate both books on the identical sample (full signal window + 2016+).
    rows = []
    for label, book in (("always_on", base), ("vvix_timed", timed)):
        for win_name, start in (("full", None), ("2016+", DECADE_START)):
            s = evaluate(book, industry_ret, start=start)
            s["book"] = label
            s["window"] = win_name
            s["pct_in_market"] = pct_in if start is None else pct_in_2016
            rows.append(s)
    comparison = pd.DataFrame(rows)[[
        "book", "window", "mean_monthly", "tstat", "sharpe", "ann_return",
        "alpha", "alpha_tstat", "beta", "beta_tstat", "sharpe_neutral",
        "pct_in_market", "n_months"]]
    comparison.to_csv(OUTPUT_DIR / "timing_comparison.csv", index=False)

    # Console summary (ASCII only -- Windows cp1252 stdout).
    print("\n  book        window  mean/mo   t     Sharpe  alpha/mo   a_t   "
          "ind_beta  bn_Sharpe   n")
    for _, r in comparison.iterrows():
        print(f"  {r['book']:<11} {r['window']:<6} "
              f"{r['mean_monthly']:+.4%} {r['tstat']:+5.2f} "
              f"{r['sharpe']:+6.2f}  {r['alpha']:+.4%} {r['alpha_tstat']:+5.2f}  "
              f"{r['beta']:+7.3f}  {r['sharpe_neutral']:+7.2f}  {int(r['n_months']):>4}")

    # Plots.
    plot_cumulative(base, timed, in_market,
                    evaluate(base, industry_ret), evaluate(timed, industry_ret),
                    OUTPUT_DIR / "cumulative.png")
    plot_signal(asof.dropna(subset=["vvix_ma"]), OUTPUT_DIR / "signal_timeline.png")

    print(f"\nSaved timing outputs -> {OUTPUT_DIR}")
    return comparison


if __name__ == "__main__":
    run()
