"""
spread_timing.py
================

Experiment 4 -- **factor-spread timing** of the top software factors.

The second timing module.  Rather than holding a factor's long/short book all the
time, it asks whether the book should be entered *only when the factor's own
cross-sectional dispersion is unusually wide* -- the classic "factor spread"
predictability idea (a wide value spread between cheap and expensive names
forecasts a larger value premium, and so on).  The rule, applied to each of
Experiment 2's top-ranked factors:

    On each monthly rebalance day (formation month-end ``t``) measure the
    *factor-value spread* -- the average raw factor value of the top quintile
    minus that of the bottom quintile, the same Q5/Q1 buckets the book trades.
    Enter the dollar-neutral long/short book for month ``t+1`` **only if** that
    spread is above its own trailing 6-month average; otherwise sit in cash
    (0 return) for that month.

Both the spread at ``t`` and its trailing average are known at ``t`` (they are
formation-date characteristics, not returns), so the rule is strictly look-ahead
free.  Because exiting and re-entering the book is itself a trade, the **turnover
cost** of the timing overlay is charged explicitly (full liquidation on exit,
re-establishment on re-entry) and the timed book is compared to the always-on
book on an after-cost basis, over the common sample where the signal is defined.

The candidate factors are exactly Experiment 2's top-factor hand-off
(``experiment2 - sw factors/top_factors/top_factors.csv``), so re-running
Experiment 2's ``collect`` step re-points this module automatically.

Engine reuse (the project's dependency-injection convention)
------------------------------------------------------------
Nothing generic is re-implemented:

* each factor's signed Q5-Q1 long/short return is read through
  ``factor_momentum.signed_spread`` (the published ``quintile_returns.csv``,
  oriented by the factor's bullish ``direction``) -- no return is recomputed;
* every performance statistic (industry-neutral alpha + t, industry beta,
  beta-neutral Sharpe) comes from ``composite.book_stats`` / ``industry_return``,
  so "alpha" is defined identically to every other long/short book in the project;
* the trading cost is Experiment 1's ``cost.long_short_cost`` -- extended once,
  backward-compatibly, with an ``active`` mask so it can price the timing overlay;
* the quintile membership and within-month winsorisation behind the value spread
  are Experiment 1's ``factors.prepare_slice`` / ``winsorize_cross_section``.

This module adds **only** the value-spread timing signal and its evaluation.

Run standalone::

    python spread_timing.py

Outputs land under ``experiment4 - timing/output/spread_timing/``:

    spread_timing_summary.csv     per factor: always-on vs timed after-cost alpha
                                  (+t) full & 2016+, average cost, % months in market
    spread_timing_comparison.png  bar chart of always-on vs timed after-cost alpha t
    <factor>/timing_comparison.csv   always-on vs timed, gross & net, full & 2016+
    <factor>/cumulative.png          after-cost growth of $1, always-on vs timed
    <factor>/signal_timeline.png     the factor-value spread vs its 6m trailing average
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths & dependency injection -- reuse Experiment 3's composite plumbing
# --------------------------------------------------------------------------- #
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
_EXP3_DIR = _PROJECT_ROOT / "experiment3 - multifactor"
sys.path.insert(0, str(_EXP3_DIR))

import composite as C        # noqa: E402  loads the engine (factors/cost/quintile/regression)
import factor_momentum as FM  # noqa: E402  signed standalone L/S books from quintile_returns.csv
import cost as COST          # noqa: E402  registered in sys.modules by composite, bound to the engine

F = C.F                      # the Experiment 1 engine, wired for the software universe

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
TOP_FACTORS_CSV = C.TOP_FACTORS_CSV
LOOKBACK = 6                                  # trailing months for the spread average
N_QUINTILES = C.N_QUINTILES
DECADE_START = C.DECADE_START                 # 2016-01-01, the project "past decade" cut-off
OUTPUT_DIR = _THIS_DIR / "output" / "spread_timing"

UNIVERSE = F.SOFTWARE_SERVICES                # every top factor lives on this cross-section


# --------------------------------------------------------------------------- #
# Step 1 -- the factor-value spread (the timing characteristic)
# --------------------------------------------------------------------------- #
def value_spread(panel: pd.DataFrame, factor: str) -> pd.Series:
    """
    Monthly factor-value spread = mean raw factor value of the top z-score
    quintile minus that of the bottom quintile, indexed by formation month.

    The buckets are the *same* equal-count z-score quintiles the book trades
    (via :func:`factors.prepare_slice`), and the raw values are winsorised within
    each month (the engine's own cross-sectional winsorisation) so the tail means
    are not dominated by a single outlier.  Measured on the raw value, not the
    z-score: z-scores are standardised to ~unit dispersion every month, so their
    Q5-Q1 gap is near-constant and carries no timing information, whereas the raw
    characteristic spread genuinely widens and narrows over time.
    """
    sub = F.prepare_slice(panel, factor, N_QUINTILES)        # date, stock_id, zscore, next_return, quintile
    vals = (panel.loc[panel["factor"] == factor, ["date", "stock_id", "value"]]
                 .dropna(subset=["value"]))
    sub = sub.merge(vals, on=["date", "stock_id"], how="inner")
    sub["value"] = F.winsorize_cross_section(sub["value"], sub["date"])
    means = sub.pivot_table(index="date", columns="quintile", values="value", aggfunc="mean")
    return (means[float(N_QUINTILES)] - means[1.0]).rename("value_spread").sort_index()


def timing_flags(spread: pd.Series, lookback: int = LOOKBACK
                 ) -> tuple[pd.Series, pd.Series]:
    """Trailing-average benchmark and the resulting in-market flag.

    Returns ``(trailing, in_market)`` where ``trailing`` is the mean spread over
    the ``lookback`` months *strictly before* ``t`` (``shift(1).rolling``) and
    ``in_market`` is ``spread_t > trailing_t``.  Months without a full trailing
    window carry NaN in ``trailing`` (the signal is undefined and the caller
    excludes them from both books).
    """
    trailing = spread.shift(1).rolling(lookback, min_periods=lookback).mean()
    in_market = (spread > trailing).where(trailing.notna())
    return trailing.rename("trailing_avg"), in_market.rename("in_market")


# --------------------------------------------------------------------------- #
# Step 2 -- build & evaluate one factor's always-on vs timed book (after cost)
# --------------------------------------------------------------------------- #
def _win(s: pd.Series, start: pd.Timestamp | None) -> pd.Series:
    return s if start is None else s[s.index >= start]


WINDOWS: list[tuple[str, pd.Timestamp | None]] = [("full", None), ("2016+", DECADE_START)]


def evaluate_factor(factor: str, subexperiment: str, direction: str,
                    panel: pd.DataFrame, cost_panel: pd.DataFrame,
                    industry: pd.Series) -> tuple[pd.DataFrame, dict]:
    """
    Build the always-on and spread-timed books for one factor and tabulate their
    gross and after-cost performance over the full sample and 2016+.

    Returns ``(comparison, series)``: ``comparison`` is the tidy stats table (one
    row per book x window), ``series`` carries the monthly series needed for the
    plots.
    """
    gross = FM.signed_spread(subexperiment, factor, direction)        # bullish Q5-Q1 book
    spread = value_spread(panel, factor)
    trailing, _ = timing_flags(spread)

    # Common sample: months with both a traded return and a defined timing signal.
    eval_index = gross.index.intersection(trailing.dropna().index).sort_values()
    base = gross.reindex(eval_index)
    # Over eval_index both spread and trailing are defined, so the flag is a clean
    # boolean (no NaN); recompute it here rather than reindexing the NaN-padded one.
    flag = spread.reindex(eval_index) > trailing.reindex(eval_index)
    timed_gross = base.where(flag, 0.0)                                # cash (0) when out of market

    # Turnover cost: always-on (continuous) vs the timing overlay (exit/re-enter).
    cost_always = COST.long_short_cost(panel, factor, cost_panel).reindex(eval_index).fillna(0.0)
    cost_timed = COST.long_short_cost(panel, factor, cost_panel, active=flag).reindex(eval_index).fillna(0.0)

    books = {
        "always_on": (base, cost_always, pd.Series(True, index=eval_index)),
        "timed":     (timed_gross, cost_timed, flag),
    }

    rows = []
    for book, (gross_s, cost_s, active_s) in books.items():
        net_s = gross_s - cost_s
        for win_name, start in WINDOWS:
            sg = C.book_stats(gross_s, industry, start=start)
            sn = C.book_stats(net_s, industry, start=start)
            rows.append({
                "factor": factor, "book": book, "window": win_name,
                "gross_mean": sg["mean_monthly"], "gross_tstat": sg["tstat"],
                "gross_sharpe": sg["sharpe"], "gross_alpha": sg["alpha"],
                "gross_alpha_tstat": sg["alpha_tstat"],
                "ind_beta": sg["ind_beta"], "ind_beta_tstat": sg["ind_beta_tstat"],
                "sharpe_neutral": sg["sharpe_neutral"],
                "avg_cost": float(_win(cost_s, start).mean()),
                "net_mean": sn["mean_monthly"], "net_tstat": sn["tstat"],
                "net_sharpe": sn["sharpe"], "net_alpha": sn["alpha"],
                "net_alpha_tstat": sn["alpha_tstat"],
                "net_sharpe_neutral": sn["sharpe_neutral"],
                "pct_in_market": float(_win(active_s, start).mean()),
                "n_months": int(sg["n_months"]),
            })

    comparison = pd.DataFrame(rows)
    series = {
        "net_always": base - cost_always,
        "net_timed": timed_gross - cost_timed,
        "flag": flag,
        "spread": spread.reindex(eval_index),
        "trailing": trailing.reindex(eval_index),
    }
    return comparison, series


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def _shade_in_market(ax, flag: pd.Series) -> None:
    """Shade contiguous spans of in-market months."""
    s = flag.sort_index().astype(bool)
    idx = s.index
    start = None
    for i, d in enumerate(idx):
        on = bool(s.iloc[i])
        if on and start is None:
            start = d
        if start is not None and (not on or i == len(idx) - 1):
            ax.axvspan(start, d, color="C1", alpha=0.12, lw=0)
            start = None


def plot_cumulative(series: dict, comparison: pd.DataFrame, factor: str,
                    path: Path) -> None:
    """After-cost cumulative growth of $1: always-on vs spread-timed."""
    full = comparison[comparison["window"] == "full"].set_index("book")
    cum_a = (1.0 + series["net_always"].fillna(0.0)).cumprod()
    cum_t = (1.0 + series["net_timed"].fillna(0.0)).cumprod()

    fig, ax = plt.subplots(figsize=(11, 5))
    _shade_in_market(ax, series["flag"])
    ax.plot(cum_a.index, cum_a, color="C0", linewidth=1.3,
            label=f"always-on  [Sharpe={full.loc['always_on', 'net_sharpe']:+.2f}, "
                  f"alpha={full.loc['always_on', 'net_alpha']:+.4%}/mo, "
                  f"t={full.loc['always_on', 'net_alpha_tstat']:+.2f}]")
    ax.plot(cum_t.index, cum_t, color="C3", linewidth=1.3,
            label=f"spread-timed  [Sharpe={full.loc['timed', 'net_sharpe']:+.2f}, "
                  f"alpha={full.loc['timed', 'net_alpha']:+.4%}/mo, "
                  f"t={full.loc['timed', 'net_alpha_tstat']:+.2f}]")
    ax.axhline(1.0, color="black", linewidth=0.6)
    ax.set_title(f"{factor} Q5-Q1 (after cost): always-on vs spread-timed\n"
                 f"(enter only when the factor-value spread > its trailing "
                 f"{LOOKBACK}m average; shaded = in-market months)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Cumulative value of $1 (net of cost)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_signal(series: dict, factor: str, path: Path) -> None:
    """The factor-value spread against its trailing average, in-market shaded."""
    spread, trailing, flag = series["spread"], series["trailing"], series["flag"]
    fig, ax = plt.subplots(figsize=(11, 4))
    _shade_in_market(ax, flag)
    ax.plot(spread.index, spread, color="C4", linewidth=1.0, label="factor-value spread (Q5-Q1)")
    ax.plot(trailing.index, trailing, color="C3", linewidth=1.0, linestyle="--",
            label=f"trailing {LOOKBACK}m average")
    ax.set_title(f"Timing signal: {factor} top-minus-bottom-quintile factor-value spread")
    ax.set_xlabel("Month")
    ax.set_ylabel("Factor-value spread")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_summary(summary: pd.DataFrame, path: Path) -> None:
    """Grouped bars: always-on vs timed after-cost alpha t-stat per factor (full)."""
    factors = summary["factor"].tolist()
    x = np.arange(len(factors))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w / 2, summary["always_net_alpha_t_full"], w, color="C0", label="always-on")
    ax.bar(x + w / 2, summary["timed_net_alpha_t_full"], w, color="C3", label="spread-timed")
    for thr in (1.65, 2.0):
        ax.axhline(thr, color="grey", linewidth=0.7, linestyle=":")
    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(factors, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("After-cost industry-neutral alpha t-stat")
    ax.set_title("Spread timing vs always-on: after-cost alpha t-stat by factor (full sample)\n"
                 f"timing = hold only when the factor-value spread exceeds its trailing {LOOKBACK}m average")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def _pick(comparison: pd.DataFrame, book: str, window: str, col: str) -> float:
    m = comparison[(comparison["book"] == book) & (comparison["window"] == window)]
    return float(m[col].iloc[0])


def run(csv_path: Path = TOP_FACTORS_CSV, out_dir: Path = OUTPUT_DIR) -> pd.DataFrame:
    """Test every top factor under the spread-timing rule and write all outputs."""
    top = FM.load_top_factors(csv_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Experiment 4: factor-spread timing of the top factors ===")
    print(f"Candidates ({len(top)}): " + ", ".join(
        f"{r.factor} [{r.subexperiment}]" for r in top.itertuples()))
    print(f"Rule: hold the Q5-Q1 book in month t+1 only if the factor-value spread "
          f"at t exceeds its trailing {LOOKBACK}m average (after-cost).")

    # Built once and shared: the within-industry "market" return and the cost panel
    # (every top factor trades the same Software & Services cross-section).
    industry = C.industry_return()
    cost_panel = COST.build_cost_panel(UNIVERSE)

    summary_rows = []
    print("\n  factor                  book        a_net/mo   a_net_t  cost/mo  %in   n")
    for r in top.itertuples():
        panel = pd.read_csv(C.EXP2_DIR / r.subexperiment / "factor_panel.csv",
                            parse_dates=["date"])
        panel["stock_id"] = panel["stock_id"].astype(str)

        comparison, series = evaluate_factor(
            r.factor, r.subexperiment, r.direction, panel, cost_panel, industry)

        factor_dir = out_dir / r.factor
        factor_dir.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(factor_dir / "timing_comparison.csv", index=False)
        plot_cumulative(series, comparison, r.factor, factor_dir / "cumulative.png")
        plot_signal(series, r.factor, factor_dir / "signal_timeline.png")

        summary_rows.append({
            "factor": r.factor, "subexperiment": r.subexperiment, "direction": r.direction,
            "always_net_alpha_full": _pick(comparison, "always_on", "full", "net_alpha"),
            "always_net_alpha_t_full": _pick(comparison, "always_on", "full", "net_alpha_tstat"),
            "timed_net_alpha_full": _pick(comparison, "timed", "full", "net_alpha"),
            "timed_net_alpha_t_full": _pick(comparison, "timed", "full", "net_alpha_tstat"),
            "always_net_alpha_t_2016": _pick(comparison, "always_on", "2016+", "net_alpha_tstat"),
            "timed_net_alpha_t_2016": _pick(comparison, "timed", "2016+", "net_alpha_tstat"),
            "avg_cost_always": _pick(comparison, "always_on", "full", "avg_cost"),
            "avg_cost_timed": _pick(comparison, "timed", "full", "avg_cost"),
            "pct_in_market": _pick(comparison, "timed", "full", "pct_in_market"),
            "n_months": int(_pick(comparison, "always_on", "full", "n_months")),
        })

        for book in ("always_on", "timed"):
            print(f"  {r.factor:<22} {book:<11} "
                  f"{_pick(comparison, book, 'full', 'net_alpha'):+.4%} "
                  f"{_pick(comparison, book, 'full', 'net_alpha_tstat'):+7.2f}  "
                  f"{_pick(comparison, book, 'full', 'avg_cost') * 100:6.4f}  "
                  f"{_pick(comparison, book, 'full', 'pct_in_market'):4.0%}  "
                  f"{int(_pick(comparison, book, 'full', 'n_months')):>4}")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "spread_timing_summary.csv", index=False)
    plot_summary(summary, out_dir / "spread_timing_comparison.png")

    print(f"\nSaved spread-timing outputs -> {out_dir}")
    return summary


def main() -> None:
    run()


if __name__ == "__main__":
    main()
