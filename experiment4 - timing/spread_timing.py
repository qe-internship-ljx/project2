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

The single output is one consolidated performance table under
``experiment4 - timing/output/spread_timing/``:

    spread_timing_performance.png  ONE consolidated performance table (project house
                                   style, one row per factor x book): gross (cost-free)
                                   L/S & beta-neutral Sharpe, the combined after-cost
                                   ("Sharpe net cost") Sharpe, industry-neutral alpha
                                   & beta, average cost -- full sample and 2016+
"""

from __future__ import annotations

import sys
from pathlib import Path

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
OUTPUT_DIR = _THIS_DIR / "output"

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
                    industry: pd.Series) -> pd.DataFrame:
    """
    Build the always-on and spread-timed books for one factor and tabulate their
    gross and after-cost performance over the full sample and 2016+.

    Returns the tidy ``comparison`` stats table (one row per book x window): the
    gross (cost-free) mean / t / Sharpe / industry-neutral alpha & beta, and the
    net-of-cost (after-cost) counterparts, plus the average turnover cost and the
    fraction of months in market.  The market-alpha regression (gross and net) is
    estimated over the in-market months only, so the timed book's alpha is not
    diluted by the exact-zero cash months; mean / t / Sharpe still cover the full
    timed series including those months.
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
        # The market-alpha regression uses only the in-market months: an
        # out-of-market month is cash (an exact 0 with no industry exposure), so
        # including it would mechanically shrink both alpha and beta toward zero.
        # For the always-on book the mask is all-True and rg/rn coincide with sg/sn.
        reg_gross, reg_net = gross_s[active_s], net_s[active_s]
        for win_name, start in WINDOWS:
            sg = C.book_stats(gross_s, industry, start=start)
            sn = C.book_stats(net_s, industry, start=start)
            rg = C.book_stats(reg_gross, industry, start=start)
            rn = C.book_stats(reg_net, industry, start=start)
            rows.append({
                "factor": factor, "book": book, "window": win_name,
                "gross_mean": sg["mean_monthly"], "gross_tstat": sg["tstat"],
                "gross_sharpe": sg["sharpe"], "gross_alpha": rg["alpha"],
                "gross_alpha_tstat": rg["alpha_tstat"],
                "ind_beta": rg["ind_beta"], "ind_beta_tstat": rg["ind_beta_tstat"],
                "sharpe_neutral": sg["sharpe_neutral"],
                "avg_cost": float(_win(cost_s, start).mean()),
                "net_mean": sn["mean_monthly"], "net_tstat": sn["tstat"],
                "net_sharpe": sn["sharpe"], "net_alpha": rn["alpha"],
                "net_alpha_tstat": rn["alpha_tstat"],
                "net_sharpe_neutral": sn["sharpe_neutral"],
                "pct_in_market": float(_win(active_s, start).mean()),
                "n_months": int(sg["n_months"]),
            })

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Consolidated performance table (project house style, one row per factor x book)
# --------------------------------------------------------------------------- #
_PERF_TITLE = (
    "Factor-spread timing: always-on vs spread-timed long-short performance\n"
    "(hold the Q5-Q1 book only when its top-minus-bottom factor-value spread exceeds "
    f"its trailing {LOOKBACK}m average;  α = industry-neutral monthly return,  "
    "Sharpe net cost = after-cost Sharpe)")


def _perf_rows(comparison: pd.DataFrame, factor: str, direction: str) -> list[dict]:
    """The always-on and spread-timed :func:`regression.render_alpha_table` rows for
    one factor.  The gross (cost-free) L/S Sharpe and β-neutral Sharpe come from the
    always-on / timed book unchanged; the combined "Sharpe net cost" column carries
    this experiment's after-cost (net) raw and β-neutral Sharpe.  Alpha and industry
    β are the gross book's, matching every other experiment's table (cost is shown
    via the net-of-cost Sharpe and the average-cost column, not netted from α)."""
    rows = []
    for book, family in (("always_on", "always-on"), ("timed", "spread-timed")):
        sub = comparison[comparison["book"] == book].set_index("window")
        full, dec = sub.loc["full"], sub.loc["2016+"]
        rows.append({
            "factor": factor, "family": family, "direction": direction,
            "alpha": full["gross_alpha"], "alpha_tstat": full["gross_alpha_tstat"],
            "beta": full["ind_beta"], "beta_tstat": full["ind_beta_tstat"],
            "sharpe": full["gross_sharpe"], "sharpe_neutral": full["sharpe_neutral"],
            "sharpe_cost": full["net_sharpe"],
            "sharpe_cost_neutral": full["net_sharpe_neutral"],
            "avg_cost_pp": full["avg_cost"] * 100.0, "n": int(full["n_months"]),
            "alpha_2016": dec["gross_alpha"], "alpha_tstat_2016": dec["gross_alpha_tstat"],
            "beta_2016": dec["ind_beta"], "beta_tstat_2016": dec["ind_beta_tstat"],
            "sharpe_2016": dec["gross_sharpe"], "sharpe_neutral_2016": dec["sharpe_neutral"],
            "sharpe_cost_2016": dec["net_sharpe"],
            "sharpe_cost_neutral_2016": dec["net_sharpe_neutral"],
        })
    return rows


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def _pick(comparison: pd.DataFrame, book: str, window: str, col: str) -> float:
    m = comparison[(comparison["book"] == book) & (comparison["window"] == window)]
    return float(m[col].iloc[0])


def run(csv_path: Path = TOP_FACTORS_CSV, out_dir: Path = OUTPUT_DIR) -> pd.DataFrame:
    """Test every top factor under the spread-timing rule and write the single
    consolidated performance table."""
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

    perf_rows = []
    print("\n  factor                  book        a_net/mo   a_net_t  cost/mo  %in   n")
    for r in top.itertuples():
        # Resolve the factor's source panel through composite (its library map),
        # so a top factor from Experiment 1 (General) resolves as readily as one
        # from an Experiment 2 subexperiment -- the hard-coded EXP2 path would miss
        # it.  Same convention factor_momentum uses for the signed book.
        panel_path = C.resolve_factors([r.factor]).iloc[0]["panel_path"]
        panel = pd.read_csv(panel_path, parse_dates=["date"])
        panel["stock_id"] = panel["stock_id"].astype(str)

        comparison = evaluate_factor(
            r.factor, r.subexperiment, r.direction, panel, cost_panel, industry)
        perf_rows.extend(_perf_rows(comparison, r.factor, r.direction))

        for book in ("always_on", "timed"):
            print(f"  {r.factor:<22} {book:<11} "
                  f"{_pick(comparison, book, 'full', 'net_alpha'):+.4%} "
                  f"{_pick(comparison, book, 'full', 'net_alpha_tstat'):+7.2f}  "
                  f"{_pick(comparison, book, 'full', 'avg_cost') * 100:6.4f}  "
                  f"{_pick(comparison, book, 'full', 'pct_in_market'):4.0%}  "
                  f"{int(_pick(comparison, book, 'full', 'n_months')):>4}")

    # One consolidated performance table for the whole experiment (all factors x
    # both books), in the same house style as every other experiment's alpha table.
    perf = pd.DataFrame(perf_rows)
    C.R.render_alpha_table(perf, out_dir / "spread_timing_performance.png", title=_PERF_TITLE)

    print(f"\nSaved consolidated performance table -> "
          f"{out_dir / 'spread_timing_performance.png'}")
    return perf


def main() -> None:
    run()


if __name__ == "__main__":
    main()
