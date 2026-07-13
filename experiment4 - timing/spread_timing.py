"""
spread_timing.py
================

Experiment 4 -- **factor-spread timing** of every ranked software factor.

The second timing module.  Rather than holding a factor's long/short book all the
time, it asks whether the book should be entered *only when the factor's own
cross-sectional dispersion is unusually wide* -- the classic "factor spread"
predictability idea (a wide value spread between cheap and expensive names
forecasts a larger value premium, and so on).  The rule, applied to every factor
in Experiment 2's cross-experiment ranking:

    On each monthly rebalance day (formation month-end ``t``) measure the
    *factor-value spread* -- the average raw factor value of the top quintile
    minus that of the bottom quintile, the same Q5/Q1 buckets the book trades.
    Enter the dollar-neutral long/short book for month ``t+1`` **only if** that
    spread is above its own trailing 12-month average; otherwise sit in cash
    (0 return) for that month.

Both the spread at ``t`` and its trailing average are known at ``t`` (they are
formation-date characteristics, not returns), so the rule is strictly look-ahead
free.  Because exiting and re-entering the book is itself a trade, the **turnover
cost** of the timing overlay is charged explicitly (full liquidation on exit,
re-establishment on re-entry) and the timed book is reported on an after-cost
basis, over the common sample where the signal is defined.

The candidate factors are exactly Experiment 2's FULL quarterly-repositioned
cross-experiment ranking (``quarter_position.ranked_factors`` with ``n=None`` --
every ranked factor, not just the top five), so re-running Experiment 2 re-points
this module automatically.  The base book each overlay gates is the factor's
**quarterly-repositioned** top-minus-bottom book (the project-wide convention); the
factor-value spread signal and the in/out gate remain monthly.

The whole rule is run at **two bucket granularities** -- the headline quintile (Q5-Q1)
book and, identically, the tertile (Q3-Q1) book -- by threading a single bucket count
``n`` through the spread signal, the traded book and the table (``n=5`` quintiles,
``n=3`` tertiles).  Each granularity reads its own ``n``-bucket ranking as the candidate
set (the ``quintile`` / ``tertile`` hand-off), matching ``composite.py``'s convention,
and writes its own table.

Engine reuse (the project's dependency-injection convention)
------------------------------------------------------------
Nothing generic is re-implemented:

* each factor's signed quarterly top-minus-bottom long/short return is the shared
  ``quarter_position.quarter_held_spread`` (the primitive behind
  ``factor_momentum.signed_spread``), oriented by the factor's bullish ``direction``
  and formed on ``n`` buckets -- no return is recomputed;
* every performance statistic (industry-neutral alpha + t, industry beta,
  beta-neutral Sharpe) comes from ``composite.book_stats`` / ``industry_return``,
  so "alpha" is defined identically to every other long/short book in the project;
* the trading cost is ``cost.turnover_cost`` on the factor's quarterly-held legs
  (``quarter_position.quarter_held_legs``) with an ``active`` mask so it prices the
  timing overlay's exit / re-entry churn;
* the quintile membership and within-month winsorisation behind the value spread
  signal are Experiment 1's ``factors.prepare_slice`` / ``winsorize_cross_section``.

This module adds **only** the value-spread timing signal and its evaluation.

Run standalone::

    python spread_timing.py

The outputs are two consolidated performance tables under
``experiment4 - timing/output/`` -- one per bucket granularity:

    spread_timing_quintile_performance.png          headline QUINTILE (Q5-Q1) book
    spread_timing_tertile_performance.png  the same rule on the TERTILE (Q3-Q1) book

Each is one consolidated performance table (project house style, one row per factor --
the spread-timed book): gross (cost-free) L/S & beta-neutral Sharpe, the combined
after-cost ("Sharpe net cost") Sharpe, the % activation (share of months the spread
gate holds the book), industry-neutral alpha & beta, average cost -- full sample and
2016+.
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
LOOKBACK = 12                                  # trailing months for the spread average
N_QUINTILES = C.N_QUINTILES
DECADE_START = C.DECADE_START                 # 2016-01-01, the project "past decade" cut-off
OUTPUT_DIR = _THIS_DIR / "output"

UNIVERSE = F.SOFTWARE_SERVICES                # every candidate factor lives on this cross-section


# --------------------------------------------------------------------------- #
# Step 1 -- the factor-value spread (the timing characteristic)
# --------------------------------------------------------------------------- #
def value_spread(panel: pd.DataFrame, factor: str, n: int = N_QUINTILES) -> pd.Series:
    """
    Monthly factor-value spread = mean raw factor value of the top z-score
    bucket minus that of the bottom bucket of an ``n``-bucket sort, indexed by
    formation month (``n=5`` quintiles, ``n=3`` tertiles).

    The buckets are the *same* equal-count z-score buckets the book trades
    (via :func:`factors.prepare_slice`), and the raw values are winsorised within
    each month (the engine's own cross-sectional winsorisation) so the tail means
    are not dominated by a single outlier.  Measured on the raw value, not the
    z-score: z-scores are standardised to ~unit dispersion every month, so their
    top-minus-bottom gap is near-constant and carries no timing information, whereas
    the raw characteristic spread genuinely widens and narrows over time.
    """
    sub = F.prepare_slice(panel, factor, n)                  # date, stock_id, zscore, next_return, quintile
    vals = (panel.loc[panel["factor"] == factor, ["date", "stock_id", "value"]]
                 .dropna(subset=["value"]))
    sub = sub.merge(vals, on=["date", "stock_id"], how="inner")
    sub["value"] = F.winsorize_cross_section(sub["value"], sub["date"])
    means = sub.pivot_table(index="date", columns="quintile", values="value", aggfunc="mean")
    return (means[float(n)] - means[1.0]).rename("value_spread").sort_index()


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
# Step 2 -- build & evaluate one factor's timed book (after cost)
# --------------------------------------------------------------------------- #
def _win(s: pd.Series, start: pd.Timestamp | None) -> pd.Series:
    return s if start is None else s[s.index >= start]


WINDOWS: list[tuple[str, pd.Timestamp | None]] = [("full", None), ("2016+", DECADE_START)]


def evaluate_factor(factor: str, subexperiment: str, direction: str,
                    panel: pd.DataFrame, cost_panel: pd.DataFrame,
                    industry: pd.Series, n: int = N_QUINTILES) -> pd.DataFrame:
    """
    Build the spread-timed book for one factor and tabulate its gross and after-cost
    performance over the full sample and 2016+.  ``n`` sets the bucket count of both
    the traded book and the timing spread (``n=5`` quintiles, ``n=3`` tertiles).

    Returns the tidy ``comparison`` stats table (one row per window): the gross
    (cost-free) mean / t / Sharpe / industry-neutral alpha & beta, and the
    net-of-cost (after-cost) counterparts, plus the average turnover cost and the
    fraction of months in market.  **Every quantity -- the raw Sharpe, the average
    cost and the net-of-cost Sharpe included -- is measured over the activated
    (in-market) months only**; the exact-zero cash months are excluded so they
    neither dilute the mean nor shrink alpha / beta.  The activated-month cost folds
    each run's exit-month liquidation back onto the run's last active month
    (:func:`cost.active_month_cost`), so a book entered and exited within a single
    month is discounted by the full round-trip (double) cost on that month.
    """
    # Orient long/short from the ranking's direction label, bucket-count free: bullish
    # top-minus-bottom labels lead with the high bucket (``Q5-Q1``/``T3-T1``/``H2-H1``),
    # the bearish reverse leads with ``1`` (``Q1-Q5``/``T1-T3``).  (A bare ``== "Q5-Q1"``
    # test would silently flip every tertile/half book, whose label is ``T3-T1`` etc.)
    sign = -1 if str(direction).strip()[1:].startswith("1-") else 1
    gross = C.QP.quarter_held_spread(panel, factor, sign, n).rename(factor)  # bullish quarterly top-bottom book
    spread = value_spread(panel, factor, n)
    trailing, _ = timing_flags(spread)

    # Common sample: months with both a traded return and a defined timing signal.
    eval_index = gross.index.intersection(trailing.dropna().index).sort_values()
    base = gross.reindex(eval_index)
    # Over eval_index both spread and trailing are defined, so the flag is a clean
    # boolean (no NaN); recompute it here rather than reindexing the NaN-padded one.
    flag = spread.reindex(eval_index) > trailing.reindex(eval_index)
    timed_gross = base.where(flag, 0.0)                                # cash (0) when out of market

    # Turnover cost of the timing overlay on the factor's quarterly-held legs (exit
    # on the out-of-market months, re-enter): the ``active`` mask charges the full
    # liquidation on exit and re-establishment on entry over the quarterly membership.
    legs = C.QP.quarter_held_legs(panel, factor, n)
    leg_dates = pd.Index(sorted(legs["date"].unique()), name="date")
    cost_timed = (COST.turnover_cost(legs, cost_panel, leg_dates, active=flag)
                      .reindex(eval_index).fillna(0.0))

    books = {
        "timed": (timed_gross, cost_timed, flag),
    }

    rows = []
    for book, (gross_s, cost_s, active_s) in books.items():
        # Everything is measured over the **activated months only**.  An out-of-market
        # month is cash (an exact 0 with no industry exposure), so including it would
        # dilute the mean, shrink alpha / beta toward zero, and drag the (neutral and
        # raw) Sharpe below what a significant alpha implies.  The activated series is
        # also where the true trading cost belongs: :func:`cost.active_month_cost`
        # folds each run's exit-month liquidation back onto its last active month, so a
        # book entered and exited within a single month pays the full round-trip
        # (double) cost on that one activated month.
        act_gross = gross_s[active_s]
        act_cost = COST.active_month_cost(cost_s, active_s)
        act_net = act_gross - act_cost
        reg_gross, reg_net = act_gross, act_net
        for win_name, start in WINDOWS:
            sg = C.book_stats(act_gross, industry, start=start)
            sn = C.book_stats(act_net, industry, start=start)
            rg = C.book_stats(reg_gross, industry, start=start)
            rn = C.book_stats(reg_net, industry, start=start)
            rows.append({
                "factor": factor, "book": book, "window": win_name,
                "gross_mean": sg["mean_monthly"], "gross_tstat": sg["tstat"],
                "gross_sharpe": sg["sharpe"], "gross_alpha": rg["alpha"],
                "gross_alpha_tstat": rg["alpha_tstat"],
                "ind_beta": rg["ind_beta"], "ind_beta_tstat": rg["ind_beta_tstat"],
                "sharpe_neutral": rg["sharpe_neutral"],
                "avg_cost": float(_win(act_cost, start).mean()),
                "net_mean": sn["mean_monthly"], "net_tstat": sn["tstat"],
                "net_sharpe": sn["sharpe"], "net_alpha": rn["alpha"],
                "net_alpha_tstat": rn["alpha_tstat"],
                "net_sharpe_neutral": rn["sharpe_neutral"],
                "pct_in_market": float(_win(active_s, start).mean()),
                "n_months": int(sg["n_months"]),
            })

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Consolidated performance table (project house style, one row per factor x book)
# --------------------------------------------------------------------------- #
def _perf_title(n: int) -> str:
    """House-style table title for the ``n``-bucket sort (``n=5`` quintiles, ``n=3``
    tertiles): only the sort word and the top-minus-bottom spread column change."""
    return (
        f"Factor-spread timing: spread-timed {C.sort_word(n)} long-short performance\n"
        f"(hold the Q{n}-Q1 book only when its top-minus-bottom factor-value spread exceeds "
        f"its trailing {LOOKBACK}m average;  α = industry-neutral monthly return,  "
        "Sharpe net cost = after-cost Sharpe)")


def _perf_rows(comparison: pd.DataFrame, factor: str, direction: str) -> list[dict]:
    """The spread-timed :func:`regression.render_alpha_table` row for one factor.  The
    gross (cost-free) L/S Sharpe and β-neutral Sharpe come from the timed book
    unchanged; the combined "Sharpe net cost" column carries this experiment's
    after-cost (net) raw and β-neutral Sharpe.  Alpha is the gross book's, matching
    every other experiment's table (cost is shown via the net-of-cost Sharpe and the
    average-cost column, not netted from α).  ``pct_activation`` (the full-sample
    fraction of months the spread gate holds the book) drives the "% activation"
    column the table renders right after "Sharpe net cost"."""
    sub = comparison[comparison["book"] == "timed"].set_index("window")
    full, dec = sub.loc["full"], sub.loc["2016+"]
    return [{
        "factor": factor, "family": "spread-timed", "direction": direction,
        "alpha": full["gross_alpha"], "alpha_tstat": full["gross_alpha_tstat"],
        "sharpe": full["gross_sharpe"], "sharpe_neutral": full["sharpe_neutral"],
        "sharpe_cost": full["net_sharpe"],
        "sharpe_cost_neutral": full["net_sharpe_neutral"],
        "pct_activation": full["pct_in_market"],
        "avg_cost_pp": full["avg_cost"] * 100.0, "n": int(full["n_months"]),
        "alpha_2016": dec["gross_alpha"], "alpha_tstat_2016": dec["gross_alpha_tstat"],
        "sharpe_2016": dec["gross_sharpe"], "sharpe_neutral_2016": dec["sharpe_neutral"],
        "sharpe_cost_2016": dec["net_sharpe"],
        "sharpe_cost_neutral_2016": dec["net_sharpe_neutral"],
    }]


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def _pick(comparison: pd.DataFrame, book: str, window: str, col: str) -> float:
    m = comparison[(comparison["book"] == book) & (comparison["window"] == window)]
    return float(m[col].iloc[0])


def run(out_dir: Path = OUTPUT_DIR, n: int = N_QUINTILES) -> pd.DataFrame:
    """Test every ranked factor under the spread-timing rule and write the single
    consolidated performance table.  ``n`` sets the bucket count of both the traded
    book and the timing spread (``n=5`` quintiles, ``n=3`` tertiles); the ``n``-bucket
    ranking is read as the candidate set, so the tertile run tests the tertile-ranked
    factors and writes ``spread_timing_tertile_performance.png``."""
    word = C.sort_word(n)
    candidates = FM.ranked_factors(None, word)           # every ranked factor, not just the top-5
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Experiment 4: factor-spread timing ({word}) of every ranked factor ===")
    print(f"Candidates ({len(candidates)}): " + ", ".join(
        f"{r.factor} [{r.subexperiment}]" for r in candidates.itertuples()))
    print(f"Rule: hold the Q{n}-Q1 book in month t+1 only if the factor-value spread "
          f"at t exceeds its trailing {LOOKBACK}m average (after-cost).")

    # Built once and shared: the within-industry "market" return and the cost panel
    # (every candidate factor trades the same Software & Services cross-section).
    industry = C.industry_return()
    cost_panel = COST.build_cost_panel(UNIVERSE)

    # Resolve every candidate to its source panel through composite (its library
    # map), so a factor from Experiment 1 (General) resolves as readily as one
    # from an Experiment 2 subexperiment.  Evaluation then walks the candidates
    # panel-by-panel -- composite's ``build_exposures`` convention -- so each
    # (large) library panel is read from disk exactly once for the whole sweep.
    resolved = C.resolve_factors(candidates["factor"].tolist())
    candidates = candidates.merge(resolved[["factor", "panel_path"]],
                                  on="factor", how="left")

    perf_by_factor: dict[str, list[dict]] = {}
    print("\n  factor                          book        a_net/mo   a_net_t  cost/mo  %in   n")
    for panel_path, grp in candidates.groupby("panel_path", sort=False):
        panel = pd.read_csv(panel_path, parse_dates=["date"],
                            usecols=["date", "stock_id", "factor", "value",
                                     "zscore", "next_return"])
        panel["stock_id"] = panel["stock_id"].astype(str)

        for r in grp.itertuples():
            comparison = evaluate_factor(
                r.factor, r.subexperiment, r.direction, panel, cost_panel, industry, n)
            perf_by_factor[r.factor] = _perf_rows(comparison, r.factor, r.direction)

            print(f"  {r.factor:<30} {'timed':<11} "
                  f"{_pick(comparison, 'timed', 'full', 'net_alpha'):+.4%} "
                  f"{_pick(comparison, 'timed', 'full', 'net_alpha_tstat'):+7.2f}  "
                  f"{_pick(comparison, 'timed', 'full', 'avg_cost') * 100:6.4f}  "
                  f"{_pick(comparison, 'timed', 'full', 'pct_in_market'):4.0%}  "
                  f"{int(_pick(comparison, 'timed', 'full', 'n_months')):>4}")

    # One consolidated performance table for the whole experiment (all factors x
    # both books), in the same house style as every other experiment's alpha table.
    # Rows are re-assembled in the hand-off's rank order (evaluation above walked
    # the candidates panel-by-panel, not by rank).
    perf = pd.DataFrame([row for factor in candidates["factor"]
                         for row in perf_by_factor[factor]])
    # Headline quintile book keeps the bare filename; other sorts (tertile) are suffixed.
    suffix = f"_{word}"
    out_png = out_dir / f"spread_timing{suffix}_performance.png"
    C.R.render_alpha_table(perf, out_png, title=_perf_title(n))

    print(f"\nSaved consolidated performance table -> {out_png}")
    return perf


def main() -> None:
    run(n=N_QUINTILES)                                   # headline quintile book
    run(n=C.N_TERTILES)                                  # tertile book


if __name__ == "__main__":
    main()
