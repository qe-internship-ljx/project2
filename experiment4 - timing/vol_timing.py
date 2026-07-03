"""
vol_timing.py
=============

Experiment 4 -- **volatility-regime timing** of the top software factors.

The first timing module.  Rather than proposing a new factor, it asks whether an
*existing* factor's long/short book can be improved by only holding it in a chosen
market regime.  The rule, applied to each of Experiment 2's top-ranked factors
(the same hand-off ``spread_timing.py`` uses):

    On each monthly rebalance day (formation month-end ``t``), enter the
    dollar-neutral Q5-Q1 book for month ``t+1`` *only if* the 10-trading-day
    moving average of the CBOE VVIX (vol-of-vol) index as of that day is above 95;
    otherwise sit in cash (0 return) for that month.

The economic prior: VVIX (the vol of VIX) is a forward-looking gauge of tail-risk
demand, so a high 10-day VVIX average flags stressed, risk-off regimes.  The
overlay tests whether the top factors' premia pay best in exactly those regimes.
Both the VVIX moving average at ``t`` and the threshold comparison are known at
``t`` (they read the last VVIX observation on or before the rebalance day), so the
rule is strictly look-ahead free.  Because exiting and re-entering the book is
itself a trade, the **turnover cost** of the timing overlay is charged explicitly
(full liquidation on exit, re-establishment on re-entry) and the timed book is
compared to the always-on book on an after-cost basis, over the common sample for
which the timing signal exists (VVIX history starts 2006-03).

The candidate factors are exactly the top five of Experiment 2's cross-experiment
ranking (``experiment2 - sw factors/factor_ranking/monthly_quintile_ranked.csv``), so
re-running Experiment 2's ``collect`` step re-points this module automatically.

Engine reuse (the project's dependency-injection convention)
------------------------------------------------------------
Nothing generic is re-implemented -- the wiring mirrors ``spread_timing.py``:

* each factor's signed Q5-Q1 long/short return is read through
  ``factor_momentum.signed_spread`` (the published ``quintile_returns.csv``,
  oriented by the factor's bullish ``direction``) -- no return is recomputed;
* every performance statistic (industry-neutral alpha + t, industry beta,
  beta-neutral Sharpe) comes from ``composite.book_stats`` / ``industry_return``,
  so "alpha" is defined identically to every other long/short book in the project;
* the trading cost is Experiment 1's ``cost.long_short_cost`` with an ``active``
  mask so it prices the timing overlay.

This module adds **only** the VVIX-regime timing signal and its evaluation.

Run standalone::

    python vol_timing.py

The single output is one consolidated performance table under
``experiment4 - timing/output/vol_timing/``:

    vol_timing_performance.png   ONE consolidated performance table (project house
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
TOP_FACTORS_CSV = C.RANKED_CSV
VVIX_SECURITY = "VVIX Index"
VVIX_MA_WINDOW = 10          # trading-day moving-average window of VVIX
VVIX_THRESHOLD = 95.0        # enter the book only when the 10d VVIX MA exceeds this
DECADE_START = C.DECADE_START                 # 2016-01-01, the project "past decade" cut-off
OUTPUT_DIR = _THIS_DIR / "output"

UNIVERSE = F.SOFTWARE_SERVICES                # every top factor lives on this cross-section


# --------------------------------------------------------------------------- #
# Step 1 -- the VVIX timing signal (a market-wide, factor-independent overlay)
# --------------------------------------------------------------------------- #
def vvix_ma_series() -> pd.DataFrame:
    """Daily 10-trading-day moving average of the VVIX close.

    Returns a frame ``[date, vvix_ma]`` (sorted, no NaNs) ready for an as-of merge
    onto the monthly rebalance dates.  The moving average requires a full
    ``VVIX_MA_WINDOW`` window, so the first few daily observations are dropped.
    """
    raw = pd.read_csv(F.DATA_DIR / "VolatilityIndexData.csv")
    vv = raw[raw["SECURITY"] == VVIX_SECURITY].copy()
    vv["date"] = pd.to_datetime(vv["DATE"])
    vv["vvix"] = pd.to_numeric(vv["INDEX_VALUE"], errors="coerce")
    vv = (vv.dropna(subset=["vvix"])
            .sort_values("date")
            .drop_duplicates("date", keep="last"))
    vv["vvix_ma"] = vv["vvix"].rolling(VVIX_MA_WINDOW, min_periods=VVIX_MA_WINDOW).mean()
    return vv.dropna(subset=["vvix_ma"])[["date", "vvix_ma"]].reset_index(drop=True)


def timing_flag(rebalance_dates: pd.DatetimeIndex) -> pd.Series:
    """In-market flag (``vvix_ma > VVIX_THRESHOLD``) for each monthly rebalance date.

    The 10d VVIX MA is read with a backward as-of merge -- the last VVIX
    observation on or before the rebalance day -- so a month-end that is not itself
    a trading day still reads the most recent value (no look-ahead).  Rebalance
    dates earlier than the first available MA carry NaN (the signal is undefined and
    the caller excludes them from both books).
    """
    ma = vvix_ma_series()
    dates = (pd.DataFrame({"date": pd.to_datetime(rebalance_dates)})
               .sort_values("date").reset_index(drop=True))
    asof = pd.merge_asof(dates, ma, on="date", direction="backward").set_index("date")
    return (asof["vvix_ma"] > VVIX_THRESHOLD).where(asof["vvix_ma"].notna()).rename("in_market")


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
    Build the always-on and VVIX-timed books for one factor and tabulate their
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
    in_market_all = timing_flag(gross.index)

    # Common sample: months with both a traded return and a defined timing signal.
    eval_index = gross.index.intersection(in_market_all.dropna().index).sort_values()
    base = gross.reindex(eval_index)
    flag = in_market_all.reindex(eval_index).astype(bool)
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
    "VVIX-regime timing: always-on vs vol-timed long-short performance\n"
    f"(hold the Q5-Q1 book only when the {VVIX_MA_WINDOW}d VVIX moving average exceeds "
    f"{VVIX_THRESHOLD:g};  α = industry-neutral monthly return,  "
    "Sharpe net cost = after-cost Sharpe)")


def _perf_rows(comparison: pd.DataFrame, factor: str, direction: str) -> list[dict]:
    """The always-on and VVIX-timed :func:`regression.render_alpha_table` rows for
    one factor.  The gross (cost-free) L/S Sharpe and β-neutral Sharpe come from the
    always-on / timed book unchanged; the combined "Sharpe net cost" column carries
    this experiment's after-cost (net) raw and β-neutral Sharpe.  Alpha is the gross
    book's, matching every other experiment's table (cost is shown via the
    net-of-cost Sharpe and the average-cost column, not netted from α)."""
    rows = []
    for book, family in (("always_on", "always-on"), ("timed", "vol-timed")):
        sub = comparison[comparison["book"] == book].set_index("window")
        full, dec = sub.loc["full"], sub.loc["2016+"]
        rows.append({
            "factor": factor, "family": family, "direction": direction,
            "alpha": full["gross_alpha"], "alpha_tstat": full["gross_alpha_tstat"],
            "sharpe": full["gross_sharpe"], "sharpe_neutral": full["sharpe_neutral"],
            "sharpe_cost": full["net_sharpe"],
            "sharpe_cost_neutral": full["net_sharpe_neutral"],
            "avg_cost_pp": full["avg_cost"] * 100.0, "n": int(full["n_months"]),
            "alpha_2016": dec["gross_alpha"], "alpha_tstat_2016": dec["gross_alpha_tstat"],
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
    """Test every top factor under the VVIX-timing rule and write the single
    consolidated performance table."""
    top = FM.load_top_factors(csv_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Experiment 4: VVIX-regime timing of the top factors ===")
    print(f"Candidates ({len(top)}): " + ", ".join(
        f"{r.factor} [{r.subexperiment}]" for r in top.itertuples()))
    print(f"Rule: hold the Q5-Q1 book in month t+1 only if the {VVIX_MA_WINDOW}d VVIX "
          f"moving average at t exceeds {VVIX_THRESHOLD:g} (after-cost).")

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
    C.R.render_alpha_table(perf, out_dir / "vol_timing_performance.png", title=_PERF_TITLE)

    print(f"\nSaved consolidated performance table -> "
          f"{out_dir / 'vol_timing_performance.png'}")
    return perf


def main() -> None:
    run()


if __name__ == "__main__":
    main()
