"""
portfolio_overlay.py
====================

Experiment 3 -- equal-capital overlay of the top factors' univariate books.

Take the five software-industry factors Experiment 2 ranks highest by
industry-neutral alpha t-stat (the ``top_factors/top_factors.csv`` hand-off,
retrieved exactly as ``factor_momentum.py`` does) and hold **all five standalone
long/short books at once**, each on 1/5 of capital:

    overlay_t = mean over factors of (bullish-oriented Q5-Q1 spread)_t

restricted to the months where every candidate has a return (the same panel the
factor-momentum rotation selects over).  Unlike ``composite.py`` -- which merges
the factors into one *signal* and re-sorts -- the overlay merges the finished
*portfolios*: it is the naive diversification baseline the fancier combinations
should beat.

Design -- reuses factor_momentum.py and composite.py wholesale
--------------------------------------------------------------
Nothing generic is re-implemented.  The candidate set and each candidate's
signed standalone long/short return come from ``factor_momentum.load_top_factors``
/ ``build_spread_matrix`` (which read each factor's published
``quintile_returns.csv``); the industry benchmark, window statistics,
cost-incorporated Sharpe and the performance table are ``composite.py``'s
(``industry_return`` / ``book_stats`` / ``attach_net_cost_sharpe`` /
``render_performance``), so "alpha" is defined identically to every other book
in the project.  This module adds **only** the 1/n capital split and the
netted-turnover cost of holding the five books jointly.

Cost of the joint book
----------------------
Each constituent book is its equal-weighted Q5/Q1 leg membership
(``factor_momentum._leg_membership``) scaled by the 1/n capital split; the five
signed weight vectors are then **netted per name** before the project cost model
charges turnover (``cost.turnover_cost``) -- a stock long in one book and short
in another is only charged on the residual position actually traded.  As
everywhere else the cost is reported (and folded into the net-of-cost Sharpe),
never netted from the gross alpha.

Outputs (``output/portfolio_overlay/``)
---------------------------------------
    performance.png     the joint book's mean / t / Sharpe / industry-neutral
                        alpha / net-of-cost Sharpe / avg cost, full sample and 2016+

Run standalone::

    python portfolio_overlay.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import composite as C                  # engine + analysis plumbing, loaded by path
import factor_momentum as FM           # top-factor retrieval + leg membership

OUTPUT_DIR = C.OUTPUT_DIR / "portfolio_overlay"
DECADE_START = C.DECADE_START          # 2016-01-01, the project "past decade" cut-off


# --------------------------------------------------------------------------- #
# The joint book and its netted turnover cost
# --------------------------------------------------------------------------- #
def overlay_return(spreads: pd.DataFrame) -> pd.Series:
    """Monthly return of the equal-capital overlay: the mean of the candidates'
    signed long/short spreads, over months where *every* candidate has a return
    (the same panel convention as the factor-momentum rotation)."""
    return spreads.dropna().mean(axis=1).rename("overlay")


def overlay_cost(factor_names: list[str], months: pd.Index) -> pd.Series:
    """
    Monthly turnover cost of holding all candidate books jointly, indexed by
    formation month and restricted to the ``months`` the overlay is booked.

    Each book contributes its equal-weighted Q5/Q1 leg membership scaled by the
    ``1/n`` capital split, signed by side; the per-name weights are then netted
    across books before ``cost.turnover_cost`` charges the weight actually
    traded.  Splitting the netted weight back into a positive long and short leg
    keeps the charge exact even when a name flips side month-over-month (the
    two legs' one-way charges sum to the full crossing trade).
    """
    resolved = C.resolve_factors(factor_names)
    pieces = []
    for r in resolved.itertuples():
        mem = FM._leg_membership(r)
        legs = C.COST.equal_weight_legs(mem[mem["date"].isin(months)])
        legs["w"] = np.where(legs["leg"] == "long", legs["w"], -legs["w"]) / len(resolved)
        pieces.append(legs[["date", "stock_id", "w"]])

    net = pd.concat(pieces).groupby(["date", "stock_id"], as_index=False)["w"].sum()
    net = net[net["w"] != 0.0]
    net["leg"] = np.where(net["w"] > 0, "long", "short")
    net["w"] = net["w"].abs()
    dates = pd.Index(sorted(months), name="date")
    return C.COST.turnover_cost(net, C.cost_panel(), dates)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run(csv_path: Path = FM.TOP_FACTORS_CSV, out_dir: Path = OUTPUT_DIR) -> dict:
    """Build the equal-capital overlay of the top factors' univariate books and
    write its performance table under ``out_dir``.  Returns the per-window
    performance dict."""
    top = FM.load_top_factors(csv_path)
    factor_names = top["factor"].tolist()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Experiment 3: portfolio overlay of the top univariate books ===")
    print(f"Constituents ({len(factor_names)}): " + ", ".join(
        f"{r.factor} [{r.subexperiment}]" for r in top.itertuples()))

    # 1. The joint book: each candidate's signed standalone L/S at 1/n capital.
    spread = overlay_return(FM.build_spread_matrix(top))

    # 2. Performance vs the industry, exactly as every project book is measured.
    industry = C.industry_return()
    windows = [("Full sample", C.book_stats(spread, industry)),
               ("Past decade (2016+)", C.book_stats(spread, industry, start=DECADE_START))]
    full, decade = windows[0][1], windows[1][1]
    start, end = spread.index.min(), spread.index.max()

    # Netted turnover cost of the joint book (reported, not netted from the gross
    # alpha) plus the cost-incorporated Sharpe from the same series.
    cost_series = overlay_cost(factor_names, spread.index)
    full["avg_cost"] = C.window_cost(cost_series)
    decade["avg_cost"] = C.window_cost(cost_series, start=DECADE_START)
    C.attach_net_cost_sharpe(full, spread, cost_series, industry)
    C.attach_net_cost_sharpe(decade, spread, cost_series, industry, start=DECADE_START)

    # --- Persist output ------------------------------------------------------ #
    C.render_performance(
        windows, "Portfolio overlay: joint top-factor long-short performance",
        f"equal-capital (1/{len(factor_names)}) overlay of the standalone books: "
        f"{', '.join(factor_names)}   |   "
        f"{len(spread)} months ({start:%Y-%m} .. {end:%Y-%m})   |   "
        "alpha from regressing the book on the market-cap-weighted industry return; "
        "cost charged on the cross-book netted weights.   "
        "Shading: |t| >= 1.65 (10%), 2.0 (5%).",
        out_dir / "performance.png")

    # --- Console summary (ASCII only -- Windows cp1252 stdout) --------------- #
    print(f"  overlay: mean {full['mean_monthly']:+.4%}/mo "
          f"(t={full['tstat']:+.2f}, Sharpe={full['sharpe']:+.2f})")
    print(f"  industry-neutral alpha: {full['alpha']:+.4%}/mo "
          f"(t={full['alpha_tstat']:+.2f}, ind beta={full['ind_beta']:+.2f})")
    print(f"  2016+: alpha={decade['alpha']:+.4%}/mo (t={decade['alpha_tstat']:+.2f})")
    print(f"Saved -> {out_dir}")
    return {lbl: s for lbl, s in windows}


def main() -> None:
    run()


if __name__ == "__main__":
    main()
