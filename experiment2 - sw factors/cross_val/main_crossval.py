"""
main_crossval.py
================

Driver for the **cross-validation** library (``crossval_factors.py``): re-test the
fixed five-factor panel (``crossval_factors.CROSSVAL_FACTORS`` -- the quality and
low-risk anchors plus one tailored factor from each of the RD / Stability / Skew
libraries) on the **Banks + Insurance + Commodity Producers** universe, using the
same engine as every other Experiment 2 subfolder.

Deliberately minimal outputs.  Unlike the full software subexperiments, this
cross-validation re-test produces **only two artefacts** (plus the factor panel):

    cross_val/
      factor_panel.csv
      quintile/long_short_market_alpha.{csv,png}       # the universe-neutral L/S table
      quintile/compare_software_vs_crossval.png        # software vs cross-val bar plot

There are no per-factor quintile-cumulative plots, no regression subtree and no
factor-correlation step: redundancy vs the general market factors is already
established per factor in each source subexperiment on the software universe, and
the only question here is whether the software-discovered premia survive on a
structurally unrelated universe -- a single L/S alpha table answers it.

Engine reuse.  ``driver_utils.wire_engine`` binds ``crossval_factors`` to the
``factors`` name and returns Experiment 1's ``regression`` module, whose per-factor
long/short helpers (``long_short_portfolio`` signed by each factor's canonical
software prior, ``market_regression`` for the industry-neutral alpha,
``beta_neutral_sharpe`` and the ``cost`` net-of-cost Sharpes) build exactly the rows
of the standard alpha table -- and its ``render_alpha_table`` renders them -- so this
driver assembles the table without re-running the per-factor plotting/regression
pipeline.

Run standalone::

    python main_crossval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# The shared driver boilerplate lives one level up (the experiment2 root).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import crossval_factors as C   # noqa: E402
import driver_utils as D       # noqa: E402

# Dependency injection: bind Experiment 1's analysis modules to the cross-validation
# factor library, then import them.  We use the regression module's helpers directly.
quintile, regression = D.wire_engine(C)
import cost as CostM           # noqa: E402  (bound to crossval_factors via wire_engine)

R = regression
UNIVERSE = C.BANKS_COMMODITY
_RANKED_CSV = C._EXP2_DIR / "Factor Ranking" / "quarter_quintile_ranked.csv"

# Bar-plot colours (match the presentation deck): software vs cross-val universe.
_SW_COLOR, _CV_COLOR = "#2f6fdb", "#22aa77"


def alpha_table(panel: pd.DataFrame) -> pd.DataFrame:
    """Build the per-factor long-short-vs-industry alpha table (the exact rows
    ``regression.run`` persists to ``long_short_market_alpha.csv``) without emitting
    any of its per-factor plots or the regression subtree."""
    industry_ret = R.industry_monthly_return(panel)
    cost_panel = CostM.build_cost_panel(UNIVERSE)
    rows = []
    for factor in C.FACTOR_NAMES:
        # Direction is the factor's canonical software prior (USE_CANONICAL_LS_DIRECTION);
        # the t-stat argument is unused in that mode, so 0.0 is a safe placeholder.
        spread, sign = R.long_short_portfolio(panel, factor, 0.0)
        ls = R.long_short_stats(spread)
        ls_2016 = R.long_short_stats(spread[spread.index >= R.DECADE_START])
        mreg = R.market_regression(spread, industry_ret)
        mreg_2016 = R.market_regression(
            spread[spread.index >= R.DECADE_START],
            industry_ret[industry_ret.index >= R.DECADE_START])
        cost_series = CostM.long_short_cost(panel, factor, cost_panel, R.N_QUINTILES)
        rows.append({
            "factor": factor, "family": C.FACTORS[factor]["family"],
            "direction": "Q5-Q1" if sign > 0 else "Q1-Q5",
            "alpha": mreg["alpha"], "alpha_tstat": mreg["alpha_tstat"],
            "sharpe": ls["sharpe"],
            "avg_cost_pp": CostM.average_cost(cost_series) * 100.0,
            "sharpe_cost": R.net_of_cost_sharpe(spread, cost_series),
            "sharpe_cost_neutral": R.net_of_cost_neutral_sharpe(
                spread, cost_series, industry_ret),
            "r2": mreg["r2"], "n": mreg["n"],
            "sharpe_neutral": R.beta_neutral_sharpe(spread, industry_ret),
            "alpha_2016": mreg_2016["alpha"],
            "alpha_tstat_2016": mreg_2016["alpha_tstat"],
            "sharpe_2016": ls_2016["sharpe"],
            "sharpe_neutral_2016": R.beta_neutral_sharpe(
                spread, industry_ret, start=R.DECADE_START),
            "sharpe_cost_2016": R.net_of_cost_sharpe(
                spread, cost_series, start=R.DECADE_START),
            "sharpe_cost_neutral_2016": R.net_of_cost_neutral_sharpe(
                spread, cost_series, industry_ret, start=R.DECADE_START),
            "r2_2016": mreg_2016["r2"],
        })
        print(f"  {factor:<26} alpha={mreg['alpha']:+.4%}/mo "
              f"(t={mreg['alpha_tstat']:+.2f})  "
              f"net-bn Sharpe={rows[-1]['sharpe_cost_neutral']:+.2f}")
    return pd.DataFrame(rows)


def compare_plot(cv: pd.DataFrame, path: Path) -> None:
    """Grouped bar chart of the full-period net-cost beta-neutral Sharpe on the
    software universe (from the quarterly-quintile ranking) vs the cross-validation
    universe (this run), one pair of bars per factor."""
    sw = pd.read_csv(_RANKED_CSV).set_index("factor")["sharpe_cost_neutral"]
    factors = list(C.FACTOR_NAMES)
    sw_v = [sw.get(f, np.nan) for f in factors]
    cv_v = cv.set_index("factor")["sharpe_cost_neutral"].reindex(factors).tolist()

    x = np.arange(len(factors))
    w = 0.4
    fig, ax = plt.subplots(figsize=(13, 6))
    b1 = ax.bar(x - w / 2, sw_v, w, label="Software & Services", color=_SW_COLOR)
    b2 = ax.bar(x + w / 2, cv_v, w, label="Banks + Insurance + Commodity", color=_CV_COLOR)
    ax.bar_label(b1, fmt="%+.2f", padding=2, fontsize=9)
    ax.bar_label(b2, fmt="%+.2f", padding=2, fontsize=9)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(factors, rotation=20, ha="right")
    ax.set_ylabel("Net-cost β-neutral Sharpe")
    ax.set_title("Full-period net-cost β-neutral Sharpe")
    fig.suptitle("Univariate factors — quarterly QUINTILE (Q5-Q1) book, "
                 "Software & Services vs Banks+Insurance+Commodity cross-validation universe",
                 fontsize=12)
    ax.legend()
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _coverage_report(panel) -> None:
    # Coverage per factor is itself part of the finding (rd_stability needs an R&D
    # programme, which banks, insurers and resource firms largely do not report).
    counts = panel.groupby("factor")["value"].size().reindex(C.FACTOR_NAMES)
    for name, c in counts.items():
        print(f"    {name:<26} {c:>8,} stock-months")


def main() -> None:
    scope = ", ".join(UNIVERSE.industries)
    print(f"=== Building cross-validation panel: {UNIVERSE.slug} ({scope}) ===")
    panel = C.build(save=True, u=UNIVERSE)
    print(f"  {len(panel):,} rows | {panel['stock_id'].nunique()} stocks | "
          f"{panel['date'].nunique()} months "
          f"({panel['date'].min():%Y-%m} .. {panel['date'].max():%Y-%m})")
    print(f"  Factors: {', '.join(C.FACTOR_NAMES)}")
    _coverage_report(panel)

    print("\n=== Long-short vs industry: universe-neutral alpha table ===")
    tbl = alpha_table(panel)

    out_dir = UNIVERSE.output_dir / "quintile"
    out_dir.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(out_dir / "long_short_market_alpha.csv", index=False)
    R.render_alpha_table(
        tbl, out_dir / "long_short_market_alpha.png",
        title=("Cross-validation: quarterly QUINTILE (Q5-Q1) long-short on "
               "Banks + Insurance + Commodity Producers"))
    compare_plot(tbl, out_dir / "compare_software_vs_crossval.png")

    print(f"\nDone. Outputs under {out_dir}:")
    print("  long_short_market_alpha.{csv,png}")
    print("  compare_software_vs_crossval.png")


if __name__ == "__main__":
    main()
