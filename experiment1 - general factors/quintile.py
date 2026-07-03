"""
quintile.py
===========

Approach 1 -- even-quintile portfolio sort.

For every factor, each month we sort the industry cross-section into five
equal-count buckets on the factor z-score, then measure the **next-period**
(month t+1) mean return of each bucket.  This yields, per factor:

    * ``output/quintile/<factor>/quintile_returns.csv`` -- months x
      {Q1..Q5, Q5-Q1 spread}, the mean next-period return in each bucket.
    * ``output/quintile/<factor>/quintile_cumulative.png`` -- the five buckets
      shown as cumulative growth of $1 (log scale).

The long-short (Q5-Q1) mean, t-stat, Sharpe and annualised spread are printed
per factor as the sort runs.  (Trading cost and the cumulative long/short
portfolio plot live with the regression outputs -- see ``regression.py`` -- which signs the book and
writes ``output/quintile/<factor>/long_short.png`` and the
``long_short_market_alpha`` table.)

This module is a library: :func:`run` is driven by the per-universe
orchestrators (``software_service.py``, ``banks_insurance.py``,
``commodity_producers.py``), which build the panel once and reuse it across
both analyses.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import factors as F

N_QUINTILES = 5
QCOLS = [f"Q{i}" for i in range(1, N_QUINTILES + 1)]
MONTHS_PER_YEAR = 12


# --------------------------------------------------------------------------- #
# Core computation
# --------------------------------------------------------------------------- #
def quintile_returns(panel: pd.DataFrame, factor: str) -> pd.DataFrame:
    """
    Wide table indexed by month: mean next-period return of each z-score
    quintile, plus the Q5-Q1 long-short spread.
    """
    sub = F.prepare_slice(panel, factor, N_QUINTILES)

    wide = (sub.pivot_table(index="date", columns="quintile",
                            values="next_return", aggfunc="mean")
               .rename(columns={i: f"Q{i}" for i in range(1, N_QUINTILES + 1)})
               .sort_index())
    wide = wide.reindex(columns=QCOLS)
    wide["Q5-Q1"] = wide["Q5"] - wide["Q1"]
    return wide


def long_short_stats(spread: pd.Series) -> dict:
    """Mean / t-stat / Sharpe / annualised summary of a monthly long-short spread.

    The Q5-Q1 book is self-financing (zero net investment), so its Sharpe ratio
    is the mean spread over its volatility -- no risk-free subtraction -- and is
    annualised by sqrt(12).
    """
    s = spread.dropna()
    n = s.size
    mean = s.mean()
    sd = s.std(ddof=1)
    tstat = mean / (sd / np.sqrt(n)) if n > 1 and sd > 0 else np.nan
    sharpe = (mean / sd) * np.sqrt(MONTHS_PER_YEAR) if n > 1 and sd > 0 else np.nan
    return {"mean_monthly": mean,
            "tstat": tstat,
            "sharpe": sharpe,
            "ann_return": mean * MONTHS_PER_YEAR,
            "n_months": n}


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def _title(factor: str) -> str:
    return f"{factor}  ({F.FACTORS[factor]['family']})"


def plot_cumulative(wide: pd.DataFrame, factor: str, path: Path) -> None:
    cum = (1.0 + wide[QCOLS].fillna(0.0)).cumprod()
    fig, ax = plt.subplots(figsize=(11, 5))
    for q in QCOLS:
        ax.plot(cum.index, cum[q], label=q, linewidth=1.3)
    ax.set_yscale("log")
    ax.set_title(f"Cumulative growth of $1 by z-score quintile\n{_title(factor)}")
    ax.set_xlabel("Month")
    ax.set_ylabel("Cumulative value (log scale)")
    ax.legend(title="Quintile", ncol=5, loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run(panel: pd.DataFrame | None = None,
        u: "F.Universe" = F.SOFTWARE_SERVICES) -> None:
    if panel is None:
        panel = F.load_panel(u=u)
    quintile_dir = u.output_dir / "quintile"
    quintile_dir.mkdir(parents=True, exist_ok=True)

    for factor in F.FACTOR_NAMES:
        factor_dir = quintile_dir / factor
        factor_dir.mkdir(parents=True, exist_ok=True)
        wide = quintile_returns(panel, factor)
        wide.to_csv(factor_dir / "quintile_returns.csv")
        plot_cumulative(wide, factor, factor_dir / "quintile_cumulative.png")

        stats = long_short_stats(wide["Q5-Q1"])
        print(f"  {factor:<22} Q5-Q1 monthly={stats['mean_monthly']:+.4%} "
              f"(t={stats['tstat']:+.2f}, Sharpe={stats['sharpe']:+.2f}, n={stats['n_months']})")

    print(f"\nSaved quintile outputs -> {quintile_dir}")
