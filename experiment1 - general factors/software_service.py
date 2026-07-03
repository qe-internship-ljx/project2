"""
software_service.py
===================

Run the full factor pipeline on the **Software & Services** cross-section
(``gics_industry_group_name == 'Software & Services'``) -- the project's
default universe.

This is a thin orchestrator: it adds no new analytics. It simply points the
existing, universe-parameterised modules at :data:`factors.SOFTWARE_SERVICES`,
so every factor definition, z-score, quintile sort and cross-sectional
regression is exactly the same code as every other universe -- only the
universe and the output location change. Results land in ``output/software/``,
mirroring the ``banks_insurance`` / ``commodity_producers`` layout one-for-one::

    output/software/
      factor_panel.csv
      quintile/    <factor>/...
      regression/  <factor>/...   + summary.csv + summary_table.png

This is the canonical entry point for the default-universe outputs (the role
``banks_insurance.py`` / ``commodity_producers.py`` play for their universes),
building the panel once and reusing it across both analyses.

Run standalone::

    python software_service.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import factors as F
import quintile
import regression

UNIVERSE = F.SOFTWARE_SERVICES


def plot_universe_size(u: "F.Universe" = UNIVERSE) -> Path:
    """
    Plot the size of the software universe over time: the number of *active*
    stocks each month (any name with price data that month) alongside the number
    that survive the investable market-cap screen -- the point-in-time
    ``factors.apply_mcap_screen`` (flat and/or relative floor) applied before the sorts.

    Counts come from the pre-screen monthly panel (``build_monthly_panel``), so
    "active" is the full cross-section and "above threshold" is the investable
    subset held by the strategy.  Writes ``output/software/universe_size.png``.
    """
    monthly = F.build_monthly_panel(u=u)
    active = monthly.groupby("period", observed=True)["stock_id"].nunique()
    # Investable subset = names surviving whatever market-cap screens the universe
    # switches on (flat floor and/or relative per-month floor) -- reuse the same
    # screen the strategy applies so this count tracks it automatically.
    above = (F.apply_mcap_screen(monthly, u)
             .groupby("period", observed=True)["stock_id"].nunique()
             .reindex(active.index, fill_value=0))
    active.index = active.index.to_timestamp(how="end")
    above.index = above.index.to_timestamp(how="end")

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(active.index, active.values, label="Active stocks", linewidth=1.3)
    ax.plot(above.index, above.values, linewidth=1.3,
            label="Investable (after market-cap screen)")
    ax.set_title("Software & Services universe size over time")
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of stocks")
    ax.set_ylim(bottom=0)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    u.output_dir.mkdir(parents=True, exist_ok=True)
    path = u.output_dir / "universe_size.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_smallest_mcap(u: "F.Universe" = UNIVERSE) -> Path:
    """
    Plot the smallest USD market cap among *active* stocks each month, before and
    after the bottom-20% relative screen.

    "Active" is the full pre-screen monthly cross-section (``build_monthly_panel``)
    restricted to names with an established USD cap.  The two lines are:
      * **without the bottom-20% filter** -- the min cap over all active names, i.e.
        the very smallest name in the universe that month.
      * **with the bottom-20% filter** -- the min cap over the names that survive
        :func:`factors.apply_mcap_screen`; because the relative floor drops the
        lowest ``min_mcap_pct`` by cap each month, this is the effective size floor
        the strategy actually trades at.

    Caps are shown in USD billions.  Writes ``output/software/smallest_mcap.png``.
    """
    monthly = F.build_monthly_panel(u=u)
    smallest_all = (monthly.groupby("period", observed=True)["security_mcap_usd"]
                    .min())
    smallest_screened = (F.apply_mcap_screen(monthly, u)
                         .groupby("period", observed=True)["security_mcap_usd"]
                         .min()
                         .reindex(smallest_all.index))
    smallest_all.index = smallest_all.index.to_timestamp(how="end")
    smallest_screened.index = smallest_screened.index.to_timestamp(how="end")

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(smallest_all.index, smallest_all.values / 1e9, linewidth=1.3,
            label="Smallest active (no filter)")
    ax.plot(smallest_screened.index, smallest_screened.values / 1e9, linewidth=1.3,
            label="Smallest after bottom-20% filter")
    ax.set_title("Software & Services: smallest active market cap over time")
    ax.set_xlabel("Month")
    ax.set_ylabel("Market cap (USD billions)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    u.output_dir.mkdir(parents=True, exist_ok=True)
    path = u.output_dir / "smallest_mcap.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main() -> None:
    label = UNIVERSE.industry_group or ", ".join(UNIVERSE.industries)
    print(f"=== Building factor panel: {UNIVERSE.slug} "
          f"(industry group: {label}) ===")
    panel = F.build(save=True, u=UNIVERSE)
    n_months = panel["date"].nunique()
    n_stocks = panel["stock_id"].nunique()
    print(f"  {len(panel):,} rows | {n_stocks} stocks | {n_months} months "
          f"({panel['date'].min():%Y-%m} .. {panel['date'].max():%Y-%m})")
    print(f"  Saved -> {UNIVERSE.panel_path}\n")

    print("=== Universe size over time ===")
    size_path = plot_universe_size(UNIVERSE)
    print(f"  Saved -> {size_path}\n")

    print("=== Smallest active market cap over time ===")
    mcap_path = plot_smallest_mcap(UNIVERSE)
    print(f"  Saved -> {mcap_path}\n")

    # Reuse the loaded panel for both analyses (avoids re-reading from disk).
    print("=== Approach 1: quintile sorts ===")
    quintile.run(panel=panel, u=UNIVERSE)

    print("\n=== Approach 2: cross-sectional regressions ===")
    regression.run(panel=panel, u=UNIVERSE)

    print(f"\nDone. All outputs under {UNIVERSE.output_dir}")


if __name__ == "__main__":
    main()
