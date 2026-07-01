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
      quintile/    <factor>/...   + summary.csv
      regression/  <factor>/...   + summary.csv + summary_table.png

This is the canonical entry point for the default-universe outputs (the role
``banks_insurance.py`` / ``commodity_producers.py`` play for their universes),
building the panel once and reusing it across both analyses.

Run standalone::

    python software_service.py
"""

from __future__ import annotations

import factors as F
import quintile
import regression

UNIVERSE = F.SOFTWARE_SERVICES


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

    # Reuse the loaded panel for both analyses (avoids re-reading from disk).
    print("=== Approach 1: quintile sorts ===")
    quintile.run(panel=panel, u=UNIVERSE)

    print("\n=== Approach 2: cross-sectional regressions ===")
    regression.run(panel=panel, u=UNIVERSE)

    print(f"\nDone. All outputs under {UNIVERSE.output_dir}")


if __name__ == "__main__":
    main()
