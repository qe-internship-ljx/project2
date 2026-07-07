"""
banks_insurance.py
==================

Run the full factor pipeline on the **Banks + Insurance** cross-section
(``gics_industry_name in {'Banks', 'Insurance'}``) instead of Software &
Services.

This is a thin orchestrator: it adds no new analytics. It simply points the
existing, universe-parameterised modules at :data:`factors.BANKS_INSURANCE`,
so every factor definition, z-score, quintile sort and cross-sectional
regression is exactly the same code as the default run -- only the universe
and the output location change. Like Software & Services, this universe applies
the bottom-20%-by-cap relative market-cap screen (``min_mcap_pct=0.20``), so the
cross-section tested is comparable across universes. Results land in
``output/banks_insurance/``, mirroring the default ``output/`` layout one-for-one::

    output/banks_insurance/
      factor_panel.csv
      quintile/    <factor>/...
      regression/  <factor>/...   + summary.csv + summary_table.png

Run standalone::

    python banks_insurance.py

(equivalently: ``python factors.py banks_insurance`` then
``python quintile.py banks_insurance`` then ``python regression.py banks_insurance``).
"""

from __future__ import annotations

import factors as F
import quintile
import regression

UNIVERSE = F.BANKS_INSURANCE


def main() -> None:
    print(f"=== Building factor panel: {UNIVERSE.slug} "
          f"(industries: {', '.join(UNIVERSE.industries)}) ===")
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
