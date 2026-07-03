"""
main_crossval.py
================

Experiment 2 driver for the **cross-validation** library (``crossval_factors.py``):
re-test Experiment 2's **top ``TOP_N`` factors** (the leaders of the
``top_factors/top_factors.csv`` hand-off, resolved to their source libraries) on
the **Banks + Insurance + Commodity Producers** universe, using the *same pipeline*
as the ``Stability/`` subfolder.

Same wiring as every Experiment 2 driver (shared in ``../driver_utils.py``):
Experiment 1's ``quintile.py`` / ``regression.py`` / ``cost.py`` bind to their
factor library via ``import factors as F``; ``driver_utils.wire_engine``
registers ``crossval_factors`` under that name in ``sys.modules`` *before*
importing them, so the entire analysis -- quintile sorts, cross-sectional
(Fama-MacBeth) regressions, dollar-neutral long/short books, industry-neutral
alpha and average turnover cost -- runs against the cross-validated top factors
with zero changes to Experiment 1.  Outputs land directly in this ``Cross_val/``
folder, mirroring the Experiment 1 / Experiment 2 layout one-for-one::

    Cross_val/
      factor_panel.csv
      quintile/    <factor>/...   + long_short_market_alpha.{csv,png}
      regression/  <factor>/...   + summary.csv + summary_table.png
      factor_correlation/<factor>/...   redundancy of each factor vs the Exp1
                                        general market factors on the SAME
                                        (banks_insurance) universe

Universe note
-------------
This driver runs on ``crossval_factors.BANKS_COMMODITY`` -- the union of Banks +
Insurance and Commodity Producers (``gics_industry_name`` in {Banks, Insurance,
Metals & Mining, Oil, Gas & Consumable Fuels}) -- with the same market-cap screen
the software libraries apply.  The "industry-neutral alpha" the quintile step
reports is therefore the L/S book's alpha vs the cap-weighted return of this
combined cross-section (a **universe-neutral** alpha).

Redundancy comparison
----------------------
The ``factor_correlation`` step is run **vs the Experiment 1 general market factors
built on this same combined universe** -- built on demand by
``crossval_factors.build_general_market_panel`` (reusing Experiment 1's engine) and
cached under ``Cross_val/general_market/factor_panel.csv``.  Only a same-universe
general-factor table is a valid redundancy benchmark, so the panel must share this
universe's cross-section.  (A top factor that *is* an Experiment 1 general factor --
e.g. ``gross_profitability`` -- is trivially explained by itself in this table; its
self-R^2 is expected ~1.)

Run standalone::

    python main_crossval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# The shared driver boilerplate lives one level up (the experiment2 root).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import crossval_factors as C   # noqa: E402
import driver_utils as D       # noqa: E402

# Dependency injection (see module docstring): bind Experiment 1's analysis
# modules to the cross-validation factor library, then import them.
quintile, regression = D.wire_engine(C)

UNIVERSE = C.BANKS_COMMODITY


def _coverage_report(panel) -> None:
    # Coverage per factor is itself part of the finding (rd_stability needs an R&D
    # programme, which banks, insurers and resource firms largely do not report).
    counts = panel.groupby("factor")["value"].size().reindex(C.FACTOR_NAMES)
    for name, c in counts.items():
        print(f"    {name:<22} {c:>8,} stock-months")


def main() -> None:
    D.run_pipeline(
        C, UNIVERSE, "cross-validation factor", quintile, regression,
        panel_report=_coverage_report,
        # Same-universe benchmark, built on demand only when the redundancy
        # step runs (a callable, not a path -- see driver_utils).
        market_panel=C.build_general_market_panel,
        market_desc=("Exp1 general market factors, "
                     "banks+insurance+commodity universe"))


if __name__ == "__main__":
    main()
