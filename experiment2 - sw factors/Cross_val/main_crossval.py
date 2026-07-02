"""
main_crossval.py
================

Experiment 2 driver for the **cross-validation** library (``crossval_factors.py``):
re-test Experiment 2's **top ``TOP_N`` factors** (the leaders of the
``top_factors/top_factors.csv`` hand-off, resolved to their source libraries) on
the **Banks + Insurance + Commodity Producers** universe, using the *same pipeline*
as the ``Stability/`` subfolder.

Identical wiring to ``main_stability.py``: Experiment 1's ``quintile.py`` /
``regression.py`` / ``cost.py`` bind to their factor library via
``import factors as F``; we register ``crossval_factors`` under that name in
``sys.modules`` *before* importing them, so the entire analysis -- quintile
sorts, cross-sectional (Fama-MacBeth) regressions, dollar-neutral long/short
books, industry-neutral alpha and average turnover cost -- runs against the
cross-validated top factors with zero changes to Experiment 1.  Outputs land
directly in this ``Cross_val/`` folder, mirroring the Experiment 1 / Experiment 2
layout one-for-one::

    Cross_val/
      factor_panel.csv
      quintile/    <factor>/...   + summary.csv
                                  + long_short_market_alpha.{csv,png}
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

import importlib.util
import sys
from pathlib import Path

import crossval_factors as C

# --- Dependency injection: make Experiment 1's analysis modules reuse the
#     cross-validation factor library (see module docstring).
sys.modules["factors"] = C

# This driver lives in experiment2's Cross_val/ subfolder, so the project root is
# three parents up (Cross_val -> "experiment2 - sw factors" -> project root).
_EXP1_DIR = Path(__file__).resolve().parent.parent.parent / "experiment1 - general factors"
_EXP3_DIR = Path(__file__).resolve().parent.parent.parent / "experiment3 - multifactor"
sys.path.insert(0, str(_EXP1_DIR))

import quintile        # noqa: E402  (import after sys.modules / sys.path wiring)
import regression      # noqa: E402

UNIVERSE = C.BANKS_COMMODITY


def _load_factor_correlation():
    """Load Experiment 3's panel-agnostic factor_correlation module by path
    (its folder name contains spaces, so it cannot be imported normally)."""
    spec = importlib.util.spec_from_file_location(
        "crossval_factor_correlation", _EXP3_DIR / "factor_correlation.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_correlation_analysis() -> None:
    """
    Quantify how much of each cross-validated factor is already explained by the
    Experiment 1 general market factors *on the same combined universe*.
    Writes one R^2 table per factor under ``Cross_val/factor_correlation/<factor>/``.
    """
    fc = _load_factor_correlation()

    target_panel = UNIVERSE.panel_path                              # Cross_val/factor_panel.csv
    general_panel = C.build_general_market_panel()                  # built on demand, same universe
    corr_root = UNIVERSE.output_dir / "factor_correlation"

    print(f"\n--- Redundancy of cross-validated factors vs general "
          f"(Exp1 general market factors, banks+insurance+commodity universe) ---")
    for factor in C.FACTOR_NAMES:
        fc.run(target_factor=factor,
               target_panel=target_panel,
               market_panel=general_panel,
               out_dir=corr_root / factor,
               include_market_cap=True)


def main() -> None:
    scope = ", ".join(UNIVERSE.industries) if UNIVERSE.industries else "entire market"
    print(f"=== Building cross-validation factor panel: {UNIVERSE.slug} "
          f"({scope}) ===")
    panel = C.build(save=True, u=UNIVERSE)
    n_months = panel["date"].nunique()
    n_stocks = panel["stock_id"].nunique()
    print(f"  {len(panel):,} rows | {n_stocks} stocks | {n_months} months "
          f"({panel['date'].min():%Y-%m} .. {panel['date'].max():%Y-%m})")
    print(f"  Factors: {', '.join(C.FACTOR_NAMES)}")
    # Coverage per factor is itself part of the finding (rd_stability needs an R&D
    # programme, which banks, insurers and resource firms largely do not report).
    counts = panel.groupby("factor")["value"].size().reindex(C.FACTOR_NAMES)
    for name, c in counts.items():
        print(f"    {name:<22} {c:>8,} stock-months")
    print(f"  Saved -> {UNIVERSE.panel_path}\n")

    print("=== Approach 1: quintile sorts (with average trading cost) ===")
    quintile.run(panel=panel, u=UNIVERSE)

    print("\n=== Approach 2: cross-sectional regressions ===")
    regression.run(panel=panel, u=UNIVERSE)

    print("\n=== Redundancy check: cross-validated factors vs general factors ===")
    run_correlation_analysis()

    print(f"\nDone. All outputs under {UNIVERSE.output_dir}")


if __name__ == "__main__":
    main()
