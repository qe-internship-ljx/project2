"""
main_crossval.py
================

Experiment 2 driver for the **cross-validation** library (``crossval_factors.py``):
re-test ``rd_stability`` (from ``RD/``) and ``revenue_stability`` (from
``Rev & Cost/``) on the **Banks + Insurance** universe, using the *same pipeline*
as the ``Stability/`` subfolder.

Identical wiring to ``main_stability.py``: Experiment 1's ``quintile.py`` /
``regression.py`` / ``cost.py`` bind to their factor library via
``import factors as F``; we register ``crossval_factors`` under that name in
``sys.modules`` *before* importing them, so the entire analysis -- quintile
sorts, cross-sectional (Fama-MacBeth) regressions, dollar-neutral long/short
books, industry-neutral alpha and average turnover cost -- runs against the two
cross-validated factors with zero changes to Experiment 1.  Outputs land directly
in this ``Cross_val/`` folder, mirroring the Experiment 1 / Experiment 2 layout
one-for-one::

    Cross_val/
      factor_panel.csv
      quintile/    <factor>/...   + summary.csv
                                  + long_short_market_alpha.{csv,png}
      regression/  <factor>/...   + summary.csv + summary_table.png
      factor_correlation/
        vs_general/<factor>/...   redundancy of each factor vs the Exp1 general
                                  market factors on the SAME (banks_insurance)
                                  universe

Universe note
-------------
This driver runs on ``crossval_factors.BANKS_INSURANCE`` -- ``gics_industry_name
in {'Banks', 'Insurance'}`` -- the same universe Experiment 1 evaluates.  The
"industry-neutral alpha" the quintile step reports is therefore the L/S book's
alpha vs the cap-weighted banks+insurance ("market") return, the natural analog
of the software industry-neutral alpha.

Redundancy comparison
----------------------
The ``factor_correlation`` step is run **only vs the Experiment 1 general market
factors built on this same banks_insurance universe**
(``experiment1 - general factors/output/banks_insurance/factor_panel.csv``).
Unlike ``Stability/``, we do NOT compare against the Exp2 software / R&D / Rev&Cost
panels: those are built on the *software* cross-section, so their stock_ids do not
overlap this universe and an R^2 against them would be meaningless.  The
same-universe general-factor table is the only valid redundancy benchmark here.

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

UNIVERSE = C.BANKS_INSURANCE


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
    Experiment 1 general market factors *on the same banks_insurance universe*.
    Writes one R^2 table per factor under ``Cross_val/factor_correlation/vs_general/``.
    """
    fc = _load_factor_correlation()

    target_panel = UNIVERSE.panel_path                              # Cross_val/factor_panel.csv
    general_panel = _EXP1_DIR / "output" / "banks_insurance" / "factor_panel.csv"
    corr_root = UNIVERSE.output_dir / "factor_correlation"

    if not general_panel.exists():
        print(f"  [skip] Exp1 general factors: {general_panel} not found "
              f"(build experiment1's banks_insurance panel first)")
        return

    print(f"\n--- Redundancy of cross-validated factors vs general "
          f"(Exp1 general market factors, banks_insurance universe) ---")
    for factor in C.FACTOR_NAMES:
        fc.run(target_factor=factor,
               target_panel=target_panel,
               market_panel=general_panel,
               out_dir=corr_root / "vs_general" / factor,
               include_market_cap=True)


def main() -> None:
    print(f"=== Building cross-validation factor panel: {UNIVERSE.slug} "
          f"(industries: {', '.join(UNIVERSE.industries)}) ===")
    panel = C.build(save=True, u=UNIVERSE)
    n_months = panel["date"].nunique()
    n_stocks = panel["stock_id"].nunique()
    print(f"  {len(panel):,} rows | {n_stocks} stocks | {n_months} months "
          f"({panel['date'].min():%Y-%m} .. {panel['date'].max():%Y-%m})")
    print(f"  Factors: {', '.join(C.FACTOR_NAMES)}")
    # Coverage per factor is itself part of the finding (rd_stability needs an R&D
    # programme, which banks/insurers rarely report).
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
