"""
main_rd.py
==========

Experiment 2 driver for the **R&D-behavior factors** (``rd_factors.py``).

Identical wiring to ``main.py`` (the software-factor driver): Experiment 1's
``quintile.py`` / ``regression.py`` / ``cost.py`` bind to their factor library
via ``import factors as F``; we register ``rd_factors`` under that name in
``sys.modules`` *before* importing them, so the entire analysis -- even-quintile
sorts, cross-sectional (Fama-MacBeth) regressions, dollar-neutral long/short
books, industry-neutral alpha and average turnover cost -- runs against the R&D
factors with zero changes to Experiment 1.  Outputs land directly in this
``RD/`` folder, mirroring the Experiment 1 / Experiment 2 layout one-for-one::

    RD/
      factor_panel.csv
      quintile/    <factor>/...   + summary.csv
                                  + long_short_market_alpha.{csv,png}
      regression/  <factor>/...   + summary.csv + summary_table.png
      factor_correlation/         redundancy of each R&D factor vs the existing
        vs_general/<factor>/...   Exp1 general market factors, and ...
        vs_software/<factor>/...  ... the Exp2 software factors

The ``factor_correlation`` step reuses Experiment 3's panel-agnostic
``factor_correlation.run`` to quantify how little of each R&D factor is spanned
by the pre-existing factors -- the empirical backing for the "not correlated
with existing factors" claim in ``R&D factors.md``.

Run standalone::

    python main_rd.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import rd_factors as R

# --- Dependency injection: make Experiment 1's analysis modules reuse the R&D
#     factor library (see module docstring).
sys.modules["factors"] = R

# This driver lives in experiment2's RD/ subfolder, so the project root is three
# parents up (RD -> "experiment2 - sw factors" -> project root).
_EXP1_DIR = Path(__file__).resolve().parent.parent.parent / "experiment1 - general factors"
_EXP2_DIR = Path(__file__).resolve().parent.parent          # the "experiment2 - sw factors" dir
_EXP3_DIR = Path(__file__).resolve().parent.parent.parent / "experiment3 - multifactor"
sys.path.insert(0, str(_EXP1_DIR))

import quintile        # noqa: E402  (import after sys.modules / sys.path wiring)
import regression      # noqa: E402

UNIVERSE = R.SOFTWARE_SERVICES


def _load_factor_correlation():
    """Load Experiment 3's panel-agnostic factor_correlation module by path
    (its folder name contains spaces, so it cannot be imported normally)."""
    spec = importlib.util.spec_from_file_location(
        "rd_factor_correlation", _EXP3_DIR / "factor_correlation.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_correlation_analysis() -> None:
    """
    Quantify how much of each R&D factor is already explained by the existing
    factors, two ways: vs Experiment 1's 9 general market factors, and vs
    Experiment 2's established software factors.  Writes one R^2 table per (comparison,
    factor) under ``RD/factor_correlation/``.
    """
    fc = _load_factor_correlation()

    rd_panel = UNIVERSE.panel_path                                   # RD/factor_panel.csv
    general_panel = _EXP1_DIR / "output" / "software" / "factor_panel.csv"
    software_panel = _EXP2_DIR / "Standard" / "factor_panel.csv"
    corr_root = UNIVERSE.output_dir / "factor_correlation"

    comparisons = [
        ("vs_general", general_panel, "Exp1 general market factors"),
        ("vs_software", software_panel, "Exp2 software factors"),
    ]
    for tag, market_panel, label in comparisons:
        if not market_panel.exists():
            print(f"  [skip] {label}: {market_panel} not found "
                  f"(build that experiment's panel first)")
            continue
        print(f"\n--- Redundancy of R&D factors {tag.replace('_', ' ')} "
              f"({label}) ---")
        for factor in R.FACTOR_NAMES:
            fc.run(target_factor=factor,
                   target_panel=rd_panel,
                   market_panel=market_panel,
                   out_dir=corr_root / tag / factor)


def main() -> None:
    print(f"=== Building R&D factor panel: {UNIVERSE.slug} "
          f"(industry group: {UNIVERSE.industry_group}) ===")
    panel = R.build(save=True, u=UNIVERSE)
    n_months = panel["date"].nunique()
    n_stocks = panel["stock_id"].nunique()
    print(f"  {len(panel):,} rows | {n_stocks} stocks | {n_months} months "
          f"({panel['date'].min():%Y-%m} .. {panel['date'].max():%Y-%m})")
    print(f"  Factors: {', '.join(R.FACTOR_NAMES)}")
    print(f"  Saved -> {UNIVERSE.panel_path}\n")

    print("=== Approach 1: quintile sorts (with average trading cost) ===")
    quintile.run(panel=panel, u=UNIVERSE)

    print("\n=== Approach 2: cross-sectional regressions ===")
    regression.run(panel=panel, u=UNIVERSE)

    print("\n=== Redundancy check: R&D factors vs existing factors ===")
    run_correlation_analysis()

    print(f"\nDone. All outputs under {UNIVERSE.output_dir}")


if __name__ == "__main__":
    main()
