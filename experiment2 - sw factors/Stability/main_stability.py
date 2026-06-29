"""
main_stability.py
=================

Experiment 2 driver for the **earnings-stability factors**
(``stability_factors.py``).

Identical wiring to ``main_rd.py``: Experiment 1's ``quintile.py`` /
``regression.py`` / ``cost.py`` bind to their factor library via
``import factors as F``; we register ``stability_factors`` under that name in
``sys.modules`` *before* importing them, so the entire analysis -- quintile
sorts, cross-sectional (Fama-MacBeth) regressions, dollar-neutral long/short
books, industry-neutral alpha and average turnover cost -- runs against the
stability factors with zero changes to Experiment 1.  Outputs land directly in
this ``Stability/`` folder, mirroring the Experiment 1 / Experiment 2 layout
one-for-one::

    Stability/
      factor_panel.csv
      quintile/    <factor>/...   + summary.csv
                                  + long_short_market_alpha.{csv,png}
      regression/  <factor>/...   + summary.csv + summary_table.png
      factor_correlation/         redundancy of each stability factor vs ...
        vs_general/<factor>/...   ... the Exp1 general market factors,
        vs_software/<factor>/...  ... the Exp2 software factors, and
        vs_rd/<factor>/...        ... the Exp2 R&D-behaviour factors

The ``factor_correlation`` step reuses Experiment 3's panel-agnostic
``factor_correlation.run`` to quantify how little of each stability factor is
spanned by the pre-existing factors.  The ``vs_rd`` comparison is included
because both factors are second-moment cousins of ``rd_stability`` and
``rd_earning_stability`` is built directly on R&D intensity, so the R&D library
is the *nearest* benchmark for novelty.

Run standalone::

    python main_stability.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import stability_factors as S

# --- Dependency injection: make Experiment 1's analysis modules reuse the
#     stability factor library (see module docstring).
sys.modules["factors"] = S

# This driver lives in experiment2's Stability/ subfolder, so the project root is
# three parents up (Stability -> "experiment2 - sw factors" -> project root).
_EXP1_DIR = Path(__file__).resolve().parent.parent.parent / "experiment1 - general factors"
_EXP2_DIR = Path(__file__).resolve().parent.parent          # the "experiment2 - sw factors" dir
_EXP3_DIR = Path(__file__).resolve().parent.parent.parent / "experiment3 - multifactor"
sys.path.insert(0, str(_EXP1_DIR))

import quintile        # noqa: E402  (import after sys.modules / sys.path wiring)
import regression      # noqa: E402

UNIVERSE = S.SOFTWARE_SERVICES


def _load_factor_correlation():
    """Load Experiment 3's panel-agnostic factor_correlation module by path
    (its folder name contains spaces, so it cannot be imported normally)."""
    spec = importlib.util.spec_from_file_location(
        "stability_factor_correlation", _EXP3_DIR / "factor_correlation.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_correlation_analysis() -> None:
    """
    Quantify how much of each stability factor is already explained by the
    existing factors, three ways: vs Experiment 1's 9 general market factors, vs
    Experiment 2's established software factors, and vs Experiment 2's
    R&D-behaviour factors (the nearest benchmark).  Writes one R^2 table per
    (comparison, factor) under ``Stability/factor_correlation/``.
    """
    fc = _load_factor_correlation()

    target_panel = UNIVERSE.panel_path                              # Stability/factor_panel.csv
    general_panel = _EXP1_DIR / "output" / "software" / "factor_panel.csv"
    software_panel = _EXP2_DIR / "Standard" / "factor_panel.csv"
    rd_panel = _EXP2_DIR / "RD" / "factor_panel.csv"
    corr_root = UNIVERSE.output_dir / "factor_correlation"

    comparisons = [
        ("vs_general", general_panel, "Exp1 general market factors"),
        ("vs_software", software_panel, "Exp2 software factors"),
        ("vs_rd", rd_panel, "Exp2 R&D-behaviour factors"),
    ]
    for tag, market_panel, label in comparisons:
        if not market_panel.exists():
            print(f"  [skip] {label}: {market_panel} not found "
                  f"(build that experiment's panel first)")
            continue
        print(f"\n--- Redundancy of stability factors {tag.replace('_', ' ')} "
              f"({label}) ---")
        for factor in S.FACTOR_NAMES:
            fc.run(target_factor=factor,
                   target_panel=target_panel,
                   market_panel=market_panel,
                   out_dir=corr_root / tag / factor)


def main() -> None:
    print(f"=== Building stability factor panel: {UNIVERSE.slug} "
          f"(industry group: {UNIVERSE.industry_group}) ===")
    panel = S.build(save=True, u=UNIVERSE)
    n_months = panel["date"].nunique()
    n_stocks = panel["stock_id"].nunique()
    print(f"  {len(panel):,} rows | {n_stocks} stocks | {n_months} months "
          f"({panel['date'].min():%Y-%m} .. {panel['date'].max():%Y-%m})")
    print(f"  Factors: {', '.join(S.FACTOR_NAMES)}")
    print(f"  Saved -> {UNIVERSE.panel_path}\n")

    print("=== Approach 1: quintile sorts (with average trading cost) ===")
    quintile.run(panel=panel, u=UNIVERSE)

    print("\n=== Approach 2: cross-sectional regressions ===")
    regression.run(panel=panel, u=UNIVERSE)

    print("\n=== Redundancy check: stability factors vs existing factors ===")
    run_correlation_analysis()

    print(f"\nDone. All outputs under {UNIVERSE.output_dir}")


if __name__ == "__main__":
    main()
