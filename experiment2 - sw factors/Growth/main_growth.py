"""
main_growth.py
==============

Experiment 2 driver for the **growth factors** (``growth_factors.py``).

Identical wiring to ``main_stability.py`` / ``main_rd.py``: Experiment 1's
``quintile.py`` / ``regression.py`` / ``cost.py`` bind to their factor library
via ``import factors as F``; we register ``growth_factors`` under that name in
``sys.modules`` *before* importing them, so the entire analysis -- quintile
sorts, cross-sectional (Fama-MacBeth) regressions, dollar-neutral long/short
books, industry-neutral alpha and average turnover cost -- runs against the
growth factors with zero changes to Experiment 1.  Outputs land directly in this
``Growth/`` folder, mirroring the Experiment 1 / Experiment 2 layout
one-for-one::

    Growth/
      factor_panel.csv
      quintile/    <factor>/...   + summary.csv
                                  + long_short_market_alpha.{csv,png}
      regression/  <factor>/...   + summary.csv + summary_table.png
      factor_correlation/         redundancy of each growth factor vs ...
        vs_general/<factor>/...   ... the Exp1 general market factors,
        vs_software/<factor>/...  ... the Exp2 software factors,
        vs_rd/<factor>/...        ... the Exp2 R&D-behaviour factors, and
        vs_revcost/<factor>/...   ... the Exp2 Rev & Cost factors (nearest:
                                      gross_margin level is the static cousin)

The ``factor_correlation`` step reuses Experiment 3's panel-agnostic
``factor_correlation.run`` to quantify how little of each growth factor is
spanned by the pre-existing factors.  The ``vs_revcost`` comparison is included
because ``gross_margin_expansion`` is the first difference of the Rev & Cost
``gross_margin`` *level* -- the static margin is its nearest cousin and the
natural benchmark for novelty.

Run standalone::

    python main_growth.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import growth_factors as G

# --- Dependency injection: make Experiment 1's analysis modules reuse the
#     growth factor library (see module docstring).
sys.modules["factors"] = G

# This driver lives in experiment2's Growth/ subfolder, so the project root is
# three parents up (Growth -> "experiment2 - sw factors" -> project root).
_EXP1_DIR = Path(__file__).resolve().parent.parent.parent / "experiment1 - general factors"
_EXP2_DIR = Path(__file__).resolve().parent.parent          # the "experiment2 - sw factors" dir
_EXP3_DIR = Path(__file__).resolve().parent.parent.parent / "experiment3 - multifactor"
sys.path.insert(0, str(_EXP1_DIR))

import quintile        # noqa: E402  (import after sys.modules / sys.path wiring)
import regression      # noqa: E402

UNIVERSE = G.SOFTWARE_SERVICES


def _load_factor_correlation():
    """Load Experiment 3's panel-agnostic factor_correlation module by path
    (its folder name contains spaces, so it cannot be imported normally)."""
    spec = importlib.util.spec_from_file_location(
        "growth_factor_correlation", _EXP3_DIR / "factor_correlation.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_correlation_analysis() -> None:
    """
    Quantify how much of each growth factor is already explained by the existing
    factors, four ways: vs Experiment 1's 9 general market factors, vs
    Experiment 2's established software factors, vs Experiment 2's R&D-behaviour
    factors, and vs Experiment 2's Rev & Cost factors (the nearest benchmark,
    since ``gross_margin_expansion`` is the first difference of the Rev & Cost
    ``gross_margin`` level).  Writes one R^2 table per (comparison, factor) under
    ``Growth/factor_correlation/``.
    """
    fc = _load_factor_correlation()

    target_panel = UNIVERSE.panel_path                              # Growth/factor_panel.csv
    general_panel = _EXP1_DIR / "output" / "software" / "factor_panel.csv"
    software_panel = _EXP2_DIR / "Standard" / "factor_panel.csv"
    rd_panel = _EXP2_DIR / "RD" / "factor_panel.csv"
    revcost_panel = _EXP2_DIR / "Rev & Cost" / "factor_panel.csv"
    corr_root = UNIVERSE.output_dir / "factor_correlation"

    comparisons = [
        ("vs_general", general_panel, "Exp1 general market factors"),
        ("vs_software", software_panel, "Exp2 software factors"),
        ("vs_rd", rd_panel, "Exp2 R&D-behaviour factors"),
        ("vs_revcost", revcost_panel, "Exp2 Rev & Cost factors"),
    ]
    for tag, market_panel, label in comparisons:
        if not market_panel.exists():
            print(f"  [skip] {label}: {market_panel} not found "
                  f"(build that experiment's panel first)")
            continue
        print(f"\n--- Redundancy of growth factors {tag.replace('_', ' ')} "
              f"({label}) ---")
        for factor in G.FACTOR_NAMES:
            fc.run(target_factor=factor,
                   target_panel=target_panel,
                   market_panel=market_panel,
                   out_dir=corr_root / tag / factor,
                   include_market_cap=(tag == "vs_general"))


def main() -> None:
    print(f"=== Building growth factor panel: {UNIVERSE.slug} "
          f"(industry group: {UNIVERSE.industry_group}) ===")
    panel = G.build(save=True, u=UNIVERSE)
    n_months = panel["date"].nunique()
    n_stocks = panel["stock_id"].nunique()
    print(f"  {len(panel):,} rows | {n_stocks} stocks | {n_months} months "
          f"({panel['date'].min():%Y-%m} .. {panel['date'].max():%Y-%m})")
    print(f"  Factors: {', '.join(G.FACTOR_NAMES)}")
    print(f"  Saved -> {UNIVERSE.panel_path}\n")

    print("=== Approach 1: quintile sorts (with average trading cost) ===")
    quintile.run(panel=panel, u=UNIVERSE)

    print("\n=== Approach 2: cross-sectional regressions ===")
    regression.run(panel=panel, u=UNIVERSE)

    print("\n=== Redundancy check: growth factors vs existing factors ===")
    run_correlation_analysis()

    print(f"\nDone. All outputs under {UNIVERSE.output_dir}")


if __name__ == "__main__":
    main()
