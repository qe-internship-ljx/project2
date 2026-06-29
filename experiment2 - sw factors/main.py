"""
main.py
=======

Experiment 2 driver: run the full factor pipeline on the **Software & Services**
cross-section using the established *software-industry* factors (``sw_factors.py``,
plan section 2.2) instead of the general market factors.

Like Experiment 1's per-universe drivers (``banks_insurance.py`` /
``commodity_producers.py``) this adds no new analytics.  It reuses Experiment 1's
``quintile.py`` and ``regression.py`` **unmodified** -- the quintile sorts (now
including the average-trading-cost summary), cross-sectional (Fama-MacBeth)
regressions, long/short books and every plot are exactly the same code path as
Experiment 1.  Only the *factor library* changes.

How the reuse works
-------------------
Experiment 1's analysis modules bind to their factor library via ``import
factors as F``.  We register our software-factor library under that name in
``sys.modules`` *before* importing them, so their ``F`` resolves to
``sw_factors`` -- a clean dependency injection that leaves Experiment 1 wholly
untouched.  Results land in ``Standard/``, mirroring the
Experiment 1 layout one-for-one::

    Standard/
      factor_panel.csv
      quintile/    <factor>/...   + summary.csv + summary_table.png
                                  + long_short_market_alpha.{csv,png}
      regression/  <factor>/...   + summary.csv + summary_table.png

After the Standard run finishes, this driver also runs every sibling
subexperiment (``RD/``, ``Rev & Cost/``, ``Growth/``, ``Stability/``,
``Cross_val/``) by invoking each one's ``main_*.py`` in its own subprocess.
Each subexperiment binds ``factors`` to its own library at import time, so they
must not share an interpreter -- a subprocess per driver keeps them isolated.

Run standalone::

    python main.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import sw_factors as S

# --- Dependency injection: make Experiment 1's analysis modules reuse our
#     software-factor library.  ``quintile`` / ``regression`` (and ``cost``) do
#     ``import factors as F``; binding ``factors`` -> ``sw_factors`` here means
#     every reference (FACTOR_NAMES, FACTORS, prepare_slice, load_panel, ols,
#     load_prices, ...) targets the software factors, with zero changes to
#     Experiment 1.
sys.modules["factors"] = S

_EXP1_DIR = Path(__file__).resolve().parent.parent / "experiment1 - general factors"
sys.path.insert(0, str(_EXP1_DIR))

import quintile        # noqa: E402  (import after sys.modules / sys.path wiring)
import regression      # noqa: E402

UNIVERSE = S.SOFTWARE_SERVICES

# --- Sibling subexperiments.  Each lives in its own subfolder with its own
#     factor library and `main_*.py` driver that does the same ``import factors
#     as F`` dependency injection this file does -- binding ``factors`` to *its*
#     library before importing quintile/regression.  Those module-level bindings
#     are cached per Python process, so the subexperiments cannot share one
#     interpreter without clobbering each other's ``factors`` (the first one
#     imported would win for all).  We therefore run each as its own subprocess,
#     exactly the documented ``python main_*.py`` standalone path.
_THIS_DIR = Path(__file__).resolve().parent
_SUBEXPERIMENTS = [
    _THIS_DIR / "RD" / "main_rd.py",
    _THIS_DIR / "Rev & Cost" / "main_revcost.py",
    _THIS_DIR / "Growth" / "main_growth.py",
    _THIS_DIR / "Stability" / "main_stability.py",
    _THIS_DIR / "Cross_val" / "main_crossval.py",
]


def run_subexperiments() -> None:
    """Run every subexperiment driver in its own subprocess, continuing past
    any that fail and reporting the roster at the end."""
    failed: list[str] = []
    for path in _SUBEXPERIMENTS:
        label = f"{path.parent.name}/{path.name}"
        print(f"\n{'#' * 72}\n# Subexperiment: {label}\n{'#' * 72}")
        result = subprocess.run([sys.executable, path.name], cwd=path.parent)
        if result.returncode != 0:
            failed.append(label)
            print(f"!! {label} exited with code {result.returncode}")

    if failed:
        print(f"\n{len(failed)} subexperiment(s) FAILED: {', '.join(failed)}")
    else:
        print(f"\nAll {len(_SUBEXPERIMENTS)} subexperiments completed.")


def main() -> None:
    print(f"=== Building software factor panel: {UNIVERSE.slug} "
          f"(industry group: {UNIVERSE.industry_group}) ===")
    panel = S.build(save=True, u=UNIVERSE)
    n_months = panel["date"].nunique()
    n_stocks = panel["stock_id"].nunique()
    print(f"  {len(panel):,} rows | {n_stocks} stocks | {n_months} months "
          f"({panel['date'].min():%Y-%m} .. {panel['date'].max():%Y-%m})")
    print(f"  Factors: {', '.join(S.FACTOR_NAMES)}")
    print(f"  Saved -> {UNIVERSE.panel_path}\n")

    # Reuse the loaded panel for both analyses (avoids re-reading from disk).
    print("=== Approach 1: quintile sorts (with average trading cost) ===")
    quintile.run(panel=panel, u=UNIVERSE)

    print("\n=== Approach 2: cross-sectional regressions ===")
    regression.run(panel=panel, u=UNIVERSE)

    print(f"\nDone (Standard). All outputs under {UNIVERSE.output_dir}")

    # Run the remaining subexperiments (each in its own process, see above).
    run_subexperiments()


if __name__ == "__main__":
    main()
