"""
driver_utils.py
===============

Shared boilerplate for Experiment 2's per-library drivers (``monthly_position.py``
and the subfolder ``main_*.py``).  Every driver does the same four things, differing
only in *which* factor library it injects and the labels it prints:

1. register its factor library as ``sys.modules["factors"]`` and import
   Experiment 1's analysis modules against it (:func:`wire_engine`);
2. build the library's factor panel;
3. run the engine's quintile + regression pipelines on the shared panel;
4. regress each factor on the general market factors for redundancy
   (:func:`run_correlation_analysis`, reusing Experiment 3's panel-agnostic
   ``factor_correlation.py``).

:func:`run_pipeline` is steps 2-4; each driver keeps only its library import,
its universe and its labels.  The ``sys.modules["factors"]`` binding is cached
per Python process, so drivers still must not share an interpreter --
``monthly_position.py`` keeps running each subexperiment in its own subprocess.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_EXP2_DIR = Path(__file__).resolve().parent
EXP1_DIR = _EXP2_DIR.parent / "experiment1 - general factors"
EXP3_DIR = _EXP2_DIR.parent / "experiment3 - multifactor"

# Experiment 1's general market factors on the software universe -- the default
# redundancy benchmark for every software-universe factor library.
GENERAL_SOFTWARE_PANEL = EXP1_DIR / "output" / "software" / "factor_panel.csv"


def wire_engine(library):
    """Bind *library* to the ``factors`` name and import Experiment 1's
    analysis modules against it.

    ``quintile`` / ``regression`` (and ``cost``) do ``import factors as F``;
    registering the library in ``sys.modules`` first means every reference
    (FACTOR_NAMES, prepare_slice, load_panel, ...) targets the injected
    library, with zero changes to Experiment 1.  Must run before anything
    else imports those modules.  Returns ``(quintile, regression)``.
    """
    sys.modules["factors"] = library
    sys.path.insert(0, str(EXP1_DIR))
    import quintile        # noqa: E402  (import after sys.modules / sys.path wiring)
    import regression      # noqa: E402
    return quintile, regression


def load_factor_correlation():
    """Load Experiment 3's panel-agnostic factor_correlation module by path
    (its folder name contains spaces, so it cannot be imported normally)."""
    spec = importlib.util.spec_from_file_location(
        "exp2_factor_correlation", EXP3_DIR / "factor_correlation.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_correlation_analysis(library, universe, label: str,
                             market_panel=None,
                             market_desc: str = "Exp1 general market factors") -> None:
    """
    Quantify how much of each library factor is already explained by the
    general market factors.  Writes one R^2 table per factor under
    ``<universe.output_dir>/factor_correlation/<factor>/``.

    ``market_panel`` defaults to Experiment 1's software-universe panel (the
    valid benchmark for every software-universe library); a library on a
    different universe passes its own same-universe benchmark -- a path or a
    zero-argument callable returning one, built only when this step runs
    (see ``Cross_val/main_crossval.py``).
    """
    fc = load_factor_correlation()

    if market_panel is None:
        market_panel = GENERAL_SOFTWARE_PANEL
        if not market_panel.exists():
            print(f"  [skip] Exp1 general market factors: {market_panel} not found "
                  f"(build experiment1's software panel first)")
            return
    elif callable(market_panel):
        market_panel = market_panel()

    corr_root = universe.output_dir / "factor_correlation"
    print(f"\n--- Redundancy of {label}s vs general ({market_desc}) ---")
    for factor in library.FACTOR_NAMES:
        fc.run(target_factor=factor,
               target_panel=universe.panel_path,
               market_panel=market_panel,
               out_dir=corr_root / factor,
               include_market_cap=True)


def run_pipeline(library, universe, label: str, quintile, regression, *,
                 panel_report=None, market_panel=None,
                 market_desc: str = "Exp1 general market factors",
                 done_suffix: str = ""):
    """
    The shared driver flow: build the factor panel, run the quintile and
    regression pipelines on it, then the redundancy check.

    ``label`` is the singular noun used in headings (e.g. ``"R&D factor"`` ->
    "Building R&D factor panel", "R&D factors vs existing factors").
    ``panel_report(panel)`` may print extra lines under the build report
    (e.g. Cross_val's per-factor coverage counts).  Returns the built panel.
    """
    scope = (f"industry group: {universe.industry_group}" if universe.industry_group
             else ", ".join(universe.industries) if universe.industries
             else "entire market")
    print(f"=== Building {label} panel: {universe.slug} ({scope}) ===")
    panel = library.build(save=True, u=universe)
    n_months = panel["date"].nunique()
    n_stocks = panel["stock_id"].nunique()
    print(f"  {len(panel):,} rows | {n_stocks} stocks | {n_months} months "
          f"({panel['date'].min():%Y-%m} .. {panel['date'].max():%Y-%m})")
    print(f"  Factors: {', '.join(library.FACTOR_NAMES)}")
    if panel_report is not None:
        panel_report(panel)
    print(f"  Saved -> {universe.panel_path}\n")

    # Reuse the loaded panel for both analyses (avoids re-reading from disk).
    print("=== Approach 1: quintile sorts (with average trading cost) ===")
    quintile.run(panel=panel, u=universe)

    print("\n=== Approach 2: cross-sectional regressions ===")
    regression.run(panel=panel, u=universe)

    print(f"\n=== Redundancy check: {label}s vs existing factors ===")
    run_correlation_analysis(library, universe, label,
                             market_panel=market_panel, market_desc=market_desc)

    print(f"\nDone{done_suffix}. All outputs under {universe.output_dir}")
    return panel
