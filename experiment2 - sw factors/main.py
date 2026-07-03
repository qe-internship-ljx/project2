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
      quintile/    <factor>/...   + long_short_market_alpha.{csv,png}
      regression/  <factor>/...   + summary.csv + summary_table.png
      factor_correlation/<factor>/...   redundancy of each software factor vs
                                        the Exp1 general market factors

After the Standard run finishes, this driver also runs every sibling
subexperiment (``RD/``, ``Stability/``, ``Skew/``) by invoking
each one's ``main_*.py`` in its own subprocess.  Each subexperiment binds
``factors`` to its own library at import time, so they must not share an
interpreter -- a subprocess per driver keeps them isolated.  (``Cross_val/`` is
run manually *after* the top-factor collection below: its library resolves the
factors to re-test from ``factor_ranking/monthly_quintile_ranked.csv``.)

Cross-subexperiment top-factor collection
-----------------------------------------
Once every subexperiment has written its ``quintile/long_short_market_alpha.csv``,
this driver collects them, ranks every tested factor by the sum of its
full-period and 2016-onward industry-neutral alpha t-stats, and reports the
leaders (see :func:`collect_top_factors`)::

    factor_ranking/
      monthly_quintile_ranked.csv             every factor, ranked by alpha t-stat
      monthly_quintile.png                    every factor's quintile book (the
                                              quintile counterpart of tertile.py's table)

The ``monthly_quintile_ranked.csv`` ranking is the single hand-off downstream
experiments read -- each slices its own top N (Experiment 3's ``factor_momentum.py``
/ ``composite.py``, Experiment 4's timing overlays, and the ``Cross_val/`` re-test).
``Cross_val/`` is excluded from the ranking itself: it re-tests factors on the
Banks+Insurance universe, so its alphas are not comparable to a Software &
Services ranking.

Run standalone::

    python main.py                  # full pipeline + subexperiments + collection
    python main.py collect          # only (re)collect the top factors from existing CSVs
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

import sw_factors as S
import driver_utils as D

# --- Dependency injection: make Experiment 1's analysis modules reuse our
#     software-factor library (``driver_utils.wire_engine`` binds ``factors``
#     -> ``sw_factors`` before importing them, so every reference targets the
#     software factors with zero changes to Experiment 1).
quintile, regression = D.wire_engine(S)

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
    _THIS_DIR / "Stability" / "main_stability.py",
    _THIS_DIR / "Skew" / "main_skew.py",
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


# --------------------------------------------------------------------------- #
# Cross-subexperiment collection
#
# Every subexperiment writes a ``quintile/long_short_market_alpha.csv`` -- the
# same table Experiment 1 produces, one row per factor, with the project's
# "alpha t-stat" in the ``alpha_tstat`` column (the t-stat of the long/short
# book's industry-neutral alpha).  We concatenate those tables across the
# software-universe subexperiments, rank by the sum of the full-period and
# 2016-onward alpha t-stats, and both (a) render every factor's quintile book in
# the *same* table format as the per-subexperiment alpha plots and
# (b) persist the full ranking (``monthly_quintile_ranked.csv``), the single
# hand-off downstream experiments read and slice their own top N from.
#
# Experiment 1's general market factors are *also* tested on the Software &
# Services universe (``experiment1 - general factors/output/software/``), so they
# belong in the same ranking -- they are directly comparable, same cross-section
# and same "alpha t-stat" definition.  They are included as one more source.
#
# ``Cross_val/`` is intentionally excluded: it re-tests software factors on the
# Banks+Insurance universe, so its alphas are not comparable to a Software &
# Services ranking (and its rows would collide on factor name with the software
# copies).
# --------------------------------------------------------------------------- #
_SOFTWARE_SUBEXPERIMENTS = ["Standard", "RD", "Stability", "Skew"]

# Sources contributing to the cross-experiment ranking, each a
# ``quintile/long_short_market_alpha.csv`` on the Software & Services universe:
# Experiment 1's general market factors first, then every Experiment 2 software
# subexperiment.  ``(label, alpha-csv path)`` -- the label tags each row's origin
# in the ranking (the ``subexperiment`` column).
_EXP1_SOFTWARE_ALPHA = (_THIS_DIR.parent / "experiment1 - general factors"
                        / "output" / "software" / "quintile"
                        / "long_short_market_alpha.csv")
_RANKING_SOURCES: list[tuple[str, Path]] = (
    [("General", _EXP1_SOFTWARE_ALPHA)]
    + [(lib, _THIS_DIR / lib / "quintile" / "long_short_market_alpha.csv")
       for lib in _SOFTWARE_SUBEXPERIMENTS])

FACTOR_RANKING_DIR = _THIS_DIR / "factor_ranking"
TOP_N = 5


def rank_factors() -> pd.DataFrame:
    """Concatenate every Software & Services factor source's long/short alpha
    table -- Experiment 1's general market factors plus every Experiment 2
    software subexperiment -- and return one frame ranked by the *sum* of the
    full-period and 2016-onward industry-neutral alpha t-stats (descending).

    Ranking on ``alpha_tstat + alpha_tstat_2016`` rewards factors that are
    significant over the full sample *and* hold up in the recent (2016-onward)
    regime, rather than full-period significance alone.  The combined score is
    persisted in the ``alpha_tstat_combined`` column.

    Each row keeps its source (``subexperiment`` column) and a 1-based ``rank``;
    sources whose alpha table is missing are skipped with a note (so a partial
    run still ranks what exists).
    """
    tables = []
    for label, alpha_csv in _RANKING_SOURCES:
        if not alpha_csv.exists():
            print(f"  (skip {label}: {alpha_csv.name} missing -- run its driver first)")
            continue
        tbl = pd.read_csv(alpha_csv)
        tbl.insert(0, "subexperiment", label)
        tables.append(tbl)
    if not tables:
        raise FileNotFoundError(
            "No long_short_market_alpha.csv found; run Experiment 1 and the "
            "Experiment 2 subexperiments first (python main.py).")

    ranked = pd.concat(tables, ignore_index=True)
    ranked["alpha_tstat_combined"] = (
        ranked["alpha_tstat"] + ranked["alpha_tstat_2016"])
    ranked = (ranked
                .sort_values("alpha_tstat_combined", ascending=False, kind="stable")
                .reset_index(drop=True))
    ranked.insert(0, "rank", ranked.index + 1)
    return ranked


def collect_top_factors(top_n: int = TOP_N) -> pd.DataFrame:
    """Rank every tested factor, persist the full ranking
    (``monthly_quintile_ranked.csv`` -- the single hand-off downstream experiments
    slice their own top N from), and render every factor's long/short alpha table
    in the standard plot format.  Returns the top-``top_n`` rows.
    """
    ranked = rank_factors()
    FACTOR_RANKING_DIR.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(FACTOR_RANKING_DIR / "monthly_quintile_ranked.csv", index=False)

    top = ranked.head(top_n).copy()

    # Plot EVERY factor's quintile book in the SAME format as each subexperiment's
    # long_short_market_alpha.png (Experiment 1's render_alpha_table) -- the quintile
    # counterpart of tertile.py's all-factor table -- tagging each family with its
    # source subexperiment for provenance.
    plot_all = ranked.copy()
    plot_all["family"] = plot_all["family"] + "  [" + plot_all["subexperiment"] + "]"
    regression.render_alpha_table(
        plot_all, FACTOR_RANKING_DIR / "monthly_quintile.png")

    print(f"\n=== Top {top_n} factors across subexperiments "
          f"(by full-period + 2016-onward alpha t-stat) ===")
    for r in top.itertuples():
        print(f"  {r.rank}. {r.factor:<24} [{r.subexperiment:<10}] "
              f"alpha={r.alpha:+.4%}/mo  t(alpha)={r.alpha_tstat:+.2f}  "
              f"t(alpha,2016+)={r.alpha_tstat_2016:+.2f}  "
              f"t(sum)={r.alpha_tstat_combined:+.2f}")
    print(f"Saved -> {FACTOR_RANKING_DIR}")
    return top


def main() -> None:
    D.run_pipeline(S, UNIVERSE, "software factor", quintile, regression,
                   done_suffix=" (Standard)")

    # Run the remaining subexperiments (each in its own process, see above).
    run_subexperiments()

    # Collect every subexperiment's factors and report/plot the leaders.
    print(f"\n{'#' * 72}\n# Collecting top factors across subexperiments\n{'#' * 72}")
    collect_top_factors()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in {"collect", "--collect-only"}:
        collect_top_factors()
    else:
        main()
