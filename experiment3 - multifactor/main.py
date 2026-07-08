"""
main.py
=======

Experiment 3 driver: run every multifactor pipeline in this folder end-to-end.

Experiment 3 combines the top single factors (the top five of Experiment 2's
quarterly-repositioned cross-experiment ranking, ``quarter_position.ranked_factors``)
into multifactor books.  Every book here is likewise **repositioned quarterly**.
Each pipeline lives in its own module and already runs standalone via
``python <module>.py``:

    composite.py           straight-sum (equal-weight) z-score composite L/S
    weighted_composite.py  regression-premia-weighted composite L/S
    bivariate_gate.py      double-sorted (tertile x tertile) intersection book
    factor_momentum.py     time-series factor momentum / rotation across the top factors
    portfolio_overlay.py   equal-capital overlay of the top factors' univariate books

``factor_correlation.py`` is intentionally omitted: it is a panel-agnostic helper
(no ``main``) invoked by other experiments to quantify factor redundancy, not a
standalone pipeline.

Like Experiment 2's ``main.py``, this adds no new analytics -- it just invokes
each module's own ``main`` with its default factor set (for
``bivariate_gate.py`` that is both default pairs -- revenue_stability and
return_stability, each gated against gross_profitability).  Each module wires
Experiment 1's engine into ``sys.modules`` at import time (via ``composite``),
so to keep every run pristine and isolated -- and to match the documented
``python <module>.py`` standalone path -- each is run in its own subprocess.  A
failure in one is reported and the rest continue.

Before running the pipelines, this driver also renders ``output/top5_quintile_factors.png``:
the five constituent factors every multifactor book below is built from (the top
five of Experiment 2's quarterly-repositioned ranking), shown in the project's
standard long/short alpha-table format -- the Experiment 3 counterpart of
Experiment 2's ``factor_ranking/quarter_quintile.png``.

Run standalone::

    python main.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import composite as C

_THIS_DIR = Path(__file__).resolve().parent

# Every runnable pipeline in dependency-agnostic logical order (each is isolated
# in its own subprocess, so ordering is for readability, not correctness).  Each
# entry is the full argv after the interpreter: module path plus any CLI args.
# ``bivariate_gate.py`` runs both default pairs (revenue_stability and
# return_stability, each x gross_profitability) in its own ``main``.
# ``factor_correlation.py`` has no ``main`` and is excluded by design.
_PIPELINES = [
    [_THIS_DIR / "composite.py"],
    [_THIS_DIR / "weighted_composite.py"],
    [_THIS_DIR / "bivariate_gate.py"],
    [_THIS_DIR / "factor_momentum.py"],
    [_THIS_DIR / "portfolio_overlay.py"],
]


def render_top5_table() -> None:
    """Render the five constituent factors -- the top five of Experiment 2's
    cross-experiment ranking, the set every multifactor book here is built from --
    as a standard long/short alpha table (``output/top5_quintile_factors.png``).

    Reuses ``composite.ranked_factors`` (the shared quarterly top-factor hand-off)
    and Experiment 1's ``render_alpha_table``, so the table is defined identically to
    Experiment 2's ``factor_ranking/quarter_quintile.png``; each factor's family is
    tagged with its source subexperiment for provenance."""
    top = C.ranked_factors(C.TOP_N).copy()
    top["family"] = top["family"] + "  [" + top["subexperiment"] + "]"
    C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = C.OUTPUT_DIR / "top5_quintile_factors.png"
    C.R.render_alpha_table(
        top, out_png,
        title="Experiment 3 constituents -- top 5 factors of Experiment 2's "
              "quarterly-repositioned ranking\n(the set every multifactor book is "
              "built from; industry-neutral alpha of each factor's standalone "
              "quarterly quintile L/S book)")
    print(f"Saved -> {out_png}")


def main() -> None:
    """Render the constituent summary, then run every pipeline in its own
    subprocess, continuing past any that fail and reporting the roster at the end."""
    print(f"\n{'#' * 72}\n# Experiment 3 constituents: top-5 factor table\n{'#' * 72}")
    render_top5_table()

    failed: list[str] = []
    for path, *args in _PIPELINES:
        label = " ".join([path.name, *args])
        print(f"\n{'#' * 72}\n# Experiment 3 pipeline: {label}\n{'#' * 72}")
        result = subprocess.run([sys.executable, path.name, *args], cwd=path.parent)
        if result.returncode != 0:
            failed.append(label)
            print(f"!! {label} exited with code {result.returncode}")

    if failed:
        print(f"\n{len(failed)} pipeline(s) FAILED: {', '.join(failed)}")
        sys.exit(1)
    print(f"\nAll {len(_PIPELINES)} Experiment 3 pipelines completed.")


if __name__ == "__main__":
    main()
