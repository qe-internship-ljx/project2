"""
main.py
=======

Experiment 3 driver: run every multifactor pipeline in this folder end-to-end.

Experiment 3 combines the top single factors (Experiment 2's cross-experiment
``top_factors.csv`` hand-off) into multifactor books.  Each pipeline lives in its
own module and already runs standalone via ``python <module>.py``:

    composite.py           straight-sum (equal-weight) z-score composite L/S
    weighted_composite.py  regression-premia-weighted composite L/S
    bivariate_tertile.py   double-sorted (tertile x tertile) intersection book
    factor_momentum.py     time-series factor momentum / rotation across the top factors

``factor_correlation.py`` is intentionally omitted: it is a panel-agnostic helper
(no ``main``) invoked by other experiments to quantify factor redundancy, not a
standalone pipeline.

Like Experiment 2's ``main.py``, this adds no new analytics -- it just invokes
each module's own ``main`` with its default factor set.  Each module wires
Experiment 1's engine into ``sys.modules`` at import time (via ``composite``),
so to keep every run pristine and isolated -- and to match the documented
``python <module>.py`` standalone path -- each is run in its own subprocess.  A
failure in one is reported and the rest continue.

Run standalone::

    python main.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent

# Every runnable pipeline in dependency-agnostic logical order (each is isolated
# in its own subprocess, so ordering is for readability, not correctness).
# ``factor_correlation.py`` has no ``main`` and is excluded by design.
_PIPELINES = [
    _THIS_DIR / "composite.py",
    _THIS_DIR / "weighted_composite.py",
    _THIS_DIR / "bivariate_tertile.py",
    _THIS_DIR / "factor_momentum.py",
]


def main() -> None:
    """Run every pipeline in its own subprocess, continuing past any that fail
    and reporting the roster at the end."""
    failed: list[str] = []
    for path in _PIPELINES:
        print(f"\n{'#' * 72}\n# Experiment 3 pipeline: {path.name}\n{'#' * 72}")
        result = subprocess.run([sys.executable, path.name], cwd=path.parent)
        if result.returncode != 0:
            failed.append(path.name)
            print(f"!! {path.name} exited with code {result.returncode}")

    if failed:
        print(f"\n{len(failed)} pipeline(s) FAILED: {', '.join(failed)}")
        sys.exit(1)
    print(f"\nAll {len(_PIPELINES)} Experiment 3 pipelines completed.")


if __name__ == "__main__":
    main()
