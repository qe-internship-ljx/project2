"""
main_revcost.py
===============

Experiment 2 driver for the **revenue- & cost-structure factors**
(``revcost_factors.py``).

Same wiring as every Experiment 2 driver (shared in ``../driver_utils.py``):
Experiment 1's ``quintile.py`` / ``regression.py`` / ``cost.py`` bind to their
factor library via ``import factors as F``; ``driver_utils.wire_engine``
registers ``revcost_factors`` under that name in ``sys.modules`` *before*
importing them, so the entire analysis -- even-quintile sorts, cross-sectional
(Fama-MacBeth) regressions, dollar-neutral long/short books, industry-neutral
alpha and average turnover cost -- runs against the Rev & Cost factors with
zero changes to Experiment 1.  Outputs land directly in this ``Rev & Cost/``
folder, mirroring the Experiment 1 / Experiment 2 layout one-for-one::

    Rev & Cost/
      factor_panel.csv
      quintile/    <factor>/...
                                  + long_short_market_alpha.{csv,png}
      regression/  <factor>/...   + summary.csv + summary_table.png
      factor_correlation/<factor>/...   redundancy of each Rev & Cost factor vs
                                        the Exp1 general market factors

The ``factor_correlation`` step reuses Experiment 3's panel-agnostic
``factor_correlation.run`` to quantify how little of each Rev & Cost factor is
spanned by the Exp1 general market factors -- the empirical backing for the "not
correlated with existing factors" claim in ``Theory & hypothesis.md``.

Run standalone::

    python main_revcost.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# The shared driver boilerplate lives one level up (the experiment2 root).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import revcost_factors as R   # noqa: E402
import driver_utils as D      # noqa: E402

# Dependency injection (see module docstring): bind Experiment 1's analysis
# modules to the Rev & Cost factor library, then import them.
quintile, regression = D.wire_engine(R)

UNIVERSE = R.SOFTWARE_SERVICES


def main() -> None:
    D.run_pipeline(R, UNIVERSE, "Rev & Cost factor", quintile, regression)


if __name__ == "__main__":
    main()
