"""
confidence_scaling.py
=====================

A sibling of ``capacity_scaling.py`` that changes the within-leg weighting to a
**conviction** tilt rather than a capacity one.  Where ``capacity_scaling`` weights
each name by a monotone function of its market cap, here each name is weighted by
the **softmax of its cross-sectional z-score** inside its leg -- so within the long
leg the names the factor likes *most* (highest z-score) carry the most capital, and
within the short leg the names it likes *least* (most negative z-score) do.

The book is the **quarter-HALF** portfolio of each factor (``N_HALVES`` = 2: long the
top half, short the bottom half), repositioned quarterly exactly as everywhere else
in the project (buckets formed at each end-Feb/May/Aug/Nov reposition date and held
three months via ``quarter_position.quarter_held_membership``).  The half book is the
coarsest sort -- every name is in one leg or the other -- so the softmax conviction
tilt does the work of picking out the high-conviction names that a quintile/tertile
sort would have isolated by bucketing.

Softmax conviction weight
-------------------------
Within each ``(date, leg)`` the unnormalised weight of a name is
``exp(TEMPERATURE * s_i)`` where ``s_i`` is the name's leg-oriented z-score
(``+zscore`` in the long/top leg, ``-zscore`` in the short/bottom leg, so a large
positive ``s_i`` always means "strong conviction for this leg"); softmax then
normalises the leg to sum to one.  ``TEMPERATURE`` controls how sharply capital
concentrates on the highest-conviction names (``->0`` is equal weighting, larger is
more concentrated).

Reuse
-----
Everything but the leg construction is ``capacity_scaling``'s (imported by path):
its :func:`evaluate_factor` accepts a ``legs_fn`` override, so the whole regression /
cost / beta-neutral-Sharpe / ranking / rendering pipeline -- and its factor universe
(Experiment 2's ``quarter_position.SOURCES``) -- is reused verbatim.  This module
adds only :func:`confidence_legs`.  The recorded per-factor orientation
(``Q5-Q1`` / ``Q1-Q5`` in each source's alpha CSV) is reused as-is; the rendered
half-book direction label is ``H2-H1`` / ``H1-H2``.

Output: one alpha table under ``output/confidence_scaling/`` --
``softmax_half_long_short_market_alpha.png``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import capacity_scaling as cs   # sibling driver; reuses Exp1/2 engine + factor universe

F = cs.F
_qp = cs._qp
_THIS_DIR = Path(__file__).resolve().parent

N_HALVES = 2                    # quarter-half book: long top half / short bottom half
TEMPERATURE = 0.5               # softmax sharpness on the leg-oriented z-score


def confidence_legs(panel: pd.DataFrame, factor: str, sign: int,
                    n: int = N_HALVES) -> pd.DataFrame:
    """Tidy ``date, stock_id, leg, w, next_return`` for the top and bottom halves of
    ``factor``, with each name's within-leg weight ``w`` the **softmax of its
    leg-oriented cross-sectional z-score**, normalised to sum to one inside its leg
    each month.

    The even ``n``-bucket sort and return winsorisation are the shared
    :func:`factors.prepare_slice` (which carries the ``zscore`` column), held
    **quarterly** via :func:`quarter_position.quarter_held_membership`; only the
    within-leg weighting differs from the standard equal-weighted half book.  ``sign``
    (+1 = long top / short bottom, -1 = reverse) orients which half is the long leg
    and flips the z-score so a large positive oriented score always means strong
    conviction *for that leg*.
    """
    held = _qp.quarter_held_membership(F.prepare_slice(panel, factor, n))
    legs = held.loc[held["leg"].isin([1.0, float(n)])].copy()
    legs["leg"] = np.where(legs["leg"] == float(n), "top", "bottom")

    # Leg-oriented conviction score: +zscore in the long leg, -zscore in the short
    # leg (accounting for the factor's own +/- orientation via ``sign``), so a large
    # positive score always means "strong conviction for this leg".
    long_leg = "top" if sign > 0 else "bottom"
    score = np.where(legs["leg"] == long_leg, legs["zscore"], -legs["zscore"])
    # Softmax within each (date, leg): subtract the per-group max for numerical
    # stability, exponentiate, renormalise to sum to one.
    grp = [legs["date"], legs["leg"]]
    z = TEMPERATURE * pd.Series(score, index=legs.index)
    z = z - z.groupby(grp, observed=True).transform("max")
    ex = np.exp(z)
    legs["w"] = ex / ex.groupby(grp, observed=True).transform("sum")
    return legs[["date", "stock_id", "leg", "w", "next_return"]]


def run() -> pd.DataFrame:
    """Re-evaluate every factor's quarter-half book under softmax-of-z-score
    conviction weighting and render the ranked alpha table, reusing
    ``capacity_scaling.run`` (its regression / cost / ranking / rendering pipeline)
    with :func:`confidence_legs` plugged in as the leg builder."""
    out_png = (_THIS_DIR / "output" / "confidence_scaling"
               / "softmax_half_long_short_market_alpha.png")
    title = (
        "Long-short HALF strategy (repositioned QUARTERLY) with "
        "softmax(z-score)-weighted legs, regressed on the industry return\n"
        "(long top half / short bottom half; within each leg "
        "w_i ∝ exp(z_i) on the leg-oriented z-score;  ls_t = α + β·industry_t + ε,  "
        "α = industry-neutral monthly return, t-stat tests α ≠ 0)")
    return cs.run(cs._equal_weight, title, out_png, "softmax",
                  n=N_HALVES, legs_fn=confidence_legs)


if __name__ == "__main__":
    print("\n===== confidence scaling: softmax(z-score) quarter-half book =====")
    run()
