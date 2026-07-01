"""
tertile.py
==========

Re-evaluate **every** factor tested on the Software & Services universe -- both
Experiment 1's general market factors and every Experiment 2 software
subexperiment (Standard, RD, Rev & Cost, Growth, Stability, Skew) -- with a
**tertile** long/short book (long the top third, short the bottom third) in
place of the quintile (top / bottom fifth) book, then render the whole set in a
single alpha table in the *same* format as
``top_factors/top_factors_long_short_market_alpha.png``.

Only the *bucketing* changes.  Each factor's long/short **orientation** (which
tail is the long leg) is a property of the factor, not of the number of buckets,
and was already decided by the quintile analysis and recorded in each source's
``quintile/long_short_market_alpha.csv`` (``direction`` = ``Q5-Q1`` for
long-top / short-bottom, ``Q1-Q5`` for the reverse).  We reuse that recorded
orientation verbatim -- so this is genuinely the same books, sorted into thirds
instead of fifths -- and relabel the direction ``T3-T1`` / ``T1-T3``.

Reuse (the project's standard "load Experiment 1's engine by path" convention)
------------------------------------------------------------------------------
Everything downstream of the sort is Experiment 1 code, unmodified:

  * ``factors.prepare_slice`` already takes the bucket count, so passing ``3``
    (via :data:`N_TERTILES`) gives even tertiles instead of quintiles with no
    new sorting code;
  * ``regression.industry_monthly_return`` / ``market_regression`` /
    ``long_short_stats`` / ``beta_neutral_sharpe`` compute the industry-neutral
    alpha, its t-stat, the Sharpe and the beta-neutral Sharpe (full sample and
    2016+) exactly as the quintile pipeline does;
  * ``cost.long_short_cost`` already accepts the bucket count, so the turnover
    cost of the T3/T1 legs reuses the same machinery;
  * ``regression.render_alpha_table`` renders the PNG (a ``title`` argument was
    added, backward-compatibly, so the caption reads "tertile" not "quintile").

Because those analytics bind their factor library through ``import factors as
F``, and we want the *generic* engine helpers (slice, winsorise, cap-weighted
means -- all library-agnostic, driven purely by the panel columns), we import
Experiment 1's ``factors`` / ``cost`` / ``regression`` straight off ``sys.path``
(the engine is ``factors.py``).  Nothing here mutates the shared modules.

Run standalone::

    python tertile.py            # write top_factors/tertile_long_short_market_alpha.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# --- Load Experiment 1's engine + analysis modules by path.  The folder name
#     has spaces but the modules are plainly named (factors/cost/regression), so
#     putting the directory on sys.path lets them import normally -- and, crucially,
#     lets their own ``import factors as F`` / ``import cost`` resolve to the
#     Experiment 1 engine (we do NOT override ``sys.modules['factors']``, unlike a
#     per-library driver, because we want the generic engine, not one factor set).
_THIS_DIR = Path(__file__).resolve().parent
_EXP1_DIR = _THIS_DIR.parent / "experiment1 - general factors"
sys.path.insert(0, str(_EXP1_DIR))

import factors as F        # noqa: E402  (import after sys.path wiring)
import cost                # noqa: E402
import regression         # noqa: E402

N_TERTILES = 3
DECADE_START = regression.DECADE_START   # 2016+ ("past decade") window, shared

# --------------------------------------------------------------------------- #
# Sources -- one ``quintile/long_short_market_alpha.csv`` (for the factor list,
# family label and recorded orientation) + its ``factor_panel.csv`` (to rebuild
# the tertile books) per Software & Services factor library.  Mirrors the ranking
# sources in ``main.py`` (Experiment 1's general factors first, then every
# Experiment 2 software subexperiment); ``Cross_val/`` is likewise excluded -- it
# re-tests factors on Banks+Insurance, a different universe.
# --------------------------------------------------------------------------- #
_EXP2_LIBS = ["Standard", "RD", "Rev & Cost", "Growth", "Stability", "Skew"]


def _source(label: str, panel: Path, alpha_csv: Path) -> dict:
    return {"label": label, "panel": panel, "alpha_csv": alpha_csv}


SOURCES: list[dict] = [
    _source("General",
            _EXP1_DIR / "output" / "software" / "factor_panel.csv",
            _EXP1_DIR / "output" / "software" / "quintile"
            / "long_short_market_alpha.csv"),
] + [
    _source(lib,
            _THIS_DIR / lib / "factor_panel.csv",
            _THIS_DIR / lib / "quintile" / "long_short_market_alpha.csv")
    for lib in _EXP2_LIBS
]

OUT_PNG = _THIS_DIR / "top_factors" / "tertile_long_short_market_alpha.png"

_TITLE = ("Long-short TERTILE strategy regressed on the industry return\n"
          "(long top third / short bottom third;  ls_t = α + β·industry_t "
          "+ ε,  α = industry-neutral monthly return, t-stat tests α ≠ 0)")


# --------------------------------------------------------------------------- #
# Core: tertile book + its industry-neutral evaluation
# --------------------------------------------------------------------------- #
def tertile_spread(panel: pd.DataFrame, factor: str, sign: int) -> pd.Series:
    """Monthly dollar-neutral long/short return of the top-third-minus-bottom-third
    book for ``factor``, oriented by ``sign`` (+1 = long T3 / short T1, -1 = the
    reverse).  Reuses ``factors.prepare_slice`` with ``N_TERTILES`` buckets, so the
    legs are the same even-count sort the quintile book uses, just into thirds."""
    sub = F.prepare_slice(panel, factor, N_TERTILES)
    legs = (sub.pivot_table(index="date", columns="quintile",
                            values="next_return", aggfunc="mean")
               .sort_index())
    top, bottom = legs.get(float(N_TERTILES)), legs.get(1.0)
    if top is None or bottom is None:
        return pd.Series(dtype=float, name=f"{factor}_ls")
    spread = (top - bottom) if sign > 0 else (bottom - top)
    return spread.rename(f"{factor}_ls")


def evaluate_factor(panel: pd.DataFrame, industry_ret: pd.Series,
                    cost_panel: pd.DataFrame, factor: str, family: str,
                    direction: str) -> dict:
    """One ``render_alpha_table`` row for ``factor``: build its tertile book with
    the recorded orientation and reuse Experiment 1's regression / cost helpers to
    measure the industry-neutral alpha, Sharpe, beta-neutral Sharpe and turnover
    cost, full sample and 2016+."""
    sign = 1 if direction == "Q5-Q1" else -1
    spread = tertile_spread(panel, factor, sign)
    recent = spread.index >= DECADE_START

    ls = regression.long_short_stats(spread)
    ls_2016 = regression.long_short_stats(spread[recent])
    mreg = regression.market_regression(spread, industry_ret)
    mreg_2016 = regression.market_regression(
        spread[recent], industry_ret[industry_ret.index >= DECADE_START])
    sr_neutral = regression.beta_neutral_sharpe(spread, industry_ret, mreg["beta"])
    sr_neutral_2016 = regression.beta_neutral_sharpe(
        spread[recent], industry_ret[industry_ret.index >= DECADE_START],
        mreg_2016["beta"])
    avg_cost_pp = cost.average_cost(
        cost.long_short_cost(panel, factor, cost_panel, N_TERTILES)) * 100.0

    return {
        "factor": factor, "family": family,
        "direction": "T3-T1" if sign > 0 else "T1-T3",
        "alpha": mreg["alpha"], "alpha_tstat": mreg["alpha_tstat"],
        "sharpe": ls["sharpe"], "sharpe_neutral": sr_neutral,
        "avg_cost_pp": avg_cost_pp, "n": mreg["n"],
        "alpha_2016": mreg_2016["alpha"], "alpha_tstat_2016": mreg_2016["alpha_tstat"],
        "sharpe_2016": ls_2016["sharpe"], "sharpe_neutral_2016": sr_neutral_2016,
    }


def _load_panel(path: Path) -> pd.DataFrame:
    """Read a tidy factor panel from disk with the engine's dtype conventions
    (``factors.load_panel`` takes a Universe, not a path, so we mirror its
    post-read normalisation directly)."""
    df = pd.read_csv(path, parse_dates=["date"])
    df["stock_id"] = df["stock_id"].astype(str)
    return df


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run() -> pd.DataFrame:
    """Re-evaluate every factor across all sources with tertile books, rank by the
    sum of the full-period and 2016+ alpha t-stats (as ``main.py`` ranks the top
    factors), and render the whole table to :data:`OUT_PNG`.  Returns the ranked
    table."""
    # One cost panel for the whole run: every software library shares the same
    # Software & Services universe, so the per-(stock, month) costs are identical.
    cost_panel = cost.build_cost_panel(F.SOFTWARE_SERVICES)

    rows: list[dict] = []
    for src in SOURCES:
        if not src["alpha_csv"].exists() or not src["panel"].exists():
            print(f"  (skip {src['label']}: panel or alpha csv missing -- "
                  f"run its driver first)")
            continue
        directions = pd.read_csv(src["alpha_csv"])[["factor", "family", "direction"]]
        panel = _load_panel(src["panel"])
        industry_ret = regression.industry_monthly_return(panel)
        print(f"--- {src['label']}: {len(directions)} factors ---")
        for r in directions.itertuples(index=False):
            row = evaluate_factor(panel, industry_ret, cost_panel,
                                  r.factor, r.family, r.direction)
            row["subexperiment"] = src["label"]
            rows.append(row)
            print(f"  {row['factor']:<26} [{src['label']:<10}] {row['direction']}  "
                  f"alpha={row['alpha']:+.4%}/mo  t={row['alpha_tstat']:+.2f}  "
                  f"(2016+ t={row['alpha_tstat_2016']:+.2f})")

    if not rows:
        raise FileNotFoundError(
            "No source alpha tables found; run Experiment 1 and the Experiment 2 "
            "subexperiments first (python main.py).")

    table = pd.DataFrame(rows)
    table["alpha_tstat_combined"] = table["alpha_tstat"] + table["alpha_tstat_2016"]
    table = (table.sort_values("alpha_tstat_combined", ascending=False, kind="stable")
                  .reset_index(drop=True))

    # Render in the standard alpha-table format, tagging each family with its
    # source subexperiment for provenance -- exactly like the top_factors PNG.
    plot_rows = table.copy()
    plot_rows["family"] = plot_rows["family"] + "  [" + plot_rows["subexperiment"] + "]"
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    regression.render_alpha_table(plot_rows, OUT_PNG, title=_TITLE)

    print(f"\nSaved tertile long-short alpha table ({len(table)} factors) -> {OUT_PNG}")
    return table


if __name__ == "__main__":
    run()
