"""
quarter_position.py
====================

Re-evaluate **exactly the same set of factors** as ``tertile.py`` -- Experiment 1's
general market factors plus every Experiment 2 software subexperiment (Standard,
RD, Stability, Skew) -- with the same long/short book, but
**repositioned quarterly instead of monthly**.

Two bucketings are produced (see :data:`BUCKETINGS`), one alpha table each:

    quintile:  long the top fifth  / short the bottom fifth  (Q5-Q1)
    tertile:   long the top third  / short the bottom third  (T3-T1, per tertile.py)

so the quintile table reads directly against Experiment 1's standard monthly
quintile books and the tertile table reads directly against the monthly
``monthly_tertile.png``.

Where the monthly book re-sorts the whole cross-section every month, here new
buckets are formed only at the **end of February, May, August and November**
(:data:`REPOSITION_MONTHS`), and that membership is **held fixed for the next three
months**.  A stock enters a leg at a reposition date and stays there -- earning its
realised monthly return each held month -- until the next reposition date re-sorts
the cross-section.  So a month ``m`` belongs to the reposition formed ``(m - 2) mod
3`` months earlier (e.g. a March or April return is earned by the end-February
book; the end-May book takes over from June).

The only question this module answers is *what quarterly (rather than monthly)
repositioning does to each factor's industry-neutral alpha, Sharpe and, especially,
its turnover cost*.  Holding the membership fixed for two of every three months
trades far less, so the turnover cost falls while the signal is allowed to "stale"
between reposition dates -- the classic rebalancing-frequency trade-off.

Each factor's long/short **orientation** (which tail is the long leg) is a property
of the factor, not of the rebalancing frequency or the bucket count, and was
already decided by the quintile analysis and recorded in each source's
``quintile/long_short_market_alpha.csv`` (``direction`` = ``Q5-Q1`` for
long-top / short-bottom, ``Q1-Q5`` for the reverse).  We reuse that verbatim -- so
these are genuinely the same books, repositioned quarterly -- and relabel the
direction ``Q5-Q1`` / ``Q1-Q5`` (quintile) or ``T3-T1`` / ``T1-T3`` (tertile) after
the bucket count.

Reuse (the project's standard conventions)
------------------------------------------
* The **factor universe is not redefined here** -- we import ``tertile.py`` by path
  and reuse its ``SOURCES`` list, ``_load_panel`` helper and 2016+ window, so "the
  same set of factors as tertile.py" stays literally true even if that list
  changes.  (This is the same reuse ``experiment5 .../capacity_scaling.py`` makes;
  like it, we render one table per variant -- there over weighting schemes, here
  over bucket counts.)
* Experiment 1's engine is imported by path (``factors`` / ``cost`` /
  ``regression`` off ``sys.path`` -- the *generic* engine, driven purely by the
  panel columns; we do **not** override ``sys.modules['factors']``, exactly like
  ``tertile.py``).
* ``factors.prepare_slice`` gives the even sort at every month for any bucket count;
  we simply keep the reposition-date assignment and forward-fill it across the
  quarter (:func:`quarter_held_membership`) -- no new sorting code.
* ``regression.industry_monthly_return`` / ``market_regression`` /
  ``long_short_stats`` / ``beta_neutral_sharpe`` compute the industry-neutral
  alpha, its t-stat and the (beta-neutral) Sharpe, full sample and 2016+, exactly
  as the monthly pipeline does.
* ``cost.turnover_cost`` charges the turnover of the **actual quarterly-held**
  legs: because membership (and the equal weight within each leg) is constant
  between reposition dates, the turnover differencing charges a round-trip only at
  the reposition months and ~0 in the held months -- automatically capturing the
  lower trading of a quarterly book, with no change to the cost machinery.
* ``regression.render_alpha_table`` renders each PNG (its ``title`` argument gives
  the quarterly caption).

Nothing here mutates the shared modules or ``tertile.py``.

Run standalone::

    python quarter_position.py    # -> factor_ranking/quarter_position[_tertile]_long_short_market_alpha.png
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- #
# Load Experiment 1's engine (generic, by path) + reuse tertile.py's factor
# universe.  Putting the Experiment 1 directory on sys.path lets ``factors`` /
# ``cost`` / ``regression`` -- and their own ``import factors as F`` -- resolve to
# the shared engine.
# --------------------------------------------------------------------------- #
_THIS_DIR = Path(__file__).resolve().parent
_EXP1_DIR = _THIS_DIR.parent / "experiment1 - general factors"
sys.path.insert(0, str(_EXP1_DIR))

import factors as F        # noqa: E402  (import after sys.path wiring)
import cost                # noqa: E402
import regression          # noqa: E402


def _load_by_path(name: str, path: Path):
    """Import a module from an explicit file path (the folder name has spaces, so
    the usual ``import`` won't find it).  Used to reuse ``tertile.py`` without
    copying its source list."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)                       # type: ignore[union-attr]
    return mod


# Reuse the tertile driver's factor universe, panel loader and 2016+ window --
# single source of truth for "the same set of factors as tertile.py".  Importing it
# is side-effect free (its ``run()`` is guarded by __main__).
_tertile = _load_by_path("exp2_tertile", _THIS_DIR / "tertile.py")
SOURCES = _tertile.SOURCES
_load_panel = _tertile._load_panel
DECADE_START = _tertile.DECADE_START

# Repositioning calendar: form new buckets only at the end of Feb, May, Aug and
# Nov (month numbers 2, 5, 8, 11 -- every third month), then hold that membership
# for the two following months.  A month ``m`` belongs to the reposition formed
# ``(m - 2) mod 3`` months earlier (see :func:`quarter_held_membership`).
REPOSITION_MONTHS = (2, 5, 8, 11)


def _title(bucket_word: str, top_leg: str) -> str:
    return (f"Long-short {bucket_word.upper()} strategy repositioned QUARTERLY "
            "(end of Feb / May / Aug / Nov), regressed on the industry return\n"
            f"(long {top_leg} / short the reverse, membership held fixed 3 months;  "
            "ls_t = α + β·industry_t + ε,  α = industry-neutral monthly return, "
            "t-stat tests α ≠ 0)")


# One quarterly alpha table per bucket count -- the standard quintile book and the
# tertile book (matching ``tertile.py``).  ``run`` iterates over this list; add a
# row to test another bucketing.
BUCKETINGS = [
    {"label": "quintile", "n": 5,
     "out_png": _THIS_DIR / "factor_ranking" / "quarter_quintile.png",
     "title": _title("quintile", "top fifth")},
    {"label": "tertile", "n": 3,
     "out_png": _THIS_DIR / "factor_ranking" / "quarter_tertile.png",
     "title": _title("tertile", "top third")},
]


def _dir_label(sign: int, n: int) -> str:
    """Long/short direction label after the bucket count: ``Q5-Q1``/``Q1-Q5`` for
    quintiles (``n`` = 5), ``T3-T1``/``T1-T3`` for tertiles (``n`` = 3)."""
    p = "Q" if n >= 5 else "T"
    return f"{p}{n}-{p}1" if sign > 0 else f"{p}1-{p}{n}"


# --------------------------------------------------------------------------- #
# Quarterly-held membership + its long/short book
# --------------------------------------------------------------------------- #
def quarter_held_membership(sub: pd.DataFrame) -> pd.DataFrame:
    """
    Turn the per-month bucket slice ``sub`` (``date, stock_id, zscore, next_return,
    quintile`` for *every* month, from :func:`factors.prepare_slice`) into a
    quarterly-repositioned book: attach a ``leg`` column holding the bucket each
    stock was assigned at the most recent reposition date (end of Feb / May / Aug /
    Nov), i.e. membership held fixed within each quarter instead of re-sorted each
    month.  (The ``quintile`` column is ``prepare_slice``'s generic even-bucket
    label; for a tertile slice it already holds 1..3.)

    Each month ``m`` maps to its quarter's reposition month via ``offset = (m - 2)
    mod 3`` (0 at a reposition month, 1 and 2 in the two held months after it); the
    reposition-date buckets (``offset == 0``) are joined back onto every month of
    their quarter by ``(stock_id, formation period)``.  A row survives only if its
    stock was present and ranked at that quarter's reposition date -- a name that
    first appears mid-quarter waits for the next reposition, and one that drops out
    mid-quarter (no return that month) simply leaves the held leg -- so the returned
    legs are exactly the positions actually held each month.
    """
    sub = sub.copy()
    period = sub["date"].dt.to_period("M")
    offset = (period.dt.month - 2) % 3            # months since this quarter's reposition
    sub["form_period"] = period - offset          # the quarter's reposition month
    forms = (sub.loc[offset == 0, ["stock_id", "form_period", "quintile"]]
                .rename(columns={"quintile": "leg"}))
    return sub.merge(forms, on=["stock_id", "form_period"], how="inner")


def evaluate_factor(panel: pd.DataFrame, industry_ret: pd.Series,
                    cost_panel: pd.DataFrame, factor: str, family: str,
                    direction: str, n: int) -> dict:
    """One :func:`regression.render_alpha_table` row for ``factor``: build its
    quarterly-repositioned ``n``-bucket book with the recorded orientation and reuse
    Experiment 1's regression / cost helpers to measure the industry-neutral alpha,
    Sharpe, beta-neutral Sharpe and turnover cost, full sample and 2016+."""
    sign = 1 if direction == "Q5-Q1" else -1
    sub = F.prepare_slice(panel, factor, n)
    held = quarter_held_membership(sub)

    # Monthly leg returns of the fixed membership: equal-weighted mean next-period
    # return of whoever is held in the top / bottom bucket that month.
    legs_ret = (held.pivot_table(index="date", columns="leg",
                                 values="next_return", aggfunc="mean").sort_index())
    top, bottom = legs_ret.get(float(n)), legs_ret.get(1.0)
    spread = (pd.Series(dtype=float, name=f"{factor}_ls") if top is None or bottom is None
              else ((top - bottom) if sign > 0 else (bottom - top)).rename(f"{factor}_ls"))

    recent = spread.index >= DECADE_START
    ind_recent = industry_ret[industry_ret.index >= DECADE_START]

    ls = regression.long_short_stats(spread)
    ls_2016 = regression.long_short_stats(spread[recent])
    mreg = regression.market_regression(spread, industry_ret)
    mreg_2016 = regression.market_regression(spread[recent], ind_recent)
    # Beta-neutral Sharpe: hedge with a walk-forward -beta*industry overlay whose
    # beta is re-estimated on an expanding, look-ahead-free window (2016+ windows the
    # same hedged series, so its betas still use all prior history).
    sr_neutral = regression.beta_neutral_sharpe(spread, industry_ret)
    sr_neutral_2016 = regression.beta_neutral_sharpe(spread, industry_ret, start=DECADE_START)

    # Turnover cost of the ACTUAL quarterly-held legs: membership (and the equal
    # weight within each leg) is constant between reposition dates, so the turnover
    # differencing in ``cost.turnover_cost`` charges a round-trip only at the
    # reposition months and ~0 in the held months -- the lower trading of a
    # quarterly book falls straight out, with no change to the cost machinery.
    leg_members = held.loc[held["leg"].isin([1.0, float(n)]),
                           ["date", "stock_id", "leg"]]
    leg_members = cost.equal_weight_legs(leg_members)
    dates = pd.Index(sorted(held["date"].unique()), name="date")
    cost_series = cost.turnover_cost(leg_members, cost_panel, dates)
    avg_cost_pp = cost.average_cost(cost_series) * 100.0
    sharpe_cost = regression.net_of_cost_sharpe(spread, cost_series)
    sharpe_cost_neutral = regression.net_of_cost_neutral_sharpe(
        spread, cost_series, industry_ret)
    sharpe_cost_2016 = regression.net_of_cost_sharpe(spread, cost_series, start=DECADE_START)
    sharpe_cost_neutral_2016 = regression.net_of_cost_neutral_sharpe(
        spread, cost_series, industry_ret, start=DECADE_START)

    return {
        "factor": factor, "family": family,
        "direction": _dir_label(sign, n),
        "alpha": mreg["alpha"], "alpha_tstat": mreg["alpha_tstat"],
        "sharpe": ls["sharpe"], "sharpe_neutral": sr_neutral,
        "sharpe_cost": sharpe_cost, "sharpe_cost_neutral": sharpe_cost_neutral,
        "avg_cost_pp": avg_cost_pp, "n": mreg["n"],
        "alpha_2016": mreg_2016["alpha"], "alpha_tstat_2016": mreg_2016["alpha_tstat"],
        "sharpe_2016": ls_2016["sharpe"], "sharpe_neutral_2016": sr_neutral_2016,
        "sharpe_cost_2016": sharpe_cost_2016,
        "sharpe_cost_neutral_2016": sharpe_cost_neutral_2016,
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run(n: int, title: str, out_png: Path, label: str) -> pd.DataFrame:
    """Re-evaluate every factor across all sources with quarterly-repositioned
    ``n``-bucket books, rank by the sum of the full-period and 2016+ alpha t-stats
    (as ``main.py`` ranks the top factors), and render the whole table to
    ``out_png``.  Returns the ranked table."""
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
                                  r.factor, r.family, r.direction, n)
            row["subexperiment"] = src["label"]
            rows.append(row)
            print(f"  {row['factor']:<26} [{src['label']:<10}] {row['direction']}  "
                  f"alpha={row['alpha']:+.4%}/mo  t={row['alpha_tstat']:+.2f}  "
                  f"(2016+ t={row['alpha_tstat_2016']:+.2f})  "
                  f"cost={row['avg_cost_pp']:.4f}pp/mo")

    if not rows:
        raise FileNotFoundError(
            "No source alpha tables found; run Experiment 1 and the Experiment 2 "
            "subexperiments first (python main.py).")

    table = pd.DataFrame(rows)
    table["alpha_tstat_combined"] = table["alpha_tstat"] + table["alpha_tstat_2016"]
    table = (table.sort_values("alpha_tstat_combined", ascending=False, kind="stable")
                  .reset_index(drop=True))

    # Render in the standard alpha-table format, tagging each family with its
    # source subexperiment for provenance -- exactly like the tertile / factor_ranking PNG.
    plot_rows = table.copy()
    plot_rows["family"] = plot_rows["family"] + "  [" + plot_rows["subexperiment"] + "]"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    regression.render_alpha_table(plot_rows, out_png, title=title)

    print(f"\nSaved quarterly-repositioned {label} long-short alpha table "
          f"({len(table)} factors) -> {out_png}")
    return table


if __name__ == "__main__":
    for _b in BUCKETINGS:
        print(f"\n===== bucketing: {_b['label']} (n={_b['n']}) =====")
        run(_b["n"], _b["title"], _b["out_png"], _b["label"])
