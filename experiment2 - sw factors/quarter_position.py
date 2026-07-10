"""
quarter_position.py
====================

Re-evaluate **exactly the same set of factors** as ``monthly_position.py`` --
Experiment 1's general market factors plus every Experiment 2 software subexperiment
(Literature, RD, Stability, Skew) -- with the same long/short book, but
**repositioned quarterly instead of monthly**.

Three bucketings are produced (see :data:`BUCKETINGS`), one alpha table each:

    quintile:  long the top fifth  / short the bottom fifth  (Q5-Q1)
    tertile:   long the top third  / short the bottom third  (T3-T1)
    half:      long the top half   / short the bottom half   (H2-H1)

so each table reads directly against ``monthly_position.py``'s monthly counterpart
(``monthly_quintile.png`` / ``monthly_tertile.png`` / ``monthly_half.png``).

Alongside the long/short books, this module also runs the **regression pipeline of
``monthly_position.py``, but quarterly** (:func:`run_regression`): at every reposition month
each factor's next-quarter return is cross-sectionally regressed on its factor
z-score, and the Fama-MacBeth t-stat of the resulting quarterly-alpha series is
reported full period and 2016+, ranked by the sum of the two t-stats and rendered as
a single PNG ``Factor Ranking/quarter_regression.png`` (the quarterly counterpart of
each subexperiment's ``regression/summary_table.png``).

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
direction after the bucket count (``Q5-Q1`` quintile / ``T3-T1`` tertile / ``H2-H1``
half, via ``monthly_position.dir_label``).

Reuse (the project's standard conventions)
------------------------------------------
* The **factor universe and shared evaluation are not redefined here** -- we import
  ``monthly_position.py`` by path and reuse its ``SOURCES`` list, ``_load_panel``
  helper, 2016+ window, direction labels and, crucially, its ``evaluate_all`` ranking
  driver, ``alpha_row`` (the shared alpha/Sharpe/cost row) and ``render_ranked``.  So
  "the same set of factors and the same statistics as the monthly re-evaluation"
  stays literally true even if those change; only the *book* (quarterly instead of
  monthly) differs here.  (Same reuse ``experiment5 .../capacity_scaling.py`` makes.)
* Experiment 1's engine is imported by path (``factors`` / ``cost`` /
  ``regression`` off ``sys.path`` -- the *generic* engine, driven purely by the
  panel columns; we do **not** override ``sys.modules['factors']``).
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

Nothing here mutates the shared modules or ``monthly_position.py``.

The single downstream hand-off
------------------------------
This module is also the **factor-selection hand-off** every downstream experiment
reads.  Running it persists each bucketing's ranked table to
``Factor Ranking/quarter_{label}_ranked.csv`` (quintile / tertile / half), and
:func:`ranked_factors` reads that CSV back -- top-``n`` or the whole set -- so a
downstream process resolves the leaders with a plain ``read_csv`` (no engine wiring,
no whole-universe re-evaluation per process).  Experiments 3-5 slice their
constituents from the *quarterly* book they will actually trade, and the
``cross_val/`` re-test reads the same CSV.  If the CSV is missing (this module was
never run standalone) :func:`ranked_factors` falls back to computing and persisting
it once, so the hand-off is self-bootstrapping.  Because the whole project now
repositions quarterly, the per-factor
quarterly long/short book itself is a shared primitive: :func:`quarter_held_spread`
(a factor's oriented top-minus-bottom return) and :func:`quarter_held_legs` (its
equal-weighted leg membership, for cost / ownership) are computed here from
:func:`quarter_held_membership` and reused by ``composite.py`` and every module that
builds on it -- so "repositioned quarterly" means one implementation, called
everywhere.

Run standalone::

    python quarter_position.py    # -> Factor Ranking/quarter_{quintile,tertile,half}.png
                                  #    + quarter_{quintile,tertile,half}_ranked.csv
                                  #      (the persisted ranking every downstream experiment reads)
                                  #    + quarter_regression.png (quarterly cross-section regression)
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Load Experiment 1's engine (generic, by path) + reuse the monthly re-evaluation's
# shared machinery.  Putting the Experiment 1 directory on sys.path lets ``factors``
# / ``cost`` / ``regression`` -- and their own ``import factors as F`` -- resolve to
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
    the usual ``import`` won't find it).  Used to reuse ``monthly_position.py``
    without copying its source list or shared evaluation."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)                       # type: ignore[union-attr]
    return mod


# Reuse the monthly re-evaluation's factor universe, panel loader, 2016+ window,
# direction labels and shared evaluation driver -- single source of truth for "the
# same set of factors and statistics as the monthly re-evaluation".  Importing it is
# side-effect free (its ``main()`` / ``run_all()`` are guarded by __main__).
_monthly = _load_by_path("exp2_monthly_position", _THIS_DIR / "monthly_position.py")
SOURCES = _monthly.SOURCES
_load_panel = _monthly._load_panel
DECADE_START = _monthly.DECADE_START
_dir_label = _monthly.dir_label       # shared bucket-count direction labels (Q/T/H)
_alpha_row = _monthly.alpha_row       # shared alpha/Sharpe/cost row from a book
_evaluate_all = _monthly.evaluate_all  # shared source-iteration + ranking driver
_render_ranked = _monthly.render_ranked

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


# One quarterly alpha table per bucket count -- the standard quintile book, the
# tertile book and the half book (each matching ``monthly_position.py``'s monthly
# counterpart).  ``run`` iterates over this list; add a row to test another
# bucketing.  Each also names the CSV ``run`` persists the ranked table to -- the
# single factor-selection hand-off :func:`ranked_factors` (hence every downstream
# experiment) reads back, instead of recomputing the whole-universe re-evaluation
# in each process.
def _ranked_csv(label: str) -> Path:
    return _THIS_DIR / "Factor Ranking" / f"quarter_{label}_ranked.csv"


BUCKETINGS = [
    {"label": "quintile", "n": 5,
     "out_png": _THIS_DIR / "Factor Ranking" / "quarter_quintile.png",
     "ranked_csv": _ranked_csv("quintile"),
     "title": _title("quintile", "top fifth")},
    {"label": "tertile", "n": 3,
     "out_png": _THIS_DIR / "Factor Ranking" / "quarter_tertile.png",
     "ranked_csv": _ranked_csv("tertile"),
     "title": _title("tertile", "top third")},
    {"label": "half", "n": 2,
     "out_png": _THIS_DIR / "Factor Ranking" / "quarter_half.png",
     "ranked_csv": _ranked_csv("half"),
     "title": _title("half", "top half")},
]


# --------------------------------------------------------------------------- #
# Quarterly-held membership + its long/short book
# --------------------------------------------------------------------------- #
def quarter_hold(frame: pd.DataFrame, bucket_cols: list[str]) -> pd.DataFrame:
    """
    Hold the given ``bucket_cols`` fixed within each quarter: replace them, on every
    month of a quarter, with the value each stock carried at that quarter's
    **reposition month** (end of Feb / May / Aug / Nov), instead of the per-month
    value.  The generic quarterly-repositioning primitive -- one bucket column for a
    single sort (:func:`quarter_held_membership`), two for an independent double sort.

    Each month ``m`` maps to its quarter's reposition month via ``offset = (m - 2)
    mod 3`` (0 at a reposition month, 1 and 2 in the two held months after it); the
    reposition-date rows (``offset == 0``) are joined back onto every month of their
    quarter by ``(stock_id, formation period)``.  A row survives only if its stock
    was present and ranked at that quarter's reposition date -- a name that first
    appears mid-quarter waits for the next reposition, and one that drops out
    mid-quarter (no row that month) simply leaves the held book -- so the result is
    exactly the positions actually held each month, with no look-ahead.
    """
    frame = frame.copy()
    period = frame["date"].dt.to_period("M")
    offset = (period.dt.month - 2) % 3            # months since this quarter's reposition
    frame["form_period"] = period - offset        # the quarter's reposition month
    forms = frame.loc[offset == 0, ["stock_id", "form_period", *bucket_cols]]
    return (frame.drop(columns=list(bucket_cols))
                 .merge(forms, on=["stock_id", "form_period"], how="inner"))


def quarter_held_membership(sub: pd.DataFrame) -> pd.DataFrame:
    """
    Turn the per-month bucket slice ``sub`` (``date, stock_id, zscore, next_return,
    quintile`` for *every* month, from :func:`factors.prepare_slice`) into a
    quarterly-repositioned book: a ``leg`` column holding the bucket each stock was
    assigned at the most recent reposition date, i.e. membership held fixed within
    each quarter instead of re-sorted each month.  A thin wrapper over
    :func:`quarter_hold` on the single generic even-bucket ``quintile`` label
    (``prepare_slice``'s 1..n column; for a tertile slice it already holds 1..3),
    renamed to ``leg``.
    """
    return quarter_hold(sub.rename(columns={"quintile": "leg"}), ["leg"])


def quarter_held_spread(panel: pd.DataFrame, factor: str, sign: int,
                        n: int = 5) -> pd.Series:
    """A factor's **quarterly-repositioned** dollar-neutral long/short return: the
    equal-weighted mean next-period return of whoever is held in the top bucket minus
    the bottom bucket of its quarterly-held ``n``-bucket book, oriented by ``sign``
    (+1 = long top / short bottom, -1 = the reverse).  Indexed by month.

    The single shared primitive behind every quarterly book downstream (the composite
    constituents' benchmark books, the momentum-rotation candidates, the timing
    overlays' base book): sorts with :func:`factors.prepare_slice`, holds membership
    with :func:`quarter_held_membership`, so there is one implementation of "the
    factor's quarterly L/S book" for the whole project."""
    held = quarter_held_membership(F.prepare_slice(panel, factor, n))
    legs_ret = (held.pivot_table(index="date", columns="leg",
                                 values="next_return", aggfunc="mean").sort_index())
    top, bottom = legs_ret.get(float(n)), legs_ret.get(1.0)
    if top is None or bottom is None:
        return pd.Series(dtype=float, name=f"{factor}_ls")
    return ((top - bottom) if sign > 0 else (bottom - top)).rename(f"{factor}_ls")


def quarter_held_legs(panel: pd.DataFrame, factor: str, n: int = 5) -> pd.DataFrame:
    """Equal-weighted top / bottom leg membership (``date, stock_id, leg, w``) of a
    factor's quarterly-repositioned ``n``-bucket book -- the same held membership
    :func:`quarter_held_spread` trades, for the turnover cost and ownership
    diagnostics.  Because membership (and the equal weight within each leg) is
    constant between reposition dates, ``cost.turnover_cost`` charges a round-trip
    only at the reposition months and ~0 in the held months."""
    held = quarter_held_membership(F.prepare_slice(panel, factor, n))
    legs = held.loc[held["leg"].isin([1.0, float(n)]), ["date", "stock_id", "leg"]]
    return cost.equal_weight_legs(legs)


# --------------------------------------------------------------------------- #
# Quarterly cross-section regression (the regression pipeline of monthly_position.py,
# but quarterly)
#
# The per-library regression pipeline re-estimates one cross-sectional OLS per *month*
# (``regression.monthly_regressions``) and Fama-MacBeths the per-month slope -- the
# factor premium reported in ``regression/summary_table.png`` (mean β + FM t-stat).
# Here we do the same thing on the *quarterly* calendar this module already trades:
# only at a reposition month (end of Feb/May/Aug/Nov) do we regress every stock's
# **next-quarter** return on its factor z-score, giving one quarterly slope (β, the
# return per 1 cross-sectional sigma of the factor) per reposition date.
# Fama-MacBeth-ing that quarterly β series yields the "quarterly alpha" t-stat --
# full period and 2016+ -- exactly the per-library summary-table statistic, but
# on the held-quarterly book's own calendar.
# --------------------------------------------------------------------------- #
def next_quarter_return(sub: pd.DataFrame) -> pd.DataFrame:
    """Attach each stock's **next-quarter** return to the per-month slice ``sub``
    (``date, stock_id, zscore, next_return``, from :func:`factors.prepare_slice`):
    the 3-month forward return compounded from the monthly ``next_return`` over the
    quarter that starts at each reposition month (end of Feb/May/Aug/Nov).

    ``next_return`` at month t is the return realised in month t+1, so compounding it
    over the reposition month ``m`` and the two months after it gives the return over
    months ``m+1..m+3`` -- the next quarter, i.e. exactly the return the
    quarterly-held book earns before the next reposition.  A stock keeps a
    next-quarter return only if it is present in all three months of the quarter (a
    name that leaves mid-quarter drops out, no look-ahead).  Returns the
    reposition-date rows (``date, stock_id, zscore, q_return``)."""
    # Gross next-quarter return: (1+r_m)(1+r_{m+1})(1+r_{m+2}) per stock, where r is
    # the monthly ``next_return``.  Join on the calendar period (m, m+1, m+2), not a
    # positional shift, so a stock with a gap in its monthly history is not compounded
    # across non-adjacent months -- a name absent in any of the three months simply
    # drops out (same present-in-all-three, no-look-ahead rule the held book uses).
    s = sub[["date", "stock_id", "zscore", "next_return"]].copy()
    s["period"] = s["date"].dt.to_period("M")
    forms = s[(s["period"].dt.month - 2) % 3 == 0].copy()   # reposition-date rows
    # Growth factor (1 + next_return) at each stock-month, to be joined onto the
    # formation row at period m by relabelling it to the formation month (m+k -> m).
    grow = s[["stock_id", "period", "next_return"]].assign(g=lambda d: 1.0 + d["next_return"])
    forms = forms.reset_index(drop=True)
    q_gross = np.ones(len(forms))
    for k in (0, 1, 2):
        gk = grow.assign(period=grow["period"] - k).rename(columns={"g": f"g{k}"})
        forms = forms.merge(gk[["stock_id", "period", f"g{k}"]],   # left merge keeps row order
                            on=["stock_id", "period"], how="left")
        q_gross = q_gross * forms[f"g{k}"].to_numpy()
    return (forms.assign(q_return=q_gross - 1.0)[["date", "stock_id", "zscore", "q_return"]]
                 .dropna(subset=["q_return"]))


def quarterly_slopes(panel: pd.DataFrame, factor: str) -> pd.Series:
    """The factor's series of **quarterly cross-section regression** slopes: at every
    reposition month, OLS-regress each stock's next-quarter return on its factor
    z-score (:func:`factors.ols`) and keep the slope (β = the factor's return per 1
    cross-sectional sigma that quarter).  Indexed by reposition month.  The quarterly
    analogue of ``regression.monthly_regressions``' per-month β."""
    nq = next_quarter_return(F.prepare_slice(panel, factor, assign_q=False))
    betas = {date: F.ols(g["zscore"].to_numpy(), g["q_return"].to_numpy())["beta"]
             for date, g in nq.groupby("date", observed=True)}
    return pd.Series(betas, name=f"{factor}_qbeta").sort_index()


def evaluate_regression(panel: pd.DataFrame, factor: str, family: str) -> dict:
    """One quarterly-cross-section-regression row for ``factor``: its quarterly-slope
    series (:func:`quarterly_slopes`) summarised into the mean β and its Fama-MacBeth
    t-stat -- the "quarterly alpha" and its t-stat -- full period and 2016+ (reusing
    ``regression.fama_macbeth``, the same aggregation ``summary_table.png`` uses)."""
    beta = quarterly_slopes(panel, factor)
    full = regression.fama_macbeth(beta)
    recent = regression.fama_macbeth(beta[beta.index >= DECADE_START])
    return {"factor": factor, "family": family,
            "alpha_q": full["mean_beta"], "alpha_tstat": full["fm_tstat"],
            "n_q": full["n_months"],
            "alpha_q_2016": recent["mean_beta"], "alpha_tstat_2016": recent["fm_tstat"],
            "n_q_2016": recent["n_months"]}


def render_regression_table(rows: pd.DataFrame, path: Path) -> None:
    """Render the quarterly cross-section regression as a single PNG (the quarterly
    counterpart of ``regression.summary_table.png``): one row per factor with the
    mean quarterly β (the factor premium -- next-quarter return per 1 cross-sectional
    sigma) and its Fama-MacBeth t-stat -- the "quarterly alpha" and its t-stat --
    full period and 2016+, ranked by the sum of the two t-stats.  The t-stats are
    shaded by significance, reusing Experiment 1's ``regression._tstat_color`` --
    same palette as every other table."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = ["factor", "family", "alpha_q", "alpha_tstat",
            "alpha_q_2016", "alpha_tstat_2016", "alpha_tstat_combined"]
    headers = ["Factor", "Family", "Mean β\n(quarterly)", "α t-stat\n(full)",
               "Mean β\n(2016+)", "α t-stat\n(2016+)", "α t-stat\n(sum)"]

    cell_text, cell_colors = [], []
    for _, r in rows[cols].iterrows():
        cell_text.append([
            r["factor"], r["family"],
            f"{r['alpha_q']:+.4f}", f"{r['alpha_tstat']:+.2f}",
            f"{r['alpha_q_2016']:+.4f}", f"{r['alpha_tstat_2016']:+.2f}",
            f"{r['alpha_tstat_combined']:+.2f}",
        ])
        cell_colors.append([
            "white", "white", "white",
            regression._tstat_color(r["alpha_tstat"]),
            "white", regression._tstat_color(r["alpha_tstat_2016"]),
            regression._tstat_color(r["alpha_tstat_combined"]),
        ])

    n = len(rows)
    fig, ax = plt.subplots(figsize=(13, 0.45 * (n + 1) + 1.4))
    ax.axis("off")
    tbl = ax.table(cellText=cell_text, colLabels=headers, cellColours=cell_colors,
                   colWidths=[0.17, 0.33, 0.10, 0.10, 0.10, 0.10, 0.10],
                   cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for j in range(len(headers)):                       # bold header row
        tbl[0, j].set_text_props(weight="bold", color="white")
        tbl[0, j].set_facecolor("#404040")
    for i in range(1, n + 1):                           # left-align text cols
        tbl[i, 0].set_text_props(ha="left")
        tbl[i, 1].set_text_props(ha="left")

    fig.suptitle("Quarterly cross-section regression: mean quarterly β (factor "
                 "premium) and its Fama-MacBeth t-stat\n(each reposition month "
                 "regress next-quarter return on factor z-score;  β = return per 1σ "
                 "of the factor;  ranked by full + 2016+ α t-stat)", fontsize=11, y=0.99)
    fig.text(0.5, 0.015, "Shaded α t-stats: |t| ≥ 1.65 (10%), darker |t| ≥ 2.0 (5%).  "
             "β = next-quarter industry return per 1 std of the factor.  "
             "2016+ columns re-estimate on the past decade only.",
             ha="center", fontsize=8, color="#555555")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.80, bottom=0.10)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate_factor(panel: pd.DataFrame, industry_ret: pd.Series,
                    cost_panel: pd.DataFrame, factor: str, family: str,
                    direction: str, n: int) -> dict:
    """One shared :func:`monthly_position.alpha_row` for ``factor``'s
    **quarterly-repositioned** ``n``-bucket book with the recorded orientation: build
    the top-minus-bottom spread and the actual quarterly-held turnover cost (the
    shared :func:`quarter_held_spread` / :func:`quarter_held_legs` primitives), then
    measure the same alpha/Sharpe/cost stats as the monthly re-evaluation.

    Turnover cost is charged on the ACTUAL quarterly-held legs: membership is constant
    between reposition dates, so ``cost.turnover_cost``'s differencing charges a
    round-trip only at the reposition months and ~0 in the held months -- the lower
    trading of a quarterly book falls straight out, with no change to the machinery."""
    sign = 1 if direction == "Q5-Q1" else -1
    spread = quarter_held_spread(panel, factor, sign, n)

    leg_members = quarter_held_legs(panel, factor, n)
    dates = pd.Index(sorted(leg_members["date"].unique()), name="date")
    cost_series = cost.turnover_cost(leg_members, cost_panel, dates)

    return _alpha_row(spread, cost_series, industry_ret, factor, family, sign, n)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
_RANK_CACHE: dict[str, pd.DataFrame] = {}


def rank(n: int, label: str) -> pd.DataFrame:
    """Re-evaluate every factor across all sources with quarterly-repositioned
    ``n``-bucket books and rank by the sum of the full-period and 2016+ net-of-cost
    beta-neutral Sharpe ratios.  Returns the ranked table (no rendering).

    Delegates the source-iteration + ranking to the shared
    :func:`monthly_position.evaluate_all`, passing this module's quarterly-book
    :func:`evaluate_factor` -- so the monthly and quarterly re-evaluations share one
    ranking driver and differ only in the book each builds.

    Cached by ``label`` so the (expensive) whole-universe re-evaluation runs at most
    once per bucketing per process.  :func:`run` computes it, persists it to the
    hand-off CSV and renders the PNG; :func:`ranked_factors` normally reads that CSV
    back, calling this only to (re)build it when the CSV is missing."""
    if label not in _RANK_CACHE:
        _RANK_CACHE[label] = _evaluate_all(evaluate_factor, n)
    return _RANK_CACHE[label]


def run(n: int, title: str, out_png: Path, label: str,
        ranked_csv: Path) -> pd.DataFrame:
    """Rank every factor's quarterly-repositioned ``n``-bucket book (:func:`rank`),
    **persist the ranked table to** ``ranked_csv`` -- the single factor-selection
    hand-off every downstream experiment reads back (:func:`ranked_factors`) instead
    of recomputing -- and render the whole table to ``out_png`` in the standard
    alpha-table format.  Returns the ranked table."""
    table = rank(n, label)

    ranked_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(ranked_csv, index=False)

    # Render in the standard alpha-table format via the shared helper (family-tagged
    # by source subexperiment, exactly like the monthly / Factor Ranking PNGs).
    _render_ranked(table, out_png, title)

    print(f"\nSaved quarterly-repositioned {label} long-short alpha table "
          f"({len(table)} factors) -> {out_png}\n"
          f"  hand-off ranking -> {ranked_csv}")
    return table


def ranked_factors(n: int | None = None, label: str = "quintile") -> pd.DataFrame:
    """The quarterly-repositioned factor ranking -- **the single hand-off** every
    downstream experiment reads (Experiments 3-5 and ``cross_val/``).

    Reads the ranked table straight from the CSV ``run`` persisted
    (``Factor Ranking/quarter_{label}_ranked.csv``) -- so a downstream process resolves
    the leaders with a plain ``read_csv``, without re-running the whole-universe
    re-evaluation (no engine wiring needed).  If that CSV is missing (this module was
    never run standalone), it falls back to computing the ranking with :func:`rank`
    and persisting it, so the hand-off is self-bootstrapping.

    ``label`` picks the bucketing (a :data:`BUCKETINGS` label, default the
    ``quintile`` book Experiments 3-5 select constituents from; ``composite.py`` also
    reads the ``tertile`` ranking).  ``n`` keeps the top-``n`` leaders (the top-book
    consumers) or, when ``None``, returns every ranked factor (the full-sweep
    consumers, e.g. Experiment 4's timing overlays).  Each row carries the factor's
    ``factor`` name, source ``subexperiment``, bullish ``direction``, ``family`` and
    alpha stats, pre-sorted by combined (full + 2016+) net-of-cost beta-neutral
    Sharpe."""
    b = next(spec for spec in BUCKETINGS if spec["label"] == label)
    csv_path = b["ranked_csv"]
    if csv_path.exists():
        table = pd.read_csv(csv_path)
    else:                                   # never run standalone -- compute + persist once
        table = run(b["n"], b["title"], b["out_png"], label, csv_path)
    return table if n is None else table.head(n).reset_index(drop=True)


REGRESSION_PNG = _THIS_DIR / "Factor Ranking" / "quarter_regression.png"


def run_regression(out_png: Path = REGRESSION_PNG) -> pd.DataFrame:
    """Run the quarterly cross-section regression for every factor across all sources
    and render the ranked single-PNG table (the quarterly counterpart of the
    per-library ``regression/summary_table.png``).

    Unlike the long/short books, the regression uses the full cross-section z-score
    (no bucketing / orientation), so this is a single sweep -- one row per factor:
    the mean quarterly alpha and its Fama-MacBeth t-stat, full period and 2016+,
    ranked by the sum of the two t-stats.  Returns the ranked table."""
    rows: list[dict] = []
    for src in SOURCES:
        if not src["alpha_csv"].exists() or not src["panel"].exists():
            print(f"  (skip {src['label']}: panel or alpha csv missing -- "
                  f"run its driver first)")
            continue
        directions = pd.read_csv(src["alpha_csv"])[["factor", "family"]]
        panel = _load_panel(src["panel"])
        print(f"--- {src['label']}: {len(directions)} factors ---")
        for r in directions.itertuples(index=False):
            row = evaluate_regression(panel, r.factor, r.family)
            row["subexperiment"] = src["label"]
            rows.append(row)
            print(f"  {row['factor']:<26} [{src['label']:<10}]  "
                  f"beta={row['alpha_q']:+.4f}/q  t={row['alpha_tstat']:+.2f}  "
                  f"(2016+ t={row['alpha_tstat_2016']:+.2f})")

    if not rows:
        raise FileNotFoundError(
            "No source alpha tables found; run Experiment 1 and the Experiment 2 "
            "subexperiments first (python monthly_position.py).")

    table = pd.DataFrame(rows)
    table["alpha_tstat_combined"] = table["alpha_tstat"] + table["alpha_tstat_2016"]
    table = (table.sort_values("alpha_tstat_combined", ascending=False, kind="stable")
                  .reset_index(drop=True))

    plot_rows = table.copy()
    plot_rows["family"] = plot_rows["family"] + "  [" + plot_rows["subexperiment"] + "]"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    render_regression_table(plot_rows, out_png)

    print(f"\nSaved quarterly cross-section regression table "
          f"({len(table)} factors) -> {out_png}")
    return table


if __name__ == "__main__":
    for _b in BUCKETINGS:
        print(f"\n===== bucketing: {_b['label']} (n={_b['n']}) =====")
        run(_b["n"], _b["title"], _b["out_png"], _b["label"], _b["ranked_csv"])
    print("\n===== quarterly cross-section regression =====")
    run_regression()
