"""
monthly_position.py
===================

Experiment 2 driver **and** the shared monthly-repositioning re-evaluation, in one
module (the former ``main.py`` + ``tertile.py``).

Two roles
---------
1. **Experiment 2 driver** (:func:`main`).  Run the full factor pipeline on the
   **Software & Services** cross-section using the software-industry factors
   (``sw_factors.py``) instead of the general market factors, then every sibling
   subexperiment (``RD/``, ``Stability/``, ``Skew/``) in its own subprocess, then
   collect every subexperiment's factors and render the leaders
   (:func:`collect_top_factors`).  Like Experiment 1's per-universe drivers this
   adds no new analytics -- it reuses Experiment 1's ``quintile.py`` /
   ``regression.py`` **unmodified** via dependency injection
   (``driver_utils.wire_engine`` binds ``factors`` -> ``sw_factors`` before
   importing them, so their ``F`` resolves to the software factors with zero
   changes to Experiment 1).  Results land in ``Standard/`` mirroring the
   Experiment 1 layout one-for-one.

2. **The monthly re-evaluation** (:func:`run` / :func:`run_all`).  Re-evaluate
   **every** factor tested on the Software & Services universe -- Experiment 1's
   general market factors and every Experiment 2 software subexperiment -- with a
   **monthly** long/short book at each bucketing, and render the whole set as one
   alpha table per bucketing:

       quintile: long the top fifth  / short the bottom fifth (Q5-Q1)  monthly_quintile.png
       tertile:  long the top third  / short the bottom third (T3-T1)  monthly_tertile.png
       half:     long the top half   / short the bottom half  (H2-H1)  monthly_half.png

   Only the *bucketing* changes.  Each factor's long/short **orientation** (which
   tail is the long leg) is a property of the factor, recorded in each source's
   ``quintile/long_short_market_alpha.csv`` (``direction`` = ``Q5-Q1`` /
   ``Q1-Q5``); we reuse it verbatim and relabel after the bucket count
   (:func:`dir_label`).  These are the *monthly* comparison tables, read against
   ``quarter_position.py``'s quarterly ones in the Experiment 2 report.

Shared machinery (reused by ``quarter_position.py``)
----------------------------------------------------
``quarter_position.py`` re-evaluates the *same* factor set with the same books,
only repositioned quarterly.  Everything that is not about the repositioning
frequency lives here and is imported by it:

  * :data:`SOURCES` -- the single source-of-truth factor universe (Experiment 1's
    general factors + every Experiment 2 software subexperiment; ``Cross_val/`` is
    excluded, being a different universe);
  * the generic engine wiring (``factors`` / ``cost`` / ``regression`` off
    ``sys.path`` -- the panel-driven engine, not a per-library injection);
  * :func:`_load_panel`, :func:`dir_label`, :data:`DECADE_START`;
  * :func:`evaluate_all` -- iterate the sources, evaluate every factor with a
    supplied per-factor evaluator and rank by combined net-of-cost beta-neutral
    Sharpe (both drivers pass their own book/cost builder);
  * :func:`alpha_row` -- the shared ``render_alpha_table`` row (industry-neutral
    alpha, Sharpe, beta-neutral Sharpe, turnover cost, full sample and 2016+) given
    a book's spread and cost series;
  * :func:`render_ranked` -- family-tag by source subexperiment + render the PNG.

The factor-selection hand-off downstream experiments read is the *quarterly*
ranking ``quarter_position.ranked_factors`` (the CSV ``quarter_position.py``
persists), not these monthly tables.

Run standalone::

    python monthly_position.py           # full pipeline + subexperiments + collection
    python monthly_position.py collect    # only (re)render the monthly quintile/tertile/half tables
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

# --- Put our own directory on sys.path so ``sw_factors`` / ``driver_utils`` resolve
#     even when this module is loaded by path from another experiment (the folder name
#     has spaces, so it is never on sys.path by default) -- ``quarter_position.py`` and
#     ``experiment5 .../capacity_scaling.py`` reuse this module's SOURCES that way.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import sw_factors as S       # noqa: E402  (after sys.path wiring)
import driver_utils as D     # noqa: E402

# --- Load Experiment 1's engine + analysis modules by path.  The folder name has
#     spaces but the modules are plainly named (factors/cost/regression), so putting
#     the directory on sys.path lets them import normally -- and lets their own
#     ``import factors as F`` / ``import cost`` resolve to the Experiment 1 engine.
#     The re-evaluation (run/run_all) wants the *generic* engine (panel-driven, not
#     one factor set); the driver (main) separately injects ``sw_factors`` via
#     ``driver_utils.wire_engine`` before touching quintile/regression.
_EXP1_DIR = _THIS_DIR.parent / "experiment1 - general factors"
sys.path.insert(0, str(_EXP1_DIR))

# --- Dependency injection for the *driver* role: bind ``factors`` -> ``sw_factors``
#     and import Experiment 1's ``quintile`` / ``regression`` against it, so the
#     Standard pipeline (:func:`main`) runs on the software factors with zero changes
#     to Experiment 1.  This must precede any ``import regression`` (its ``F`` binds
#     at first import) -- so ``wire_engine`` runs *before* the ``cost`` / ``regression``
#     / ``factors`` imports below.
#
#     The *re-evaluation* role (run/run_all, and everything ``quarter_position.py``
#     reuses) only calls ``regression``'s stateless, panel-driven helpers
#     (market_regression, long_short_stats, beta_neutral_sharpe, ...) and ``F``'s
#     generic engine helpers (prepare_slice, SOFTWARE_SERVICES) -- never ``.run`` /
#     ``FACTOR_NAMES`` / ``FACTORS`` -- and ``sw_factors`` re-exports those generic
#     helpers verbatim, so the injection is transparent to it.
quintile, regression = D.wire_engine(S)

import cost                # noqa: E402  (after wire_engine + sys.path wiring)
import factors as F        # noqa: E402

DECADE_START = regression.DECADE_START   # 2016+ ("past decade") window, shared
UNIVERSE = S.SOFTWARE_SERVICES


def dir_label(sign: int, n: int) -> str:
    """Long/short direction label after the bucket count, shared with
    ``quarter_position.py``: ``Q5-Q1``/``Q1-Q5`` for quintiles (``n`` = 5),
    ``T3-T1``/``T1-T3`` for tertiles (``n`` = 3), ``H2-H1``/``H1-H2`` for halves
    (``n`` = 2).  ``sign`` +1 = long top / short bottom, -1 = the reverse."""
    p = {5: "Q", 3: "T", 2: "H"}.get(n, "Q")
    return f"{p}{n}-{p}1" if sign > 0 else f"{p}1-{p}{n}"


# --------------------------------------------------------------------------- #
# Sources -- one ``quintile/long_short_market_alpha.csv`` (for the factor list,
# family label and recorded orientation) + its ``factor_panel.csv`` (to rebuild the
# books) per Software & Services factor library: Experiment 1's general factors
# first, then every Experiment 2 software subexperiment.  ``Cross_val/`` is excluded
# -- it re-tests factors on Banks+Insurance, a different universe.  This is the
# single source of truth for "the tested factor set"; ``quarter_position.py`` and
# ``experiment5 .../capacity_scaling.py`` import it by path.
# --------------------------------------------------------------------------- #
_EXP2_LIBS = ["Standard", "RD", "Stability", "Skew"]


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

FACTOR_RANKING_DIR = _THIS_DIR / "factor_ranking"


def _load_panel(path: Path) -> pd.DataFrame:
    """Read a tidy factor panel from disk with the engine's dtype conventions
    (``factors.load_panel`` takes a Universe, not a path, so we mirror its
    post-read normalisation directly).  Shared with ``quarter_position.py``."""
    df = pd.read_csv(path, parse_dates=["date"])
    df["stock_id"] = df["stock_id"].astype(str)
    return df


# --------------------------------------------------------------------------- #
# Shared per-factor evaluation + ranking driver
#
# The monthly and quarterly re-evaluations differ only in how each factor's book
# (its long/short spread) and turnover cost are built; everything downstream -- the
# industry-neutral regression, the Sharpe / beta-neutral Sharpe / net-of-cost Sharpe
# stats, the ranking by combined net-of-cost beta-neutral Sharpe, the family-tagged
# render -- is identical.  ``alpha_row`` is that identical tail; ``evaluate_all`` is
# the identical source-iteration + ranking driver, parameterised by a per-factor
# evaluator.  ``quarter_position.py`` imports both and supplies its own evaluator.
# --------------------------------------------------------------------------- #
def alpha_row(spread: pd.Series, cost_series: pd.Series, industry_ret: pd.Series,
              factor: str, family: str, sign: int, n: int) -> dict:
    """One :func:`regression.render_alpha_table` row from a book's ``spread`` and its
    ``cost_series``: the industry-neutral alpha and its t-stat, the raw / beta-neutral
    / net-of-cost Sharpe and the average turnover cost, full sample and 2016+.  Reuses
    Experiment 1's regression / cost helpers; the caller supplies the book (so the
    monthly and quarterly drivers share this whole tail)."""
    recent = spread.index >= DECADE_START
    ind_recent = industry_ret[industry_ret.index >= DECADE_START]

    ls = regression.long_short_stats(spread)
    ls_2016 = regression.long_short_stats(spread[recent])
    mreg = regression.market_regression(spread, industry_ret)
    mreg_2016 = regression.market_regression(spread[recent], ind_recent)
    # Beta-neutral Sharpe: hedge with a walk-forward -beta*industry overlay whose beta
    # is re-estimated on an expanding, look-ahead-free window (2016+ windows the same
    # hedged series, so its betas still use all prior history).
    sr_neutral = regression.beta_neutral_sharpe(spread, industry_ret)
    sr_neutral_2016 = regression.beta_neutral_sharpe(spread, industry_ret, start=DECADE_START)
    # Turnover cost of the book: its mean (pp/month) plus the cost-incorporated Sharpe
    # (raw + beta-neutral, full & 2016+).
    avg_cost_pp = cost.average_cost(cost_series) * 100.0
    sharpe_cost = regression.net_of_cost_sharpe(spread, cost_series)
    sharpe_cost_neutral = regression.net_of_cost_neutral_sharpe(
        spread, cost_series, industry_ret)
    sharpe_cost_2016 = regression.net_of_cost_sharpe(spread, cost_series, start=DECADE_START)
    sharpe_cost_neutral_2016 = regression.net_of_cost_neutral_sharpe(
        spread, cost_series, industry_ret, start=DECADE_START)

    return {
        "factor": factor, "family": family,
        "direction": dir_label(sign, n),
        "alpha": mreg["alpha"], "alpha_tstat": mreg["alpha_tstat"],
        "sharpe": ls["sharpe"], "sharpe_neutral": sr_neutral,
        "sharpe_cost": sharpe_cost, "sharpe_cost_neutral": sharpe_cost_neutral,
        "avg_cost_pp": avg_cost_pp, "n": mreg["n"],
        "alpha_2016": mreg_2016["alpha"], "alpha_tstat_2016": mreg_2016["alpha_tstat"],
        "sharpe_2016": ls_2016["sharpe"], "sharpe_neutral_2016": sr_neutral_2016,
        "sharpe_cost_2016": sharpe_cost_2016,
        "sharpe_cost_neutral_2016": sharpe_cost_neutral_2016,
    }


def evaluate_all(evaluate_factor, n: int) -> pd.DataFrame:
    """Re-evaluate every factor across all :data:`SOURCES` with the per-factor
    ``evaluate_factor(panel, industry_ret, cost_panel, factor, family, direction, n)``
    callback, and rank by the sum of the full-period and 2016+ net-of-cost
    beta-neutral Sharpe ratios (as :func:`collect_top_factors` ranks the leaders).
    Returns the ranked table (no rendering).  Shared by the monthly re-evaluation
    (:func:`run`) and ``quarter_position.py`` (its quarterly-book evaluator)."""
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
                  f"(2016+ t={row['alpha_tstat_2016']:+.2f})")

    if not rows:
        raise FileNotFoundError(
            "No source alpha tables found; run Experiment 1 and the Experiment 2 "
            "subexperiments first (python monthly_position.py).")

    table = pd.DataFrame(rows)
    table["sharpe_combined"] = (
        table["sharpe_cost_neutral"] + table["sharpe_cost_neutral_2016"])
    return (table.sort_values("sharpe_combined", ascending=False, kind="stable")
                 .reset_index(drop=True))


def render_ranked(table: pd.DataFrame, out_png: Path, title: str | None = None) -> None:
    """Render a ranked factor table in the standard alpha-table format, tagging each
    family with its source subexperiment for provenance -- exactly like the monthly /
    quarterly / factor_ranking PNGs.  Shared with ``quarter_position.py``."""
    plot_rows = table.copy()
    plot_rows["family"] = plot_rows["family"] + "  [" + plot_rows["subexperiment"] + "]"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    regression.render_alpha_table(plot_rows, out_png, title=title)


# --------------------------------------------------------------------------- #
# Monthly long/short book + its bucketings
# --------------------------------------------------------------------------- #
def bucket_spread(panel: pd.DataFrame, factor: str, sign: int, n: int) -> pd.Series:
    """Monthly dollar-neutral long/short return of the top-minus-bottom book for
    ``factor`` at ``n`` buckets, oriented by ``sign`` (+1 = long top / short bottom,
    -1 = the reverse).  Reuses ``factors.prepare_slice`` with ``n`` buckets, so the
    legs are the same even-count sort the quintile book uses, just into ``n`` groups
    (``n`` = 5 quintiles, 3 tertiles, 2 halves)."""
    sub = F.prepare_slice(panel, factor, n)
    legs = (sub.pivot_table(index="date", columns="quintile",
                            values="next_return", aggfunc="mean")
               .sort_index())
    top, bottom = legs.get(float(n)), legs.get(1.0)
    if top is None or bottom is None:
        return pd.Series(dtype=float, name=f"{factor}_ls")
    spread = (top - bottom) if sign > 0 else (bottom - top)
    return spread.rename(f"{factor}_ls")


def evaluate_factor(panel: pd.DataFrame, industry_ret: pd.Series,
                    cost_panel: pd.DataFrame, factor: str, family: str,
                    direction: str, n: int) -> dict:
    """One :func:`alpha_row` for ``factor``'s **monthly** ``n``-bucket book with the
    recorded orientation: build the top-minus-bottom spread (:func:`bucket_spread`)
    and the per-month turnover cost (``cost.long_short_cost``, which reuses the same
    bucket count), then measure the shared alpha/Sharpe/cost stats."""
    sign = 1 if direction == "Q5-Q1" else -1
    spread = bucket_spread(panel, factor, sign, n)
    cost_series = cost.long_short_cost(panel, factor, cost_panel, n)
    return alpha_row(spread, cost_series, industry_ret, factor, family, sign, n)


def _title(bucket_word: str, top_leg: str) -> str:
    return (f"Long-short {bucket_word.upper()} strategy regressed on the industry "
            f"return\n(long {top_leg} / short the reverse;  ls_t = α + β·industry_t "
            "+ ε,  α = industry-neutral monthly return, t-stat tests α ≠ 0)")


# One monthly alpha table per bucketing -- the standard quintile book (the
# cross-experiment ranking table), plus the coarser tertile and half books, each read
# against the quarterly counterparts.  ``run_all`` iterates over this list.
BUCKETINGS = [
    {"label": "quintile", "n": 5,
     "out_png": FACTOR_RANKING_DIR / "monthly_quintile.png",
     "title": _title("quintile", "top fifth")},
    {"label": "tertile", "n": 3,
     "out_png": FACTOR_RANKING_DIR / "monthly_tertile.png",
     "title": _title("tertile", "top third")},
    {"label": "half", "n": 2,
     "out_png": FACTOR_RANKING_DIR / "monthly_half.png",
     "title": _title("half", "top half")},
]


def run(n: int, title: str, out_png: Path, label: str) -> pd.DataFrame:
    """Re-evaluate every factor's monthly ``n``-bucket book (:func:`evaluate_all` with
    the monthly :func:`evaluate_factor`) and render the ranked table to ``out_png``.
    Returns the ranked table."""
    table = evaluate_all(evaluate_factor, n)
    render_ranked(table, out_png, title)
    print(f"\nSaved monthly {label} long-short alpha table ({len(table)} factors) "
          f"-> {out_png}")
    return table


def run_all() -> dict[str, pd.DataFrame]:
    """Render every monthly bucketing table in :data:`BUCKETINGS`
    (``monthly_quintile.png`` + ``monthly_tertile.png`` + ``monthly_half.png``) and
    return the ranked tables keyed by bucketing label."""
    tables: dict[str, pd.DataFrame] = {}
    for b in BUCKETINGS:
        print(f"\n===== bucketing: {b['label']} (n={b['n']}) =====")
        tables[b["label"]] = run(b["n"], b["title"], b["out_png"], b["label"])
    return tables


# --------------------------------------------------------------------------- #
# Experiment 2 driver: Standard pipeline -> subexperiments -> collection
# --------------------------------------------------------------------------- #
# Each subexperiment lives in its own subfolder with its own factor library and
# ``main_*.py`` driver doing the same ``import factors as F`` injection this file
# does; those module-level bindings are cached per process, so the subexperiments
# cannot share one interpreter -- we run each as its own subprocess.
_SUBEXPERIMENTS = [
    _THIS_DIR / "RD" / "main_rd.py",
    _THIS_DIR / "Stability" / "main_stability.py",
    _THIS_DIR / "Skew" / "main_skew.py",
]


def run_subexperiments() -> None:
    """Run every subexperiment driver in its own subprocess, continuing past any
    that fail and reporting the roster at the end."""
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


TOP_N = 5


def collect_top_factors(top_n: int = TOP_N) -> pd.DataFrame:
    """Rank every tested factor and render every factor's monthly long/short alpha
    table via :func:`run_all` -- the standard quintile table (``monthly_quintile.png``,
    the cross-experiment ranking) plus the coarser tertile (``monthly_tertile.png``)
    and half (``monthly_half.png``) books, the monthly-repositioning comparison tables
    read against the quarterly tables in the Experiment 2 report.  The downstream
    factor-selection hand-off is the *quarterly* ranking
    (``quarter_position.ranked_factors``).  Returns the top-``top_n`` rows of the
    quintile ranking.

    Experiment 1's general market factors are included in the ranking -- they are
    *also* tested on the Software & Services universe (as the ``General`` source),
    directly comparable, same cross-section and same alpha definition.
    ``Cross_val/`` is excluded (a different universe; see :data:`SOURCES`).
    """
    top = run_all()["quintile"].head(top_n).copy()
    print(f"\n=== Top {top_n} factors across subexperiments "
          f"(by full-period + 2016-onward net-of-cost beta-neutral Sharpe) ===")
    for r in top.itertuples():
        print(f"  {r.factor:<24} [{r.subexperiment:<10}] "
              f"alpha={r.alpha:+.4%}/mo  srCN={r.sharpe_cost_neutral:+.2f}  "
              f"srCN(2016+)={r.sharpe_cost_neutral_2016:+.2f}  "
              f"srCN(sum)={r.sharpe_combined:+.2f}")
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
