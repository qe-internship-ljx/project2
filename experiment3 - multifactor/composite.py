"""
composite.py
============

Experiment 3 -- composite z-score quintile long/short.

Combine an arbitrary *set* of factors into one cross-sectional score -- the sum
of each stock's sign-oriented factor z-scores -- and run Experiment 1's quintile
workflow on that composite::

    composite_{i,t} = sum_f  sign_f * zscore_{f,i,t}

Each month the industry cross-section is sorted into five equal-count buckets on
the composite, the next-period (month t+1) return of each bucket is tracked, and
the Q5-Q1 long/short book's performance is measured: mean, t-stat, annualised
Sharpe, the industry-neutral alpha and its t-stat, and the cumulative growth
path.

``sign_f`` orients every factor to its *bullish* direction before summing, read
from the ``direction`` column of that factor's standalone long/short alpha table
(``Q5-Q1`` -> +1, long the high-z names;  ``Q1-Q5`` -> -1).  A high composite is
therefore "attractive across the whole set", so the book is always long Q5 /
short Q1 and no per-factor sign bookkeeping is left to the caller.

This *replaces* Experiment 3's earlier t-stat-gated multivariate regression:
there is **no significance gate and no regression model** -- the caller passes
the factor set explicitly and the factors are combined by standardised
aggregation, the textbook "composite signal" construction.

Design -- maximal reuse, zero duplication of the engine
-------------------------------------------------------
Nothing generic is re-implemented:

* the cross-section mechanics -- within-month winsorisation, equal-count
  quintile labels, the quintile-return table -- are Experiment 1's:
  ``quintile.py`` and the ``factors.py`` engine are reused *unmodified* through
  the project's dependency-injection convention (register the engine as
  ``sys.modules["factors"]`` before importing the analysis modules, exactly as
  Experiment 2's ``main.py`` does);
* the long/short book's industry-neutral alpha is measured by Experiment 1's own
  ``regression.market_regression`` / ``industry_monthly_return``, so "alpha" is
  defined identically to every other long/short book in the project;
* the factor exposures and their bullish orientation are read straight from the
  ``factor_panel.csv`` / ``long_short_market_alpha.csv`` files Experiments 1 & 2
  already wrote.

This module therefore adds only the *composite construction* and the per-set
reporting; every input it consumes was produced upstream.

Outputs (``output/composite/<slug>/``)
--------------------------------------
    factor_set.png              the constituents: family, source, sign, standalone alpha
    exposure_correlation.csv    pairwise correlation of the oriented constituent z-scores
    quintile_returns.csv        months x {Q1..Q5, Q5-Q1}, mean next-period return
    quintile_cumulative.png     the five buckets as cumulative growth of $1 (log scale)
    long_short.png              the Q5-Q1 book's cumulative growth of $1
    performance.png             the L/S book's mean / t / Sharpe / industry-neutral alpha

Run standalone::

    python composite.py                                          # default set
    python composite.py buyback_quality gross_profitability rd_stability
    python composite.py top                                       # the active top-5 (Exp 1 + Exp 2)
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
EXP3_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXP3_DIR.parent
EXP1_DIR = PROJECT_ROOT / "experiment1 - general factors"
EXP2_DIR = PROJECT_ROOT / "experiment2 - sw factors"
OUTPUT_DIR = EXP3_DIR / "output"

# Factor libraries that can supply a constituent.  Each contributes a tidy panel
# (date, stock_id, factor, value, zscore, next_return) carrying the exposures and
# an alpha table whose ``direction`` column records each factor's bullish sign.
# A requested factor is resolved against these in order (first match wins), so
# its source -- panel and orientation -- is always unambiguous.  The list spans
# every Software & Services factor library (Experiment 1's general factors plus
# all of Experiment 2's software subexperiments), so any factor in Experiment 2's
# top-factor hand-off resolves.  ``Cross_val/`` is omitted: it lives on the
# Banks+Insurance universe, a different cross-section.
def _exp2_lib(folder: str, label: str) -> dict:
    return {"label": label,
            "panel": EXP2_DIR / folder / "factor_panel.csv",
            "alpha": EXP2_DIR / folder / "quintile" / "long_short_market_alpha.csv"}


LIBRARIES: list[dict] = [
    {"label": "general market factors",
     "panel": EXP1_DIR / "output" / "software" / "factor_panel.csv",
     "alpha": EXP1_DIR / "output" / "software" / "quintile" / "long_short_market_alpha.csv"},
    _exp2_lib("Standard", "software-industry factors"),
    _exp2_lib("RD", "R&D-behaviour factors"),
    _exp2_lib("Rev & Cost", "revenue/cost factors"),
    _exp2_lib("Growth", "growth factors"),
    _exp2_lib("Stability", "stability factors"),
    _exp2_lib("Skew", "skew factors"),
]

# The cross-experiment top-factor hand-off (written by
# ``experiment2 - sw factors/main.py``, ranking Experiment 1's general market
# factors together with every Experiment 2 software subexperiment by alpha
# t-stat); ``top_factors()`` reads its factor list.
TOP_FACTORS_CSV = EXP2_DIR / "top_factors" / "top_factors.csv"
TOP_N = 5

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DEFAULT_FACTORS = ["buyback_quality", "gross_profitability", "rd_stability"]
COMPOSITE_FACTOR = "composite"          # synthetic factor name fed to the reused sort
N_QUINTILES = 5
MONTHS_PER_YEAR = 12
DECADE_START = pd.Timestamp("2016-01-01")   # "past decade" cut-off (project convention)


# --------------------------------------------------------------------------- #
# Experiment 1 engine + analysis modules, loaded by path and wired for reuse
# --------------------------------------------------------------------------- #
def _load(name: str, path: Path):
    """Import a module by file path under ``name`` and register it in sys.modules."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_exp1():
    """
    Load Experiment 1's engine and analysis modules by path (their folder name
    has spaces, so they cannot be imported normally), wired for reuse.

    ``quintile`` / ``regression`` (and ``cost``) bind their factor library via
    ``import factors as F`` / ``import cost``; registering the engine and cost
    under those names *first* makes those imports resolve here -- the project's
    standard dependency-injection convention (cf. Experiment 2's ``main.py``).
    """
    engine = _load("factors", EXP1_DIR / "factors.py")
    _load("cost", EXP1_DIR / "cost.py")              # imported by regression.py
    quintile = _load("quintile", EXP1_DIR / "quintile.py")
    regression = _load("regression", EXP1_DIR / "regression.py")
    return engine, quintile, regression


F, Q, R = _load_exp1()
import cost as COST                       # registered by _load_exp1; the turnover-cost model


# --------------------------------------------------------------------------- #
# Step 1 -- resolve the requested factors to (source panel, bullish sign)
# --------------------------------------------------------------------------- #
def top_factors(csv_path: Path = TOP_FACTORS_CSV, n: int = TOP_N) -> list[str]:
    """The active top-``n`` factor names from the cross-experiment top-factor
    hand-off (``top_factors.csv``), ranked by industry-neutral alpha t-stat.

    Reads the *current* file each call, so the composite always tracks whatever
    factors rank highest after the latest Experiment 1/2 run; the rows are
    pre-sorted by rank, so the first ``n`` are the leaders.  Raises a clear error
    if the hand-off is missing."""
    if not Path(csv_path).exists():
        raise FileNotFoundError(
            f"{csv_path} not found.  Run Experiment 2 first -- `python main.py` "
            "(or `python main.py collect`) in 'experiment2 - sw factors' writes "
            "the top-factor hand-off.")
    return pd.read_csv(csv_path)["factor"].head(n).tolist()


def resolve_factors(factor_names: list[str]) -> pd.DataFrame:
    """
    Map each requested factor to its source library, family, bullish sign and
    standalone long/short alpha, read from the libraries' alpha tables.

    The sign orients the factor so a higher score is bullish: ``+1`` when the
    factor's standalone book is long the top z-score quintile (``direction`` ==
    ``Q5-Q1``), ``-1`` otherwise.  Returns one row per factor in the requested
    order; raises if a factor is in none of the libraries.
    """
    catalog: dict[str, dict] = {}
    for lib in LIBRARIES:
        if not Path(lib["alpha"]).exists():
            continue
        tbl = pd.read_csv(lib["alpha"])
        for _, r in tbl.iterrows():
            catalog.setdefault(r["factor"], {           # first library wins
                "factor": r["factor"], "family": r["family"],
                "direction": r["direction"], "alpha": r["alpha"],
                "alpha_tstat": r["alpha_tstat"],
                "library": lib["label"], "panel_path": str(lib["panel"])})

    rows = []
    for name in factor_names:
        if name not in catalog:
            libs = ", ".join(lib["label"] for lib in LIBRARIES)
            raise KeyError(f"factor {name!r} not found in any library ({libs}). "
                           "Run Experiments 1-2 (and the R&D extension) first.")
        meta = dict(catalog[name])
        meta["sign"] = 1 if str(meta["direction"]).strip() == "Q5-Q1" else -1
        rows.append(meta)

    return pd.DataFrame(rows, columns=["factor", "family", "library", "direction",
                                       "sign", "alpha", "alpha_tstat", "panel_path"])


def factor_quintile_dir(factor: str) -> Path:
    """Locate a factor's ``quintile/`` output directory -- the parent of the
    library alpha table that lists it.  Resolves factors from Experiment 1
    (general market factors) and any Experiment 2 software subexperiment alike,
    so a standalone book written by either experiment is found.  Raises if the
    factor is in no library."""
    for lib in LIBRARIES:
        ap = Path(lib["alpha"])
        if ap.exists() and factor in set(pd.read_csv(ap, usecols=["factor"])["factor"]):
            return ap.parent
    raise KeyError(f"factor {factor!r} not found in any library alpha table "
                   f"({', '.join(l['label'] for l in LIBRARIES)}).")


def factor_long_short(factor: str) -> pd.Series:
    """
    A single factor's bullish-oriented standalone monthly long/short return,
    read from its ``quintile/<factor>/quintile_returns.csv`` (the ``Q5-Q1``
    column) and signed by its bullish ``direction`` (``+`` when the book is long
    the top z-score quintile, ``-`` otherwise).  Indexed by formation month.

    This is the exact dollar-neutral book each factor's published ``long_short.png``
    tracks, so it is the natural benchmark to regress another strategy against;
    no return is recomputed here.  Shared by ``factor_momentum.py`` (rotation
    candidates) and ``bivariate_tertile.py`` (benchmark-relative alpha).
    """
    sign = int(resolve_factors([factor]).iloc[0]["sign"])
    qr_path = factor_quintile_dir(factor) / factor / "quintile_returns.csv"
    qr = (pd.read_csv(qr_path, parse_dates=["date"])
            .set_index("date").sort_index())
    return (sign * qr["Q5-Q1"]).rename(factor)


# --------------------------------------------------------------------------- #
# Step 2 -- assemble the composite score
#
# These three helpers are the reusable spine of every composite variant:
# ``load_exposures`` reads the raw constituent matrix, ``scored_frame`` attaches
# the realised return to any per-stock-month score, and ``load_composite`` is the
# straight (sign-oriented, equal-weighted) sum built from them.  A weighted
# variant just supplies a different score over the same exposures (see
# ``weighted_composite.py``).
# --------------------------------------------------------------------------- #
def load_exposures(resolved: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Wide ``(date, stock_id) x factor`` matrix of the constituents' **raw**
    cross-sectional z-scores (in the requested order), plus the per-stock-month
    ``next_return`` series, assembled across the constituents' source panels.

    The per-library blocks are aligned on the shared ``(date, stock_id)`` key and
    rows are kept only where **every** factor is present, so any downstream
    combination (straight sum, sign-oriented sum, coefficient-weighted sum) scores
    each stock on the complete set.  Orientation/weighting is *not* applied here --
    callers weight the columns themselves (by sign, by regression coefficient...).
    """
    blocks: dict[str, pd.Series] = {}        # factor -> z-score, indexed by key
    next_ret: pd.Series | None = None        # next_return, indexed by key

    for panel_path, grp in resolved.groupby("panel_path", sort=False):
        names = grp["factor"].tolist()
        panel = pd.read_csv(panel_path, parse_dates=["date"],
                            usecols=["date", "stock_id", "factor", "zscore", "next_return"])
        panel["stock_id"] = panel["stock_id"].astype(str)
        panel = panel[panel["factor"].isin(names)]

        wide = panel.pivot_table(index=["date", "stock_id"], columns="factor",
                                 values="zscore")
        for name in names:
            blocks[name] = wide[name]

        # next_return is identical across factors for a given stock-month; take
        # it once per (date, stock_id) and union across libraries.
        nr = (panel.dropna(subset=["next_return"])
                   .drop_duplicates(["date", "stock_id"])
                   .set_index(["date", "stock_id"])["next_return"])
        next_ret = nr if next_ret is None else next_ret.combine_first(nr)

    exposures = pd.concat(blocks, axis=1)[resolved["factor"].tolist()].dropna()
    return exposures, next_ret


def scored_frame(score: pd.Series, next_ret: pd.Series) -> pd.DataFrame:
    """
    Attach the realised ``next_return`` to a per-``(date, stock_id)`` composite
    ``score`` and return a tidy ``date, stock_id, composite, next_return`` frame
    (dropping the final month, which has no t+1 return).  Shared by every variant.
    """
    return (pd.concat([score.rename("composite"),
                       next_ret.rename("next_return")], axis=1)
              .loc[score.index]                  # scored rows only
              .dropna(subset=["next_return"])    # drop the last month (no t+1 return)
              .reset_index()
              .sort_values(["date", "stock_id"]))


def load_composite(resolved: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Straight (equal-weighted) composite: orient each constituent z-score to its
    bullish sign and sum across factors.

    Returns ``(composite, exposures)`` where ``composite`` has columns
    ``date, stock_id, composite, next_return`` and ``exposures`` is the wide
    matrix of oriented constituent z-scores (for the correlation diagnostic).
    """
    exposures, next_ret = load_exposures(resolved)
    signs = pd.Series(dict(zip(resolved["factor"], resolved["sign"])))
    oriented = exposures.mul(signs, axis=1)              # +/-1 per factor
    score = oriented.sum(axis=1)
    return scored_frame(score, next_ret), oriented


def as_factor_panel(composite: pd.DataFrame) -> pd.DataFrame:
    """
    Shape the composite as a one-factor tidy panel so Experiment 1's quintile
    code (which keys off ``factor`` and reads ``zscore`` / ``next_return``) runs
    on it unmodified -- the composite plays the role of the factor z-score.
    """
    panel = composite.copy()
    panel["factor"] = COMPOSITE_FACTOR
    panel["value"] = panel["composite"]
    panel["zscore"] = panel["composite"]
    return panel


# --------------------------------------------------------------------------- #
# Step 3 -- long/short performance (reusing the engine's alpha definition)
# --------------------------------------------------------------------------- #
def industry_return() -> pd.Series:
    """
    Market-cap-weighted Software & Services next-period return -- the within-
    industry "market" -- from Experiment 1's saved panel.  This is the exact series
    every project long/short book is regressed on for its industry-neutral alpha.
    """
    return R.industry_monthly_return(F.load_panel(u=F.SOFTWARE_SERVICES))


_COST_PANEL: pd.DataFrame | None = None


def cost_panel() -> pd.DataFrame:
    """The Software & Services per-(stock, month) round-trip cost panel, built once
    and cached.  Shared by every Experiment 3 book so the reported turnover cost
    uses the project's one cost model (``cost.py``)."""
    global _COST_PANEL
    if _COST_PANEL is None:
        _COST_PANEL = COST.build_cost_panel(F.SOFTWARE_SERVICES)
    return _COST_PANEL


_MCAP_PANEL: pd.DataFrame | None = None


def market_cap_panel() -> pd.DataFrame:
    """Per-``(date, stock_id)`` formation-date USD market cap, built once and
    cached (tidy ``date, stock_id, mcap``).

    ``mcap`` is the ``weight`` column of the saved Software & Services panel --
    the month-end USD cap that weights each stock's next-period return, i.e. its
    market cap at the formation of the return every book earns.  Shared by any
    Experiment 3 book that needs to describe or size the names it holds."""
    global _MCAP_PANEL
    if _MCAP_PANEL is None:
        panel = F.load_panel(u=F.SOFTWARE_SERVICES)
        _MCAP_PANEL = (panel.dropna(subset=["weight"])
                            .drop_duplicates(["date", "stock_id"])
                            [["date", "stock_id", "weight"]]
                            .rename(columns={"weight": "mcap"})
                            .reset_index(drop=True))
    return _MCAP_PANEL


def window_cost(cost_series: pd.Series,
                start: pd.Timestamp | None = None,
                end: pd.Timestamp | None = None) -> float:
    """Average monthly turnover cost over an optional ``[start, end]`` window
    (fraction of notional).  Reported alongside performance, **not** netted from
    the alpha or Sharpe -- those stay gross, as in every other project book."""
    s = cost_series.dropna()
    if start is not None:
        s = s[s.index >= start]
    if end is not None:
        s = s[s.index <= end]
    return float(s.mean()) if len(s) else float("nan")


def book_stats(spread: pd.Series, industry: pd.Series,
               start: pd.Timestamp | None = None,
               end: pd.Timestamp | None = None) -> dict:
    """
    Performance of a long/short spread over an optional ``[start, end]`` window:
    mean, t-stat, annualised Sharpe, the industry-neutral alpha (and its t-stat),
    the industry beta and the beta-neutralised Sharpe.

    Window-parameterised so the same helper serves the full / past-decade split
    here and the walk-forward out-of-sample window in ``weighted_composite.py``.
    All quantities use Experiment 1's own helpers, so "alpha" is defined
    identically to every other long/short book in the project.
    """
    s, mkt = spread, industry
    if start is not None:
        s, mkt = s[s.index >= start], mkt[mkt.index >= start]
    if end is not None:
        s, mkt = s[s.index <= end], mkt[mkt.index <= end]
    stats = Q.long_short_stats(s)
    mreg = R.market_regression(s, mkt)
    return {**stats,
            "alpha": mreg["alpha"], "alpha_tstat": mreg["alpha_tstat"],
            "ind_beta": mreg["beta"], "ind_beta_tstat": mreg["beta_tstat"],
            "sharpe_neutral": R.beta_neutral_sharpe(s, mkt, mreg["beta"])}


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
QCOLS = [f"Q{i}" for i in range(1, N_QUINTILES + 1)]


def _set_label(factor_names: list[str]) -> str:
    return " + ".join(factor_names)


def plot_cumulative(wide: pd.DataFrame, factor_names: list[str], path: Path,
                    score_desc: str = "sum of z-scores",
                    boundary: pd.Timestamp | None = None) -> None:
    """Cumulative growth of $1 in each composite quintile (log scale).

    ``score_desc`` labels how the composite was built; ``boundary``, if given,
    draws a vertical rule (e.g. an in-sample / out-of-sample split).
    """
    cum = (1.0 + wide[QCOLS].fillna(0.0)).cumprod()
    fig, ax = plt.subplots(figsize=(11, 5))
    for q in QCOLS:
        ax.plot(cum.index, cum[q], label=q, linewidth=1.3)
    if boundary is not None:
        ax.axvline(boundary, color="black", linestyle="--", linewidth=0.9, alpha=0.7)
    ax.set_yscale("log")
    ax.set_title("Cumulative growth of $1 by composite-score quintile\n"
                 f"composite = {score_desc}: {_set_label(factor_names)}")
    ax.set_xlabel("Month")
    ax.set_ylabel("Cumulative value (log scale)")
    ax.legend(title="Quintile", ncol=5, loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_long_short(spread: pd.Series, factor_names: list[str], sharpe: float,
                    alpha: float, alpha_tstat: float, path: Path,
                    boundary: pd.Timestamp | None = None,
                    stat_window: str = "") -> None:
    """Cumulative growth of $1 in the dollar-neutral Q5-Q1 composite book.

    ``boundary`` draws a vertical rule (e.g. the IS/OOS split); ``stat_window``
    annotates which window the header Sharpe / alpha refer to.
    """
    cum = (1.0 + spread.fillna(0.0)).cumprod()
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(cum.index, cum, color="C2", linewidth=1.3)
    ax.axhline(1.0, color="black", linewidth=0.6)
    if boundary is not None:
        ax.axvline(boundary, color="black", linestyle="--", linewidth=0.9, alpha=0.7)
    win = f"  {stat_window}" if stat_window else ""
    ax.set_title("Dollar-neutral composite long-short (long Q5 / short Q1): "
                 "cumulative growth of $1\n"
                 f"{_set_label(factor_names)}  "
                 f"[Sharpe={sharpe:+.2f}, alpha={alpha:+.4%}/mo, "
                 f"t(alpha)={alpha_tstat:+.2f}{win}]")
    ax.set_xlabel("Month")
    ax.set_ylabel("Cumulative value of $1")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def render_factor_set(resolved: pd.DataFrame, path: Path) -> None:
    """Render the composite's constituents as a shaded PNG (project house style)."""
    headers = ["Factor", "Family", "Source", "Sign", "Standalone α", "Standalone α t"]
    cell_text, cell_colors = [], []
    for _, r in resolved.iterrows():
        cell_text.append([
            r["factor"], r["family"], r["library"],
            "+1 (long high)" if r["sign"] > 0 else "-1 (long low)",
            R._fmt_pct(r["alpha"]), R._fmt_num(r["alpha_tstat"])])
        cell_colors.append(["white", "white", "white", "white", "white",
                            R._tstat_color(r["alpha_tstat"])])

    n = len(resolved)
    fig, ax = plt.subplots(figsize=(13, 0.5 * (n + 1) + 1.2))
    ax.axis("off")
    tbl = ax.table(cellText=cell_text, colLabels=headers, cellColours=cell_colors,
                   colWidths=[0.19, 0.24, 0.21, 0.14, 0.11, 0.11],
                   cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for j in range(len(headers)):
        tbl[0, j].set_text_props(weight="bold", color="white")
        tbl[0, j].set_facecolor("#404040")
    for i in range(1, n + 1):
        for j in (0, 1, 2):
            tbl[i, j].set_text_props(ha="left")
    fig.suptitle("Experiment 3 -- composite constituents (each oriented to its "
                 "bullish direction before summing)", fontsize=11, y=0.99)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.84, bottom=0.06)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# Metric row = (label, stats-key, format, shade).  ``format`` is int/pct/num;
# ``shade`` True tints the cell by the |t| significance of its own value (used for
# t-stat rows).  ``book_stats`` keys are shared by every variant so this table
# renders any window set; callers may append further rows via ``extra_metrics``.
PERF_METRICS: list[tuple[str, str, str, bool]] = [
    ("Months (n)",                   "n_months",       "int", False),
    ("Mean monthly return",          "mean_monthly",   "pct", False),
    ("t-stat (mean ≠ 0)",            "tstat",          "num", False),
    ("Sharpe (annualised)",          "sharpe",         "num", False),
    ("Industry-neutral α (monthly)", "alpha",          "pct", False),
    ("α t-stat",                     "alpha_tstat",    "num", True),
    ("Industry β",                   "ind_beta",       "num", False),
    ("β-neutral Sharpe",             "sharpe_neutral", "num", False),
]


def _fmt_cell(value, kind: str) -> str:
    if kind == "int":
        return f"{int(value)}"
    return R._fmt_pct(value) if kind == "pct" else R._fmt_num(value)


def render_performance(windows: list[tuple[str, dict]], title: str,
                       subtitle: str, path: Path,
                       extra_metrics: list[tuple[str, str, str, bool]] | None = None
                       ) -> None:
    """
    Render long/short performance across one or more named windows as a PNG.

    ``windows`` is a list of ``(column_label, stats)`` where ``stats`` is a
    :func:`book_stats` result -- e.g. full vs past-decade, or in-sample vs
    out-of-sample.  Rows flagged ``shade`` (the α t-stat, and any shaded
    ``extra_metrics``) are tinted by significance per column.

    ``extra_metrics`` appends caller-specific rows (same
    ``(label, key, format, shade)`` shape as :data:`PERF_METRICS`) -- e.g. a
    strategy's alpha above a benchmark book -- but only those whose key every
    window carries, so a partially-populated stat is silently skipped rather than
    raising.  When every window also carries an ``avg_cost`` key an extra trailing
    row reports it, for information only -- never netted from the gross alpha /
    Sharpe above.
    """
    metrics = list(PERF_METRICS)
    if extra_metrics:
        metrics += [m for m in extra_metrics if all(m[1] in s for _, s in windows)]
    if windows and all("avg_cost" in s for _, s in windows):
        metrics.append(("Avg monthly cost (turnover)", "avg_cost", "pct", False))

    headers = ["Metric"] + [label for label, _ in windows]
    cell_text, cell_colors = [], []
    for name, key, kind, shade in metrics:
        cell_text.append([name] + [_fmt_cell(s[key], kind) for _, s in windows])
        colors = ["white"] * (len(windows) + 1)
        if shade:
            for j, (_, s) in enumerate(windows, start=1):
                colors[j] = R._tstat_color(s[key])
        cell_colors.append(colors)

    nrows, ncols = len(metrics), len(windows) + 1
    metric_w = 0.40
    col_w = [metric_w] + [(1 - metric_w) / len(windows)] * len(windows)
    fig, ax = plt.subplots(figsize=(4.2 + 2.8 * len(windows), 0.5 * (nrows + 1) + 1.6))
    ax.axis("off")
    tbl = ax.table(cellText=cell_text, colLabels=headers, cellColours=cell_colors,
                   colWidths=col_w, cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for j in range(ncols):
        tbl[0, j].set_text_props(weight="bold", color="white")
        tbl[0, j].set_facecolor("#404040")
    for i in range(1, nrows + 1):
        tbl[i, 0].set_text_props(ha="left")

    fig.suptitle(title, fontsize=11, y=0.99)
    # Wrap the "   |   "-separated clauses onto their own lines so the caption
    # never overflows the figure width regardless of the number of windows.
    fig.text(0.5, 0.03, subtitle.replace("   |   ", "\n"),
             ha="center", va="bottom", fontsize=8, color="#555555")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.80, bottom=0.16)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run(factor_names: list[str] = DEFAULT_FACTORS,
        label: str | None = None,
        out_root: Path = OUTPUT_DIR) -> dict:
    """
    Run the straight-sum composite pipeline for ``factor_names`` and write every
    output under ``out_root / "composite" / <slug>``.  ``label`` overrides the
    slug (default: the factor names joined by ``__``).  Returns the per-window
    performance dict.
    """
    resolved = resolve_factors(factor_names)
    slug = label or "__".join(factor_names)
    out_dir = out_root / "composite" / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Experiment 3: composite z-score quintile L/S ===")
    print(f"Factors ({len(factor_names)}): " +
          ", ".join(f"{r.factor} [{'+' if r.sign > 0 else '-'}]"
                    for r in resolved.itertuples()))

    # 1-2. Assemble the (sign-oriented, equal-weighted) composite and shape it.
    composite, exposures = load_composite(resolved)
    panel = as_factor_panel(composite)

    # 3. Quintile sort (Experiment 1's code, unmodified).
    wide = Q.quintile_returns(panel, COMPOSITE_FACTOR)
    spread = wide["Q5-Q1"]

    # 4. Long/short performance over the full sample and the past decade.
    industry = industry_return()
    windows = [("Full sample", book_stats(spread, industry)),
               ("Past decade (2016+)", book_stats(spread, industry, start=DECADE_START))]
    full, decade = windows[0][1], windows[1][1]

    # Average monthly turnover cost of the Q5-Q1 book (reported, not netted).
    cost_series = COST.long_short_cost(panel, COMPOSITE_FACTOR, cost_panel(),
                                       n_quintiles=N_QUINTILES)
    full["avg_cost"] = window_cost(cost_series)
    decade["avg_cost"] = window_cost(cost_series, start=DECADE_START)
    meta = {"n_stocks": composite["stock_id"].nunique(),
            "n_months": int(full["n_months"]),
            "start": composite["date"].min(), "end": composite["date"].max()}

    # --- Persist outputs --------------------------------------------------- #
    render_factor_set(resolved, out_dir / "factor_set.png")
    exposures.corr().to_csv(out_dir / "exposure_correlation.csv")
    wide.to_csv(out_dir / "quintile_returns.csv")
    plot_cumulative(wide, factor_names, out_dir / "quintile_cumulative.png")
    plot_long_short(spread, factor_names, full["sharpe"], full["alpha"],
                    full["alpha_tstat"], out_dir / "long_short.png")
    render_performance(
        windows, "Composite long-short (Q5-Q1) performance",
        f"{_set_label(factor_names)}   |   {meta['n_stocks']} stocks over "
        f"{meta['n_months']} months ({meta['start']:%Y-%m} .. {meta['end']:%Y-%m})   |   "
        "α from regressing the book on the market-cap-weighted industry return.   "
        "Shading: |t| ≥ 1.65 (10%), 2.0 (5%).",
        out_dir / "performance.png")

    # --- Console summary --------------------------------------------------- #
    qmeans = {q: wide[q].mean() for q in QCOLS}
    print("  quintile mean next-month return: " +
          "  ".join(f"{q}={qmeans[q]:+.3%}" for q in QCOLS))
    print(f"  Q5-Q1 long/short: {full['mean_monthly']:+.4%}/mo "
          f"(t={full['tstat']:+.2f}, Sharpe={full['sharpe']:+.2f})")
    print(f"  industry-neutral alpha: {full['alpha']:+.4%}/mo "
          f"(t={full['alpha_tstat']:+.2f}, ind beta={full['ind_beta']:+.2f})")
    print(f"  2016+: alpha={decade['alpha']:+.4%}/mo (t={decade['alpha_tstat']:+.2f})")
    print(f"Saved -> {out_dir}")
    return {lbl: s for lbl, s in windows}


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] in {"top", "--top"}:
        # Combine the active cross-experiment top factors into one composite.
        names = top_factors()
        run(names, label=f"top{len(names)}")
    else:
        run(args or DEFAULT_FACTORS)


if __name__ == "__main__":
    main()
