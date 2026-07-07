"""
weighted_composite.py
=====================

Experiment 3 -- coefficient-weighted composite, evaluated walk-forward with an
**expanding-window** regression.

Same shape as ``composite.py``, but the constituents are combined with
**data-driven weights re-estimated quarterly** on all history seen so far, and the
book is repositioned quarterly and traded strictly out-of-sample.  At each
**reposition date** ``t`` (end of Feb / May / Aug / Nov, the project-wide quarterly
cadence) after an initial training period (through ``INITIAL_TRAIN_END``):

  1. Pool every stock-month **strictly before t** -- look-ahead free, only returns
     already realised by ``t`` enter -- and regress the normalised return on the
     constituent z-scores to get that quarter's premium vector ``b_f(t)``::

         (r_{i,t+1} − market_{t+1})  =  a  +  Σ_f  b_f · z_{f,i,t}  +  ε

  2. Score month ``t``'s cross-section with those weights,
     ``score_{i,t} = Σ_f b_f(t)·z_{f,i,t}`` (intercept dropped -- constant across
     stocks, so it never changes the ranking), sort into quintiles, and **hold that
     Q5-Q1 membership for the three months of the quarter** (via
     ``composite.bucket_returns`` / ``quarter_position``), the same quarterly
     repositioning every other Experiment 2-5 book uses.

The window expands each quarter, so **every traded month is a genuine out-of-sample
step**: the weights forming a quarter's book never saw that quarter's (or any later)
return.  The first traded quarter is the first reposition date after
``INITIAL_TRAIN_END`` (default: initial training through 2006, trading 2007+).

Normalised return -- the dependent variable
-------------------------------------------
The stock's month-(t+1) return minus that month's **unweighted (equal-weighted)**
industry average (the within-industry "market"), so the normalisation is not
dominated by the handful of mega-cap software names.  This is
``regression.normalized_return`` fed the equal-weighted benchmark built here
(``unweighted_industry_return``) rather than Experiment 1's cap-weighted market.

Design -- reuses composite.py wholesale
---------------------------------------
The constituent loading (``load_exposures``), the score → tidy-panel shaping
(``scored_frame`` / ``as_factor_panel``), the quarterly-held quintile sort
(``bucket_returns`` / ``quintile_legs``, which reposition quarterly via
``quarter_position``), the industry-neutral alpha and windowed performance
(``industry_return`` / ``book_stats``), the turnover cost, and the performance-
table rendering (``render_performance``) are all imported
from ``composite.py``.  The dependent variable (``regression.normalized_return``)
and the pooled month-clustered OLS (``regression.pooled_ols``) are reused from
Experiment 1's ``regression.py``.  This module adds **only** the equal-weighted
benchmark and the expanding-window, quarterly-refit walk-forward.

Outputs (``output/weighted/<slug>/``)
-------------------------------------
    beta_path.png                both paths in one stacked figure -- weights on top,
                             clustered t-stats (with ±1.65/±2.0 bands) below
    performance.png          the walk-forward Q5-Q1 book's performance summary
                             (incl. the alpha earned above each constituent's
                             standalone factor book + largest single-name ownership
                             for a $100M book)

Run standalone::

    python weighted_composite.py                                          # default set (revenue_stability + gross_profitability)
    python weighted_composite.py buyback_quality gross_profitability rd_stability
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import composite as C
import bivariate_gate as BT            # reused benchmark-relative alpha estimator
from composite import F, R             # reused engine + analysis handles

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
INITIAL_TRAIN_END = pd.Timestamp("2006-12-31")   # expanding window: trade formation months after this
MIN_TRAIN_MONTHS = 2                             # need >=2 month-clusters for the clustered SEs
OUTPUT_DIR = C.OUTPUT_DIR

# The weighted deliverable's factor set (the factors whose marginal premia we
# weight by).  Distinct from composite.py's equal-weighted top-five set.
DEFAULT_FACTORS = ["revenue_stability", "gross_profitability"]


# --------------------------------------------------------------------------- #
# Step 0 -- the equal-weighted industry benchmark the return is normalised against
# --------------------------------------------------------------------------- #
def unweighted_industry_return(panel: pd.DataFrame) -> pd.Series:
    """
    **Equal-weighted** next-period return of the whole industry cross-section,
    indexed by formation month -- the unweighted counterpart of Experiment 1's
    cap-weighted :func:`R.industry_monthly_return`.

    Built the same way (drop the last-month rows without a t+1 return, winsorise
    ``next_return`` within each month at ``F.WINSOR_PCT``), but averages every name
    equally instead of by formation-date USD cap -- so the "market" the composite
    normalises against is not dominated by the handful of mega-cap software names.
    """
    uniq = (panel.drop_duplicates(["date", "stock_id"])
                 .dropna(subset=["next_return"]).copy())
    uniq["next_return"] = F.winsorize_cross_section(
        uniq["next_return"], uniq["date"], F.WINSOR_PCT)
    return (uniq.groupby("date")["next_return"].mean()
                .rename_axis("date").rename("industry_ret").sort_index())


# --------------------------------------------------------------------------- #
# Step 1 -- the pooled-regression frame + premium estimator
#
# The dependent variable (``R.normalized_return``, fed the equal-weighted
# benchmark above) and the pooled month-clustered OLS (``R.pooled_ols``) both live
# in Experiment 1's ``regression.py``.  ``build_fit`` assembles the frame once;
# ``_premia`` runs the regression over any slice of it (full sample or an
# expanding window), so both the reference table and the walk-forward share one
# estimator.
# --------------------------------------------------------------------------- #
def build_fit(resolved: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, list[str]]:
    """
    Assemble everything the pipeline regresses on.

    Returns ``(exposures, next_ret, fit, names)`` where ``exposures`` is the wide
    ``(date, stock_id) × factor`` z-score matrix (for scoring the full
    cross-section, including the final month), ``next_ret`` the realised return,
    ``fit`` the same z-scores inner-joined to the unweighted-normalised
    ``norm_return`` (the regression's rows), and ``names`` the factor order.
    """
    names = resolved["factor"].tolist()
    exposures, next_ret = C.load_exposures(resolved)
    panel = F.load_panel(u=F.SOFTWARE_SERVICES)
    y = (R.normalized_return(panel, industry_ret=unweighted_industry_return(panel))
          .set_index(["date", "stock_id"])["norm_return"])
    fit = exposures.join(y, how="inner").dropna()
    return exposures, next_ret, fit, names


def _premia(fit_slice: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Pooled month-clustered OLS of ``norm_return`` on the constituent z-scores
    over a slice of the fit frame (indexed by ``(date, stock_id)``).  Returns
    :func:`R.pooled_ols`'s per-term table (intercept + factors)."""
    return R.pooled_ols(fit_slice["norm_return"].to_numpy(),
                        fit_slice[names].to_numpy(),
                        fit_slice.index.get_level_values("date").to_numpy(), names)


# --------------------------------------------------------------------------- #
# Step 2 -- expanding-window walk-forward
# --------------------------------------------------------------------------- #
def expanding_weights(exposures: pd.DataFrame, fit: pd.DataFrame, names: list[str],
                      initial_train_end: pd.Timestamp = INITIAL_TRAIN_END,
                      min_train_months: int = MIN_TRAIN_MONTHS
                      ) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Walk forward **quarter by quarter**.  At each reposition date ``t`` (end of Feb /
    May / Aug / Nov) after ``initial_train_end``, refit the premium regression on
    every stock-month **strictly before t** and hold those slopes over the three
    months of the quarter, weighting each held month's z-scores with them.

    Returns ``(score, beta_path, tstat_path)``:
      * ``score``      -- the quarterly-weighted score per ``(date, stock_id)`` for
                          every month of every traded quarter.  The reposition-month
                          score uses that quarter's fresh weights; the held months
                          carry them forward, so when the reused quarterly sort
                          (``composite.bucket_returns``) keeps the reposition-date
                          buckets it re-forms membership only quarterly;
      * ``beta_path``  -- reposition date × factor, the weight ``b_f(t)``;
      * ``tstat_path`` -- reposition date × factor, the clustered t-stat of ``b_f(t)``.

    Look-ahead free: the weights forming a quarter's book never see that quarter's own
    (or any later) return, so every traded month is out-of-sample.
    """
    dates = fit.index.get_level_values("date")
    all_months = sorted(exposures.index.get_level_values("date").unique())
    reposition = [t for t in all_months
                  if t > initial_train_end and t.month in C.QP.REPOSITION_MONTHS]

    score_parts, betas, tstats = [], {}, {}
    for i, t in enumerate(reposition):
        past = fit[dates < t]
        if past.index.get_level_values("date").nunique() < min_train_months:
            continue
        coef = _premia(past, names).set_index("term")
        w = coef.loc[names, "coef"]
        betas[t] = w
        tstats[t] = coef.loc[names, "t_cluster"]

        # Hold this quarter's weights over its months (t and the two calendar months
        # after it, up to the next reposition), scoring each held cross-section.
        upper = reposition[i + 1] if i + 1 < len(reposition) else None
        held = [m for m in all_months if m >= t and (upper is None or m < upper)]
        for m in held:
            z_m = exposures.xs(m, level="date")             # stock_id × factor
            s = z_m.mul(w, axis=1).sum(axis=1)
            s.index = pd.MultiIndex.from_product([[m], s.index], names=["date", "stock_id"])
            score_parts.append(s)

    score = pd.concat(score_parts)
    beta_path = pd.DataFrame(betas).T.rename_axis("date").sort_index()
    tstat_path = pd.DataFrame(tstats).T.rename_axis("date").sort_index()
    return score, beta_path, tstat_path


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def plot_paths(beta_path: pd.DataFrame, tstat_path: pd.DataFrame,
               factor_names: list[str], out_path: Path) -> None:
    """Plot both expanding-window paths in a **single** stacked figure sharing the
    formation-month x-axis: the factor weights ``b_f(t)`` on top and their
    clustered t-stats (with the ±1.65/±2.0 significance bands) below.  One legend
    covers both panels since the two share the same factor colours."""
    fig, (ax_w, ax_t) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for col in beta_path.columns:
        ax_w.plot(beta_path.index, beta_path[col], label=col, linewidth=1.3)
    for col in tstat_path.columns:
        ax_t.plot(tstat_path.index, tstat_path[col], label=col, linewidth=1.3)

    ax_w.axhline(0.0, color="black", linewidth=0.6)
    ax_w.set_ylabel("coefficient = weight\n(ind-rel /mo per 1σ)")
    ax_w.set_title("weight  b_f(t)")
    ax_w.grid(True, alpha=0.3)
    ax_w.legend(title="Factor", ncol=min(len(beta_path.columns), 4),
                loc="best", fontsize=8)

    ax_t.axhline(0.0, color="black", linewidth=0.6)
    for y in (-2.0, -1.65, 1.65, 2.0):
        ax_t.axhline(y, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
    ax_t.set_ylabel("clustered t-stat")
    ax_t.set_title("clustered t-stat")
    ax_t.set_xlabel("Formation month (expanding-window train end)")
    ax_t.grid(True, alpha=0.3)

    fig.suptitle("Experiment 3 -- expanding-window factor weights & t-stats over time\n"
                 f"{' + '.join(factor_names)}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _benchmark_extra_metrics(factor_names: list[str]) -> list[tuple[str, str, str, bool]]:
    """``render_performance`` rows for the alpha (+ t-stat) the weighted book earns
    above each constituent's standalone factor book -- keyed to match
    :func:`bivariate_gate.benchmark_alphas`' ``alpha_vs_<f>`` output."""
    extra: list[tuple[str, str, str, bool]] = []
    for f in factor_names:
        extra.append((f"α vs {f} book (monthly)", f"alpha_vs_{f}", "pct", False))
        extra.append((f"    α t-stat vs {f}", f"alpha_tstat_vs_{f}", "num", True))
    return extra


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run(factor_names: list[str] = DEFAULT_FACTORS,
        initial_train_end: pd.Timestamp = INITIAL_TRAIN_END,
        label: str | None = None,
        out_root: Path = OUTPUT_DIR) -> dict:
    """
    Build the expanding-window coefficient-weighted composite (weights refit every
    month on all history before it) and evaluate the walk-forward Q5-Q1 book.
    Writes every output under ``out_root / "weighted" / <slug>`` and returns the
    walk-forward performance.
    """
    resolved = C.resolve_factors(factor_names)
    slug = label or "__".join(factor_names)
    out_dir = out_root / "weighted" / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Experiment 3: coefficient-weighted composite "
          f"(expanding window, initial train <= {initial_train_end:%Y}, trade after) ===")
    print(f"Factors ({len(factor_names)}): {', '.join(factor_names)}")

    # 1. Exposures + pooled-regression frame (normalised vs the unweighted industry mean).
    exposures, next_ret, fit, names = build_fit(resolved)

    # 2. Full-sample reference premia (t-stats over the entire period, printed to
    #    the console as a sanity check; the traded book uses the expanding-window
    #    weights, not these).
    coef_full = _premia(fit, names)

    # 3. Expanding-window weights -> walk-forward score + beta / t-stat paths.
    score, beta_path, tstat_path = expanding_weights(exposures, fit, names, initial_train_end)

    # 4. Shape for the reused quarterly sort and build the Q5-Q1 book: the composite
    #    is bucketed with ``composite.bucket_returns``, which forms quintiles at each
    #    reposition date and holds them for the quarter (the same quarterly membership
    #    the weights were refit on).  The sort is internal -- no quintile files written.
    panel = C.as_factor_panel(C.scored_frame(score, next_ret))
    wide = C.bucket_returns(panel)
    spread = wide["Q5-Q1"]

    # 5. Walk-forward (out-of-sample by construction) performance of the book.
    industry = C.industry_return()
    traded = spread.dropna()
    oos_start, oos_end = traded.index.min(), traded.index.max()
    win_label = f"Walk-forward OOS ({oos_start:%Y}+)"
    stats = C.book_stats(spread, industry)
    legs = C.quintile_legs(panel)
    cost_series = C.COST.turnover_cost(legs, C.cost_panel())
    stats["avg_cost"] = C.window_cost(cost_series)
    C.attach_net_cost_sharpe(stats, spread, cost_series, industry)
    # Largest single-name ownership for a $100M dollar-neutral book (worst case over
    # the walk-forward window), from the same Q5/Q1 leg membership the cost uses.
    C.attach_ownership(stats, C.leg_ownership(legs))
    # Alpha the walk-forward book earns *above* each constituent's standalone factor
    # book -- does weighting the constituents add return beyond simply holding them?
    # (same benchmark-relative estimator bivariate_gate uses; regressed only over
    # the overlapping OOS months since market_regression inner-joins on the index).
    benchmarks = {f: C.factor_long_short(f) for f in factor_names}
    stats.update(BT.benchmark_alphas(spread, benchmarks))
    windows = [(win_label, stats)]

    # --- Persist outputs --------------------------------------------------- #
    plot_paths(beta_path, tstat_path, factor_names, out_dir / "beta_path.png")

    C.render_performance(
        windows,
        "Coefficient-weighted composite long-short (Q5-Q1): expanding-window walk-forward",
        f"expanding window: weights refit each month on all prior history "
        f"(initial train <= {initial_train_end:%Y})   |   "
        f"factors: {' + '.join(factor_names)}   |   "
        "every traded month is out-of-sample   |   "
        "Shading: |t| ≥ 1.65 (10%), 2.0 (5%).",
        out_dir / "performance.png",
        extra_metrics=_benchmark_extra_metrics(factor_names) + [C.ownership_metric()])

    # --- Console summary --------------------------------------------------- #
    print("Full-sample premia (reference t-stats over the entire period):")
    print(coef_full[["term", "coef", "t_ols", "t_cluster"]].to_string(index=False))
    print(f"Traded months: {len(traded)} ({oos_start:%Y-%m} .. {oos_end:%Y-%m})")
    print(f"  OOS Q5-Q1: mean={stats['mean_monthly']:+.4%}/mo (t={stats['tstat']:+.2f}), "
          f"Sharpe={stats['sharpe']:+.2f}, alpha={stats['alpha']:+.4%}/mo "
          f"(t={stats['alpha_tstat']:+.2f}), ind beta={stats['ind_beta']:+.2f}")
    print(f"Saved -> {out_dir}")
    return {lbl: s for lbl, s in windows}


def main() -> None:
    factor_names = sys.argv[1:] or DEFAULT_FACTORS
    run(factor_names)


if __name__ == "__main__":
    main()
