"""
regression.py
=============

Approach 2 -- monthly cross-sectional regression.

For every factor, each month we regress the **next-period** (month t+1) return
of all securities on the factor z-score via OLS:

    next_return_i = alpha_t + beta_t * zscore_i + eps_i

Re-estimating every month gives one time series of slopes (``beta_t``) per
factor -- the cross-sectional "factor premium".

Alongside this per-month view, a single **pooled** regression complements it: the
**industry-relative** return (each stock's month-t+1 return minus the cap-weighted
industry return -- :func:`normalized_return`) regressed on the factor z-score over
**every stock-month at once** (pooled over both time and the cross section), with
month-clustered standard errors.  Where the monthly slopes answer "how does the
premium move over time", the pooled slope answers "what is the factor's average
industry-relative payoff per 1 sigma across the whole panel".  The
``normalized_return`` / ``pooled_ols`` helpers are reusable -- Experiment 3's
coefficient-weighted composite consumes them directly.

Outputs, per factor:

    * ``output/regression/<factor>/regression.csv`` -- months x {beta,
      alpha, tstat, r2, n}, the monthly regression diagnostics.
    * ``output/regression/<factor>/beta.png`` -- the beta over time.
    * ``output/quintile/<factor>/long_short.png`` -- a
      dollar-neutral long-short quintile book and its cumulative growth-of-$1
      path over time.  Its direction follows each factor's canonical literature
      sign when the factor library opts in (``USE_CANONICAL_LS_DIRECTION``;
      Experiment 1: long high-bullish factors Q5 / short Q1, else Q1 / short
      Q5); otherwise it is inferred from the factor's full-sample Fama-MacBeth
      t-stat (t>0: long Q5 / short Q1; t<0: long Q1 / short Q5).  The header
      reports the book's Sharpe ratio and its industry-neutral alpha (with
      t-stat).

A cross-factor ``output/regression/summary.csv`` reports each factor's mean
monthly beta and its Fama-MacBeth t-statistic (time-series mean of the monthly
betas divided by its standard error), plus the directional long-short book's
mean return, t-stat and annualised return.  A companion
``output/regression/normalized_regression.csv`` (+ ``..._table.png``) reports the
pooled time-and-cross-section slope of the industry-relative return on each factor
z-score, with OLS and month-clustered t-stats (full sample and 2016+).  The
``output/quintile/long_short_market_alpha`` table additionally carries each
book's industry-neutral alpha and its **average turnover cost** (pp/month).

This module is a library: :func:`run` is driven by the per-universe
orchestrators (``software_service.py``, ``banks_insurance.py``,
``commodity_producers.py``), which build the panel once and reuse it across
both analyses.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import cost
import factors as F

DECADE_START = pd.Timestamp("2016-01-01")  # "past decade" cut-off
N_QUINTILES = 5
MONTHS_PER_YEAR = 12

# Minimum months of history before the walk-forward industry hedge activates.  The
# beta-neutral Sharpe hedges each month with a beta estimated on an *expanding*
# window of only the data available up to that month (no look-ahead), so the first
# HEDGE_MIN_MONTHS months -- too little history for a stable slope -- are left
# unhedged and drop out of the hedged series.  36 months (3 years) is the textbook
# monthly-beta estimation window.
HEDGE_MIN_MONTHS = 36


# --------------------------------------------------------------------------- #
# Core computation
# --------------------------------------------------------------------------- #
def monthly_regressions(panel: pd.DataFrame, factor: str) -> pd.DataFrame:
    """
    Monthly OLS diagnostics for a factor: each month regress next_return on the
    factor z-score across the whole industry cross-section.  Indexed by month.
    """
    sub = F.prepare_slice(panel, factor, assign_q=False)

    records = []
    for date, grp in sub.groupby("date", observed=True):
        stats = F.ols(grp["zscore"].to_numpy(), grp["next_return"].to_numpy())
        stats["date"] = date
        records.append(stats)

    out = pd.DataFrame.from_records(records).set_index("date").sort_index()
    return out[["beta", "alpha", "tstat", "r2", "n"]]


def fama_macbeth(beta: pd.Series) -> dict:
    """Time-series summary of a monthly beta series (the factor premium)."""
    b = beta.dropna()
    n = b.size
    mean = b.mean()
    fm_t = mean / (b.std(ddof=1) / np.sqrt(n)) if n > 1 and b.std() > 0 else np.nan
    return {"mean_beta": mean, "fm_tstat": fm_t,
            "pct_positive": (b > 0).mean() if n else np.nan, "n_months": n}


# --------------------------------------------------------------------------- #
# Directional long-short portfolio
# --------------------------------------------------------------------------- #
def long_short_portfolio(panel: pd.DataFrame, factor: str,
                         fm_tstat: float) -> tuple[pd.Series, int]:
    """
    Build a dollar-neutral (long-leg minus short-leg) monthly return series for
    ``factor`` and return it with the trade sign.  The direction is set one of
    two ways, depending on the factor library:

      * Canonical (``F.USE_CANONICAL_LS_DIRECTION`` is truthy -- Experiment 1):
        the literature-expected direction from ``F.FACTORS[factor]
        ['higher_is_bullish']`` -- long Q5 / short Q1 if bullish-high, else
        long Q1 / short Q5.  Independent of the in-sample data.

      * Inferred (attribute absent -- Experiment 2): the sign of the factor's
        full-sample Fama-MacBeth t-stat::

            fm_tstat > 0  ->  long top quintile (Q5), short bottom quintile (Q1)
            fm_tstat < 0  ->  long bottom quintile (Q1), short top quintile (Q5)

    Each leg is the equal-weighted next-period mean return of its quintile, so
    the spread is the return of a $1-long / $1-short, zero-net-investment book.
    Returns the monthly spread series (indexed by month) and the trade sign.
    """
    sub = F.prepare_slice(panel, factor, N_QUINTILES)
    legs = (sub.pivot_table(index="date", columns="quintile",
                            values="next_return", aggfunc="mean")
               .sort_index())
    top, bottom = legs.get(float(N_QUINTILES)), legs.get(1.0)

    if getattr(F, "USE_CANONICAL_LS_DIRECTION", False):
        sign = 1 if F.FACTORS[factor]["higher_is_bullish"] else -1
    else:
        sign = -1 if (np.isfinite(fm_tstat) and fm_tstat < 0) else 1
    spread = (top - bottom) if sign > 0 else (bottom - top)
    return spread.rename(f"{factor}_ls"), sign


def long_short_stats(spread: pd.Series) -> dict:
    """Mean / t-stat / Sharpe / annualised summary of the long-short return series."""
    s = spread.dropna()
    n = s.size
    mean = s.mean()
    sd = s.std(ddof=1)
    tstat = mean / (sd / np.sqrt(n)) if n > 1 and sd > 0 else np.nan
    sharpe = (mean / sd) * np.sqrt(MONTHS_PER_YEAR) if n > 1 and sd > 0 else np.nan
    return {"mean_monthly": mean, "tstat": tstat, "sharpe": sharpe,
            "ann_return": mean * MONTHS_PER_YEAR, "n_months": n}


def _window(series: pd.Series, start: pd.Timestamp | None = None,
            end: pd.Timestamp | None = None) -> pd.Series:
    """Restrict a month-indexed series to an optional ``[start, end]`` window."""
    if start is not None:
        series = series[series.index >= start]
    if end is not None:
        series = series[series.index <= end]
    return series


def expanding_betas(spread: pd.Series, market: pd.Series,
                    min_months: int = HEDGE_MIN_MONTHS) -> pd.Series:
    """
    Walk-forward industry betas, free of look-ahead.

    For each formation month ``t`` with at least ``min_months`` earlier joint
    observations, the industry beta is the OLS slope of the long-short spread on the
    industry ("market") return estimated over **every month strictly before t** -- an
    expanding window of exactly the data available at t.  It is re-estimated every
    month via the closed-form ``cov(spread, market) / var(market)`` accumulated with
    prefix sums (O(n) over the whole series).  Returns one beta per hedgeable month
    (indexed by t); months without ``min_months`` of prior history are omitted.
    """
    df = pd.concat([spread.rename("y"), market.rename("x")], axis=1).dropna().sort_index()
    n = len(df)
    if n <= min_months:
        return pd.Series(dtype=float, name="beta")
    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    # Prefix sums: cs*[k] aggregates the first k observations (the months strictly
    # before position k), so the beta hedging month k is fit on prior data only.
    csx = np.concatenate([[0.0], np.cumsum(x)])
    csy = np.concatenate([[0.0], np.cumsum(y)])
    csxx = np.concatenate([[0.0], np.cumsum(x * x)])
    csxy = np.concatenate([[0.0], np.cumsum(x * y)])
    k = np.arange(min_months, n)
    m = k.astype(float)                                   # prior-obs count = position
    denom = m * csxx[k] - csx[k] ** 2
    beta = np.where(denom > 0, (m * csxy[k] - csx[k] * csy[k]) / denom, np.nan)
    return pd.Series(beta, index=df.index[k], name="beta")


def _hedge_with_betas(returns: pd.Series, market: pd.Series,
                      betas: pd.Series) -> pd.Series:
    """Industry-hedged return ``returns_t - beta_t * market_t`` over the months where
    the return, the market return and a walk-forward beta are all defined."""
    df = pd.concat([returns.rename("r"), market.rename("m"), betas.rename("b")],
                   axis=1).dropna()
    return (df["r"] - df["b"] * df["m"]).rename("hedged")


def beta_neutral_sharpe(spread: pd.Series, market: pd.Series,
                        start: pd.Timestamp | None = None,
                        end: pd.Timestamp | None = None,
                        min_months: int = HEDGE_MIN_MONTHS) -> float:
    """
    Annualised Sharpe of the industry-beta-neutralised book, hedged **without
    look-ahead**.

    The book's industry exposure is neutralised by overlaying a ``-beta`` position in
    the industry ("market") return, but ``beta`` is re-estimated every month on an
    *expanding window* of only the data available up to that month
    (:func:`expanding_betas`) rather than a single full-sample slope -- so the hedge
    an investor could actually have put on carries no future information.  The Sharpe
    is of that walk-forward hedged series (:func:`_hedge_with_betas`) restricted to
    the optional ``[start, end]`` window; because each month's beta always uses all
    prior history, a window starting in 2016 still hedges with betas fit on
    2001-onward data.  Returns NaN if no month has enough history to hedge.
    """
    hedged = _hedge_with_betas(spread, market,
                               expanding_betas(spread, market, min_months))
    return long_short_stats(_window(hedged, start, end))["sharpe"]


def _net_of_cost_spread(spread: pd.Series, cost_series: pd.Series,
                        start: pd.Timestamp | None = None,
                        end: pd.Timestamp | None = None) -> pd.Series:
    """The long-short book's monthly return **net of turnover cost** over an optional
    ``[start, end]`` window: the gross spread less the per-formation-month turnover
    cost (:func:`cost.long_short_cost` / :func:`cost.turnover_cost`), aligned on the
    spread's month index (a month with no recorded cost is charged zero)."""
    net = spread.subtract(cost_series.reindex(spread.index).fillna(0.0))
    if start is not None:
        net = net[net.index >= start]
    if end is not None:
        net = net[net.index <= end]
    return net


def net_of_cost_sharpe(spread: pd.Series, cost_series: pd.Series,
                       start: pd.Timestamp | None = None,
                       end: pd.Timestamp | None = None) -> float:
    """
    Annualised Sharpe of the long-short book's return **net of turnover cost**.

    Trading cost is a per-month drag charged on turnover, so netting it from the
    gross spread (:func:`_net_of_cost_spread`) and taking the standard annualised
    Sharpe gives the risk-adjusted return an investor actually realises after paying
    to trade the book -- the cost-incorporated counterpart of the gross ``sharpe``
    every book also reports.  Returns NaN if the net series is empty.
    """
    return long_short_stats(_net_of_cost_spread(spread, cost_series, start, end))["sharpe"]


def net_of_cost_neutral_sharpe(spread: pd.Series, cost_series: pd.Series,
                               market: pd.Series,
                               start: pd.Timestamp | None = None,
                               end: pd.Timestamp | None = None,
                               min_months: int = HEDGE_MIN_MONTHS) -> float:
    """
    Annualised Sharpe of the **industry-beta-neutralised, net-of-cost** book, hedged
    without look-ahead.

    Combines :func:`net_of_cost_sharpe` and :func:`beta_neutral_sharpe`: net the
    turnover cost from the gross spread (:func:`_net_of_cost_spread`), then hedge that
    net series with the same walk-forward industry betas (:func:`expanding_betas`,
    estimated on the *gross* book -- its true industry exposure) that back
    ``sharpe_neutral``.  The Sharpe is of the hedged net series over the optional
    ``[start, end]`` window.  This is the cost-incorporated counterpart of the
    beta-neutral Sharpe.  Returns NaN if no month has enough history to hedge.
    """
    net = _net_of_cost_spread(spread, cost_series)
    hedged = _hedge_with_betas(net, market,
                               expanding_betas(spread, market, min_months))
    return long_short_stats(_window(hedged, start, end))["sharpe"]


# --------------------------------------------------------------------------- #
# Long-short return regressed on the industry ("market") return
# --------------------------------------------------------------------------- #
def industry_monthly_return(panel: pd.DataFrame) -> pd.Series:
    """
    Market-cap-weighted next-period return of the whole industry cross-section,
    indexed by formation month.

    This is the within-industry "market" return.  It is built from the same
    ``next_return`` column the long-short spread is -- including the same
    within-month winsorisation the spread legs get via ``prepare_slice`` -- so
    both series share the formation-date index, the identical (month t+1) return
    period, and the same tail treatment, and can be regressed directly.  Each
    stock is weighted by its formation-date USD market cap (``weight`` in the
    panel), which is the cap at the start of the t+1 return period, so the index is
    look-ahead free; names without a cap drop out of the weighted mean.
    """
    uniq = (panel.drop_duplicates(["date", "stock_id"])
                 .dropna(subset=["next_return"])
                 .copy())
    uniq["next_return"] = F.winsorize_cross_section(
        uniq["next_return"], uniq["date"], F.WINSOR_PCT)
    return (F.weighted_group_mean(uniq["next_return"], uniq["weight"], uniq["date"])
                .rename_axis("date").rename("industry_ret").sort_index())


def market_regression(spread: pd.Series, market: pd.Series) -> dict:
    """
    Time-series OLS of a long-short strategy's monthly return on the industry's
    market-cap-weighted monthly return:

        ls_t = alpha + beta * industry_t + eps_t

    ``alpha`` is the strategy return left unexplained by industry exposure (its
    industry-neutral mean return); ``alpha_tstat`` tests whether that alpha is
    non-zero.  ``beta`` is the book's net exposure to the industry.
    """
    df = pd.concat([spread.rename("y"), market.rename("x")], axis=1).dropna()
    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    n = x.size
    nan = {"alpha": np.nan, "alpha_tstat": np.nan, "beta": np.nan,
           "beta_tstat": np.nan, "r2": np.nan, "n": n}
    if n < 3 or np.ptp(x) == 0:
        return nan

    X = np.column_stack([np.ones(n), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha, beta = coef
    resid = y - X @ coef
    dof = n - 2
    sigma2 = (resid @ resid) / dof
    cov = sigma2 * np.linalg.inv(X.T @ X)         # 2x2 coefficient covariance
    se_alpha, se_beta = np.sqrt(np.diag(cov))
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1.0 - (resid @ resid) / ss_tot if ss_tot > 0 else np.nan
    return {"alpha": alpha,
            "alpha_tstat": alpha / se_alpha if se_alpha > 0 else np.nan,
            "beta": beta,
            "beta_tstat": beta / se_beta if se_beta > 0 else np.nan,
            "r2": r2, "n": n}


# --------------------------------------------------------------------------- #
# Normalised (industry-relative) return + pooled time-and-cross-section
# regression
#
# The monthly regressions above re-estimate one slope per month and Fama-MacBeth
# the series ("beta over time").  These helpers add the complementary view: a
# single pooled regression of the **industry-relative** return on the factor
# z-score across every stock-month at once (pooled over both time and the cross
# section), clustered by month.  ``normalized_return`` and ``pooled_ols`` are the
# reusable spine -- Experiment 3's coefficient-weighted composite consumes them
# directly instead of re-implementing its own.
# --------------------------------------------------------------------------- #
def normalized_return(panel: pd.DataFrame,
                      industry_ret: pd.Series | None = None) -> pd.DataFrame:
    """
    Per stock-month industry-relative next return -- the winsorised month-(t+1)
    return less that month's market-cap-weighted industry average -- indexed by
    formation month ``t``.  Returns ``date, stock_id, norm_return``.

    The subtracted average is :func:`industry_monthly_return` (cap-weighted), the
    same within-industry "market" used for every industry-neutral alpha in the
    project, so the normalised return is industry-relative against an identical
    benchmark.  ``next_return`` is winsorised within each month at ``F.WINSOR_PCT``
    -- the project's standard tail treatment, the same clip :func:`prepare_slice`
    applies -- before the industry mean is removed.  Pass a precomputed
    ``industry_ret`` (e.g. the one :func:`run` already built) to skip recomputing
    it.

    Reusable across experiments: Experiment 3's ``weighted_composite`` uses this
    as the dependent variable of its in-sample premium regression.
    """
    if industry_ret is None:
        industry_ret = industry_monthly_return(panel)
    uniq = (panel.drop_duplicates(["date", "stock_id"])
                 .dropna(subset=["next_return"]).copy())
    uniq["next_return"] = F.winsorize_cross_section(
        uniq["next_return"], uniq["date"], F.WINSOR_PCT)
    uniq["norm_return"] = uniq["next_return"] - uniq["date"].map(industry_ret)
    return uniq[["date", "stock_id", "norm_return"]]


def _norm_cdf(x: float) -> float:
    """Standard-normal CDF (large-sample p-values for the clustered t-stats)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def pooled_ols(y: np.ndarray, X: np.ndarray, clusters: np.ndarray,
               factor_names: list[str]) -> pd.DataFrame:
    """
    Pooled OLS of ``y`` on ``[1, X]`` with homoskedastic-OLS and cluster-robust
    (CR1, clustered on ``clusters`` = month) standard errors.

    Clustering by month is the appropriate standard error for a pooled
    cross-section of returns -- same-month residuals are cross-sectionally
    correlated, which plain OLS ignores.  Returns one row per term (intercept +
    factors) with coef, both SEs / t-stats, and a normal-approx clustered p-value.
    """
    y = np.asarray(y, dtype=float)
    Xf = np.asarray(X, dtype=float)
    n, k = Xf.shape
    Xd = np.column_stack([np.ones(n), Xf])            # design: intercept + factors
    p = k + 1

    XtX_inv = np.linalg.pinv(Xd.T @ Xd)
    beta = XtX_inv @ (Xd.T @ y)
    resid = y - Xd @ beta

    # Homoskedastic (classical OLS) covariance.
    sigma2 = (resid @ resid) / (n - p)
    se_ols = np.sqrt(np.diag(sigma2 * XtX_inv))

    # Cluster-robust (CR1) covariance, clustered by month.
    meat = np.zeros((p, p))
    groups = pd.Series(np.arange(n)).groupby(np.asarray(clusters)).groups
    for idx in groups.values():
        rows = np.asarray(idx)
        s = Xd[rows].T @ resid[rows]                  # cluster score sum
        meat += np.outer(s, s)
    G = len(groups)
    correction = (G / (G - 1)) * ((n - 1) / (n - p)) if G > 1 else 1.0
    se_cl = np.sqrt(np.diag(correction * (XtX_inv @ meat @ XtX_inv)))

    t_ols = beta / se_ols
    t_cl = beta / se_cl
    p_cl = [2.0 * (1.0 - _norm_cdf(abs(t))) if np.isfinite(t) else np.nan for t in t_cl]

    return pd.DataFrame({
        "term": ["intercept"] + list(factor_names),
        "coef": beta,
        "se_ols": se_ols, "t_ols": t_ols,
        "se_cluster": se_cl, "t_cluster": t_cl, "p_cluster": p_cl,
        "n_obs": n, "n_clusters_months": G,
    })


def normalized_regression(panel: pd.DataFrame, factor: str, norm: pd.DataFrame,
                          start: pd.Timestamp | None = None,
                          end: pd.Timestamp | None = None) -> pd.DataFrame:
    """
    Pooled (time x cross-section) OLS of the industry-relative return on a single
    factor's z-score, clustered by month, over an optional ``[start, end]`` window.

    Complements :func:`monthly_regressions`: where that re-estimates a fresh slope
    each month and Fama-MacBeths the series, this pools every stock-month into one
    regression of the **normalised** return (``norm``, from
    :func:`normalized_return`) on the z-score, so the slope is the factor's average
    industry-relative return per 1 cross-sectional sigma across the whole panel.
    The z-score is the panel's (build-time winsorised) ``zscore`` joined to the
    same-key normalised return; both sides therefore carry the project's standard
    winsorisation.  Returns the intercept + factor coefficient rows from
    :func:`pooled_ols`.
    """
    sub = F.prepare_slice(panel, factor, assign_q=False)
    merged = (sub.merge(norm, on=["date", "stock_id"], how="inner")
                 .dropna(subset=["zscore", "norm_return"]))
    if start is not None:
        merged = merged[merged["date"] >= start]
    if end is not None:
        merged = merged[merged["date"] <= end]
    return pooled_ols(merged["norm_return"].to_numpy(),
                      merged[["zscore"]].to_numpy(),
                      merged["date"].to_numpy(), [factor])


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_long_short(spread: pd.Series, factor: str, sign: int,
                    sharpe: float, alpha: float, alpha_tstat: float,
                    path: Path) -> None:
    """Cumulative growth of $1 in the directional dollar-neutral L/S book.

    The header reports the portfolio's annualised Sharpe ratio and its
    industry-neutral alpha (monthly) with that alpha's t-statistic.
    """
    direction = ("long Q5 / short Q1" if sign > 0 else "long Q1 / short Q5")
    cum = (1.0 + spread.fillna(0.0)).cumprod()
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(cum.index, cum, color="C2", linewidth=1.3)
    ax.axhline(1.0, color="black", linewidth=0.6)
    ax.set_title("Dollar-neutral long-short portfolio: cumulative growth of $1\n"
                 f"{factor}  ({F.FACTORS[factor]['family']})  --  {direction}  "
                 f"[Sharpe={sharpe:+.2f}, alpha={alpha:+.4%}/mo, "
                 f"t(alpha)={alpha_tstat:+.2f}]")
    ax.set_xlabel("Month")
    ax.set_ylabel("Cumulative value of $1")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_beta(reg: pd.DataFrame, factor: str, path: Path) -> None:
    beta = reg["beta"]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(beta.index, beta, color="C0", linewidth=0.9, label="monthly beta")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.axhline(beta.mean(), color="C3", linewidth=1.2, linestyle="--",
               label=f"mean = {beta.mean():+.4f}")
    ax.set_title("Cross-sectional regression slope (next return on factor z-score)\n"
                 f"{factor}  ({F.FACTORS[factor]['family']})")
    ax.set_xlabel("Month")
    ax.set_ylabel("Regression beta")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _tstat_color(t: float) -> str:
    """Light green / amber shading by absolute significance of a t-stat."""
    if not np.isfinite(t):
        return "white"
    a = abs(t)
    if a >= 2.0:
        return "#bfe3bf"   # significant at ~5%
    if a >= 1.65:
        return "#e8f2d8"   # significant at ~10%
    return "white"


def render_summary_table(summary: pd.DataFrame, path: Path) -> None:
    """Render the per-factor mean-beta / FM-t table (full + 2016+) as a PNG."""
    cols = ["factor", "family",
            "mean_beta_full", "fm_t_full",
            "mean_beta_2016", "fm_t_2016"]
    headers = ["Factor", "Family",
               "Mean β\n(full)", "FM t\n(full)",
               "Mean β\n(2016+)", "FM t\n(2016+)"]

    cell_text, cell_colors = [], []
    for _, r in summary[cols].iterrows():
        cell_text.append([
            r["factor"], r["family"],
            f"{r['mean_beta_full']:+.4f}", f"{r['fm_t_full']:+.2f}",
            f"{r['mean_beta_2016']:+.4f}", f"{r['fm_t_2016']:+.2f}",
        ])
        cell_colors.append([
            "white", "white",
            "white", _tstat_color(r["fm_t_full"]),
            "white", _tstat_color(r["fm_t_2016"]),
        ])

    n = len(summary)
    fig, ax = plt.subplots(figsize=(11, 0.45 * (n + 1) + 1.4))
    ax.axis("off")

    tbl = ax.table(cellText=cell_text, colLabels=headers, cellColours=cell_colors,
                   colWidths=[0.16, 0.23, 0.1525, 0.1525, 0.1525, 0.1525],
                   cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for j in range(len(headers)):                       # bold header row
        tbl[0, j].set_text_props(weight="bold", color="white")
        tbl[0, j].set_facecolor("#404040")
    for i in range(1, n + 1):                           # left-align text cols
        tbl[i, 0].set_text_props(ha="left")
        tbl[i, 1].set_text_props(ha="left")

    fig.suptitle("Cross-sectional factor premia: mean monthly β and "
                 "Fama-MacBeth t-stat\n(next-period return on factor z-score; "
                 "β = return per 1 std of factor)", fontsize=11, y=0.99)
    fig.text(0.5, 0.015, "Shaded FM t-stats: |t| ≥ 1.65 (10%), darker |t| ≥ 2.0 (5%)",
             ha="center", fontsize=8, color="#555555")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.80, bottom=0.10)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render_normalized_table(rows: pd.DataFrame, path: Path) -> None:
    """
    Render the per-factor pooled normalised-return regression as a PNG.

    One row per factor: the pooled slope (industry-relative monthly return per 1
    cross-sectional sigma) with its OLS and month-clustered t-stats over the full
    sample, the pooled observation count, and the slope / clustered t-stat
    re-estimated on the past decade (2016+).  Clustered t-stats are shaded by
    absolute significance.
    """
    cols = ["factor", "family", "coef_full", "t_ols_full", "t_cluster_full",
            "n_obs", "coef_2016", "t_cluster_2016"]
    headers = ["Factor", "Family", "Coef\n(ind-rel /mo per 1σ)",
               "t (OLS)\n(full)", "t (cluster)\n(full)", "n obs",
               "Coef\n(2016+)", "t (cluster)\n(2016+)"]

    cell_text, cell_colors = [], []
    for _, r in rows[cols].iterrows():
        cell_text.append([
            r["factor"], r["family"],
            f"{r['coef_full']:+.4%}", f"{r['t_ols_full']:+.2f}",
            f"{r['t_cluster_full']:+.2f}", f"{int(r['n_obs']):,}",
            f"{r['coef_2016']:+.4%}", f"{r['t_cluster_2016']:+.2f}",
        ])
        cell_colors.append([
            "white", "white", "white",
            _tstat_color(r["t_ols_full"]), _tstat_color(r["t_cluster_full"]),
            "white", "white", _tstat_color(r["t_cluster_2016"]),
        ])

    n = len(rows)
    fig, ax = plt.subplots(figsize=(13, 0.45 * (n + 1) + 1.4))
    ax.axis("off")

    tbl = ax.table(cellText=cell_text, colLabels=headers, cellColours=cell_colors,
                   colWidths=[0.16, 0.21, 0.155, 0.105, 0.115, 0.10, 0.085, 0.115],
                   cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for j in range(len(headers)):                       # bold header row
        tbl[0, j].set_text_props(weight="bold", color="white")
        tbl[0, j].set_facecolor("#404040")
    for i in range(1, n + 1):                           # left-align text cols
        tbl[i, 0].set_text_props(ha="left")
        tbl[i, 1].set_text_props(ha="left")

    fig.suptitle("Pooled time-and-cross-section regression of the industry-relative "
                 "return on factor z-score\n(norm_return = next return − cap-weighted "
                 "industry return;  pooled over all stock-months, t clustered by month)",
                 fontsize=11, y=0.99)
    fig.text(0.5, 0.015, "Shaded t-stats: |t| ≥ 1.65 (10%), darker |t| ≥ 2.0 (5%).  "
             "Coef = industry-relative monthly return per 1 std of the factor.  "
             "2016+ columns re-estimate on the past decade only.",
             ha="center", fontsize=8, color="#555555")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.80, bottom=0.10)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fmt_pct(x: float) -> str:
    return f"{x:+.4%}" if np.isfinite(x) else "—"


def _fmt_num(x: float, prec: int = 2) -> str:
    return f"{x:+.{prec}f}" if np.isfinite(x) else "—"


def _fmt_stat_cell(value: float, tstat: float, is_pct: bool) -> str:
    """Stack a coefficient over its t-stat in one table cell, e.g.
    ``"+1.25%\\n(t +4.33)"`` (alpha) or ``"-0.30\\n(t -5.20)"`` (beta)."""
    head = _fmt_pct(value) if is_pct else _fmt_num(value)
    tail = f"(t {tstat:+.2f})" if np.isfinite(tstat) else "(t —)"
    return f"{head}\n{tail}"


def _fmt_net_sharpe_cell(raw: float, neutral: float) -> str:
    """Stack the two cost-incorporated Sharpes in one table cell: the raw
    net-of-cost Sharpe over the industry-beta-neutralised net-of-cost Sharpe, e.g.
    ``"+0.58\\n(βn +0.92)"``.  Both are the return net of turnover cost."""
    bot = f"(βn {neutral:+.2f})" if np.isfinite(neutral) else "(βn —)"
    return f"{_fmt_num(raw)}\n{bot}"


def render_alpha_table(rows: pd.DataFrame, path: Path,
                       title: str | None = None) -> None:
    """
    Render the per-factor long-short-vs-industry regression as a PNG table.

    Each window (full sample, then the trailing 2016+ block) reports the
    industry-neutral **alpha** stacked over its t-stat, alongside the annualised
    Sharpe, the industry-beta-neutralised Sharpe (the book hedged with a
    walk-forward ``-beta*industry`` overlay whose beta is re-estimated on an
    expanding, look-ahead-free window -- :func:`beta_neutral_sharpe`), a combined
    **cost-incorporated Sharpe** cell (the raw net-of-cost Sharpe stacked over its
    beta-neutral net-of-cost counterpart) and the average turnover cost.  Alpha
    cells are shaded by the absolute significance of their t-stat.

    The cost-incorporated Sharpe fields (``sharpe_cost`` / ``sharpe_cost_neutral``
    full, ``sharpe_cost_2016`` / ``sharpe_cost_neutral_2016`` for the decade) are
    read defensively so a table built before they were populated still renders (the
    cell shows "—").

    ``title`` overrides the figure's suptitle; when ``None`` (the default) the
    standard quintile-book caption is used, so existing callers are unchanged.
    A book built on a different bucketing (e.g. tertiles) passes its own title.
    """
    headers = ["Factor", "Family", "L/S\ndirection",
               "Alpha (monthly)\n& t-stat",
               "Sharpe\n(ann.)", "β-neutral\nSharpe",
               "Sharpe net cost\n(raw / β-neut)", "Avg cost\n(monthly)", "n",
               "Alpha (2016+)\n& t-stat",
               "Sharpe\n(2016+)", "β-neutral\nSh (2016+)",
               "Sharpe net cost\n(2016+, raw/β-n)"]

    cell_text, cell_colors = [], []
    for _, r in rows.iterrows():
        cell_text.append([
            r["factor"], r["family"], r["direction"],
            _fmt_stat_cell(r["alpha"], r["alpha_tstat"], True),
            _fmt_num(r["sharpe"]), _fmt_num(r["sharpe_neutral"]),
            _fmt_net_sharpe_cell(r.get("sharpe_cost", np.nan),
                                 r.get("sharpe_cost_neutral", np.nan)),
            f"{-r['avg_cost_pp'] / 100:+.4%}", f"{int(r['n'])}",
            _fmt_stat_cell(r["alpha_2016"], r["alpha_tstat_2016"], True),
            _fmt_num(r["sharpe_2016"]), _fmt_num(r["sharpe_neutral_2016"]),
            _fmt_net_sharpe_cell(r.get("sharpe_cost_2016", np.nan),
                                 r.get("sharpe_cost_neutral_2016", np.nan)),
        ])
        cell_colors.append([
            "white", "white", "white",
            _tstat_color(r["alpha_tstat"]),
            "white", "white", "white", "#fde7d6", "white",
            _tstat_color(r["alpha_tstat_2016"]), "white", "white", "white",
        ])

    n = len(rows)
    fig, ax = plt.subplots(figsize=(16.0, 0.62 * (n + 1) + 1.4))
    ax.axis("off")

    tbl = ax.table(cellText=cell_text, colLabels=headers, cellColours=cell_colors,
                   colWidths=[0.10, 0.12, 0.055, 0.09, 0.05, 0.058,
                              0.085, 0.06, 0.03, 0.09, 0.05, 0.058, 0.085],
                   cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for j in range(len(headers)):                       # bold header row
        tbl[0, j].set_text_props(weight="bold", color="white")
        tbl[0, j].set_facecolor("#404040")
    for i in range(1, n + 1):                           # left-align text cols
        tbl[i, 0].set_text_props(ha="left")
        tbl[i, 1].set_text_props(ha="left")

    default_title = ("Long-short quintile strategy regressed on the industry return\n"
                     "(ls_t = α + β·industry_t + ε;  α = industry-neutral monthly return)")
    fig.suptitle(default_title if title is None else title, fontsize=11, y=0.99)
    fig.text(0.5, 0.015, "The alpha cell stacks the estimate over its t-stat.  "
             "Shaded alpha t-stats: |t| ≥ 1.65 (10%), darker |t| ≥ 2.0 (5%).  "
             "Sharpe = annualised Sharpe of the L/S book.  β-neutral Sharpe hedges "
             "the book with a walk-forward -β·industry overlay (β re-estimated on an "
             "expanding, look-ahead-free window).  Sharpe net cost = annualised Sharpe of "
             "the book's return net of turnover cost (top = raw L/S, bottom βn = "
             "β-neutralised).  Avg cost = mean monthly turnover cost (one-way, traded "
             "weight only).  2016+ columns re-estimate on the past decade only.",
             ha="center", fontsize=8, color="#555555")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.80, bottom=0.10)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run(panel: pd.DataFrame | None = None,
        u: "F.Universe" = F.SOFTWARE_SERVICES) -> pd.DataFrame:
    if panel is None:
        panel = F.load_panel(u=u)
    regression_dir = u.output_dir / "regression"
    quintile_dir = u.output_dir / "quintile"   # long-short books live alongside the sorts
    regression_dir.mkdir(parents=True, exist_ok=True)
    quintile_dir.mkdir(parents=True, exist_ok=True)

    # Industry ("market") monthly return, shared across all factor regressions.
    industry_ret = industry_monthly_return(panel)

    # Industry-relative (normalised) next return per stock-month -- the dependent
    # variable of the pooled time-and-cross-section regression below.  Built once
    # off the shared industry return.
    norm = normalized_return(panel, industry_ret)

    # Per-(stock, month) trading cost, shared across all factors (the average
    # turnover cost of each book is reported in the long-short alpha table).
    cost_panel = cost.build_cost_panel(u)

    summary_rows = []
    alpha_rows = []
    norm_rows = []
    for factor in F.FACTOR_NAMES:
        factor_dir = regression_dir / factor
        factor_dir.mkdir(parents=True, exist_ok=True)
        reg = monthly_regressions(panel, factor)
        reg.to_csv(factor_dir / "regression.csv")
        plot_beta(reg, factor, factor_dir / "beta.png")

        full = fama_macbeth(reg["beta"])
        decade = fama_macbeth(reg.loc[reg.index >= DECADE_START, "beta"])

        # Pooled (time x cross-section) regression of the industry-relative return
        # on the factor z-score: the slope and its clustered t-stat, full sample
        # and past decade.  Complements the monthly "beta over time" above.
        nfull = normalized_regression(panel, factor, norm).set_index("term").loc[factor]
        n2016 = (normalized_regression(panel, factor, norm, start=DECADE_START)
                 .set_index("term").loc[factor])
        norm_rows.append({
            "factor": factor, "family": F.FACTORS[factor]["family"],
            "coef_full": nfull["coef"], "t_ols_full": nfull["t_ols"],
            "t_cluster_full": nfull["t_cluster"], "p_cluster_full": nfull["p_cluster"],
            "n_obs": nfull["n_obs"], "n_months": nfull["n_clusters_months"],
            "coef_2016": n2016["coef"], "t_ols_2016": n2016["t_ols"],
            "t_cluster_2016": n2016["t_cluster"], "p_cluster_2016": n2016["p_cluster"],
            "n_obs_2016": n2016["n_obs"], "n_months_2016": n2016["n_clusters_months"],
        })

        # Directional dollar-neutral long-short book, signed by the full-sample
        # Fama-MacBeth t-stat, with its return path plotted over time.
        spread, sign = long_short_portfolio(panel, factor, full["fm_tstat"])
        ls = long_short_stats(spread)
        ls_2016 = long_short_stats(spread[spread.index >= DECADE_START])

        # Regress the strategy's monthly return on the industry's monthly return
        # to isolate its industry-neutral alpha (and t-stat).
        mreg = market_regression(spread, industry_ret)
        # Same regression restricted to the past decade (2016+): its alpha,
        # alpha t-stat and R^2 (the industry beta is not reported here).
        mreg_2016 = market_regression(
            spread[spread.index >= DECADE_START],
            industry_ret[industry_ret.index >= DECADE_START])
        # Industry-beta-neutralised Sharpe: hedge each month with a -beta * industry
        # overlay whose beta is re-estimated on an expanding window of only the data
        # available up to that month (no look-ahead).  The 2016+ figure windows the
        # same walk-forward hedged series, so its betas still use all prior history.
        sr_neutral = beta_neutral_sharpe(spread, industry_ret)
        sr_neutral_2016 = beta_neutral_sharpe(spread, industry_ret, start=DECADE_START)

        # Turnover cost incurred by the book (per formation month): its time-series
        # mean (pp/month) is reported, and it also nets the gross spread for the
        # cost-incorporated Sharpe (raw and beta-neutral, full sample and 2016+).
        cost_series = cost.long_short_cost(panel, factor, cost_panel, N_QUINTILES)
        avg_cost_pp = cost.average_cost(cost_series) * 100.0
        sharpe_cost = net_of_cost_sharpe(spread, cost_series)
        sharpe_cost_neutral = net_of_cost_neutral_sharpe(
            spread, cost_series, industry_ret)
        sharpe_cost_2016 = net_of_cost_sharpe(spread, cost_series, start=DECADE_START)
        sharpe_cost_neutral_2016 = net_of_cost_neutral_sharpe(
            spread, cost_series, industry_ret, start=DECADE_START)

        ls_dir = quintile_dir / factor
        ls_dir.mkdir(parents=True, exist_ok=True)
        plot_long_short(spread, factor, sign, ls["sharpe"],
                        mreg["alpha"], mreg["alpha_tstat"],
                        ls_dir / "long_short.png")

        alpha_rows.append({
            "factor": factor, "family": F.FACTORS[factor]["family"],
            "direction": "Q5-Q1" if sign > 0 else "Q1-Q5",
            "alpha": mreg["alpha"], "alpha_tstat": mreg["alpha_tstat"],
            "sharpe": ls["sharpe"],
            "avg_cost_pp": avg_cost_pp,
            "sharpe_cost": sharpe_cost,
            "sharpe_cost_neutral": sharpe_cost_neutral,
            "r2": mreg["r2"], "n": mreg["n"],
            "sharpe_neutral": sr_neutral,
            "alpha_2016": mreg_2016["alpha"],
            "alpha_tstat_2016": mreg_2016["alpha_tstat"],
            "sharpe_2016": ls_2016["sharpe"],
            "sharpe_neutral_2016": sr_neutral_2016,
            "sharpe_cost_2016": sharpe_cost_2016,
            "sharpe_cost_neutral_2016": sharpe_cost_neutral_2016,
            "r2_2016": mreg_2016["r2"],
        })

        summary_rows.append({
            "factor": factor, "family": F.FACTORS[factor]["family"],
            "mean_beta_full": full["mean_beta"], "fm_t_full": full["fm_tstat"],
            "n_full": full["n_months"],
            "mean_beta_2016": decade["mean_beta"], "fm_t_2016": decade["fm_tstat"],
            "n_2016": decade["n_months"],
            "ls_direction": "long_Q5_short_Q1" if sign > 0 else "long_Q1_short_Q5",
            "ls_mean_monthly": ls["mean_monthly"], "ls_tstat": ls["tstat"],
            "ls_ann_return": ls["ann_return"], "ls_n_months": ls["n_months"],
        })
        print(f"  {factor:<22} full: beta={full['mean_beta']:+.5f} "
              f"(t={full['fm_tstat']:+.2f})   "
              f"2016+: beta={decade['mean_beta']:+.5f} "
              f"(t={decade['fm_tstat']:+.2f})   "
              f"L/S [{'Q5-Q1' if sign > 0 else 'Q1-Q5'}]: "
              f"{ls['mean_monthly']:+.4%}/mo (t={ls['tstat']:+.2f}, "
              f"ann={ls['ann_return']:+.2%})")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(regression_dir / "summary.csv", index=False)
    render_summary_table(summary, regression_dir / "summary_table.png")

    # Pooled normalised-return regression table (time x cross-section), alongside
    # the per-month "beta over time" summary.
    norm_tbl = pd.DataFrame(norm_rows)
    norm_tbl.to_csv(regression_dir / "normalized_regression.csv", index=False)
    render_normalized_table(norm_tbl, regression_dir / "normalized_regression_table.png")

    # Long-short-vs-industry alpha table, saved alongside the quintile books.
    alpha_tbl = pd.DataFrame(alpha_rows)
    alpha_tbl.to_csv(quintile_dir / "long_short_market_alpha.csv", index=False)
    render_alpha_table(alpha_tbl, quintile_dir / "long_short_market_alpha.png")

    print(f"\nSaved regression outputs -> {regression_dir}")
    print(f"Saved normalized-return regression table -> "
          f"{regression_dir / 'normalized_regression_table.png'}")
    print(f"Saved long-short alpha table -> "
          f"{quintile_dir / 'long_short_market_alpha.png'}")
    return summary
