"""
regression.py
=============

Approach 2 -- monthly cross-sectional regression.

For every factor, each month we regress the **next-period** (month t+1) return
of all securities on the factor z-score via OLS:

    next_return_i = alpha_t + beta_t * zscore_i + eps_i

Re-estimating every month gives one time series of slopes (``beta_t``) per
factor -- the cross-sectional "factor premium". Outputs, per factor:

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
mean return, t-stat and annualised return.  The
``output/quintile/long_short_market_alpha`` table additionally carries each
book's industry-neutral alpha and its **average turnover cost** (pp/month).

Run standalone::

    python regression.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import cost
import factors as F

REGRESSION_DIR = F.OUTPUT_DIR / "regression"
QUINTILE_DIR = F.OUTPUT_DIR / "quintile"   # long-short books live alongside the sorts
DECADE_START = pd.Timestamp("2016-01-01")  # "past decade" cut-off
N_QUINTILES = 5
MONTHS_PER_YEAR = 12


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


def beta_neutral_sharpe(spread: pd.Series, market: pd.Series, beta: float) -> float:
    """
    Annualised Sharpe of the industry-beta-neutralised book.

    Hedge the long-short book's industry exposure by overlaying a ``-beta``
    position in the industry ("market") return: ``r_hedged_t = ls_t - beta *
    industry_t``.  ``beta`` is the exposure estimated by :func:`market_regression`
    over the relevant window (full sample or 2016+), so the hedged series has
    (in-sample) zero industry beta and its Sharpe measures risk-adjusted return
    net of industry risk.  Returns NaN if ``beta`` is not finite.
    """
    df = pd.concat([spread.rename("y"), market.rename("x")], axis=1).dropna()
    if df.empty or not np.isfinite(beta):
        return np.nan
    hedged = df["y"] - beta * df["x"]
    return long_short_stats(hedged)["sharpe"]


# --------------------------------------------------------------------------- #
# Long-short return regressed on the industry ("market") return
# --------------------------------------------------------------------------- #
def industry_monthly_return(panel: pd.DataFrame) -> pd.Series:
    """
    Equal-weighted next-period return of the whole industry cross-section,
    indexed by formation month.

    This is the within-industry "market" return.  It is built from the same
    ``next_return`` column the long-short spread is -- including the same
    within-month winsorisation the spread legs get via ``prepare_slice`` -- so
    both series share the formation-date index, the identical (month t+1) return
    period, and the same tail treatment, and can be regressed directly.
    """
    uniq = (panel.drop_duplicates(["date", "stock_id"])
                 .dropna(subset=["next_return"])
                 .copy())
    uniq["next_return"] = F.winsorize_cross_section(
        uniq["next_return"], uniq["date"], F.WINSOR_PCT)
    return (uniq.groupby("date", observed=True)["next_return"].mean()
                .sort_index().rename("industry_ret"))


def market_regression(spread: pd.Series, market: pd.Series) -> dict:
    """
    Time-series OLS of a long-short strategy's monthly return on the industry's
    equal-weighted monthly return:

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


def _fmt_pct(x: float) -> str:
    return f"{x:+.4%}" if np.isfinite(x) else "—"


def _fmt_num(x: float, prec: int = 2) -> str:
    return f"{x:+.{prec}f}" if np.isfinite(x) else "—"


def render_alpha_table(rows: pd.DataFrame, path: Path) -> None:
    """
    Render the per-factor long-short-vs-industry regression as a PNG table.

    Full-sample columns: the industry-neutral alpha and its t-stat, the book's
    annualised Sharpe ratio, its average turnover cost and the industry-beta-
    neutralised Sharpe (the book hedged with -beta*industry).  A trailing block
    repeats the alpha, its t-stat, the annualised Sharpe and the beta-neutralised
    Sharpe estimated on the past decade (2016+) only.  Alpha t-stats are shaded
    by absolute significance.
    """
    cols = ["factor", "family", "direction",
            "alpha", "alpha_tstat", "sharpe", "sharpe_neutral", "avg_cost_pp", "n",
            "alpha_2016", "alpha_tstat_2016", "sharpe_2016", "sharpe_neutral_2016"]
    headers = ["Factor", "Family", "L/S\ndirection",
               "Alpha\n(monthly)", "Alpha\nt-stat",
               "Sharpe\n(ann.)", "β-neutral\nSharpe", "Avg cost\n(monthly)", "n",
               "Alpha\n(2016+)", "Alpha t\n(2016+)", "Sharpe\n(2016+)", "β-neutral\nSh (2016+)"]

    cell_text, cell_colors = [], []
    for _, r in rows[cols].iterrows():
        cell_text.append([
            r["factor"], r["family"], r["direction"],
            _fmt_pct(r["alpha"]), _fmt_num(r["alpha_tstat"]),
            _fmt_num(r["sharpe"]), _fmt_num(r["sharpe_neutral"]),
            f"{-r['avg_cost_pp'] / 100:+.4%}", f"{int(r['n'])}",
            _fmt_pct(r["alpha_2016"]), _fmt_num(r["alpha_tstat_2016"]),
            _fmt_num(r["sharpe_2016"]), _fmt_num(r["sharpe_neutral_2016"]),
        ])
        cell_colors.append([
            "white", "white", "white",
            "white", _tstat_color(r["alpha_tstat"]),
            "white", "white", "#fde7d6", "white",
            "white", _tstat_color(r["alpha_tstat_2016"]), "white", "white",
        ])

    n = len(rows)
    fig, ax = plt.subplots(figsize=(16.5, 0.45 * (n + 1) + 1.4))
    ax.axis("off")

    tbl = ax.table(cellText=cell_text, colLabels=headers, cellColours=cell_colors,
                   colWidths=[0.13, 0.15, 0.07, 0.075, 0.06, 0.06, 0.078,
                              0.068, 0.036, 0.075, 0.06, 0.058, 0.09],
                   cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for j in range(len(headers)):                       # bold header row
        tbl[0, j].set_text_props(weight="bold", color="white")
        tbl[0, j].set_facecolor("#404040")
    for i in range(1, n + 1):                           # left-align text cols
        tbl[i, 0].set_text_props(ha="left")
        tbl[i, 1].set_text_props(ha="left")

    fig.suptitle("Long-short quintile strategy regressed on the industry return\n"
                 "(ls_t = α + β·industry_t + ε;  α = industry-neutral monthly "
                 "return, t-stat tests α ≠ 0)", fontsize=11, y=0.99)
    fig.text(0.5, 0.015, "Shaded alpha t-stats: |t| ≥ 1.65 (10%), darker |t| ≥ 2.0 (5%).  "
             "Sharpe = annualised Sharpe of the L/S book.  Avg cost = mean monthly "
             "turnover cost (one-way, traded weight only).  2016+ columns re-estimate "
             "α on the past decade only.",
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

    # Per-(stock, month) trading cost, shared across all factors (the average
    # turnover cost of each book is reported in the long-short alpha table).
    cost_panel = cost.build_cost_panel(u)

    summary_rows = []
    alpha_rows = []
    for factor in F.FACTOR_NAMES:
        factor_dir = regression_dir / factor
        factor_dir.mkdir(parents=True, exist_ok=True)
        reg = monthly_regressions(panel, factor)
        reg.to_csv(factor_dir / "regression.csv")
        plot_beta(reg, factor, factor_dir / "beta.png")

        full = fama_macbeth(reg["beta"])
        decade = fama_macbeth(reg.loc[reg.index >= DECADE_START, "beta"])

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
        # Industry-beta-neutralised Sharpe: hedge the book with -beta * industry,
        # using the beta estimated over each window (full sample / 2016+).
        sr_neutral = beta_neutral_sharpe(spread, industry_ret, mreg["beta"])
        sr_neutral_2016 = beta_neutral_sharpe(
            spread[spread.index >= DECADE_START],
            industry_ret[industry_ret.index >= DECADE_START],
            mreg_2016["beta"])

        # Average turnover cost incurred by the book (pp/month).
        avg_cost_pp = cost.average_cost(
            cost.long_short_cost(panel, factor, cost_panel, N_QUINTILES)) * 100.0

        ls_dir = quintile_dir / factor
        ls_dir.mkdir(parents=True, exist_ok=True)
        plot_long_short(spread, factor, sign, ls["sharpe"],
                        mreg["alpha"], mreg["alpha_tstat"],
                        ls_dir / "long_short.png")

        alpha_rows.append({
            "factor": factor, "family": F.FACTORS[factor]["family"],
            "direction": "Q5-Q1" if sign > 0 else "Q1-Q5",
            "alpha": mreg["alpha"], "alpha_tstat": mreg["alpha_tstat"],
            "beta": mreg["beta"], "beta_tstat": mreg["beta_tstat"],
            "sharpe": ls["sharpe"],
            "avg_cost_pp": avg_cost_pp,
            "r2": mreg["r2"], "n": mreg["n"],
            "sharpe_neutral": sr_neutral,
            "alpha_2016": mreg_2016["alpha"],
            "alpha_tstat_2016": mreg_2016["alpha_tstat"],
            "sharpe_2016": ls_2016["sharpe"],
            "sharpe_neutral_2016": sr_neutral_2016,
            "beta_tstat_2016": mreg_2016["beta_tstat"],
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

    # Long-short-vs-industry alpha table, saved alongside the quintile books.
    alpha_tbl = pd.DataFrame(alpha_rows)
    alpha_tbl.to_csv(quintile_dir / "long_short_market_alpha.csv", index=False)
    render_alpha_table(alpha_tbl, quintile_dir / "long_short_market_alpha.png")

    print(f"\nSaved regression outputs -> {regression_dir}")
    print(f"Saved long-short alpha table -> "
          f"{quintile_dir / 'long_short_market_alpha.png'}")
    return summary


if __name__ == "__main__":
    run(u=F.universe_from_argv())
