"""
weighted_composite.py
=====================

Experiment 3 -- coefficient-weighted composite, evaluated out-of-sample.

Same shape as ``composite.py``, but the constituents are combined with
**data-driven weights** instead of a straight (sign-oriented) sum: estimate each
factor's return premium on an in-sample window (through ``IS_END``), use those
regression coefficients to weight the z-scores into one score, sort into
quintiles, and trade the Q5-Q1 book -- with performance judged strictly
**out-of-sample** (from ``OOS_START``).  The split defaults to in-sample ≤ 2015 /
out-of-sample 2016 onwards.

Step 1 -- in-sample premia.  Pool every in-sample stock-month and regress the
**normalised return** -- the stock's month-(t+1) return minus that month's
market-cap-weighted industry average (the within-industry "market") -- on the
formation-date factor z-scores::

    (r_{i,t+1} − market_{t+1})  =  a  +  Σ_f  b_f · z_{f,i,t}  +  ε

The slope vector ``b_f`` is each factor's in-sample premium (industry-relative
monthly return per 1 cross-sectional σ), reported with OLS and month-clustered
t-stats (the right standard error for a pooled cross-section of returns).

Step 2 -- weighted score.  For every stock-month (full sample) the score is the
premium-weighted sum of exposures -- the model's predicted industry-relative
return::

    score_{i,t}  =  Σ_f  b_f · z_{f,i,t}

(The intercept is constant across stocks and does not affect the cross-sectional
ranking, so it is dropped.)  Higher score = higher predicted return, so the book
is long Q5 / short Q1, exactly as in the straight-sum pipeline.

Step 3 -- out-of-sample test.  The fixed in-sample weights are applied to the
out-of-sample cross-sections; the Q5-Q1 book's out-of-sample performance is the
headline (the in-sample window is reported alongside for contrast).  Because the
weights never see post-``IS_END`` data, the result is a genuine out-of-sample
test of the weighting.

Design -- reuses composite.py wholesale
---------------------------------------
The constituent loading (``load_exposures``), the score → tidy-panel shaping
(``scored_frame`` / ``as_factor_panel``), the quintile sort (Experiment 1's
``quintile.py``), the industry-neutral alpha and window-split performance
(``industry_return`` / ``book_stats``, Experiment 1's ``regression.py``), and the
plotting / table rendering (``plot_cumulative`` / ``plot_long_short`` /
``render_performance``) are all imported from ``composite.py``.  The dependent
variable (``regression.normalized_return``) and the pooled month-clustered OLS
(``regression.pooled_ols``) are reused from Experiment 1's ``regression.py`` -- the
same spine that powers the pooled normalised-return regression in Experiments 1 &
2.  This module adds **only** the in-sample window and the train/test split.

Outputs (``output/weighted/<slug>/``)
-------------------------------------
    coefficients.{csv,png}      the in-sample premia = the composite weights (+ t-stats)
    quintile_returns.csv        months × {Q1..Q5, Q5−Q1}, mean next-period return
    quintile_cumulative.png     the five buckets as growth of $1, IS/OOS split marked
    long_short.png              the Q5−Q1 book's growth of $1, IS/OOS split marked
    performance.png             in-sample vs out-of-sample book performance (headline = OOS)

Run standalone::

    python weighted_composite.py                                          # default set (buyback_quality + rd_stability)
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
from composite import F, Q, R          # reused engine + analysis handles

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
IS_END = pd.Timestamp("2015-12-31")        # in-sample: formation months through 2015
OOS_START = pd.Timestamp("2016-01-01")     # out-of-sample: formation months from 2016
OUTPUT_DIR = C.OUTPUT_DIR

# The weighted deliverable's factor set (the two factors whose marginal premia we
# weight by).  Distinct from composite.py's equal-weighted DEFAULT_FACTORS.
DEFAULT_FACTORS = ["buyback_quality", "rd_stability"]


# --------------------------------------------------------------------------- #
# Step 1 -- in-sample premia
#
# The dependent variable (industry-relative return, ``R.normalized_return``) and
# the pooled month-clustered OLS (``R.pooled_ols``) both live in Experiment 1's
# ``regression.py`` -- the same reusable spine that powers the pooled
# normalised-return regression added to Experiments 1 & 2.  This module only
# supplies the in-sample window and turns the slopes into composite weights.
# --------------------------------------------------------------------------- #
def fit_weights(resolved: pd.DataFrame, exposures: pd.DataFrame,
                is_end: pd.Timestamp = IS_END) -> tuple[pd.DataFrame, pd.Series, int, int]:
    """
    Fit the in-sample (≤ ``is_end``) pooled regression of normalised return on the
    constituent z-scores.

    Returns ``(coef_table, weights, n_obs, n_months)`` where ``weights`` is the
    slope vector (intercept dropped) indexed by factor name -- the premia used to
    weight the composite.
    """
    names = resolved["factor"].tolist()
    panel = F.load_panel(u=F.SOFTWARE_SERVICES)
    y = R.normalized_return(panel).set_index(["date", "stock_id"])["norm_return"]
    fit = exposures.join(y, how="inner").dropna()
    fit = fit[fit.index.get_level_values("date") <= is_end]

    coef = R.pooled_ols(fit["norm_return"].to_numpy(), fit[names].to_numpy(),
                        fit.index.get_level_values("date").to_numpy(), names)
    weights = coef.set_index("term").loc[names, "coef"]
    n_months = fit.index.get_level_values("date").nunique()
    return coef, weights, len(fit), n_months


# --------------------------------------------------------------------------- #
# Rendering -- the coefficient (weight) table
# --------------------------------------------------------------------------- #
def render_coefficients(coef: pd.DataFrame, resolved: pd.DataFrame,
                        is_end: pd.Timestamp, path: Path) -> None:
    """Render the in-sample premia (= composite weights) as a shaded PNG."""
    family = resolved.set_index("factor")["family"]
    headers = ["Term", "Family", "Coef = weight\n(ind-rel /mo per 1σ)",
               "t (OLS)", "t (cluster)", "p (cluster)"]
    cell_text, cell_colors = [], []
    for _, r in coef.iterrows():
        fam = "" if r["term"] == "intercept" else str(family.get(r["term"], ""))
        cell_text.append([r["term"], fam, f"{r['coef']:+.4%}",
                          f"{r['t_ols']:+.2f}", f"{r['t_cluster']:+.2f}",
                          f"{r['p_cluster']:.3f}"])
        cell_colors.append(["white", "white", "white",
                            R._tstat_color(r["t_ols"]), R._tstat_color(r["t_cluster"]),
                            "white"])

    n = len(coef)
    fig, ax = plt.subplots(figsize=(12, 0.5 * (n + 1) + 1.7))
    ax.axis("off")
    tbl = ax.table(cellText=cell_text, colLabels=headers, cellColours=cell_colors,
                   colWidths=[0.20, 0.26, 0.22, 0.10, 0.12, 0.10],
                   cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for j in range(len(headers)):
        tbl[0, j].set_text_props(weight="bold", color="white")
        tbl[0, j].set_facecolor("#404040")
    for i in range(1, n + 1):
        tbl[i, 0].set_text_props(ha="left")
        tbl[i, 1].set_text_props(ha="left")

    n_obs = int(coef["n_obs"].iloc[0])
    n_months = int(coef["n_clusters_months"].iloc[0])
    fig.suptitle("Experiment 3 -- in-sample factor premia = the composite weights\n"
                 "normalised next return regressed on factor z-scores (pooled across "
                 "stocks & months)", fontsize=11, y=0.99)
    fig.text(0.5, 0.05,
             f"in-sample: {n_obs:,} stock-months over {n_months} months "
             f"(through {is_end:%Y-%m}).   t clustered by month.   "
             "Shading: |t| ≥ 1.65 (10%), 2.0 (5%).",
             ha="center", fontsize=8, color="#555555")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.80, bottom=0.10)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run(factor_names: list[str] = DEFAULT_FACTORS,
        is_end: pd.Timestamp = IS_END,
        oos_start: pd.Timestamp = OOS_START,
        label: str | None = None,
        out_root: Path = OUTPUT_DIR) -> dict:
    """
    Fit in-sample premia (≤ ``is_end``), build the coefficient-weighted composite,
    and evaluate the Q5-Q1 book out-of-sample (≥ ``oos_start``).  Writes every
    output under ``out_root / "weighted" / <slug>`` and returns the per-window
    performance.
    """
    resolved = C.resolve_factors(factor_names)
    slug = label or "__".join(factor_names)
    out_dir = out_root / "weighted" / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Experiment 3: coefficient-weighted composite "
          f"(IS <= {is_end:%Y} / OOS >= {oos_start:%Y}) ===")
    print(f"Factors ({len(factor_names)}): {', '.join(factor_names)}")

    # 1. Constituent exposures (raw z-scores) and in-sample premia (the weights).
    exposures, next_ret = C.load_exposures(resolved)
    coef, weights, n_is, m_is = fit_weights(resolved, exposures, is_end)

    # 2. Weighted score over the full sample, shaped for the reused quintile sort.
    score = exposures.mul(weights, axis=1).sum(axis=1)
    df = C.scored_frame(score, next_ret)
    panel = C.as_factor_panel(df)

    # 3. Quintile sort (Experiment 1's code, unmodified).
    wide = Q.quintile_returns(panel, C.COMPOSITE_FACTOR)
    spread = wide["Q5-Q1"]

    # 4. In-sample vs out-of-sample performance (OOS is the headline).  Window
    #    labels derive from the split dates, so changing IS_END/OOS_START is enough.
    industry = C.industry_return()
    is_label = f"In-sample (≤{is_end:%Y})"
    oos_label = f"Out-of-sample ({oos_start:%Y}+)"
    windows = [(is_label, C.book_stats(spread, industry, end=is_end)),
               (oos_label, C.book_stats(spread, industry, start=oos_start))]
    is_stats, oos = windows[0][1], windows[1][1]

    # --- Persist outputs --------------------------------------------------- #
    coef.to_csv(out_dir / "coefficients.csv", index=False)
    render_coefficients(coef, resolved, is_end, out_dir / "coefficients.png")
    wide.to_csv(out_dir / "quintile_returns.csv")
    C.plot_cumulative(wide, factor_names, out_dir / "quintile_cumulative.png",
                      score_desc="coefficient-weighted z-scores", boundary=oos_start)
    C.plot_long_short(spread, factor_names, oos["sharpe"], oos["alpha"],
                      oos["alpha_tstat"], out_dir / "long_short.png",
                      boundary=oos_start, stat_window=f"({oos_start:%Y}+ OOS)")
    C.render_performance(
        windows,
        "Coefficient-weighted composite long-short (Q5-Q1): in-sample vs out-of-sample",
        f"weights = in-sample (≤{is_end:%Y}) premia of: {' + '.join(factor_names)}   |   "
        f"fit on {n_is:,} stock-months over {m_is} months.   Headline = OOS.   "
        "Shading: |t| ≥ 1.65 (10%), 2.0 (5%).",
        out_dir / "performance.png")

    # --- Console summary --------------------------------------------------- #
    print("In-sample premia (composite weights):")
    print(coef[["term", "coef", "t_ols", "t_cluster"]].to_string(index=False))
    print(f"  IS  Q5-Q1: alpha={is_stats['alpha']:+.4%}/mo (t={is_stats['alpha_tstat']:+.2f}), "
          f"Sharpe={is_stats['sharpe']:+.2f}")
    print(f"  OOS Q5-Q1: mean={oos['mean_monthly']:+.4%}/mo (t={oos['tstat']:+.2f}), "
          f"Sharpe={oos['sharpe']:+.2f}, alpha={oos['alpha']:+.4%}/mo "
          f"(t={oos['alpha_tstat']:+.2f}), ind beta={oos['ind_beta']:+.2f}")
    print(f"Saved -> {out_dir}")
    return {lbl: s for lbl, s in windows}


def main() -> None:
    factor_names = sys.argv[1:] or DEFAULT_FACTORS
    run(factor_names)


if __name__ == "__main__":
    main()
