"""
factor_correlation.py
=====================

Experiment 3 -- **how much of an industry factor is already explained by the
general market factors of Experiment 1?**

For one *target* factor (an industry-specific signal -- e.g. the software
``buyback_quality`` of Experiment 2) this measures its association with each of
Experiment 1's general market factors through the **R-squared of a linear
regression** of the target exposure on the market factor.

Because each market factor is regressed on its own (a single regressor plus an
intercept), the regression R-squared equals the squared Pearson correlation,
``R^2 = corr^2`` -- so the table reads directly as "fraction of the target
factor's cross-sectional variation that this market factor linearly explains".
A high R-squared means the industry factor is largely redundant with that
general factor; a low one means it carries distinct information.

A final **joint** row regresses the target on *all* market factors together
(multivariate OLS), giving the fraction of the target spanned by the general
factor set as a whole -- the headline redundancy number.

Reusable
--------
The module is factor- and panel-agnostic.  :func:`run` takes the target factor
name and the two panels (target + market), so the same code evaluates any other
industry factor against the general set::

    from factor_correlation import run
    run(target_factor="rd_productivity")                  # another Exp-2 factor
    run(target_factor="operating_leverage",
        out_dir=OUTPUT_DIR / "operating_leverage")

All it needs is the project-standard ``factor_panel.csv`` schema
(``date, stock_id, factor, value, zscore, next_return``) that Experiments 1-2
already write; the regressor column defaults to the project's ``zscore``
representation.

Outputs (``output/factor_correlation/<target>/``)
-------------------------------------------------
    r2_table.png     per market factor: n, corr, R^2, slope (+ joint row), rendered
                     and shaded by R^2

Run standalone::

    python factor_correlation.py            # buyback_quality vs the 9 market factors
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths -- mirror multifactor.py
# --------------------------------------------------------------------------- #
EXP3_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXP3_DIR.parent
EXP1_DIR = PROJECT_ROOT / "experiment1 - general factors"
EXP2_DIR = PROJECT_ROOT / "experiment2 - sw factors"
OUTPUT_DIR = EXP3_DIR / "output" / "factor_correlation"

# Default panels: target = Experiment 2 (software-industry factors),
# market = Experiment 1 (general market factors) on the software universe.
MARKET_PANEL = EXP1_DIR / "output" / "software" / "factor_panel.csv"
TARGET_PANEL = EXP2_DIR / "output" / "factor_panel.csv"
# Family labels for the market factors (nice-to-have; read if present).
MARKET_ALPHA = EXP1_DIR / "output" / "software" / "quintile" / "long_short_market_alpha.csv"

REGRESSOR_COL = "zscore"        # project convention: factors are z-scores vs the industry mean
WINSOR_PCT = 0.01               # cross-sectional winsorisation before z-scoring (mirrors factors.py)
SIZE_FACTOR_LABEL = "log_market_cap"   # row name for the market-cap (size) exposure


# --------------------------------------------------------------------------- #
# Panel loading
# --------------------------------------------------------------------------- #
def _load_exposures(panel_path: Path, regressor_col: str) -> pd.DataFrame:
    """
    Read a ``factor_panel.csv`` and pivot it to a wide
    ``(date, stock_id) x factor`` matrix of the chosen exposure column.
    """
    panel = pd.read_csv(panel_path, parse_dates=["date"],
                        usecols=["date", "stock_id", "factor", regressor_col])
    panel["stock_id"] = panel["stock_id"].astype(str)
    return panel.pivot_table(index=["date", "stock_id"],
                             columns="factor", values=regressor_col)


def _market_cap_zscore(panel_path: Path, winsor: float = WINSOR_PCT) -> pd.Series:
    """
    Cross-sectional z-score of **log** market cap, from a panel's ``weight``
    column (= formation-date month-end USD market cap, identical across factor
    rows for a given date/stock).

    Log is taken first because raw market cap is extremely right-skewed -- a raw
    z-score would be dominated by a handful of mega-caps; log size is the standard
    size exposure.  It is then winsorised per date at ``[winsor, 1-winsor]`` and
    standardised, matching the project's ``cross_sectional_zscore`` convention.

    Returns a Series indexed by ``(date, stock_id)`` named ``log_market_cap``.
    """
    raw = pd.read_csv(panel_path, parse_dates=["date"],
                      usecols=["date", "stock_id", "weight"])
    raw["stock_id"] = raw["stock_id"].astype(str)
    # weight is identical across a stock-month's factor rows -> one row per key.
    cap = (raw.dropna(subset=["weight"])
              .drop_duplicates(["date", "stock_id"])
              .set_index(["date", "stock_id"])["weight"])
    cap = cap[cap > 0]
    logcap = np.log(cap)

    grp = logcap.groupby(level="date")
    lo = grp.transform(lambda s: s.quantile(winsor))
    hi = grp.transform(lambda s: s.quantile(1.0 - winsor))
    clipped = logcap.clip(lower=lo, upper=hi)
    cgrp = clipped.groupby(level="date")
    z = (clipped - cgrp.transform("mean")) / cgrp.transform("std").replace(0.0, np.nan)
    z.name = SIZE_FACTOR_LABEL
    return z


def _market_families() -> dict[str, str]:
    """Map market factor -> family from Exp 1's alpha table, if available."""
    if not MARKET_ALPHA.exists():
        return {}
    tbl = pd.read_csv(MARKET_ALPHA, usecols=["factor", "family"])
    return dict(zip(tbl["factor"], tbl["family"]))


# --------------------------------------------------------------------------- #
# Core -- the reusable computation
# --------------------------------------------------------------------------- #
def _univariate_r2(y: np.ndarray, x: np.ndarray) -> tuple[float, float, float, int]:
    """
    OLS of ``y`` on ``[1, x]`` over the rows where both are finite.

    Returns ``(corr, r2, slope, n)``.  With a single regressor and an intercept
    the regression R-squared is exactly the squared Pearson correlation.
    """
    m = np.isfinite(y) & np.isfinite(x)
    n = int(m.sum())
    if n < 3:
        return np.nan, np.nan, np.nan, n
    yy, xx = y[m], x[m]
    sx = xx.std()
    sy = yy.std()
    if sx == 0 or sy == 0:
        return np.nan, np.nan, np.nan, n
    corr = float(np.corrcoef(xx, yy)[0, 1])
    slope = float(np.cov(xx, yy, bias=True)[0, 1] / (sx ** 2))
    return corr, corr ** 2, slope, n


def _joint_r2(y: np.ndarray, X: np.ndarray) -> tuple[float, int]:
    """R-squared of the multivariate OLS of ``y`` on ``[1, X]`` (complete rows)."""
    m = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    n = int(m.sum())
    if n <= X.shape[1] + 1:
        return np.nan, n
    yy, XX = y[m], X[m]
    Xd = np.column_stack([np.ones(n), XX])
    beta, *_ = np.linalg.lstsq(Xd, yy, rcond=None)
    resid = yy - Xd @ beta
    ss_tot = ((yy - yy.mean()) ** 2).sum()
    r2 = 1.0 - (resid @ resid) / ss_tot if ss_tot > 0 else np.nan
    return float(r2), n


def factor_r2_table(target_factor: str,
                    target_panel: Path = TARGET_PANEL,
                    market_panel: Path = MARKET_PANEL,
                    regressor_col: str = REGRESSOR_COL,
                    market_factors: list[str] | None = None,
                    include_market_cap: bool = False) -> pd.DataFrame:
    """
    Build the R-squared table associating ``target_factor`` with each general
    market factor.

    Parameters
    ----------
    target_factor : the industry factor to evaluate (e.g. ``"buyback_quality"``).
    target_panel  : panel holding the target factor's exposures.
    market_panel  : panel holding the general market factors.
    regressor_col : exposure column to use (default ``"zscore"``).
    market_factors: which market factors to test; default = all in ``market_panel``.
    include_market_cap : if True, append an extra ``log_market_cap`` row giving
        the target factor's association with the z-score of log USD market cap
        (size), computed from the target panel's ``weight`` column.  This row is
        purely informational -- it is *not* added to the JOINT multivariate fit,
        so the documented joint-redundancy numbers are unchanged.

    Returns one row per market factor (n, corr, R^2, slope), sorted by
    descending R^2, optionally followed by the ``log_market_cap`` row, plus a
    final ``__joint__`` row carrying the multivariate R^2 of the target on all
    tested market factors together.  ``family`` is filled from Exp 1's alpha
    table when available.
    """
    target_wide = _load_exposures(target_panel, regressor_col)
    market_wide = _load_exposures(market_panel, regressor_col)

    if target_factor not in target_wide.columns:
        raise KeyError(
            f"target factor {target_factor!r} not in {target_panel} "
            f"(available: {', '.join(map(str, target_wide.columns))})")

    if market_factors is None:
        market_factors = list(market_wide.columns)
    missing = [f for f in market_factors if f not in market_wide.columns]
    if missing:
        raise KeyError(f"market factors not in {market_panel}: {missing}")

    # Align target and market exposures on the shared (date, stock_id) key.
    joined = market_wide[market_factors].join(
        target_wide[[target_factor]].rename(columns={target_factor: "__target__"}),
        how="inner")

    families = _market_families()
    y = joined["__target__"].to_numpy()

    rows = []
    for f in market_factors:
        corr, r2, slope, n = _univariate_r2(y, joined[f].to_numpy())
        rows.append({"market_factor": f, "family": families.get(f, ""),
                     "n_obs": n, "corr": corr, "r2": r2, "slope": slope})
    table = (pd.DataFrame(rows)
             .sort_values("r2", ascending=False, na_position="last")
             .reset_index(drop=True))

    extra_rows = []
    if include_market_cap:
        size_z = _market_cap_zscore(target_panel)
        # Align target exposure with size over their own shared coverage.
        pair = pd.concat([target_wide[target_factor].rename("y"),
                          size_z.rename("x")], axis=1, join="inner")
        s_corr, s_r2, s_slope, s_n = _univariate_r2(pair["y"].to_numpy(),
                                                    pair["x"].to_numpy())
        extra_rows.append({"market_factor": SIZE_FACTOR_LABEL,
                           "family": "size (market cap)", "n_obs": s_n,
                           "corr": s_corr, "r2": s_r2, "slope": s_slope})

    joint_r2, joint_n = _joint_r2(y, joined[market_factors].to_numpy())
    joint_row = {"market_factor": "__joint__", "family": "all market factors",
                 "n_obs": joint_n, "corr": np.nan, "r2": joint_r2, "slope": np.nan}
    # JOINT closes off the market-factor set; the size row sits below it as an
    # informational add-on (it is not part of the joint fit).
    table = pd.concat([table, pd.DataFrame([joint_row] + extra_rows)],
                      ignore_index=True)
    table.attrs["target_factor"] = target_factor
    table.attrs["regressor_col"] = regressor_col
    return table


# --------------------------------------------------------------------------- #
# Rendering (project house style)
# --------------------------------------------------------------------------- #
def _r2_color(r2: float) -> str:
    """Shade an R^2 cell by magnitude (redundancy)."""
    if not np.isfinite(r2):
        return "white"
    if r2 >= 0.25:
        return "#9ed49e"
    if r2 >= 0.10:
        return "#bfe3bf"
    if r2 >= 0.03:
        return "#e8f2d8"
    return "white"


def render_r2_table(table: pd.DataFrame, path: Path) -> None:
    """Render the R^2 table as a shaded PNG."""
    target = table.attrs.get("target_factor", "target")
    regressor = table.attrs.get("regressor_col", REGRESSOR_COL)
    headers = ["Market factor", "Family", "n", "corr", "R²", "slope"]
    cell_text, cell_colors = [], []
    for _, r in table.iterrows():
        name = "JOINT (all factors)" if r["market_factor"] == "__joint__" else r["market_factor"]
        corr = "" if not np.isfinite(r["corr"]) else f"{r['corr']:+.3f}"
        slope = "" if not np.isfinite(r["slope"]) else f"{r['slope']:+.3f}"
        r2 = "" if not np.isfinite(r["r2"]) else f"{r['r2']:.4f}"
        cell_text.append([name, str(r["family"]), f"{int(r['n_obs']):,}", corr, r2, slope])
        cell_colors.append(["white", "white", "white", "white", _r2_color(r["r2"]), "white"])

    n = len(table)
    fig, ax = plt.subplots(figsize=(10, 0.5 * (n + 1) + 1.6))
    ax.axis("off")
    tbl = ax.table(cellText=cell_text, colLabels=headers, cellColours=cell_colors,
                   colWidths=[0.26, 0.24, 0.13, 0.12, 0.12, 0.13],
                   cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for j in range(len(headers)):
        tbl[0, j].set_text_props(weight="bold", color="white")
        tbl[0, j].set_facecolor("#404040")
    has_size = (table["market_factor"] == SIZE_FACTOR_LABEL).any()
    for i in range(1, n + 1):
        tbl[i, 0].set_text_props(ha="left")
        tbl[i, 1].set_text_props(ha="left")
        if cell_text[i - 1][0].startswith("JOINT"):
            for j in range(len(headers)):
                tbl[i, j].set_text_props(weight="bold")
        elif cell_text[i - 1][0] == SIZE_FACTOR_LABEL:
            for j in range(len(headers)):
                tbl[i, j].set_text_props(style="italic")

    unit = "z-score" if regressor == "zscore" else regressor
    fig.suptitle(
        f"Experiment 3 -- redundancy of '{target}' with the general market factors\n"
        f"R² of OLS  {target}({unit}) ~ market factor({unit})   "
        "(single regressor: R² = corr²)",
        fontsize=11, y=0.99)
    size_note = ("  log_market_cap = z-score of log USD market cap (size), "
                 "informational, not in JOINT." if has_size else "")
    fig.text(0.5, 0.04,
             "R² = fraction of the target factor's cross-sectional variation linearly "
             "explained.  JOINT = multivariate fit on all factors.\n"
             "Shading: R² ≥ 0.03, 0.10, 0.25." + size_note,
             ha="center", fontsize=8, color="#555555")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.84, bottom=0.13)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run(target_factor: str,
        target_panel: Path = TARGET_PANEL,
        market_panel: Path = MARKET_PANEL,
        regressor_col: str = REGRESSOR_COL,
        market_factors: list[str] | None = None,
        out_dir: Path | None = None,
        include_market_cap: bool = False) -> pd.DataFrame:
    """
    Compute the R^2 table for ``target_factor`` and render it as ``r2_table.png``
    under ``out_dir`` (default ``OUTPUT_DIR/<target_factor>``).
    Set ``include_market_cap=True`` to append the ``log_market_cap`` (size) row.
    Returns the table.
    """
    out_dir = (OUTPUT_DIR / target_factor) if out_dir is None else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    table = factor_r2_table(target_factor, target_panel, market_panel,
                            regressor_col, market_factors, include_market_cap)
    render_r2_table(table, out_dir / "r2_table.png")

    joint = table.loc[table["market_factor"] == "__joint__", "r2"].iloc[0]
    print(f"=== '{target_factor}' vs general market factors "
          f"(R² of {regressor_col} regression) ===")
    show = table[table["market_factor"] != "__joint__"]
    print(show[["market_factor", "family", "n_obs", "corr", "r2"]].to_string(index=False))
    print(f"JOINT (all market factors together): R² = {joint:.4f}")
    print(f"Saved -> {out_dir}\n")
    return table

