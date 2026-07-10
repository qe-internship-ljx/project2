"""
correlation_matrix.py
======================

Experiment 2's redundancy view: for a whole factor **library** at once, how
correlated is every chosen factor with every Experiment 1 general market factor?

Two complementary lenses, one PNG each, written directly under a subexperiment's
``factor_correlation/`` (no per-factor subfolders):

    zscore_correlation.png   Pearson corr of the two factors' cross-sectional
                             ``zscore`` exposures, pooled over all (date, stock).
                             "Do the two factors *rank* stocks the same way?"
    return_correlation.png   Pearson corr of the two factors' univariate Q5-Q1
                             long-short monthly return series.
                             "Do the two factors *earn* the same way?"

Each PNG is a chosen-factor (rows) × general-factor (columns) matrix, shaded by
|corr|.  Both lenses reuse machinery that already exists elsewhere:

* the zscore matrix pivots exposures with Experiment 3's panel-agnostic
  ``factor_correlation._load_exposures`` (same ``factor_panel.csv`` schema);
* the return matrix builds each factor's Q5-Q1 spread with :func:`_quintile_spread`,
  the same even-quintile / next-return logic Experiment 1's ``quintile.py`` uses.

Panel- and library-agnostic: :func:`run` takes the chosen-factor names and the two
panels, so it serves every Experiment 2 subexperiment (and any same-schema panel).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

N_QUINTILES = 5


# --------------------------------------------------------------------------- #
# Core -- the two correlation matrices
# --------------------------------------------------------------------------- #
def _quintile_spread(panel: pd.DataFrame, factor: str,
                     n: int = N_QUINTILES) -> pd.Series:
    """
    Monthly Q5-Q1 long-short return of one factor: even ``zscore`` quintiles per
    month, mean ``next_return`` in the top minus the bottom bucket.  Mirrors
    Experiment 1's ``quintile.quintile_returns`` without the engine's file I/O.
    """
    sub = panel.loc[panel["factor"] == factor, ["date", "stock_id", "zscore", "next_return"]]
    sub = sub.dropna(subset=["zscore", "next_return"])

    def _bucket(s: pd.Series) -> pd.Series:
        if s.notna().sum() < n:
            return pd.Series(np.nan, index=s.index)
        return pd.qcut(s.rank(method="first"), n, labels=range(1, n + 1)).astype("float")

    q = sub.groupby("date", observed=True)["zscore"].transform(_bucket)
    means = sub.assign(quintile=q).dropna(subset=["quintile"]).pivot_table(
        index="date", columns="quintile", values="next_return", aggfunc="mean")
    if float(n) not in means.columns or 1.0 not in means.columns:
        return pd.Series(dtype=float)
    return (means[float(n)] - means[1.0]).sort_index()


def _corr_matrix(chosen: dict[str, pd.Series],
                 market: dict[str, pd.Series]) -> pd.DataFrame:
    """Pearson corr of every chosen series (rows) vs every market series (columns),
    each pair aligned on its shared index (inner join)."""
    return pd.DataFrame(
        {mf: {cf: cs.corr(ms) for cf, cs in chosen.items()} for mf, ms in market.items()},
        index=list(chosen), columns=list(market))


def zscore_correlation(chosen_factors: list[str],
                       target_panel: Path, market_panel: Path) -> pd.DataFrame:
    """Chosen × general correlation of cross-sectional ``zscore`` exposures,
    pooled over every shared (date, stock_id)."""
    from factor_correlation import _load_exposures  # exp3 module, wired in by the caller
    tgt = _load_exposures(target_panel, "zscore")
    mkt = _load_exposures(market_panel, "zscore")
    # Align on the shared (date, stock_id) index; keep the two sides separate so a
    # name shared by both (e.g. cross_val's 'beta') is never collapsed by a join.
    idx = tgt.index.intersection(mkt.index)
    tgt, mkt = tgt.loc[idx], mkt.loc[idx]
    return _corr_matrix({c: tgt[c] for c in chosen_factors},
                        {m: mkt[m] for m in mkt.columns})


def return_correlation(chosen_factors: list[str],
                       target_panel: Path, market_panel: Path) -> pd.DataFrame:
    """Chosen × general correlation of univariate Q5-Q1 long-short monthly
    return series."""
    tgt = pd.read_csv(target_panel, parse_dates=["date"],
                      usecols=["date", "stock_id", "factor", "zscore", "next_return"])
    tgt["stock_id"] = tgt["stock_id"].astype(str)
    mkt = pd.read_csv(market_panel, parse_dates=["date"],
                      usecols=["date", "stock_id", "factor", "zscore", "next_return"])
    mkt["stock_id"] = mkt["stock_id"].astype(str)
    chosen = {c: _quintile_spread(tgt, c) for c in chosen_factors}
    market = {m: _quintile_spread(mkt, m) for m in mkt["factor"].unique()}
    return _corr_matrix(chosen, market)


# --------------------------------------------------------------------------- #
# Rendering (project house style -- shaded by |corr|)
# --------------------------------------------------------------------------- #
def _corr_color(c: float) -> str:
    """Shade a correlation cell by absolute magnitude (redundancy)."""
    if not np.isfinite(c):
        return "white"
    a = abs(c)
    if a >= 0.5:
        return "#9ed49e"
    if a >= 0.3:
        return "#bfe3bf"
    if a >= 0.15:
        return "#e8f2d8"
    return "white"


def render_matrix(matrix: pd.DataFrame, path: Path, title: str, subtitle: str) -> None:
    """Render a chosen (rows) × general (columns) correlation matrix as a shaded PNG."""
    rows, cols = list(matrix.index), list(matrix.columns)
    headers = ["Chosen factor"] + cols
    cell_text, cell_colors = [], []
    for cf in rows:
        text = [cf]
        colors = ["white"]
        for mf in cols:
            v = matrix.loc[cf, mf]
            text.append("" if not np.isfinite(v) else f"{v:+.2f}")
            colors.append(_corr_color(v))
        cell_text.append(text)
        cell_colors.append(colors)

    n_r, n_c = len(rows), len(cols)
    # First column holds the (long) factor names; give it a fixed share and split
    # the rest evenly among the general-factor columns.
    name_share = 0.20
    col_widths = [name_share] + [(1.0 - name_share) / n_c] * n_c
    label_w = max(len(r) for r in rows)
    fig, ax = plt.subplots(figsize=(0.16 * label_w + 0.9 * n_c, 0.5 * (n_r + 1) + 1.7))
    ax.axis("off")
    tbl = ax.table(cellText=cell_text, colLabels=headers, cellColours=cell_colors,
                   colWidths=col_widths,
                   cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for j in range(len(headers)):
        tbl[0, j].set_text_props(weight="bold", color="white",
                                 rotation=0 if j == 0 else 30)
        tbl[0, j].set_facecolor("#404040")
    for i in range(1, n_r + 1):
        tbl[i, 0].set_text_props(ha="left", weight="bold")

    fig.suptitle(title, fontsize=11, y=0.99)
    fig.text(0.5, 0.04, subtitle, ha="center", fontsize=8, color="#555555")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.84, bottom=0.13)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run(chosen_factors: list[str], target_panel: Path, market_panel: Path,
        out_dir: Path, label: str = "factor") -> None:
    """
    Write the two chosen × general correlation PNGs directly under ``out_dir``:
    ``zscore_correlation.png`` and ``return_correlation.png``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    zmat = zscore_correlation(chosen_factors, target_panel, market_panel)
    render_matrix(
        zmat, out_dir / "zscore_correlation.png",
        f"Chosen {label}s vs general market factors -- z-score correlation",
        "Pearson corr of cross-sectional z-score exposures, pooled over all "
        "(date, stock).\nShading: |corr| ≥ 0.15, 0.30, 0.50.")

    rmat = return_correlation(chosen_factors, target_panel, market_panel)
    render_matrix(
        rmat, out_dir / "return_correlation.png",
        f"Chosen {label}s vs general market factors -- Q5-Q1 return correlation",
        "Pearson corr of univariate Q5-Q1 long-short monthly return series.\n"
        "Shading: |corr| ≥ 0.15, 0.30, 0.50.")

    print(f"=== Chosen {label}s vs general market factors (correlation matrices) ===")
    print("z-score correlation:")
    print(zmat.round(2).to_string())
    print("\nQ5-Q1 return correlation:")
    print(rmat.round(2).to_string())
    print(f"Saved -> {out_dir}\n")
