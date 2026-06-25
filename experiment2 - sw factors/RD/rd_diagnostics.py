"""
rd_diagnostics.py
=================

Supplementary diagnostics for the R&D-behavior factors, supporting the
distinctness claims in ``R&D factors.md``.  Produces, under
``RD/factor_correlation/``:

    rd_factor_crosscorr.{csv,png}   pairwise correlation of the 5 R&D factor
                                    z-scores (are they distinct from each other?)
    redundancy_summary.{csv,png}    each R&D factor's JOINT R^2 vs the Exp1
                                    general factors and the Exp2 software factors
                                    (how little of each is spanned by the existing
                                    book), read from the per-factor r2_table.csv
                                    written by main_rd.py's correlation step.

Run after ``python main_rd.py`` (which writes the inputs)::

    python rd_diagnostics.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import rd_factors as R

OUTPUT_DIR = R.SOFTWARE_SERVICES.output_dir
CORR_DIR = OUTPUT_DIR / "factor_correlation"
ORDER = ["rd_growth", "rd_conversion", "rd_stability", "innovation_mix", "rd_intensity"]


# --------------------------------------------------------------------------- #
# Cross-correlation among the R&D factors
# --------------------------------------------------------------------------- #
def cross_correlation() -> pd.DataFrame:
    panel = pd.read_csv(OUTPUT_DIR / "factor_panel.csv",
                        usecols=["date", "stock_id", "factor", "zscore"])
    wide = panel.pivot_table(index=["date", "stock_id"], columns="factor",
                             values="zscore")
    cols = [c for c in ORDER if c in wide.columns]
    return wide[cols].corr()


def render_heatmap(corr: pd.DataFrame, path: Path) -> None:
    n = len(corr)
    fig, ax = plt.subplots(figsize=(1.3 * n + 2, 1.3 * n + 1.2))
    im = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(corr.index, fontsize=9)
    for i in range(n):
        for j in range(n):
            v = corr.iloc[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    color="white" if abs(v) > 0.55 else "black", fontsize=9)
    ax.set_title("Pairwise correlation of R&D-factor z-scores\n"
                 "(shared stock-months; off-diagonal magnitudes are the\n"
                 "redundancy among the five factors)", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Redundancy summary (JOINT R^2 vs each existing factor set)
# --------------------------------------------------------------------------- #
def redundancy_summary() -> pd.DataFrame:
    rows = []
    for f in ORDER:
        row = {"factor": f}
        for tag, label in [("vs_general", "joint_r2_vs_general"),
                           ("vs_software", "joint_r2_vs_software")]:
            p = CORR_DIR / tag / f / "r2_table.csv"
            if p.exists():
                t = pd.read_csv(p)
                row[label] = float(t.loc[t.market_factor == "__joint__", "r2"].iloc[0])
                sub = t[t.market_factor != "__joint__"]
                top = sub.loc[sub["r2"].idxmax()]
                row[f"top_{tag}"] = f"{top.market_factor} (R2={top.r2:.3f})"
            else:
                row[label] = np.nan
                row[f"top_{tag}"] = ""
        rows.append(row)
    return pd.DataFrame(rows)


def _r2_color(r2: float) -> str:
    if not np.isfinite(r2):
        return "white"
    if r2 >= 0.25:
        return "#9ed49e"
    if r2 >= 0.10:
        return "#bfe3bf"
    if r2 >= 0.03:
        return "#e8f2d8"
    return "white"


def render_redundancy(tbl: pd.DataFrame, path: Path) -> None:
    headers = ["R&D factor", "JOINT R²\nvs Exp1 general", "top general\nexplainer",
               "JOINT R²\nvs Exp2 software", "top software\nexplainer"]
    cell_text, cell_colors = [], []
    for _, r in tbl.iterrows():
        cell_text.append([r["factor"],
                          f"{r['joint_r2_vs_general']:.3f}", r["top_vs_general"],
                          f"{r['joint_r2_vs_software']:.3f}", r["top_vs_software"]])
        cell_colors.append(["white",
                            _r2_color(r["joint_r2_vs_general"]), "white",
                            _r2_color(r["joint_r2_vs_software"]), "white"])
    n = len(tbl)
    fig, ax = plt.subplots(figsize=(12.5, 0.5 * (n + 1) + 1.5))
    ax.axis("off")
    tbl_ax = ax.table(cellText=cell_text, colLabels=headers, cellColours=cell_colors,
                      colWidths=[0.18, 0.16, 0.25, 0.16, 0.25],
                      cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    tbl_ax.auto_set_font_size(False)
    tbl_ax.set_fontsize(9)
    for j in range(len(headers)):
        tbl_ax[0, j].set_text_props(weight="bold", color="white")
        tbl_ax[0, j].set_facecolor("#404040")
    for i in range(1, n + 1):
        tbl_ax[i, 0].set_text_props(ha="left")
        tbl_ax[i, 2].set_text_props(ha="left")
        tbl_ax[i, 4].set_text_props(ha="left")
    fig.suptitle("Redundancy of the R&D factors with the existing book\n"
                 "JOINT R² = fraction of each R&D factor spanned by ALL of that "
                 "set's factors together (lower = more distinct)", fontsize=11, y=0.99)
    fig.text(0.5, 0.04, "Benchmark: the existing rd_productivity sits at JOINT R²≈0.03 "
             "vs the general set.  Shading: R² ≥ 0.03, 0.10, 0.25.",
             ha="center", fontsize=8, color="#555555")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.80, bottom=0.13)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    CORR_DIR.mkdir(parents=True, exist_ok=True)

    corr = cross_correlation()
    corr.to_csv(CORR_DIR / "rd_factor_crosscorr.csv")
    render_heatmap(corr, CORR_DIR / "rd_factor_crosscorr.png")
    print("Cross-correlation of R&D factor z-scores:")
    print(corr.round(3).to_string())

    summ = redundancy_summary()
    summ.to_csv(CORR_DIR / "redundancy_summary.csv", index=False)
    render_redundancy(summ, CORR_DIR / "redundancy_summary.png")
    print("\nRedundancy summary (JOINT R² vs each existing set):")
    print(summ.to_string(index=False))
    print(f"\nSaved -> {CORR_DIR}")


if __name__ == "__main__":
    main()
