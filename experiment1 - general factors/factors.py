"""
factors.py
==========

Compute equity style factors for the **Software & Services** GICS industry
group, standardise each factor cross-sectionally (z-score relative to the
industry mean), and write a tidy monthly panel to ``output/software/factor_panel.csv``.

Per the project spec we implement the *main* signal of each factor family:

    family                 signal name           definition
    ---------------------  --------------------  ---------------------------------
    Value                  earnings_yield        earnings_ltm / market cap  (E/P)
    Momentum               momentum_12m              cum. total return over [t-12, t-1]
    Short-term reversal    reversal_1m                most recent 1-month total return
    Profitability/quality  gross_profitability   gross_income_ltm / assets (GP/A)
    Low-risk               beta                  36m trailing market beta
    Investment             asset_growth          assets_t / assets_{t-12m} - 1
    Net issuance           net_issuance          diluted shares YoY growth
    Earnings mom. / PEAD   sue                   YoY LTM earnings change, scaled
    Accruals (Sloan)       accruals              (earnings_ltm - op CF_ltm) / assets

All signals are ratios or returns and are therefore currency-neutral, so no FX
conversion is required for within-industry comparison.

This module is also the shared library for ``quintile.py`` and
``regression.py`` -- it exposes :func:`load_panel`, :func:`assign_quintiles`,
:func:`ols`, and the path / factor-name constants they rely on.

Run standalone to (re)build the panel::

    python factors.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths & configuration
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
PANEL_PATH = OUTPUT_DIR / "factor_panel.csv"

INDUSTRY_GROUP = "Software & Services"

# Estimation windows (in months unless noted).
MOM_LOOKBACK = 12          # momentum: cumulate the 12 months before formation
YOY_LAG = 12               # year-over-year lag for growth / SUE signals
BETA_WINDOW = 36           # trailing months for rolling market beta
BETA_MIN_PERIODS = 24
SUE_STD_WINDOW = 24        # trailing window for SUE standardisation
SUE_STD_MIN_PERIODS = 6
WINSOR_PCT = 0.01          # cross-sectional winsorisation before z-scoring

# Ordered list of the factors produced by this module.  ``higher_is_bullish``
# is the academically expected sign of the long-leg (top quintile): True means
# the canonical trade is long Q5 / short Q1, False means long Q1 / short Q5.
# It does not affect any factor *value*; it only fixes the long/short book's
# direction when ``USE_CANONICAL_LS_DIRECTION`` is set (see below).
FACTORS: dict[str, dict] = {
    "earnings_yield":      {"family": "Value",                 "higher_is_bullish": True},
    "momentum_12m":            {"family": "Momentum",              "higher_is_bullish": True},
    "reversal_1m":              {"family": "Short-term reversal",   "higher_is_bullish": False},
    "gross_profitability": {"family": "Profitability/quality", "higher_is_bullish": True},
    "beta":                {"family": "Low-risk",              "higher_is_bullish": False},
    "asset_growth":        {"family": "Investment",            "higher_is_bullish": False},
    "net_issuance":        {"family": "Net issuance",          "higher_is_bullish": False},
    "sue":                 {"family": "Earnings momentum/PEAD", "higher_is_bullish": True},
    "accruals":            {"family": "Accruals (Sloan)",       "higher_is_bullish": False},
}
FACTOR_NAMES = list(FACTORS)

# Long/short book direction.  When True (Experiment 1), the directional L/S book
# is signed by each factor's canonical literature direction (``higher_is_bullish``
# above) rather than inferred from the in-sample Fama-MacBeth t-stat.  Factor
# libraries that omit this attribute (e.g. Experiment 2's ``sw_factors``) keep
# the t-stat-inferred direction, so this change is scoped to Experiment 1 only.
USE_CANONICAL_LS_DIRECTION = True


# --------------------------------------------------------------------------- #
# Universe configuration
# --------------------------------------------------------------------------- #
# A :class:`Universe` bundles everything that differs between one GICS
# cross-section and another: how its membership is selected in the security
# master, which (pre-filtered) price file to read, and where its outputs land.
# Everything downstream (factor maths, z-scoring, sorts, regressions) is
# universe-agnostic, so the same code runs unchanged against any of these.
@dataclass(frozen=True)
class Universe:
    slug: str                                    # output subfolder / label
    price_file: str                              # pre-filtered price feather
    output_dir: Path                             # where this universe writes
    industry_group: str | None = None            # match gics_industry_group_name
    industries: tuple[str, ...] | None = None    # match gics_industry_name

    @property
    def panel_path(self) -> Path:
        return self.output_dir / "factor_panel.csv"


# Default universe: the original Software & Services industry group, writing to
# a dedicated output/software/ subfolder mirroring the other universes' layout.
SOFTWARE_SERVICES = Universe(
    slug="software_services",
    price_file="price_software_services.feather",
    output_dir=OUTPUT_DIR / "software",
    industry_group=INDUSTRY_GROUP,
)

# New universe: Banks + Insurance (gics_industry_name), writing to a dedicated
# output/banks_insurance/ subfolder mirroring the default layout.
BANKS_INSURANCE = Universe(
    slug="banks_insurance",
    price_file="price_banks_insurance.feather",
    output_dir=OUTPUT_DIR / "banks_insurance",
    industries=("Banks", "Insurance"),
)

# New universe: Commodity Producers (gics_industry_name), writing to a dedicated
# output/commodity_producers/ subfolder mirroring the default layout.
COMMODITY_PRODUCERS = Universe(
    slug="commodity_producers",
    price_file="price_commodity_producers.feather",
    output_dir=OUTPUT_DIR / "commodity_producers",
    industries=("Metals & Mining", "Oil, Gas & Consumable Fuels"),
)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def _to_datetime(df: pd.DataFrame, cols) -> pd.DataFrame:
    for c in cols:
        df[c] = pd.to_datetime(df[c])
    return df


def load_universe(u: Universe = SOFTWARE_SERVICES) -> pd.Index:
    """stock_id index of every security in the target GICS cross-section.

    Selected either by industry *group* (``gics_industry_group_name``) or by an
    explicit list of industries (``gics_industry_name``), per the universe spec.
    """
    col = "gics_industry_group_name" if u.industry_group else "gics_industry_name"
    sm = pd.read_feather(DATA_DIR / "security_master.feather",
                         columns=["stock_id", col])
    if u.industry_group:
        mask = sm[col] == u.industry_group
    else:
        mask = sm[col].isin(u.industries)
    return pd.Index(sm.loc[mask, "stock_id"].astype(str).unique(), name="stock_id")


def load_prices(universe: pd.Index, u: Universe = SOFTWARE_SERVICES) -> pd.DataFrame:
    """Daily prices/returns for the universe (already pre-filtered on disk)."""
    px = pd.read_feather(DATA_DIR / u.price_file,
                         columns=["stock_id", "date", "total_return",
                                  "security_mcap_local", "price_local"])
    px["stock_id"] = px["stock_id"].astype(str)
    px = px[px["stock_id"].isin(universe)]
    return _to_datetime(px, ["date"])


def load_fundamentals(universe: pd.Index) -> pd.DataFrame:
    """
    Point-in-time monthly fundamentals for the universe.

    Each record carries ``observation_date`` -- the date the underlying report
    became *observable* (i.e. publicly available).  ``date_fundamental`` is the
    fiscal-period stamp and, for ~a third of software stock-months, precedes the
    report's publication, so it cannot be used as the as-of date without
    look-ahead.  We carry ``observation_date`` (100% populated) and align on it
    in :func:`build_monthly_panel`.
    """
    base_cols = ["date_fundamental", "observation_date", "stock_id", "assets",
                 "gross_income_ltm", "earnings_ltm", "operating_cf_ltm"]
    fm = pd.read_feather(DATA_DIR / "fundamental_master.feather", columns=base_cols)
    fm["stock_id"] = fm["stock_id"].astype(str)
    fm = fm[fm["stock_id"].isin(universe)].copy()

    # The extended table has no observation_date; it shares the date_fundamental
    # grid, so it merges on that key and inherits the observation_date above.
    ext = pd.read_feather(
        DATA_DIR / "Industry Fundamentals Data" / "fundamental_master_extended.feather",
        columns=["date_fundamental", "stock_id", "diluted_shares_outstanding"])
    ext["stock_id"] = ext["stock_id"].astype(str)
    ext = ext[ext["stock_id"].isin(universe)]

    fm = fm.merge(ext, on=["stock_id", "date_fundamental"], how="left")
    return _to_datetime(fm, ["date_fundamental", "observation_date"])


def attach_pit_fundamentals(monthly: pd.DataFrame, fund: pd.DataFrame) -> pd.DataFrame:
    """
    Attach fundamentals to a monthly ``(stock_id, period)`` panel point-in-time:
    a record may enter month-end *t* only once it was observable
    (``observation_date <= t``).  For each stock-month we take the most recently
    observed report (a backward as-of join on ``observation_date``), which
    forward-fills the last published figure and leaves months before a stock's
    first report missing -- removing the look-ahead that aligning on
    ``date_fundamental`` introduces.

    ``fund`` must carry ``stock_id``, ``observation_date`` and the fundamental
    columns (plus ``date_fundamental``, used only to break ties).  ``monthly``
    must carry ``stock_id`` and ``period`` (a monthly ``PeriodIndex`` column).
    """
    fund = (fund.dropna(subset=["observation_date"])
                .sort_values(["stock_id", "observation_date", "date_fundamental"])
                .drop_duplicates(["stock_id", "observation_date"], keep="last")
                .drop(columns=["date_fundamental"])
                .rename(columns={"observation_date": "asof"}))

    # month-end timestamp of each formation period for the as-of comparison
    monthly = monthly.copy()
    monthly["asof"] = monthly["period"].dt.to_timestamp(how="end").dt.normalize()

    # merge_asof requires both sides globally sorted on the key.
    monthly = monthly.sort_values("asof")
    fund = fund.sort_values("asof")
    merged = pd.merge_asof(monthly, fund, on="asof", by="stock_id",
                           direction="backward")
    return merged.drop(columns=["asof"])


# --------------------------------------------------------------------------- #
# Monthly panel construction
# --------------------------------------------------------------------------- #
def build_monthly_panel(universe: pd.Index | None = None,
                        u: Universe = SOFTWARE_SERVICES) -> pd.DataFrame:
    """
    Assemble a (stock_id, period) monthly panel carrying everything the factor
    functions need: monthly total return, month-end market cap, the equal-
    weighted universe ("market") return, and the point-in-time fundamentals.
    """
    if universe is None:
        universe = load_universe(u)

    px = load_prices(universe, u)
    px["period"] = px["date"].dt.to_period("M")
    px["gross"] = 1.0 + px["total_return"].fillna(0.0)

    # Compound daily returns to a monthly total return; carry month-end levels.
    monthly = (px.groupby(["stock_id", "period"], observed=True)
                 .agg(mret=("gross", "prod"),
                      security_mcap_local=("security_mcap_local", "last"),
                      price_local=("price_local", "last"),
                      n_days=("date", "size"))
                 .reset_index())
    monthly["mret"] = monthly["mret"] - 1.0

    # Equal-weighted universe return = within-industry "market" proxy for beta.
    mkt = monthly.groupby("period", observed=True)["mret"].mean().rename("mkt_ret")
    monthly = monthly.merge(mkt, on="period", how="left")

    # Attach point-in-time fundamentals, aligned on observation_date so a report
    # only enters a month-end once it was actually observable (no look-ahead).
    fund = load_fundamentals(universe)
    monthly = attach_pit_fundamentals(monthly, fund)

    monthly = monthly.sort_values(["stock_id", "period"]).reset_index(drop=True)
    return monthly


# --------------------------------------------------------------------------- #
# Factor definitions
# --------------------------------------------------------------------------- #
# Each function takes the sorted monthly panel and returns a Series aligned to
# its index.  Time-series operations are done per stock; cross-sectional ones
# are handled later in :func:`add_zscores`.
def _f_earnings_yield(p: pd.DataFrame) -> pd.Series:
    return p["earnings_ltm"] / p["security_mcap_local"]


def _f_gross_profitability(p: pd.DataFrame) -> pd.Series:
    return p["gross_income_ltm"] / p["assets"]


def _f_accruals(p: pd.DataFrame) -> pd.Series:
    # Sloan (1996) accruals: the non-cash component of earnings, scaled by
    # assets.  High accruals (earnings >> cash flow) predict low future returns.
    return (p["earnings_ltm"] - p["operating_cf_ltm"]) / p["assets"]


def _f_reversal_1m(p: pd.DataFrame) -> pd.Series:
    return p["mret"]


def _f_momentum_12m(p: pd.DataFrame) -> pd.Series:
    # Cumulative total return over months [t-12, t-1], i.e. the 12 months
    # before formation -- this excludes the most recent month (reversal).
    cum = p.groupby("stock_id", observed=True)["mret"].apply(
        lambda s: (1.0 + s).cumprod())
    cum.index = p.index
    g = cum.groupby(p["stock_id"], observed=True)
    return g.shift(1) / g.shift(MOM_LOOKBACK + 1) - 1.0


def _f_asset_growth(p: pd.DataFrame) -> pd.Series:
    prev = p.groupby("stock_id", observed=True)["assets"].shift(YOY_LAG)
    return p["assets"] / prev - 1.0


def _f_net_issuance(p: pd.DataFrame) -> pd.Series:
    sh = p["diluted_shares_outstanding"]
    prev = sh.groupby(p["stock_id"], observed=True).shift(YOY_LAG)
    return sh / prev - 1.0


def _f_sue(p: pd.DataFrame) -> pd.Series:
    g = p.groupby("stock_id", observed=True)["earnings_ltm"]
    yoy = p["earnings_ltm"] - g.shift(YOY_LAG)
    scale = (yoy.groupby(p["stock_id"], observed=True)
                .transform(lambda s: s.rolling(SUE_STD_WINDOW,
                                                min_periods=SUE_STD_MIN_PERIODS).std()))
    return yoy / scale.replace(0.0, np.nan)


def _f_beta(p: pd.DataFrame) -> pd.Series:
    # Rolling 36-month beta = cov(r_i, r_mkt) / var(r_mkt), computed per stock
    # via rolling means of the relevant products (fully vectorised).
    ri, rm = p["mret"], p["mkt_ret"]
    tmp = pd.DataFrame({"stock_id": p["stock_id"], "ri": ri, "rm": rm,
                        "ri_rm": ri * rm, "rm2": rm * rm})

    def roll_mean(col):
        return (tmp.groupby("stock_id", observed=True)[col]
                   .transform(lambda s: s.rolling(BETA_WINDOW,
                                                   min_periods=BETA_MIN_PERIODS).mean()))

    m_ri, m_rm = roll_mean("ri"), roll_mean("rm")
    cov = roll_mean("ri_rm") - m_ri * m_rm
    var = roll_mean("rm2") - m_rm * m_rm
    return cov / var.replace(0.0, np.nan)


_FACTOR_FUNCS = {
    "earnings_yield": _f_earnings_yield,
    "momentum_12m": _f_momentum_12m,
    "reversal_1m": _f_reversal_1m,
    "gross_profitability": _f_gross_profitability,
    "beta": _f_beta,
    "asset_growth": _f_asset_growth,
    "net_issuance": _f_net_issuance,
    "sue": _f_sue,
    "accruals": _f_accruals,
}


def compute_factors(panel: pd.DataFrame) -> pd.DataFrame:
    """Add a raw value column for every factor in :data:`FACTOR_NAMES`."""
    for name in FACTOR_NAMES:
        panel[name] = _FACTOR_FUNCS[name](panel).replace([np.inf, -np.inf], np.nan)
    return panel


# --------------------------------------------------------------------------- #
# Cross-sectional standardisation
# --------------------------------------------------------------------------- #
def winsorize_cross_section(values: pd.Series, period: pd.Series,
                            pct: float = WINSOR_PCT) -> pd.Series:
    """Clip ``values`` to the [pct, 1-pct] quantiles within each ``period``."""
    if not pct:
        return values
    grp = values.groupby(period, observed=True)
    lo = grp.transform(lambda s: s.quantile(pct))
    hi = grp.transform(lambda s: s.quantile(1.0 - pct))
    return values.clip(lower=lo, upper=hi)


def cross_sectional_zscore(values: pd.Series, period: pd.Series,
                           winsor: float = WINSOR_PCT) -> pd.Series:
    """
    Standardise ``values`` within each ``period`` (the industry cross-section):
    winsorise at the [winsor, 1-winsor] quantiles, then (x - mean) / std.
    """
    clipped = winsorize_cross_section(values, period, winsor)
    cgrp = clipped.groupby(period, observed=True)
    mean = cgrp.transform("mean")
    std = cgrp.transform("std")
    return (clipped - mean) / std.replace(0.0, np.nan)


def add_zscores(panel: pd.DataFrame) -> pd.DataFrame:
    for name in FACTOR_NAMES:
        panel[f"{name}_z"] = cross_sectional_zscore(panel[name], panel["period"])
    return panel


def add_next_return(panel: pd.DataFrame) -> pd.DataFrame:
    """next_return at month t = the stock's total return realised in month t+1."""
    panel["next_return"] = (panel.groupby("stock_id", observed=True)["mret"]
                                 .shift(-1))
    return panel


# --------------------------------------------------------------------------- #
# Tidy output
# --------------------------------------------------------------------------- #
def to_long_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape to one row per (date, stock_id, factor) with the factor value, its
    z-score, and the next-period return.  Rows with a missing factor value are
    dropped to keep the file lean.
    """
    panel = panel.copy()
    panel["date"] = panel["period"].dt.to_timestamp(how="end").dt.normalize()
    frames = []
    for name in FACTOR_NAMES:
        f = panel[["date", "stock_id", name, f"{name}_z", "next_return"]].copy()
        f.columns = ["date", "stock_id", "value", "zscore", "next_return"]
        f.insert(2, "factor", name)
        frames.append(f.dropna(subset=["value"]))
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["factor", "date", "stock_id"]).reset_index(drop=True)


def build(save: bool = True, u: Universe = SOFTWARE_SERVICES) -> pd.DataFrame:
    """Full build: panel -> factors -> z-scores -> next return -> tidy long."""
    panel = build_monthly_panel(u=u)
    panel = compute_factors(panel)
    panel = add_zscores(panel)
    panel = add_next_return(panel)
    long = to_long_panel(panel)
    if save:
        u.output_dir.mkdir(parents=True, exist_ok=True)
        long.to_csv(u.panel_path, index=False)
    return long


# --------------------------------------------------------------------------- #
# Shared helpers used by quintile.py / regression.py
# --------------------------------------------------------------------------- #
def load_panel(rebuild: bool = False,
               u: Universe = SOFTWARE_SERVICES) -> pd.DataFrame:
    """Load the tidy factor panel, building it first if needed."""
    if rebuild or not u.panel_path.exists():
        return build(save=True, u=u)
    df = pd.read_csv(u.panel_path, parse_dates=["date"])
    df["stock_id"] = df["stock_id"].astype(str)
    return df


def prepare_slice(panel: pd.DataFrame, factor: str, n_quintiles: int = 5,
                  winsor_returns: float = WINSOR_PCT,
                  assign_q: bool = True) -> pd.DataFrame:
    """
    Shared input prep for both analyses: take one factor's rows, drop missing
    z-score / next-return, and winsorise next_return within each month
    (symmetric with the factor-side winsorisation, so extreme microcap return
    outliers do not dominate the means / regressions).

    When ``assign_q`` is True an even-quintile label is added (used by the
    quintile sort); the regression works on the full cross-section and leaves
    it off.
    """
    sub = panel.loc[panel["factor"] == factor,
                    ["date", "stock_id", "zscore", "next_return"]].copy()
    sub = sub.dropna(subset=["zscore", "next_return"])
    sub["next_return"] = winsorize_cross_section(sub["next_return"], sub["date"],
                                                 winsor_returns)
    if assign_q:
        sub["quintile"] = assign_quintiles(sub, "zscore", "date", n_quintiles)
        sub = sub.dropna(subset=["quintile"])
    return sub


def assign_quintiles(df: pd.DataFrame, value_col: str = "zscore",
                     date_col: str = "date", n: int = 5,
                     min_obs: int | None = None) -> pd.Series:
    """
    Even (equal-count) quintile labels 1..n assigned within each ``date``.

    Ranking with method='first' guarantees evenly sized buckets even when the
    factor has ties.  Months with fewer than ``min_obs`` (default ``n``)
    observations get NaN.
    """
    min_obs = n if min_obs is None else min_obs

    def _bucket(s: pd.Series) -> pd.Series:
        if s.notna().sum() < min_obs:
            return pd.Series(np.nan, index=s.index)
        ranks = s.rank(method="first")
        return pd.qcut(ranks, n, labels=range(1, n + 1)).astype("float")

    return df.groupby(date_col, observed=True)[value_col].transform(_bucket)


def ols(x: np.ndarray, y: np.ndarray) -> dict:
    """
    Univariate OLS of y on x.  Returns slope (beta), intercept (alpha), the
    t-statistic of the slope, R^2 and the observation count.  Returns NaNs when
    there are too few points or no variation in x.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = x.size
    nan = {"beta": np.nan, "alpha": np.nan, "tstat": np.nan, "r2": np.nan, "n": n}
    if n < 3 or np.ptp(x) == 0:
        return nan

    X = np.column_stack([np.ones(n), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha, beta = coef
    resid = y - X @ coef
    dof = n - 2
    sigma2 = (resid @ resid) / dof
    xc = x - x.mean()
    sxx = xc @ xc
    se_beta = np.sqrt(sigma2 / sxx)
    tstat = beta / se_beta if se_beta > 0 else np.nan
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1.0 - (resid @ resid) / ss_tot if ss_tot > 0 else np.nan
    return {"beta": beta, "alpha": alpha, "tstat": tstat, "r2": r2, "n": n}


# Registry so the drivers / CLI can pick a universe by slug.
UNIVERSES: dict[str, Universe] = {
    SOFTWARE_SERVICES.slug: SOFTWARE_SERVICES,
    BANKS_INSURANCE.slug: BANKS_INSURANCE,
    COMMODITY_PRODUCERS.slug: COMMODITY_PRODUCERS,
}


def universe_from_argv(default: Universe = SOFTWARE_SERVICES) -> Universe:
    """Pick a universe from argv[1] (its slug); fall back to ``default``."""
    import sys
    if len(sys.argv) > 1:
        return UNIVERSES[sys.argv[1]]
    return default


if __name__ == "__main__":
    u = universe_from_argv()
    long = build(save=True, u=u)
    n_months = long["date"].nunique()
    n_stocks = long["stock_id"].nunique()
    print(f"Built factor panel: {len(long):,} rows | "
          f"{n_stocks} stocks | {n_months} months "
          f"({long['date'].min():%Y-%m} .. {long['date'].max():%Y-%m})")
    print(f"Saved -> {u.panel_path}")
    counts = (long.groupby("factor")["value"].size()
                  .reindex(FACTOR_NAMES))
    print("\nObservations per factor:")
    for name, c in counts.items():
        print(f"  {name:<22} {c:>9,}")
