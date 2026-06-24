from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PLOTS_DIR = ROOT / "extenddatafigure" / "plots"
FIGURES_DIR = ROOT / "extenddatafigure" / "figures"

LORENZ_XLSX = ROOT / "Fig1" / "data" / "洛伦兹曲线&基尼系数数据.xlsx"
LORENZ_SHEET = "洛伦兹曲线"
NATIONAL_PV_XLSX = ROOT / "Fig2" / "excel" / "nationalPV.xlsx"
GDP_IRRADIANCE_CAPACITY_XLSX = ROOT / "Fig1" / "excel" / "GDP&Irradiance&capacity.xlsx"
GDP_IRRADIANCE_CAPACITY_FALLBACK_XLSX = ROOT / "Fig3" / "GDP&Irradiance&capacity.xlsx"

GDP_COL = "2024 GDP (constant 2015 US$)"
GDP_PC_COL = "2024 GDP per capita (constant 2015 US$)"
GDP_PC_CURRENT_COL = "2023 GDP per capita (current US$)"

COLORS = {
    "total": "#4b0082",
    "total_bar": "#297c78",
    "utility": "#2878b8",
    "distributed": "#d74b9b",
    "other": "#b7b7b7",
    "gdp": "#ff8c00",
    "grid": "#e5e5e5",
    "text": "#1d1d1f",
    "axis": "#4c4c4c",
}

GDP_PC_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "extended_gdp_per_capita",
    ["#143bd6", "#f7f4ea", "#c71164"],
)

INCOME_COLORS = OrderedDict(
    [
        ("Low", "#143bd6"),
        ("Low-middle", "#75a1ff"),
        ("Upper-middle", "#f49abd"),
        ("High", "#c71164"),
    ]
)
INCOME_BINS = [-np.inf, 1135, 4465, 13845, np.inf]
INCOME6_COLORS = OrderedDict(
    [
        ("<2,000", "#2454c6"),
        ("2,000-5,000", "#39a2ae"),
        ("5,000-10,000", "#69b34c"),
        ("10,000-20,000", "#e6b23f"),
        ("20,000-40,000", "#e97832"),
        (">40,000", "#c71164"),
    ]
)
INCOME6_BINS = [0, 2_000, 5_000, 10_000, 20_000, 40_000, np.inf]

PV_KIND_CONFIG = {
    "utility": {
        "label": "Utility-scale PV",
        "color": COLORS["utility"],
        "capacity_col": "Utility-scale PV GW",
        "per_capita_col": "集中式人均装机 MW/万人",
        "irradiance_col": "加权辐照强度-集中式潜力",
        "figure_token": "utility",
    },
    "distributed": {
        "label": "Distributed PV",
        "color": COLORS["distributed"],
        "capacity_col": "Distributed PV GW",
        "per_capita_col": "分布式人均装机 MW/万人",
        "irradiance_col": "加权辐照强度-分布式潜力",
        "figure_token": "distributed",
    },
}


COUNTRY_LABELS = {
    "BOSNIA AND HERZEGOVINA": "Bosnia & Herz.",
    "BRUNEI DARUSSALAM": "Brunei",
    "CZECH REPUBLIC": "Czechia",
    "EGYPT, ARAB REPUBLIC OF": "Egypt",
    "IRAN, ISLAMIC REPUBLIC OF": "Iran",
    "KOREA, REPUBLIC OF": "South Korea",
    "MACEDONIA,THE FORMER YUGOSLAV REPUBLIC OF": "North Macedonia",
    "RUSSIAN FEDERATION": "Russia",
    "SAUDI ARABIA": "Saudi Arabia",
    "SOUTH AFRICA": "South Africa",
    "UNITED ARAB EMIRATES": "UAE",
    "UNITED KINGDOM": "UK",
    "UNITED STATES": "USA",
    "VIET NAM": "Viet Nam",
}


def mm(value: float) -> float:
    return value / 25.4


def set_style(font_size: float = 6) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size,
            "axes.linewidth": 0.55,
            "axes.edgecolor": COLORS["axis"],
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.fontsize": font_size,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def country_key(value: object) -> str:
    key = str(value).strip().upper()
    return " ".join(key.split())


def display_country(value: object) -> str:
    key = country_key(value)
    return COUNTRY_LABELS.get(key, key.title())


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def get_pv_kind_config(kind: str) -> dict[str, str]:
    if kind not in PV_KIND_CONFIG:
        raise ValueError("kind must be 'utility' or 'distributed'.")
    return PV_KIND_CONFIG[kind]


def gdp_irradiance_capacity_path() -> Path:
    if GDP_IRRADIANCE_CAPACITY_XLSX.exists():
        return GDP_IRRADIANCE_CAPACITY_XLSX
    return GDP_IRRADIANCE_CAPACITY_FALLBACK_XLSX


def load_pv_irradiance_income_data(kind: str, require_positive_capacity: bool = False) -> pd.DataFrame:
    config = get_pv_kind_config(kind)
    df = pd.read_excel(gdp_irradiance_capacity_path())
    df.columns = [str(col).strip() for col in df.columns]

    required_cols = [
        "地区",
        config["capacity_col"],
        config["per_capita_col"],
        config["irradiance_col"],
        GDP_PC_CURRENT_COL,
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {gdp_irradiance_capacity_path()}: {missing}")

    cleaned = pd.DataFrame(
        {
            "country": df["地区"].map(country_key),
            "display_country": df["地区"].map(display_country),
            "capacity_gw": to_numeric(df[config["capacity_col"]]).fillna(0),
            "pv_per_capita_mw_per_10k": to_numeric(df[config["per_capita_col"]]),
            "weighted_irradiance_mj_m2": to_numeric(df[config["irradiance_col"]]),
            "gdp_per_capita_usd": to_numeric(df[GDP_PC_CURRENT_COL]),
        }
    )
    cleaned["income_group"] = pd.cut(
        cleaned["gdp_per_capita_usd"],
        bins=INCOME_BINS,
        labels=list(INCOME_COLORS.keys()),
    )
    cleaned = cleaned.dropna(subset=["country", "weighted_irradiance_mj_m2"]).copy()
    cleaned = cleaned[cleaned["weighted_irradiance_mj_m2"] > 0].copy()
    if require_positive_capacity:
        cleaned = cleaned[
            (cleaned["capacity_gw"] > 0)
            & (cleaned["pv_per_capita_mw_per_10k"] > 0)
            & (cleaned["gdp_per_capita_usd"] > 0)
            & cleaned["income_group"].notna()
        ].copy()
    cleaned["income_group"] = cleaned["income_group"].astype(str)
    return cleaned.reset_index(drop=True)


def capacity_marker_sizes(
    values: pd.Series | np.ndarray,
    min_size: float = 13,
    max_size: float = 88,
    percentile: float = 98,
    vmax: float | None = None,
) -> np.ndarray:
    raw_values = np.asarray(values, dtype=float)
    transformed = np.sqrt(np.clip(raw_values, 0, None))
    if vmax is None:
        positive = transformed[np.isfinite(transformed) & (transformed > 0)]
        vmax = np.nanpercentile(positive, percentile) if len(positive) else 1.0
    vmax = max(float(vmax), 1e-9)
    return np.interp(np.clip(transformed, 0, vmax), [0, vmax], [min_size, max_size])


def capacity_size_vmax(values: pd.Series | np.ndarray, percentile: float = 98) -> float:
    raw_values = np.asarray(values, dtype=float)
    transformed = np.sqrt(np.clip(raw_values, 0, None))
    positive = transformed[np.isfinite(transformed) & (transformed > 0)]
    if not len(positive):
        return 1.0
    return max(float(np.nanpercentile(positive, percentile)), 1e-9)


def load_total_pv_gdp_data() -> pd.DataFrame:
    df = pd.read_excel(LORENZ_XLSX, sheet_name=LORENZ_SHEET)
    df = df.dropna(subset=["地区"]).copy()

    numeric_cols = [
        "Utility-scale PV GW",
        "Distributed PV GW",
        GDP_COL,
        GDP_PC_COL,
        "人均装机 MW/万人",
    ]
    for col in numeric_cols:
        df[col] = to_numeric(df[col])

    cleaned = pd.DataFrame(
        {
            "country": df["地区"].map(country_key),
            "utility_pv_gw": df["Utility-scale PV GW"],
            "distributed_pv_gw": df["Distributed PV GW"],
            "gdp_2024_usd": df[GDP_COL],
            "gdp_per_capita_2024_usd": df[GDP_PC_COL],
            "pv_per_capita_mw_per_10k": df["人均装机 MW/万人"],
        }
    )
    cleaned["total_pv_gw"] = cleaned["utility_pv_gw"].fillna(0) + cleaned["distributed_pv_gw"].fillna(0)
    cleaned = cleaned.dropna(
        subset=[
            "country",
            "total_pv_gw",
            "gdp_2024_usd",
            "gdp_per_capita_2024_usd",
            "pv_per_capita_mw_per_10k",
        ]
    )
    cleaned = cleaned[
        (cleaned["total_pv_gw"] > 0)
        & (cleaned["gdp_2024_usd"] > 0)
        & (cleaned["gdp_per_capita_2024_usd"] > 0)
        & (cleaned["pv_per_capita_mw_per_10k"] >= 0)
    ].copy()
    cleaned["display_country"] = cleaned["country"].map(display_country)
    return cleaned.reset_index(drop=True)


def load_total_pv_per_capita_data() -> pd.DataFrame:
    df = pd.read_excel(NATIONAL_PV_XLSX)
    df = df.dropna(subset=["地区"]).copy()
    df["total_pv_gw"] = to_numeric(df["total PV GW"])
    df["population"] = to_numeric(df["Population"])
    df = df.dropna(subset=["total_pv_gw", "population"])
    df = df[(df["total_pv_gw"] > 0) & (df["population"] > 0)].copy()
    df["country"] = df["地区"].map(country_key)
    df["display_country"] = df["country"].map(display_country)
    df["pv_per_capita_mw_per_10k"] = df["total_pv_gw"] * 1e7 / df["population"]
    return df[["country", "display_country", "total_pv_gw", "population", "pv_per_capita_mw_per_10k"]].reset_index(
        drop=True
    )


def aggregate_top_with_others(df: pd.DataFrame, value_col: str, top_n: int) -> pd.DataFrame:
    ranked = df.sort_values(value_col, ascending=False).reset_index(drop=True)
    top = ranked.head(top_n).copy()
    other = ranked.iloc[top_n:].copy()
    if not other.empty:
        other_row = pd.DataFrame(
            [
                {
                    "country": "OTHERS",
                    "display_country": "Others",
                    "total_pv_gw": other["total_pv_gw"].sum(),
                    "population": other["population"].sum(),
                    "pv_per_capita_mw_per_10k": other["total_pv_gw"].sum() * 1e7 / other["population"].sum(),
                }
            ]
        )
        top = pd.concat([top, other_row], ignore_index=True)
    return top


def gini_from_sorted_values(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    vals = np.sort(np.clip(vals, 0, None))
    if len(vals) == 0 or vals.sum() <= 0:
        raise ValueError("Gini calculation requires positive finite values.")
    cumulative_x = np.linspace(0, 100, len(vals) + 1)
    cumulative_y = np.concatenate([[0.0], np.cumsum(vals)])
    cumulative_y = cumulative_y / cumulative_y[-1] * 100
    area = np.trapezoid(cumulative_y / 100, cumulative_x / 100)
    return cumulative_x, cumulative_y, (0.5 - area) / 0.5


def format_si(value: float) -> str:
    value = float(value)
    if abs(value) >= 1e12:
        return f"{value / 1e12:.0f}T"
    if abs(value) >= 1e9:
        return f"{value / 1e9:.0f}B"
    if abs(value) >= 1e6:
        return f"{value / 1e6:.0f}M"
    if abs(value) >= 1e3:
        return f"{value / 1e3:.0f}k"
    return f"{value:g}"


def annotate_label(
    ax,
    x: float,
    y: float,
    text: str,
    offset: tuple[float, float],
    ha: str = "left",
    va: str = "center",
) -> None:
    import matplotlib.patheffects as pe

    label = ax.annotate(
        text,
        xy=(x, y),
        xytext=offset,
        textcoords="offset points",
        ha=ha,
        va=va,
        fontsize=6,
        color=COLORS["text"],
        arrowprops=dict(arrowstyle="-", color=COLORS["text"], linewidth=0.35, shrinkA=1, shrinkB=2),
        zorder=6,
    )
    label.set_path_effects([pe.withStroke(linewidth=1.4, foreground="white")])
