from __future__ import annotations

from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from matplotlib.lines import Line2D
from matplotlib import patheffects as pe
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pyproj import Transformer
from scipy.interpolate import PchipInterpolator
from shapely.geometry import LineString, box


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "Fig2"
EXPORT_DIR = FIG_DIR / "exported_plots"
DATA_DIR = ROOT / "data"
FIG2_DATA_DIR = FIG_DIR / "data"

WORLD_SHP = DATA_DIR / "国家边界矢量" / "World_countries.shp"
SOLAR_SHP = DATA_DIR / "10km" / "Solar_10km.shp"
SOLAR_DISTRIBUTION_XLSX = FIG_DIR / "excel" / "SolarDistributedAll.xlsx"
GDP_PV_XLSX = FIG_DIR / "excel" / "GDPvsPV.xlsx"
NATIONAL_PV_XLSX = FIG_DIR / "excel" / "nationalPV.xlsx"
WEIGHTED_IRRADIANCE_XLSX = FIG2_DATA_DIR / "4- GDP&Irradiance&capacity.xlsx"
SOLAR_MAP_CACHE = FIG2_DATA_DIR / "solar_map_cache.npz"

TARGET_CRS = "ESRI:54030"
SOURCE_CRS = "EPSG:4326"
MAP_CLIP = box(-179.9, -58, 179.9, 84)
CAPACITY_FACTOR = 0.2 / 1e6

COUNTRY_COLUMN_ORDER = [
    "China",
    "United States",
    "India",
    "Germany",
    "Japan",
    "Spain",
    "Australia",
    "Mexico",
    "Chile",
]

LAYOUT_COUNTRIES = COUNTRY_COLUMN_ORDER[:8]

COUNTRY_LABEL_POSITIONS = {
    "China": (0.06, 0.96),
    "United States": (0.06, 0.96),
    "Japan": (0.52, 0.90),
    "Germany": (0.55, 0.78),
}

COLORS = {
    "utility": "#2878b8",
    "distributed": "#d74b9b",
    "both": "#00a087",
    "land": "#f5f5f2",
    "border": "#cacaca",
    "grid": "#d7d7d7",
    "text": "#222222",
    "axis": "#4c4c4c",
}

RADIATION_BINS = np.array([2200, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000], dtype=float)
RADIATION_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "fig2_radiation",
    ["#d9f0f7", "#8cc9d8", "#b8de79", "#ffe16a", "#f18a4d", "#982bb9"],
)
RADIATION_NORM = mcolors.BoundaryNorm(RADIATION_BINS, RADIATION_CMAP.N, extend="both")

INCOME_COLORS = OrderedDict(
    [
        ("Low", "#143bd6"),
        ("Low-middle", "#75a1ff"),
        ("Upper-middle", "#f49abd"),
        ("High", "#c71164"),
    ]
)

GDP_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "fig2_gdp",
    ["#143bd6", "#75a1ff", "#f49abd", "#c71164"],
)
GDP_NORM = mcolors.LogNorm(vmin=300, vmax=150000)

COUNTRY_KEY_ALIASES = {
    "CONGO, THE DEMOCRATIC REPUBLIC OF THE": "CONGO,THE DEMOCRATIC REPUBLIC OF THE",
    "COTE D'IVOIRE": "COTE D’IVOIRE",
    "CÔTE D’IVOIRE": "COTE D’IVOIRE",
    "MACEDONIA": "MACEDONIA,THE FORMER YUGOSLAV REPUBLIC OF",
    "SRI LANKA": "SRILANKA",
}


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


def mm(value: float) -> float:
    return value / 25.4


def add_panel_label(
    ax: plt.Axes,
    label: str,
    x: float = -0.045,
    y: float = 1.055,
    fontsize: float = 8,
) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=fontsize,
        fontweight="bold",
        color=COLORS["text"],
        clip_on=False,
    )


def degree_label(value: int, axis: str) -> str:
    if value == 0:
        return "0°"
    suffix = ("W" if value < 0 else "E") if axis == "lon" else ("S" if value < 0 else "N")
    return f"{abs(value)}°{suffix}"


def project_lonlat(lons: np.ndarray | list[float], lats: np.ndarray | list[float]) -> tuple[np.ndarray, np.ndarray]:
    transformer = Transformer.from_crs(SOURCE_CRS, TARGET_CRS, always_xy=True)
    return transformer.transform(np.asarray(lons, dtype=float), np.asarray(lats, dtype=float))


def load_world_projected() -> gpd.GeoDataFrame:
    world = gpd.read_file(WORLD_SHP, columns=["geometry"])
    world = world[world.geometry.intersects(MAP_CLIP)].copy()
    world = world.clip(MAP_CLIP)
    return world.to_crs(TARGET_CRS)


def load_solar_map_data(force_rebuild: bool = False) -> pd.DataFrame:
    if (
        not force_rebuild
        and SOLAR_MAP_CACHE.exists()
        and SOLAR_MAP_CACHE.stat().st_mtime > SOLAR_SHP.stat().st_mtime
    ):
        cached = np.load(SOLAR_MAP_CACHE)
        return pd.DataFrame({key: cached[key] for key in cached.files})

    FIG2_DATA_DIR.mkdir(parents=True, exist_ok=True)
    gdf = gpd.read_file(
        SOLAR_SHP,
        columns=["光照强", "jizhong_ar", "fenbu_area", "geometry"],
    )
    bounds = shapely.bounds(gdf.geometry.array)
    lon = ((bounds[:, 0] + bounds[:, 2]) / 2).astype("float32")
    lat = ((bounds[:, 1] + bounds[:, 3]) / 2).astype("float32")
    keep = (lon >= -179.9) & (lon <= 179.9) & (lat >= -58) & (lat <= 84)

    x, y = project_lonlat(lon[keep], lat[keep])
    df = pd.DataFrame(
        {
            "lon": lon[keep],
            "lat": lat[keep],
            "x": np.asarray(x, dtype="float32"),
            "y": np.asarray(y, dtype="float32"),
            "radiation": (pd.to_numeric(gdf.loc[keep, "光照强"], errors="coerce").to_numpy() * 12 / 1e6).astype(
                "float32"
            ),
            "utility": (
                pd.to_numeric(gdf.loc[keep, "jizhong_ar"], errors="coerce").fillna(0).to_numpy()
                * CAPACITY_FACTOR
            ).astype("float32"),
            "distributed": (
                pd.to_numeric(gdf.loc[keep, "fenbu_area"], errors="coerce").fillna(0).to_numpy()
                * CAPACITY_FACTOR
            ).astype("float32"),
        }
    )
    np.savez_compressed(SOLAR_MAP_CACHE, **{col: df[col].to_numpy() for col in df.columns})
    return df


def add_graticule(ax: plt.Axes, label: bool = True) -> None:
    meridians = range(-180, 181, 45)
    parallels = range(-60, 91, 30)
    lines: list[LineString] = []
    for lon in meridians:
        lines.append(LineString([(lon, lat) for lat in np.linspace(-58, 84, 180)]))
    for lat in parallels:
        lines.append(LineString([(lon, lat) for lon in np.linspace(-179.9, 179.9, 260)]))

    grid = gpd.GeoSeries(lines, crs=SOURCE_CRS).to_crs(TARGET_CRS)
    for geom in grid:
        gx, gy = geom.xy
        ax.plot(gx, gy, color=COLORS["grid"], linewidth=0.35, zorder=0)

    if not label:
        return

    lon_labels = [-180, -135, -90, -45, 0, 45, 90, 135, 180]
    xs, ys = project_lonlat(lon_labels, [-55.5] * len(lon_labels))
    for lon, x, y in zip(lon_labels, xs, ys, strict=True):
        ax.text(x, y, degree_label(lon, "lon"), ha="center", va="top", fontsize=5.2, color="#666666")

    lat_labels = [-30, 0, 30, 60]
    xs, ys = project_lonlat([-177.5] * len(lat_labels), lat_labels)
    for lat, x, y in zip(lat_labels, xs, ys, strict=True):
        ax.text(x, y, degree_label(lat, "lat"), ha="right", va="center", fontsize=5.2, color="#666666")


def _scatter_pv_layer(ax: plt.Axes, df: pd.DataFrame, mask: np.ndarray, color: str, label: str, zorder: int) -> None:
    selected = df.loc[mask, ["x", "y"]]
    if selected.empty:
        return
    if len(selected) > 25000:
        selected = selected.sample(25000, random_state=7)
    ax.scatter(
        selected["x"],
        selected["y"],
        s=0.16,
        c=color,
        marker="o",
        alpha=0.42,
        linewidths=0,
        zorder=zorder,
        label=label,
    )


def plot_radiation_installation_map(
    fig: plt.Figure,
    ax: plt.Axes,
    world: gpd.GeoDataFrame,
    solar_df: pd.DataFrame,
    panel_label: str | None = None,
) -> None:
    ax.set_facecolor("white")
    add_graticule(ax, label=True)
    world.plot(ax=ax, facecolor=COLORS["land"], edgecolor="none", linewidth=0, zorder=1)

    hb = ax.hexbin(
        solar_df["x"],
        solar_df["y"],
        C=solar_df["radiation"],
        gridsize=430,
        reduce_C_function=np.mean,
        mincnt=1,
        cmap=RADIATION_CMAP,
        norm=RADIATION_NORM,
        linewidths=0,
        zorder=2,
    )
    world.plot(ax=ax, facecolor="none", edgecolor="#ffffff", linewidth=0.18, zorder=3)
    world.plot(ax=ax, facecolor="none", edgecolor=COLORS["border"], linewidth=0.08, zorder=4)

    utility = solar_df["utility"].to_numpy() > 1e-5
    distributed = solar_df["distributed"].to_numpy() > 1e-5
    _scatter_pv_layer(ax, solar_df, distributed & ~utility, COLORS["distributed"], "Distributed PV only", 5)
    _scatter_pv_layer(ax, solar_df, utility & ~distributed, COLORS["utility"], "Utility-scale PV only", 6)
    _scatter_pv_layer(ax, solar_df, utility & distributed, COLORS["both"], "Both PV types", 7)

    bounds = world.total_bounds
    pad_x = (bounds[2] - bounds[0]) * 0.015
    pad_y = (bounds[3] - bounds[1]) * 0.04
    ax.set_xlim(bounds[0] - pad_x, bounds[2] + pad_x)
    ax.set_ylim(bounds[1] - pad_y, bounds[3] + pad_y)
    ax.set_axis_off()

    cax = inset_axes(
        ax,
        width="43%",
        height="5%",
        loc="upper center",
        bbox_to_anchor=(0.02, -0.005, 0.96, 1.0),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    cb = fig.colorbar(hb, cax=cax, orientation="horizontal", extend="both")
    cb.outline.set_linewidth(0.35)
    cb.ax.tick_params(length=1.5, pad=1, width=0.35, labelsize=6)
    cb.set_ticks([3000, 5000, 7000, 9000])
    cb.ax.set_title("Annual solar irradiance (MJ m$^{-2}$ yr$^{-1}$)", fontsize=6, pad=1)

    pv_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=COLORS["utility"],
            markeredgecolor="none",
            markersize=3.6,
            label="Utility-scale PV",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=COLORS["distributed"],
            markeredgecolor="none",
            markersize=3.6,
            label="Distributed PV",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=COLORS["both"],
            markeredgecolor="none",
            markersize=3.6,
            label="Both PV types",
        ),
    ]
    ax.legend(
        handles=pv_handles,
        loc="lower left",
        bbox_to_anchor=(0.058, 0.11),
        frameon=False,
        handlelength=0.8,
        handletextpad=0.25,
        labelspacing=0.22,
        fontsize=6,
        borderpad=0,
    )
    if panel_label:
        add_panel_label(ax, panel_label, x=-0.018, y=1.095)


def load_distribution_data() -> tuple[pd.DataFrame, dict[str, tuple[str, str]]]:
    raw = pd.read_excel(SOLAR_DISTRIBUTION_XLSX)
    raw.columns = [str(col).strip() if not str(col).startswith("Unnamed") else str(col) for col in raw.columns]
    raw["光照"] = pd.to_numeric(raw["光照"], errors="coerce")
    raw = raw.dropna(subset=["光照"]).sort_values("光照").reset_index(drop=True)
    for col in raw.columns[1:]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    columns = raw.columns.tolist()
    country_cols: dict[str, tuple[str, str]] = {}
    start = 3
    for idx, country in enumerate(COUNTRY_COLUMN_ORDER):
        first = start + idx * 2
        second = first + 1
        if second < len(columns):
            country_cols[country] = (columns[first], columns[second])
    return raw, country_cols


def smooth_xy(x: np.ndarray, y: np.ndarray, points: int = 260) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    x_valid = x[mask]
    y_valid = y[mask]
    order = np.argsort(x_valid)
    x_valid = x_valid[order]
    y_valid = y_valid[order]
    keep = np.r_[True, np.diff(x_valid) > 0]
    x_valid = x_valid[keep]
    y_valid = y_valid[keep]
    if len(x_valid) < 4:
        return x_valid, np.clip(y_valid, 0, None)
    xs = np.linspace(x_valid.min(), x_valid.max(), points)
    ys = PchipInterpolator(x_valid, np.clip(y_valid, 0, None), extrapolate=False)(xs)
    return xs, np.clip(ys, 0, None)


def smooth_log_xy(x: np.ndarray, y: np.ndarray, points: int = 260) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (y > 0)
    if mask.sum() < 2:
        return np.array([]), np.array([])

    x_valid = x[mask]
    log_y = np.log10(y[mask])
    order = np.argsort(x_valid)
    x_valid = x_valid[order]
    log_y = log_y[order]
    keep = np.r_[True, np.diff(x_valid) > 0]
    x_valid = x_valid[keep]
    log_y = log_y[keep]
    if len(x_valid) < 2:
        return np.array([]), np.array([])

    xs = np.linspace(x_valid.min(), x_valid.max(), points)
    if len(x_valid) < 4:
        ys = np.interp(xs, x_valid, log_y)
    else:
        ys = PchipInterpolator(x_valid, log_y, extrapolate=False)(xs)
    return xs, ys


def nice_upper(value: float) -> float:
    if value <= 0:
        return 1.0
    magnitude = 10 ** np.floor(np.log10(value))
    scaled = value / magnitude
    if scaled <= 1.5:
        return 1.5 * magnitude
    if scaled <= 2:
        return 2 * magnitude
    if scaled <= 3:
        return 3 * magnitude
    if scaled <= 5:
        return 5 * magnitude
    return 10 * magnitude


def format_tick(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) >= 10:
        return f"{value:.0f}"
    if abs(value) >= 1:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    return f"{value:.2f}".rstrip("0").rstrip(".")


def format_log_tick(value: float) -> str:
    if float(value).is_integer():
        return f"{int(value)}"
    return f"{value:.1f}"


def plot_country_distribution(
    ax: plt.Axes,
    country: str,
    distribution_df: pd.DataFrame,
    country_cols: dict[str, tuple[str, str]],
    show_xlabel: bool = False,
) -> None:
    x = distribution_df["光照"].to_numpy(dtype=float)
    utility_col, distributed_col = country_cols[country]
    series = [
        (distribution_df[utility_col].to_numpy(dtype=float), COLORS["utility"], "Utility-scale PV"),
        (distribution_df[distributed_col].to_numpy(dtype=float), COLORS["distributed"], "Distributed PV"),
    ]
    curves: list[tuple[np.ndarray, np.ndarray, str, str]] = []
    for y, color, label in series:
        xs, ys = smooth_log_xy(x, y)
        if len(xs):
            curves.append((xs, ys, color, label))

    if curves:
        all_logs = np.concatenate([ys[np.isfinite(ys)] for _, ys, _, _ in curves])
        ybottom = float(np.floor(np.nanpercentile(all_logs, 5)))
        ytop = float(np.ceil(np.nanmax(all_logs)))
        if ytop <= ybottom:
            ytop = ybottom + 1
    else:
        ybottom, ytop = -3.0, 1.0

    for xs, ys, color, label in curves:
        ys_plot = np.clip(ys, ybottom, ytop)
        ax.fill_between(xs, ybottom, ys_plot, color=color, alpha=0.14, linewidth=0)
        ax.plot(xs, ys_plot, color=color, linewidth=0.72, label=label)

    label_x, label_y = COUNTRY_LABEL_POSITIONS.get(country, (0.12, 0.78))
    ax.text(
        label_x,
        label_y,
        country,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
        fontsize=6,
        color=COLORS["text"],
    )
    ax.set_xlim(2500, 9800)
    ax.set_ylim(ybottom, ytop)
    ax.set_xticks([3000, 6000, 9000])
    major_ticks = np.arange(np.floor(ybottom), np.ceil(ytop) + 1, 1.0)
    ax.set_yticks(major_ticks)
    ax.set_yticklabels(
        [
            format_log_tick(tick) if np.isclose(tick, major_ticks[0]) or np.isclose(tick, major_ticks[-1]) else ""
            for tick in major_ticks
        ]
    )
    ax.set_xlabel("Annual solar irradiance\n(MJ m$^{-2}$ yr$^{-1}$)", labelpad=0.4, fontsize=6)
    ax.set_ylabel("PV capacity\n(log$_{10}$ GW)", labelpad=0.3, fontsize=6)
    ax.tick_params(axis="x", length=1.8, pad=0.5, labelsize=6)
    ax.tick_params(axis="y", length=1.8, pad=0.5, labelsize=6)
    ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    ax.tick_params(axis="y", which="minor", length=1.1, width=0.4, labelleft=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(0.45)


def plot_global_distribution(
    ax: plt.Axes,
    distribution_df: pd.DataFrame,
    panel_label: str | None = None,
    compact: bool = False,
) -> None:
    x = distribution_df["光照"].to_numpy(dtype=float)
    series = [
        (distribution_df["集中式"].to_numpy(dtype=float), COLORS["utility"], "Utility-scale PV"),
        (distribution_df["分布式"].to_numpy(dtype=float), COLORS["distributed"], "Distributed PV"),
    ]
    curves: list[tuple[np.ndarray, np.ndarray, str, str]] = []
    for y, color, label in series:
        xs, ys = smooth_log_xy(x, y, points=360)
        if len(xs):
            curves.append((xs, ys, color, label))

    if curves:
        all_logs = np.concatenate([ys[np.isfinite(ys)] for _, ys, _, _ in curves])
        ybottom = float(np.floor(np.nanpercentile(all_logs, 5)))
        ytop = float(np.ceil(np.nanmax(all_logs)))
        if ytop <= ybottom:
            ytop = ybottom + 1
    else:
        ybottom, ytop = -4.0, 2.0

    for xs, ys, color, label in curves:
        ys_plot = np.clip(ys, ybottom, ytop)
        ax.fill_between(xs, ybottom, ys_plot, color=color, alpha=0.20, linewidth=0)
        ax.plot(xs, ys_plot, color=color, linewidth=0.85, label=label)

    ax.set_xlim(2500, 9800)
    ax.set_ylim(ybottom, ytop)
    ax.set_xticks([3000, 6000, 9000])
    major_ticks = np.arange(np.floor(ybottom), np.ceil(ytop) + 1, 1.0)
    ax.set_yticks(major_ticks)
    ax.set_yticklabels(
        [
            format_log_tick(tick) if np.isclose(tick, major_ticks[0]) or np.isclose(tick, major_ticks[-1]) else ""
            for tick in major_ticks
        ]
    )
    ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    ax.set_xlabel("Annual solar irradiance\n(MJ m$^{-2}$ yr$^{-1}$)", labelpad=0.4 if compact else 1)
    ax.set_ylabel("Global PV capacity\n(log$_{10}$ GW)", labelpad=0.5 if compact else 1)
    ax.grid(axis="y", color="#e7e7e7", linewidth=0.35)
    legend_loc = "lower right" if compact else "upper right"
    legend_anchor = (1.0, 1.02) if compact else None
    ax.legend(
        frameon=False,
        loc=legend_loc,
        bbox_to_anchor=legend_anchor,
        ncol=2,
        handlelength=1.4,
        columnspacing=0.8,
        fontsize=6,
        borderaxespad=0,
    )
    ax.tick_params(length=2, pad=1, labelsize=6)
    ax.tick_params(axis="y", which="minor", length=1.1, width=0.4, labelleft=False)
    if compact:
        ax.tick_params(labelsize=6)
    ax.spines[["top", "right"]].set_visible(False)
    if panel_label:
        add_panel_label(ax, panel_label, x=-0.103 if compact else -0.035, y=1.34)


def normalize_country_key(value: object) -> str:
    key = str(value).upper().strip()
    key = " ".join(key.split())
    return COUNTRY_KEY_ALIASES.get(key, key)


@lru_cache(maxsize=1)
def load_weighted_irradiance_targets() -> dict[str, tuple[float, float]]:
    df = pd.read_excel(WEIGHTED_IRRADIANCE_XLSX)
    df.columns = [str(col).strip() for col in df.columns]
    utility_col = "加权辐照强度-集中式潜力"
    distributed_col = "加权辐照强度-分布式潜力"
    targets: dict[str, tuple[float, float]] = {}
    for _, row in df.iterrows():
        key = normalize_country_key(row["地区"])
        utility_value = pd.to_numeric(row.get(utility_col), errors="coerce")
        distributed_value = pd.to_numeric(row.get(distributed_col), errors="coerce")
        targets[key] = (float(utility_value), float(distributed_value))
    return targets


@lru_cache(maxsize=2)
def load_country_solar_irradiance(kind: str) -> dict[str, float]:
    if kind not in {"utility", "distributed"}:
        raise ValueError("kind must be 'utility' or 'distributed'")

    solar = gpd.read_file(
        SOLAR_SHP,
        columns=["NAME", "光照强"],
        ignore_geometry=True,
    )
    solar["country_key"] = solar["NAME"].map(normalize_country_key)
    solar["annual_irradiance"] = pd.to_numeric(solar["光照强"], errors="coerce") * 12 / 1e6

    irradiance: dict[str, float] = {}
    for country_key, group in solar.groupby("country_key"):
        values = group["annual_irradiance"].to_numpy(dtype=float)
        mask = np.isfinite(values)
        if not mask.any():
            continue
        irradiance[country_key] = float(np.nanmean(values[mask]))
    return irradiance


def load_scatter_data(kind: str) -> pd.DataFrame:
    if kind not in {"utility", "distributed"}:
        raise ValueError("kind must be 'utility' or 'distributed'")

    sheet = "UtilityScalePV" if kind == "utility" else "DistributedPV"
    df = pd.read_excel(GDP_PV_XLSX, sheet_name=sheet)
    df.columns = [str(col).strip() for col in df.columns]
    df["GDPPerCapita"] = pd.to_numeric(df["GDPPerCapita"], errors="coerce")
    df["PVCapPerCapita"] = pd.to_numeric(df["PVCapPerCapita"], errors="coerce")

    national = pd.read_excel(NATIONAL_PV_XLSX)
    national["key"] = national["地区"].map(normalize_country_key)
    cap_col = "Utility-scale PV GW" if kind == "utility" else "Distributed PV GW"
    capacity_map = national.set_index("key")[cap_col].to_dict()
    gdp_pc_map = national.set_index("key")["2023 GDP per capita (current US$) 排序"].to_dict()

    keys = df["Nation"].map(normalize_country_key)
    irradiance_map = load_country_solar_irradiance(kind)
    df["capacity_gw"] = keys.map(capacity_map).fillna(0).astype(float)
    df["gdp_pc_raw"] = keys.map(gdp_pc_map).astype(float)
    df["gdp_pc_raw"] = df["gdp_pc_raw"].fillna(np.power(10, df["GDPPerCapita"]))
    df["annual_irradiance"] = keys.map(irradiance_map).astype(float)
    df["pv_cap_pc_raw"] = np.power(10, df["PVCapPerCapita"])
    df["income_group"] = pd.cut(
        df["gdp_pc_raw"],
        bins=[-np.inf, 1135, 4465, 13845, np.inf],
        labels=list(INCOME_COLORS.keys()),
    ).astype(str)
    return df.dropna(subset=["annual_irradiance", "PVCapPerCapita"])


def marker_sizes(
    capacity: pd.Series,
    reference_capacity: pd.Series | np.ndarray | None = None,
) -> np.ndarray:
    cap = np.sqrt(np.clip(capacity.to_numpy(dtype=float), 0, None))
    reference = cap if reference_capacity is None else np.sqrt(np.clip(np.asarray(reference_capacity, dtype=float), 0, None))
    vmax = np.nanpercentile(reference[reference > 0], 98) if np.any(reference > 0) else 1
    vmax = max(vmax, 1e-6)
    return np.interp(np.clip(cap, 0, vmax), [0, vmax], [7, 58])


def annotate_scatter_extremes(ax: plt.Axes, df: pd.DataFrame, title: str) -> None:
    x_col = "annual_irradiance"
    y_col = "PVCapPerCapita"
    bottom_indices = df.nsmallest(3, y_col).index.tolist()
    criteria = [
        ("capacity", df["capacity_gw"].idxmax()),
        ("top", df[y_col].idxmax()),
        ("right", df[x_col].idxmax()),
        ("left", df[x_col].idxmin()),
        *[(f"bottom{i + 1}", idx) for i, idx in enumerate(bottom_indices)],
    ]
    offsets_by_title = {
        "Utility-scale PV": {
            "capacity": (-13, 8, "right", "bottom"),
            "top": (8, 7, "left", "bottom"),
            "right": (-8, -9, "right", "top"),
            "left": (8, 10, "left", "bottom"),
            "bottom1": (5, 8, "left", "bottom"),
            "bottom2": (-7, 8, "right", "bottom"),
            "bottom3": (7, 8, "left", "bottom"),
        },
        "Distributed PV": {
            "capacity": (-12, -9, "right", "top"),
            "top": (-8, 8, "right", "bottom"),
            "right": (-8, -11, "right", "top"),
            "left": (8, 10, "left", "bottom"),
            "bottom1": (7, 8, "left", "bottom"),
            "bottom2": (6, 8, "left", "bottom"),
            "bottom3": (7, 8, "left", "bottom"),
        },
    }
    offsets = offsets_by_title.get(title, {})
    seen: set[int] = set()
    for criterion, idx in criteria:
        if idx in seen:
            continue
        seen.add(idx)
        row = df.loc[idx]
        label = str(row.get("ShortName") or row.get("Nation"))
        dx, dy, ha, va = offsets.get(criterion, (6, 6, "left", "bottom"))
        if criterion != "left":
            ax.scatter(
                [row[x_col]],
                [row[y_col]],
                s=[marker_sizes(pd.Series([row["capacity_gw"]]))[0] + 10],
                facecolors="none",
                edgecolors=COLORS["text"],
                linewidths=0.45,
                zorder=5,
            )
        text = ax.annotate(
            label,
            xy=(row[x_col], row[y_col]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=6,
            fontweight="bold",
            color=COLORS["text"],
            arrowprops={
                "arrowstyle": "-",
                "color": COLORS["text"],
                "linewidth": 0.35,
                "shrinkA": 1,
                "shrinkB": 2,
            },
            zorder=6,
        )
        text.set_path_effects([pe.withStroke(linewidth=1.4, foreground="white")])


def plot_capacity_scatter(
    ax: plt.Axes,
    df: pd.DataFrame,
    title: str,
    panel_label: str | None = None,
    colorbar_x: float = 1.015,
    ylabel_x: float = -0.04,
    show_ylabel: bool = True,
) -> None:
    sizes = marker_sizes(df["capacity_gw"])
    x_col = "annual_irradiance"
    y_col = "PVCapPerCapita"
    color_values = np.clip(df["gdp_pc_raw"].to_numpy(dtype=float), GDP_NORM.vmin, GDP_NORM.vmax)
    scatter = ax.scatter(
        df[x_col],
        df[y_col],
        s=sizes,
        c=color_values,
        cmap=GDP_CMAP,
        norm=GDP_NORM,
        alpha=0.88,
        edgecolors="none",
        linewidths=0,
        zorder=3,
    )

    ax.axhline(0, color="#bfbfbf", linewidth=0.55, zorder=1)
    ax.grid(color="#e9e9e9", linewidth=0.35, zorder=0)
    ax.set_title(title, loc="left", fontweight="bold", pad=2)
    ax.set_xlabel("Annual solar irradiance\n(MJ m$^{-2}$ yr$^{-1}$)", labelpad=1)
    if show_ylabel:
        ax.set_ylabel("PV capacity per capita\n(log10 MW per 10,000 people)", labelpad=3)
        ax.yaxis.set_label_coords(ylabel_x, 0.5)
    else:
        ax.set_ylabel("")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2, pad=1)

    y_pad = (df[y_col].max() - df[y_col].min()) * 0.10
    ax.set_xlim(2500, 9000)
    ax.set_xticks([3000, 6000, 9000])
    ax.set_ylim(df[y_col].min() - y_pad, df[y_col].max() + y_pad)
    cax = ax.inset_axes([colorbar_x, 0.0, 0.018, 1.0], transform=ax.transAxes)
    cbar = ax.figure.colorbar(scatter, cax=cax, orientation="vertical")
    cbar.outline.set_linewidth(0.35)
    cbar.set_ticks([1_000, 10_000, 100_000])
    cbar.ax.set_yticklabels(["1k", "10k", "100k"])
    cbar.ax.tick_params(length=1.4, pad=0.25, width=0.35, labelsize=6)
    cbar.ax.yaxis.set_ticks_position("right")
    cbar.ax.yaxis.set_label_position("right")
    cbar.ax.set_ylabel("GDP per capita ($)", fontsize=6, labelpad=0.3)
    annotate_scatter_extremes(ax, df, title)

    if panel_label:
        label_x = -0.085
        add_panel_label(ax, panel_label, x=label_x, y=1.15)


def add_income_legend(
    ax: plt.Axes,
    loc: str = "upper left",
    bbox_to_anchor: tuple[float, float] = (0.02, 0.98),
):
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markeredgecolor="none", markersize=4.0, label=label)
        for label, color in INCOME_COLORS.items()
    ]
    return ax.legend(
        handles=handles,
        title="Income level",
        frameon=False,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        borderpad=0,
        labelspacing=0.25,
        title_fontsize=6,
    )


def add_size_legend(
    ax: plt.Axes,
    values: tuple[float, float, float],
    loc: str = "lower right",
    bbox_to_anchor: tuple[float, float] = (0.98, 0.03),
):
    max_value = max(values)
    marker_values = pd.Series(values)
    sizes = marker_sizes(marker_values / max_value * max_value)
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#202020",
            markeredgecolor="none",
            markersize=np.sqrt(size),
            label=f"{value:g}",
        )
        for size, value in zip(sizes, values, strict=True)
    ]
    legend = ax.legend(
        handles=handles,
        title="PV capacity (GW)",
        frameon=False,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        borderpad=0,
        handletextpad=0.45,
        labelspacing=0.35,
        title_fontsize=6,
    )
    legend._legend_box.align = "left"
    return legend
