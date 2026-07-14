from __future__ import annotations

from pathlib import Path
from typing import Iterable

import geopandas as gpd
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from shapely.geometry import LineString, box


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "Fig1"
DATA_DIR = ROOT / "data"

WORLD_SHP = DATA_DIR / "国家边界矢量" / "World_countries.shp"
SOLAR_SHP = DATA_DIR / "10km" / "Solar_10km.shp"
COUNTRY_XLSX = FIG_DIR / "excel" / "lorendata.xlsx"
GDP_2024_XLSX = FIG_DIR / "data" / "洛伦兹曲线&基尼系数数据.xlsx"
GDP_2024_SHEET = "洛伦兹曲线"
GDP_2024_RAW_SHEET = "2024GDP"
GDP_2024_COL = "2024 GDP (constant 2015 US$)"
GDP_2024_PC_COL = "2024 GDP per capita (constant 2015 US$)"
REGION_XLSX = FIG_DIR / "excel" / "barchartFig1.xlsx"
UTILITY_PER_CAPITA = FIG_DIR / "data" / "percapita_utility.csv"
DISTRIBUTED_PER_CAPITA = FIG_DIR / "data" / "percapita_distributed.csv"
CACHE_CSV = FIG_DIR / "data" / "solar_points_cache.csv"

TARGET_CRS = "ESRI:54030"
CAPACITY_FACTOR = 0.2 / 1e6
MAP_CLIP = box(-180, -58, 180, 85)

COLORS = {
    "utility": "#2b7bba",
    "distributed": "#d74b9b",
    "utility_light": "#a9d6f5",
    "distributed_light": "#ebb4ec",
    "land": "#f2f2f0",
    "border": "#c8c8c8",
    "grid": "#d6d6d6",
    "text": "#1d1d1f",
}

PV_CMAP = LinearSegmentedColormap.from_list(
    "pv_potential",
    ["#d9f1fb", "#79bdd6", "#f1d46a", "#e86f43", "#8c1aa3"],
)


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
            "font.size": 6,
            "axes.labelsize": 6,
            "axes.titlesize": 6,
            "axes.titleweight": "normal",
            "axes.linewidth": 0.5,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "legend.fontsize": 6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
        }
    )


def degree_label(value: int, axis: str) -> str:
    if value == 0:
        return "0°"
    if axis == "lon":
        return f"{abs(value)}°{'W' if value < 0 else 'E'}"
    return f"{abs(value)}°{'S' if value < 0 else 'N'}"


def project_points(lons: Iterable[float], lats: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    series = gpd.GeoSeries(gpd.points_from_xy(lons, lats), crs="EPSG:4326").to_crs(TARGET_CRS)
    return series.x.to_numpy(), series.y.to_numpy()


def load_world() -> gpd.GeoDataFrame:
    world = gpd.read_file(WORLD_SHP, columns=["NAME", "geometry"])
    world = world.clip(MAP_CLIP)
    return world.to_crs(TARGET_CRS)


def load_solar_points() -> pd.DataFrame:
    if CACHE_CSV.exists() and CACHE_CSV.stat().st_mtime > SOLAR_SHP.stat().st_mtime:
        return pd.read_csv(CACHE_CSV)

    gdf = gpd.read_file(
        SOLAR_SHP,
        columns=["jizhong_ar", "fenbu_area", "geometry"],
    )
    gdf.columns = [col.lower() for col in gdf.columns]
    gdf = gdf[(gdf["jizhong_ar"] > 0) | (gdf["fenbu_area"] > 0)].copy()
    gdf = gdf[gdf.geometry.intersects(MAP_CLIP)].copy()

    bounds = gdf.geometry.bounds
    gdf["lon"] = (bounds["minx"] + bounds["maxx"]) / 2
    gdf["lat"] = (bounds["miny"] + bounds["maxy"]) / 2

    projected = gdf.to_crs(TARGET_CRS)
    centroids = projected.geometry.centroid

    points = pd.DataFrame(
        {
            "lon": gdf["lon"].to_numpy(),
            "lat": gdf["lat"].to_numpy(),
            "x": centroids.x.to_numpy(),
            "y": centroids.y.to_numpy(),
            "utility": gdf["jizhong_ar"].to_numpy() * CAPACITY_FACTOR,
            "distributed": gdf["fenbu_area"].to_numpy() * CAPACITY_FACTOR,
        }
    )
    points.to_csv(CACHE_CSV, index=False)
    return points


def add_panel_label(fig: plt.Figure, label: str, x: float, y: float) -> None:
    fig.text(
        x,
        y,
        label,
        ha="left",
        va="top",
        fontweight="bold",
        fontsize=8,
        color=COLORS["text"],
    )


def longitude_aligned_axes_position(fig: plt.Figure, map_ax: plt.Axes, y: float, height: float) -> list[float]:
    """Return a figure-positioned axis whose longitude ticks align with the map."""
    xs, ys = project_points([-135, 135], [0, 0])
    display_points = map_ax.transData.transform(np.column_stack([xs, ys]))
    fig_points = fig.transFigure.inverted().transform(display_points)
    x_135w, x_135e = fig_points[:, 0]
    width = (x_135e - x_135w) / 0.75
    left = x_135w - width * 0.125
    return [left, y, width, height]


def latitude_aligned_axes_position(
    fig: plt.Figure,
    map_ax: plt.Axes,
    x: float,
    width: float,
    lat_min: float = -60,
    lat_max: float | None = None,
) -> list[float]:
    """Return an axis whose latitude ticks align horizontally with the map."""
    _, y_bottom = project_points([0], [lat_min])
    if lat_max is None:
        y_top = map_ax.get_ylim()[1]
    else:
        _, projected_top = project_points([0], [lat_max])
        y_top = projected_top[0]

    display_points = map_ax.transData.transform(np.array([[0, y_bottom[0]], [0, y_top]]))
    fig_points = fig.transFigure.inverted().transform(display_points)
    bottom = fig_points[0, 1]
    top = fig_points[1, 1]
    return [x, bottom, width, top - bottom]


def nice_upper(value: float) -> float:
    if value <= 0:
        return 1.0
    magnitude = 10 ** np.floor(np.log10(value))
    scaled = value / magnitude
    if scaled <= 1.5:
        step = 1.5
    elif scaled <= 2:
        step = 2
    elif scaled <= 3:
        step = 3
    elif scaled <= 5:
        step = 5
    else:
        step = 10
    return step * magnitude


def add_graticule(ax: plt.Axes) -> None:
    meridians = range(-180, 181, 45)
    parallels = range(-60, 91, 30)

    lines = []
    for lon in meridians:
        lines.append(LineString([(lon, lat) for lat in np.linspace(-58, 85, 160)]))
    for lat in parallels:
        lines.append(LineString([(lon, lat) for lon in np.linspace(-180, 180, 260)]))

    grid = gpd.GeoSeries(lines, crs="EPSG:4326").to_crs(TARGET_CRS)
    for geom in grid:
        x, y = geom.xy
        ax.plot(x, y, color=COLORS["grid"], linewidth=0.35, zorder=0)

    label_lons = [-135, -90, -45, 0, 45, 90, 135]
    xs, ys = project_points(label_lons, [-57] * len(label_lons))
    for lon, x, y in zip(label_lons, xs, ys, strict=True):
        ax.text(x, y, degree_label(lon, "lon"), ha="center", va="top", fontsize=6, color="#6f6f6f")

    label_lats = [-60, -30, 0, 30, 60]
    xs, ys = project_points([177] * len(label_lats), label_lats)
    for lat, x, y in zip(label_lats, xs, ys, strict=True):
        ax.text(x, y, degree_label(lat, "lat"), ha="right", va="center", fontsize=6, color="#6f6f6f")


def plot_map(
    fig: plt.Figure,
    ax: plt.Axes,
    world: gpd.GeoDataFrame,
    points: pd.DataFrame,
    kind: str,
) -> None:
    values = points[kind].to_numpy()
    mask = values > 0
    selected = points.loc[mask]
    selected_values = values[mask]

    ax.set_facecolor("white")
    add_graticule(ax)
    world.plot(ax=ax, facecolor=COLORS["land"], edgecolor=COLORS["border"], linewidth=0.25, zorder=1)

    vmin = max(np.nanpercentile(selected_values, 5), np.nanmin(selected_values[selected_values > 0]))
    vmax = np.nanpercentile(selected_values, 99.5)
    hb = ax.hexbin(
        selected["x"],
        selected["y"],
        C=selected_values,
        gridsize=360,
        cmap=PV_CMAP,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        reduce_C_function=np.sum,
        linewidths=0,
        mincnt=1,
        rasterized=True,
        zorder=2,
    )

    ax.set_axis_off()

    cax = inset_axes(
        ax,
        width="42%",
        height="5%",
        loc="upper center",
        bbox_to_anchor=(0, 0.02, 1.0, 1.0),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    cb = fig.colorbar(hb, cax=cax, orientation="horizontal")
    cb.outline.set_linewidth(0.35)
    cb.ax.xaxis.set_major_locator(mpl.ticker.LogLocator(base=10, numticks=5))
    cb.ax.xaxis.set_minor_locator(mpl.ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=100))
    cb.ax.tick_params(length=1.5, pad=1, width=0.35, labelsize=6)
    cb.ax.tick_params(which="minor", length=1.2, width=0.35)
    cb.ax.set_title("PV Capacity (GW per cell)", fontsize=6, pad=1, fontweight="normal")


def plot_lorenz_panel(
    fig: plt.Figure,
    ax: plt.Axes,
    country_df: pd.DataFrame,
    value_col: str,
    gdp_col: str,
    gdp_pc_col: str,
    compact: bool = False,
) -> None:
    valid_df = country_df[[value_col, gdp_col, gdp_pc_col]].dropna()

    df_var = valid_df.sort_values(by=value_col).reset_index(drop=True)
    n = len(df_var)
    cum_pop = np.linspace(0, 100, n + 1)
    vals = df_var[value_col].to_numpy()
    cum_vals = np.concatenate([[0], np.cumsum(vals)])
    cum_vals_pct = cum_vals / cum_vals[-1] * 100

    try:
        area_lorenz_var = np.trapezoid(cum_vals_pct / 100, cum_pop / 100)
    except AttributeError:
        area_lorenz_var = np.trapz(cum_vals_pct / 100, cum_pop / 100)
    gini_var = (0.5 - area_lorenz_var) / 0.5

    df_gdp = valid_df.sort_values(by=gdp_col).reset_index(drop=True)
    gdp_vals = df_gdp[gdp_col].to_numpy()
    cum_gdp = np.concatenate([[0], np.cumsum(gdp_vals)])
    cum_gdp_pct = cum_gdp / cum_gdp[-1] * 100

    try:
        area_lorenz_gdp = np.trapezoid(cum_gdp_pct / 100, cum_pop / 100)
    except AttributeError:
        area_lorenz_gdp = np.trapz(cum_gdp_pct / 100, cum_pop / 100)
    gini_gdp = (0.5 - area_lorenz_gdp) / 0.5

    y_at_90 = np.interp(90, cum_pop, cum_vals_pct)
    y_gap = 100 - y_at_90

    ax.fill_between(cum_pop, cum_pop, cum_vals_pct, color="#c0c0c0", alpha=0.1, zorder=1)

    # cmap = plt.get_cmap("RdPu")

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "gdp_per_capita",
        ["#143bd6", "#F7F4EA", "#c71164"],
    )
    norm = mcolors.LogNorm(vmin=valid_df[gdp_pc_col].min(), vmax=valid_df[gdp_pc_col].max())
    for i in range(n):
        x1, x2 = cum_pop[i], cum_pop[i + 1]
        y1, y2 = cum_vals_pct[i], cum_vals_pct[i + 1]
        color = cmap(norm(df_var[gdp_pc_col].iloc[i]))
        ax.fill_between([x1, x2], 0, [y1, y2], color=color, alpha=1.0, edgecolor="none", zorder=2)

    line_width = 0.75 if compact else 1
    tick_size = 6
    label_size = 6
    note_size = 6
    legend_size = 6
    ax.set_facecolor("white")

    ax.plot([0, 100], [0, 100], color="black", linewidth=line_width, zorder=3)

    var_name = "Utility-scale PV" if "Utility" in value_col else "Distributed PV"
    ax.plot(cum_pop, cum_vals_pct, color="#4b0082", linewidth=line_width, zorder=5, label=f"{var_name} (Gini={gini_var:.2f})")
    ax.plot(cum_pop, cum_gdp_pct, color="#ff8c00", linewidth=line_width, linestyle="--", zorder=4, label=f"2024 GDP (Gini={gini_gdp:.2f})")

    ax.axvline(90, color="#1e90ff", linestyle="--", linewidth=0.4 if compact else 0.5, zorder=6)
    ax.annotate(
        "",
        xy=(89, y_at_90),
        xytext=(89, 100),
        arrowprops=dict(arrowstyle="<->, widthB=1.2, lengthB=0.3", color="black", lw=0.45 if compact else 0.6),
    )
    ax.text(78, (y_at_90 + 100) / 2, f"{y_gap:.0f}%", ha="left", va="center", fontweight="bold", fontsize=note_size)
    ax.text(95, 100.5, "10%", ha="center", va="bottom", fontweight="bold", fontsize=note_size)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4.5%" if compact else "5%", pad=0.018 if compact else 0.05)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("GDP per capita\n(constant 2015 US$)", fontsize=6, labelpad=0 if compact else 1)
    cbar.ax.tick_params(length=1.5 if compact else 2, pad=0.5 if compact else 1, labelsize=6)

    ax.text(34, 40, "Perfect equality", rotation=46, ha="center", va="center", alpha=0.7, fontsize=note_size)
    ax.text(50, 10, "Lorenz curve", rotation=12, ha="center", va="center", alpha=0.7, fontsize=note_size)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xticks([0, 50, 100])
    ax.set_yticks([0, 50, 100])
    ax.set_xlabel("Cumulative region percentage (%)", labelpad=0 if compact else 1, fontsize=label_size)
    ax.set_ylabel("Cumulative percentage (%)", labelpad=0 if compact else 1, fontsize=label_size)
    ax.tick_params(axis="both", length=1.5 if compact else 2, pad=0.5 if compact else 1, labelsize=tick_size)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(-0.10, 1.115) if compact else (0, 1.09),
        borderaxespad=0,
        frameon=compact,
        edgecolor="none",
        facecolor="white",
        framealpha=0.88 if compact else 0.8,
        fontsize=legend_size,
        handlelength=1.2 if compact else 1.5,
        labelspacing=0.18 if compact else 0.5,
    )


def binned_profile(points: pd.DataFrame, coord: str, kind: str, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    profile = (
        points.loc[points[kind] > 0]
        .groupby(pd.cut(points.loc[points[kind] > 0, coord], bins=bins, include_lowest=True), observed=True)[kind]
        .sum()
        .to_numpy()
    )
    centers = (bins[:-1] + bins[1:]) / 2
    if len(profile) != len(centers):
        complete = pd.Series(0.0, index=pd.IntervalIndex.from_breaks(bins))
        grouped = (
            points.loc[points[kind] > 0]
            .groupby(pd.cut(points.loc[points[kind] > 0, coord], bins=bins, include_lowest=True), observed=True)[kind]
            .sum()
        )
        complete.loc[grouped.index] = grouped
        profile = complete.to_numpy()
    return centers, profile


def plot_latitude_profile(
    ax: plt.Axes,
    points: pd.DataFrame,
    kind: str,
    color: str,
    show_xlabel: bool = True,
    map_ax: plt.Axes | None = None,
) -> None:
    bins = np.arange(-60, 86, 1)
    lat, profile = binned_profile(points, "lat", kind, bins)
    if map_ax is None:
        y_values = lat
        y_ticks = [-60, -30, 0, 30, 60]
        y_tick_labels = [degree_label(v, "lat") for v in y_ticks]
        y_lim = (-60, 85)
    else:
        _, y_values = project_points([0] * len(lat), lat)
        y_ticks_lat = [-60, -30, 0, 30, 60]
        _, y_ticks = project_points([0] * len(y_ticks_lat), y_ticks_lat)
        y_tick_labels = [degree_label(v, "lat") for v in y_ticks_lat]
        _, y_min = project_points([0], [-60])
        y_lim = (y_min[0], map_ax.get_ylim()[1])
    xmax = nice_upper(profile.max())
    ax.fill_betweenx(y_values, 0, profile, color=color, alpha=0.72, linewidth=0)
    ax.plot(profile, y_values, color=color, linewidth=0.8)
    ax.set_ylim(y_lim)
    ax.set_xlim(0, xmax)
    if xmax <= 20:
        ax.set_xticks(np.arange(0, xmax + 0.1, 5))
    else:
        ax.set_xticks([0, xmax / 2, xmax])
    ax.set_yticks(y_ticks)
    ax.grid(axis="y", color="#e1e1e1", linewidth=0.35)
    ax.tick_params(length=2, pad=1)
    if show_xlabel:
        ax.set_xlabel("PV Capacity (GW)", labelpad=0)
    else:
        ax.set_xlabel("", labelpad=1)
    ax.set_ylabel("", labelpad=1)
    ax.set_yticklabels(y_tick_labels)


def plot_longitude_profile(ax: plt.Axes, points: pd.DataFrame, kind: str, color: str) -> None:
    bins = np.arange(-180, 181, 1)
    lon, profile = binned_profile(points, "lon", kind, bins)
    ymax = nice_upper(profile.max())

    ax.fill_between(lon, 0, profile, color=color, alpha=0.58, linewidth=0)
    ax.plot(lon, profile, color=color, linewidth=0.85)
    ax.axhline(0, color="#2b2b2b", linewidth=0.5)
    ax.set_xlim(-180, 180)
    ax.set_ylim(0, ymax * 1.05)
    if ymax <= 20:
        yticks = np.arange(0, ymax + 0.1, 5)
    else:
        yticks = [0, ymax / 2, ymax]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{tick:g}" for tick in yticks])
    ax.set_xticks([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    ax.set_xticklabels([degree_label(x, "lon") for x in [-180, -135, -90, -45, 0, 45, 90, 135, 180]])
    ax.grid(axis="x", color="#e1e1e1", linewidth=0.35)
    ax.grid(axis="y", color="#eeeeee", linewidth=0.3)
    ax.set_ylabel("PV Capacity\n(GW per 1° longitude)", labelpad=0)
    ax.tick_params(length=2, pad=1)
    ax.spines[["top", "right"]].set_visible(False)


def plot_total_panel(ax: plt.Axes, country_df: pd.DataFrame, compact: bool = False) -> None:
    totals = pd.Series(
        {
            "Distributed PV": country_df["Distributed PV GW"].sum(),
            "Utility-scale PV": country_df["Utility-scale PV GW"].sum(),
        }
    )
    colors = [COLORS["distributed"], COLORS["utility"]]
    y = np.arange(len(totals))
    ax.barh(y, totals.values, color=colors, height=0.48 if compact else 0.52)
    max_total = totals.max()
    for idx, value in enumerate(totals.values):
        if compact and value > max_total * 0.35:
            ax.text(value * 0.965, idx, f"{value:,.0f}", va="center", ha="right", fontsize=6, color="white")
        else:
            ax.text(value + max_total * 0.035, idx, f"{value:,.0f}", va="center", ha="left", fontsize=6)
    ax.set_xlim(0, max_total * (1.10 if compact else 1.34))
    ax.set_yticks(y)
    if compact:
        ax.set_yticklabels(["Distributed\nPV", "Utility-scale\nPV"])
        ax.set_xticks([0, 2000])
        ax.set_xlabel("PV Capacity\n(GW)", labelpad=0)
    else:
        ax.set_yticklabels(totals.index)
        ax.set_xlabel("PV Capacity (GW)", labelpad=1)
    ax.tick_params(axis="y", length=0, pad=1)
    ax.tick_params(axis="x", length=2, pad=1)
    ax.spines[["top", "right"]].set_visible(False)


def plot_country_bar(ax: plt.Axes, country_df: pd.DataFrame) -> None:
    df = country_df.copy()
    df["Total PV GW"] = df["Utility-scale PV GW"] + df["Distributed PV GW"]
    ranked = df.sort_values("Total PV GW", ascending=False)
    top = ranked.head(14).copy()
    other = ranked.iloc[14:]
    if not other.empty:
        top = pd.concat(
            [
                top,
                pd.DataFrame(
                    [
                        {
                            "Country": "Other",
                            "Utility-scale PV GW": other["Utility-scale PV GW"].sum(),
                            "Distributed PV GW": other["Distributed PV GW"].sum(),
                            "Total PV GW": other["Total PV GW"].sum(),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    x = np.arange(len(top))
    ax.bar(x, top["Utility-scale PV GW"], color=COLORS["utility"], width=0.68, label="Utility-scale PV")
    ax.bar(
        x,
        top["Distributed PV GW"],
        bottom=top["Utility-scale PV GW"],
        color=COLORS["distributed"],
        width=0.68,
        label="Distributed PV",
    )
    for xi, total in zip(x, top["Total PV GW"], strict=True):
        ax.annotate(
            f"{total:.0f}",
            xy=(xi, total),
            xytext=(0, 1.5),
            textcoords="offset points",
            va="bottom",
            ha="center",
            fontsize=6,
        )

    label_map = {
        "UNITED STATES": "USA",
        "UNITED KINGDOM": "UK",
        "SOUTH AFRICA": "South Africa",
        "OTHER": "Other",
    }
    labels = [label_map.get(str(name), str(name).title()) for name in top["Country"]]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=34, ha="right", rotation_mode="anchor", fontsize=6)
    ax.set_ylim(0, top["Total PV GW"].max() * 1.22)
    ax.set_ylabel("PV Capacity (GW)", labelpad=1)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2, pad=1)
    ax.legend(frameon=False, ncol=2, loc="upper right", bbox_to_anchor=(0.995, 0.995), handlelength=1.5)
    ax.margins(x=0.02)


def country_key(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.upper()


def load_2024_gdp_by_country() -> pd.DataFrame:
    gdp_df = pd.read_excel(
        GDP_2024_XLSX,
        sheet_name=GDP_2024_SHEET,
        usecols=["地区", GDP_2024_COL, GDP_2024_PC_COL],
    )
    gdp_df = gdp_df.rename(columns={"地区": "Country"})
    gdp_df = gdp_df.dropna(subset=["Country"]).copy()
    gdp_df[GDP_2024_COL] = pd.to_numeric(gdp_df[GDP_2024_COL], errors="coerce")
    gdp_df[GDP_2024_PC_COL] = pd.to_numeric(gdp_df[GDP_2024_PC_COL], errors="coerce")
    gdp_df = gdp_df.dropna(subset=[GDP_2024_COL, GDP_2024_PC_COL]).copy()
    gdp_df["_country_key"] = country_key(gdp_df["Country"])
    return gdp_df[["_country_key", GDP_2024_COL, GDP_2024_PC_COL]]


def load_2024_gdp_per_capita_by_code() -> pd.Series:
    gdp_df = pd.read_excel(
        GDP_2024_XLSX,
        sheet_name=GDP_2024_RAW_SHEET,
        usecols=["Country Code", GDP_2024_PC_COL],
    )
    gdp_df["Country Code"] = gdp_df["Country Code"].astype(str).str.strip().str.upper()
    gdp_df[GDP_2024_PC_COL] = pd.to_numeric(gdp_df[GDP_2024_PC_COL], errors="coerce")
    gdp_df = gdp_df.dropna(subset=[GDP_2024_PC_COL])
    return gdp_df.set_index("Country Code")[GDP_2024_PC_COL]


def load_country_df() -> pd.DataFrame:
    df = pd.read_excel(COUNTRY_XLSX)
    df = df.rename(columns={"地区": "Country"})
    df = df.dropna(subset=["Country"]).copy()
    for col in ["Utility-scale PV GW", "Distributed PV GW"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["_country_key"] = country_key(df["Country"])
    df = df.merge(load_2024_gdp_by_country(), on="_country_key", how="left")
    df = df.drop(columns=["_country_key"])
    return df


def load_per_capita(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {"Country", "GDP Per Capita", "PV Capacity Per Capita"}
    if not expected.issubset(df.columns):
        df = pd.read_csv(
            path,
            header=None,
            names=["Country", "Name", "Name2", "GDP Per Capita", "PV Capacity Per Capita"],
        )
    gdp_per_capita = load_2024_gdp_per_capita_by_code()
    gdp_pc = pd.Series(np.nan, index=df.index, dtype="float64")
    for code_col in ["Name", "Name2"]:
        if code_col in df.columns:
            codes = df[code_col].astype(str).str.strip().str.upper()
            gdp_pc = gdp_pc.fillna(codes.map(gdp_per_capita))
    df["GDP Per Capita"] = np.log10(gdp_pc)
    df["PV Capacity Per Capita"] = pd.to_numeric(df["PV Capacity Per Capita"], errors="coerce")
    return df.dropna(subset=["GDP Per Capita", "PV Capacity Per Capita"])


def plot_gdp_quadrant(
    ax: plt.Axes,
    df: pd.DataFrame,
    country_df: pd.DataFrame,
    capacity_col: str,
    kind_label: str,
) -> None:
    plot_df = df.copy()
    capacity_df = country_df[["Country", capacity_col]].copy()
    capacity_df["_country_key"] = capacity_df["Country"].astype(str).str.strip().str.upper()
    plot_df["_country_key"] = plot_df["Country"].astype(str).str.strip().str.upper()
    plot_df = plot_df.merge(capacity_df[["_country_key", capacity_col]], on="_country_key", how="left")
    plot_df[capacity_col] = plot_df[capacity_col].fillna(0)

    x = plot_df["GDP Per Capita"]
    y = plot_df["PV Capacity Per Capita"]
    capacity = plot_df[capacity_col].clip(lower=0)
    kind_color = COLORS["utility"] if "Utility" in kind_label else COLORS["distributed"]
    x_mid = x.median()
    y_mid = y.median()
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    x_pad = x_range * 0.06
    y_pad = y_range * 0.12
    x_min, x_max = x.min() - x_pad, x.max() + x_pad
    y_min, y_max = y.min() - y_pad, y.max() + y_pad
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    quadrant_colors = ["#f1e6d7", "#d9ebe5", "#e3c0d8", "#f5f5f2"]
    quadrant_boxes = [
        (x_min, y_min, x_mid - x_min, y_mid - y_min, quadrant_colors[0]),
        (x_mid, y_min, x_max - x_mid, y_mid - y_min, quadrant_colors[1]),
        (x_min, y_mid, x_mid - x_min, y_max - y_mid, quadrant_colors[2]),
        (x_mid, y_mid, x_max - x_mid, y_max - y_mid, quadrant_colors[3]),
    ]
    for x0, y0, width, height, facecolor in quadrant_boxes:
        ax.add_patch(Rectangle((x0, y0), width, height, facecolor=facecolor, edgecolor="none", alpha=0.55, zorder=0))

    positive_capacity = capacity[capacity > 0]
    color_values = np.ma.masked_less_equal(capacity.to_numpy(), 0)
    scatter_cmap = PV_CMAP.copy()
    scatter_cmap.set_bad("#d8d8d8")
    norm = mcolors.LogNorm(vmin=float(positive_capacity.min()), vmax=float(positive_capacity.max()))
    scatter = ax.scatter(
        x,
        y,
        c=color_values,
        cmap=scatter_cmap,
        norm=norm,
        s=14,
        edgecolor="white",
        linewidth=0.25,
        alpha=0.95,
        zorder=2,
    )

    top = plot_df.sort_values("PV Capacity Per Capita", ascending=False).head(5)
    ax.scatter(
        top["GDP Per Capita"],
        top["PV Capacity Per Capita"],
        s=24,
        facecolors="none",
        edgecolors=kind_color,
        linewidth=0.6,
        zorder=3,
    )

    eq_x0 = max(x_min, x_mid + y_min - y_mid)
    eq_x1 = min(x_max, x_mid + y_max - y_mid)
    if eq_x1 > eq_x0:
        eq_x = np.array([eq_x0, eq_x1])
        eq_y = y_mid + (eq_x - x_mid)
        ax.plot(eq_x, eq_y, color="#8eb8aa", linewidth=0.6, zorder=1.4)
        ax.text(
            eq_x0 + (eq_x1 - eq_x0) * 0.28,
            eq_y[0] + (eq_y[1] - eq_y[0]) * 0.28 + y_range * 0.06,
            "45°",
            color="#3aa28c",
            fontsize=5,
            ha="center",
            va="center",
            zorder=2.5,
        )

    label_offsets = {
        "Utility-scale PV": {
            "CHL": (-0.410, 0.230, "right"),
            "ARE": (-0.180, 0.115, "right"),
            "AUS": (0.095, 0.235, "left"),
            "USA": (0.100, -0.120, "left"),
            "CHN": (-0.210, 0.060, "right"),
            "IND": (-0.170, 0.210, "right"),
            "AFG": (0.090, -0.430, "left"),
            "MDG": (0.105, 0.360, "left"),
            "ETH": (0.110, -0.320, "left"),
            "NGA": (0.090, 0.250, "left"),
            "CHE": (-0.170, -0.250, "right"),
            "LUX": (-0.160, 0.300, "right"),
        },
        "Distributed PV": {
            "MLT": (0.055, 0.360, "left"),
            "AUS": (0.100, 0.220, "left"),
            "ITA": (-0.160, -0.230, "right"),
            "CHN": (-0.160, 0.240, "right"),
            "USA": (0.085, 0.210, "left"),
            "JPN": (-0.170, 0.250, "right"),
            "AFG": (0.130, 0.260, "left"),
            "CAF": (0.150, -0.260, "left"),
            "GNQ": (0.090, -0.300, "left"),
            "GUY": (0.090, -0.280, "left"),
            "IRL": (-0.190, -0.230, "right"),
            "LUX": (-0.170, 0.330, "right"),
        },
    }
    label_rows = (
        pd.concat(
            [
                top.head(3),
                plot_df.nlargest(3, capacity_col),
                plot_df.nsmallest(2, "GDP Per Capita"),
                plot_df.nlargest(2, "GDP Per Capita"),
                plot_df.nsmallest(2, "PV Capacity Per Capita"),
            ]
        )
        .drop_duplicates(subset=["Country", "Name"], keep="first")
        .reset_index(drop=True)
    )
    y_offsets = [0.018, -0.026, 0.038, -0.044, 0.058]
    for rank, (_, row) in enumerate(label_rows.iterrows()):
        label = str(row.get("Name", row["Country"]))[:6]
        if label in label_offsets.get(kind_label, {}):
            dx, dy, ha = label_offsets[kind_label][label]
        else:
            if row["GDP Per Capita"] <= x_min + x_range * 0.22:
                dx, ha = x_range * 0.018, "left"
            elif row["GDP Per Capita"] >= x_max - x_range * 0.22:
                dx, ha = -x_range * 0.018, "right"
            else:
                dx, ha = x_range * 0.014, "left"

            if row["PV Capacity Per Capita"] <= y_min + y_range * 0.20:
                dy = y_range * (0.030 + 0.010 * (rank % 2))
            elif row["PV Capacity Per Capita"] >= y_max - y_range * 0.24:
                dy = y_range * (0.018 + 0.010 * (rank % 2))
            else:
                dy = y_range * y_offsets[rank % len(y_offsets)]
        ax.annotate(
            label,
            xy=(row["GDP Per Capita"], row["PV Capacity Per Capita"]),
            xytext=(row["GDP Per Capita"] + dx, row["PV Capacity Per Capita"] + dy),
            textcoords="data",
            fontsize=6,
            color=COLORS["text"],
            ha=ha,
            va="center",
            zorder=4,
            clip_on=False,
            arrowprops=dict(
                arrowstyle="-",
                color="#5f5f5f",
                linewidth=0.35,
                shrinkA=1.0,
                shrinkB=2.2,
            ),
        )

    ax.axvline(x_mid, color="#333333", linestyle="--", linewidth=0.55)
    ax.axhline(y_mid, color="#333333", linestyle="--", linewidth=0.55)
    ax.text(
        x_mid,
        y_max + y_range * 0.010,
        "Median",
        ha="center",
        va="bottom",
        fontsize=5,
        color=COLORS["text"],
        zorder=5,
        clip_on=False,
    )
    ax.text(x_max - x_range * 0.018, y_mid, "Median", ha="right", va="center", rotation=90, fontsize=5, color=COLORS["text"], zorder=5)
    ax.text(x_max - x_range * 0.030, y_max - y_range * 0.030, "I", ha="right", va="top", fontsize=7, color=COLORS["text"], zorder=5)
    ax.text(x_min + x_range * 0.030, y_max - y_range * 0.075, "II", ha="left", va="top", fontsize=7, color=COLORS["text"], zorder=5)
    ax.text(x_min + x_range * 0.030, y_min + y_range * 0.075, "III", ha="left", va="bottom", fontsize=7, color=COLORS["text"], zorder=5)
    ax.text(x_max - x_range * 0.018, y_min + y_range * 0.075, "IV", ha="right", va="bottom", fontsize=7, color=COLORS["text"], zorder=5)
    ax.grid(color="#e7e7e7", linewidth=0.35)
    ax.text(
        0.03,
        1.035,
        kind_label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color=COLORS["text"],
        fontsize=6,
        clip_on=False,
    )
    ax.set_xlabel("GDP per capita (log10 constant 2015 US$)", labelpad=1)
    ax.set_ylabel("PV Capacity per capita (log10)", labelpad=1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2, pad=1)

    pos = ax.get_position()
    cax_x = pos.x1 + 0.008
    cax = ax.figure.add_axes([cax_x, pos.y0, 0.006, pos.height])
    cbar = ax.figure.colorbar(scatter, cax=cax)
    cbar.locator = mpl.ticker.LogLocator(base=10, numticks=5)
    cbar.formatter = mpl.ticker.LogFormatterMathtext(base=10)
    cbar.update_ticks()
    cbar.outline.set_linewidth(0.35)
    cbar.ax.tick_params(length=1.5, pad=1, width=0.35, labelsize=5)
    cbar.set_label("PV Capacity (GW)", fontsize=5, labelpad=1)


def main() -> None:
    set_style()
    print("Loading country and gridded PV data...")
    world = load_world()
    points = load_solar_points()
    country_df = load_country_df()
    utility_pc = load_per_capita(UTILITY_PER_CAPITA)
    distributed_pc = load_per_capita(DISTRIBUTED_PER_CAPITA)

    width_mm = 180
    height_mm = 230
    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4), constrained_layout=False)

    # V4 keeps the maps as the dominant panels, then places one longitude
    # profile directly below each map using the equator-projected longitude grid.
    ax_map_u = fig.add_axes([0.060, 0.745, 0.805, 0.240])
    ax_map_d = fig.add_axes([0.060, 0.405, 0.805, 0.240])

    ax_total = fig.add_axes([0.895, 0.310, 0.085, 0.060])
    ax_country = fig.add_axes([0.060, 0.205, 0.910, 0.065])
    # ax_gdp_u = fig.add_axes([0.060, 0.030, 0.380, 0.125])
    # ax_gdp_d = fig.add_axes([0.570, 0.030, 0.360, 0.125])

    plot_map(fig, ax_map_u, world, points, "utility")
    ax_lat_u = fig.add_axes(latitude_aligned_axes_position(fig, ax_map_u, 0.895, 0.085, lat_min=-60))
    plot_latitude_profile(ax_lat_u, points, "utility", COLORS["utility"], map_ax=ax_map_u)
    lorenz_y_shift = 0.015
    ax_lorenz_u = fig.add_axes([0.060, 0.768 + lorenz_y_shift, 0.150, 0.125])
    ax_lorenz_u.set_zorder(20)
    plot_lorenz_panel(
        fig,
        ax_lorenz_u,
        country_df,
        "Utility-scale PV GW",
        GDP_2024_COL,
        GDP_2024_PC_COL,
        compact=True,
    )
    ax_lon_u = fig.add_axes(longitude_aligned_axes_position(fig, ax_map_u, 0.680, 0.045))
    plot_longitude_profile(ax_lon_u, points, "utility", COLORS["utility"])

    plot_map(fig, ax_map_d, world, points, "distributed")
    ax_lat_d = fig.add_axes(latitude_aligned_axes_position(fig, ax_map_d, 0.895, 0.085, lat_min=-60), sharey=ax_lat_u)
    plot_latitude_profile(ax_lat_d, points, "distributed", COLORS["distributed"], map_ax=ax_map_d)
    ax_lorenz_d = fig.add_axes([0.060, 0.428 + lorenz_y_shift, 0.150, 0.125])
    ax_lorenz_d.set_zorder(20)
    plot_lorenz_panel(
        fig,
        ax_lorenz_d,
        country_df,
        "Distributed PV GW",
        GDP_2024_COL,
        GDP_2024_PC_COL,
        compact=True,
    )
    ax_lon_d = fig.add_axes(longitude_aligned_axes_position(fig, ax_map_d, 0.310, 0.060))
    plot_longitude_profile(ax_lon_d, points, "distributed", COLORS["distributed"])

    plot_total_panel(ax_total, country_df, compact=True)
    plot_country_bar(ax_country, country_df)

    # plot_gdp_quadrant(ax_gdp_u, utility_pc, country_df, "Utility-scale PV GW", "Utility-scale PV")
    # plot_gdp_quadrant(ax_gdp_d, distributed_pc, country_df, "Distributed PV GW", "Distributed PV")

    label_y_top = 0.992
    label_y_mid = 0.655
    label_y_lon_u = 0.735
    label_y_lon_d = 0.382
    label_y_total = 0.382
    label_y_country = 0.282
    label_y_gdp = 0.168
    label_x_left = 0.000
    add_panel_label(fig, "a", label_x_left, label_y_top)
    add_panel_label(fig, "b", 0.025, 0.902 + lorenz_y_shift)
    add_panel_label(fig, "c", 0.865, label_y_top)
    add_panel_label(fig, "d", label_x_left, label_y_lon_u)
    add_panel_label(fig, "e", label_x_left, label_y_mid)
    add_panel_label(fig, "f", 0.025, 0.562 + lorenz_y_shift)
    add_panel_label(fig, "g", 0.865, label_y_mid)
    add_panel_label(fig, "h", label_x_left, label_y_lon_d)
    add_panel_label(fig, "i", 0.865, label_y_total)
    add_panel_label(fig, "j", label_x_left, label_y_country)
    # add_panel_label(fig, "k", label_x_left, label_y_gdp)
    # add_panel_label(fig, "l", 0.515, label_y_gdp)

    out_pdf = FIG_DIR / "PDFs" / "Fig1_composite.pdf"
    out_png = FIG_DIR / "Fig1_composite.png"
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    # 3. 将自定义的 bbox 传给 bbox_inches
    fig.savefig(out_pdf, dpi=600,)
    fig.savefig(out_png, dpi=300,)

    plt.close(fig)
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
