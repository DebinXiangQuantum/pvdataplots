from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import geopandas as gpd
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from shapely.geometry import LineString, box


ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "Fig3"
DATA_XLSX = FIG_DIR / "全球总装机 0518.xlsx"
# Manuscript maps use the user-specified country-boundary dataset.
WORLD_SHP = ROOT / "data" / "国家边界矢量" / "World_countries.shp"

TARGET_CRS = "ESRI:54030"
SOURCE_CRS = "EPSG:4326"
MAP_CLIP = box(-179.9, -58, 179.9, 84)

COLORS = {
    "utility": "#2878b8",
    "distributed": "#d74b9b",
    "both": "#00a087",
    "utility_light": "#b9d8ee",
    "distributed_light": "#efbfd8",
    "total_light": "#b8dfd5",
    "land": "#F6F5F0",
    "border": "#C9C7BE",
    "grid": "#D7D5CE",
    "axis": "#4F4F4F",
    "text": "#202020",
    "target_red": "#D64F4B",
    "target_blue": "#3A71B8",
    "target_orange": "#E6A45C",
}

CHANGE_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "fig3_relative_change",
    [COLORS["utility"], "#F7F4EA", COLORS["distributed"]],
)


@dataclass(frozen=True)
class Scenario:
    name: str
    title: str
    group: str


SCENARIOS = [
    Scenario("CHN-Global", "CHN-Global", "Single-model"),
    Scenario("USA-Global", "USA-Global", "Single-model"),
    Scenario("DEU-Global", "DEU-Global", "Single-model"),
    Scenario("EU-Global", "EU-Global", "Single-model"),
    Scenario("Cont-Lead", "Cont-Lead", "Regional-leader"),
    Scenario("Subcont-Lead", "Subcont-Lead", "Regional-leader"),
    Scenario("Clus5-Lead", "Clus5-Lead", "Cluster-leader"),
    Scenario("Clus10-Lead", "Clus10-Lead", "Cluster-leader"),
    Scenario("Clus20-Lead", "Clus20-Lead", "Cluster-leader"),
]


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 6,
            "axes.labelsize": 6,
            "axes.titlesize": 6,
            "axes.linewidth": 0.5,
            "axes.edgecolor": COLORS["axis"],
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.fontsize": 6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def country_key(values: pd.Series) -> pd.Series:
    return values.astype(str).str.upper().str.strip()


def degree_label(value: int, axis: str) -> str:
    if value == 0:
        return "0°"
    suffix = ("W" if value < 0 else "E") if axis == "lon" else ("S" if value < 0 else "N")
    return f"{abs(value)}°{suffix}"


def load_world() -> gpd.GeoDataFrame:
    world = gpd.read_file(WORLD_SHP, columns=["NAME", "geometry"])
    world = world[world["NAME"].notna() & world.geometry.notna()].copy()
    world = world[world.geometry.intersects(MAP_CLIP)].copy()
    world = world.clip(MAP_CLIP)
    world["country_key"] = country_key(world["NAME"])
    world = world.to_crs(TARGET_CRS)
    world["geometry"] = world.geometry.simplify(8000, preserve_topology=True)
    return world


def add_graticule(ax: plt.Axes, lon_labels: bool, lat_labels: bool) -> None:
    meridians = range(-180, 181, 60)
    parallels = range(-60, 91, 30)
    lines: list[LineString] = []
    for lon in meridians:
        lines.append(LineString([(lon, lat) for lat in np.linspace(-58, 84, 150)]))
    for lat in parallels:
        lines.append(LineString([(lon, lat) for lon in np.linspace(-179.9, 179.9, 220)]))

    grid = gpd.GeoSeries(lines, crs=SOURCE_CRS).to_crs(TARGET_CRS)
    for geom in grid:
        x, y = geom.xy
        ax.plot(x, y, color=COLORS["grid"], linewidth=0.28, zorder=0)

    if lon_labels:
        lon_vals = [-120, -60, 0, 60, 120]
        pts = gpd.GeoSeries(gpd.points_from_xy(lon_vals, [-56] * len(lon_vals)), crs=SOURCE_CRS).to_crs(TARGET_CRS)
        for lon, pt in zip(lon_vals, pts, strict=True):
            ax.text(pt.x, pt.y, degree_label(lon, "lon"), ha="center", va="top", fontsize=6, color="#666666")

    if lat_labels:
        lat_vals = [-30, 0, 30, 60]
        pts = gpd.GeoSeries(gpd.points_from_xy([-176] * len(lat_vals), lat_vals), crs=SOURCE_CRS).to_crs(TARGET_CRS)
        for lat, pt in zip(lat_vals, pts, strict=True):
            ax.text(pt.x, pt.y, degree_label(lat, "lat"), ha="right", va="center", fontsize=6, color="#666666")


def load_country_data(kind: str) -> pd.DataFrame:
    if kind not in {"utility", "distributed"}:
        raise ValueError("kind must be 'utility' or 'distributed'")

    sheet = "集中式" if kind == "utility" else "分布式"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        raw = pd.read_excel(DATA_XLSX, sheet_name=sheet, header=None).iloc[3:].copy()
    raw = raw[raw[0].notna()].copy()
    raw["country_key"] = country_key(raw[0])
    raw = raw[~raw["country_key"].isin({"TOTAL", "NAN"})].copy()

    df = pd.DataFrame(
        {
            "country_key": raw["country_key"],
            "actual_capacity": pd.to_numeric(raw[2], errors="coerce"),
        }
    )
    actual = df["actual_capacity"].where(df["actual_capacity"] > 0)
    for index, scenario in enumerate(SCENARIOS):
        source_col = 5 + index * 2 if kind == "utility" else 5 + index
        cap = pd.to_numeric(raw[source_col], errors="coerce")
        df[f"{scenario.name}_capacity"] = cap
        df[f"{scenario.name}_relative_change"] = (cap - df["actual_capacity"]) / actual
    return df


def make_total_country_data(utility_df: pd.DataFrame, distributed_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["country_key", "actual_capacity"] + [f"{s.name}_capacity" for s in SCENARIOS]
    merged = utility_df[cols].merge(
        distributed_df[cols],
        on="country_key",
        how="outer",
        suffixes=("_utility", "_distributed"),
    )
    out = pd.DataFrame({"country_key": merged["country_key"]})
    out["actual_capacity"] = merged[["actual_capacity_utility", "actual_capacity_distributed"]].sum(axis=1, min_count=1)
    actual = out["actual_capacity"].where(out["actual_capacity"] > 0)
    for scenario in SCENARIOS:
        cap_cols = [f"{scenario.name}_capacity_utility", f"{scenario.name}_capacity_distributed"]
        out[f"{scenario.name}_capacity"] = merged[cap_cols].sum(axis=1, min_count=1)
        out[f"{scenario.name}_relative_change"] = (out[f"{scenario.name}_capacity"] - out["actual_capacity"]) / actual
    return out


def load_global_data() -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        raw = pd.read_excel(DATA_XLSX, sheet_name="全球", header=None)
    utility = raw.iloc[1:10, [0, 5, 9]].copy()
    utility.columns = ["scenario", "utility", "actual_utility"]
    distributed = raw.iloc[13:22, [0, 5, 9]].copy()
    distributed.columns = ["scenario", "distributed", "actual_distributed"]
    total = raw.iloc[30:39, [0, 4, 5, 6]].copy()
    total.columns = ["scenario", "total", "actual_total", "ratio"]
    df = utility.merge(distributed, on="scenario").merge(total, on="scenario")
    numeric_cols = ["utility", "actual_utility", "distributed", "actual_distributed", "total", "actual_total", "ratio"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df["scenario"] = pd.Categorical(df["scenario"], [s.name for s in SCENARIOS], ordered=True)
    return df.sort_values("scenario").reset_index(drop=True)


def ratio_norm(values: pd.Series) -> tuple[mcolors.SymLogNorm, float]:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        vmax = 10.0
    else:
        vmax = float(np.nanpercentile(clean, 98))
        vmax = min(10000.0, max(10.0, 10 ** np.ceil(np.log10(max(vmax, 1.01)))))
    return mcolors.SymLogNorm(linthresh=0.25, linscale=0.7, vmin=-1.0, vmax=vmax, base=10), vmax


def ratio_ticks(vmax: float) -> list[float]:
    if vmax >= 1000:
        return [-1.0, 0.0, 1.0, 1000.0]
    if vmax >= 100:
        return [-1.0, 0.0, 1.0, 100.0]
    return [-1.0, 0.0, 1.0, 10.0]


def format_ratio_tick(value: float, _pos: int | None = None) -> str:
    if value == -1:
        return "-100%"
    if value == 0:
        return "0"
    if value == 1:
        return "+100%"
    return f"+{value:g}x"


def plot_violin_box_inset(
    ax: plt.Axes,
    values: pd.Series | np.ndarray,
    color: str,
    norm: mcolors.SymLogNorm,
    vmax: float,
    show_xlabel: bool = True,
) -> None:
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if values.empty:
        return
    clipped = values.clip(lower=norm.vmin, upper=norm.vmax)
    axins = inset_axes(
        ax,
        width="31%",
        height="26%",
        loc="lower left",
        bbox_to_anchor=(0.055, 0.045, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    parts = axins.violinplot([clipped.to_numpy()], positions=[0], vert=False, widths=0.74, showmeans=False, showmedians=False, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(0.28)
    q10, q33, med, q67, q90 = np.percentile(clipped, [10, 33, 50, 67, 90])
    axins.bxp(
        [{"med": med, "q1": q33, "q3": q67, "whislo": q10, "whishi": q90, "fliers": []}],
        positions=[0],
        vert=False,
        widths=0.34,
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": color, "alpha": 0.45, "edgecolor": COLORS["axis"], "linewidth": 0.45},
        medianprops={"color": "#1A1A1A", "linewidth": 0.7},
        whiskerprops={"color": COLORS["axis"], "linewidth": 0.45},
        capprops={"color": COLORS["axis"], "linewidth": 0.45},
    )
    axins.set_xscale("symlog", linthresh=0.25, linscale=0.7, base=10)
    axins.set_xlim(norm.vmin, norm.vmax)
    axins.set_ylim(-0.62, 0.62)
    axins.set_yticks([])
    ticks = [-1.0, 0.0, 10.0] if vmax >= 100 else [-1.0, 0.0, 1.0]
    axins.set_xticks(ticks)
    axins.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda value, _pos: "-1" if value == -1 else ("0" if value == 0 else f"{value:g}")))
    axins.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    axins.set_xlabel("Relative change" if show_xlabel else "", fontsize=6, labelpad=0.2)
    axins.tick_params(axis="x", length=1.6, pad=0.4, labelsize=6, width=0.45)
    axins.spines[["top", "left", "right"]].set_visible(False)
    axins.spines["bottom"].set_linewidth(0.45)
    axins.set_facecolor((1, 1, 1, 0.78))
