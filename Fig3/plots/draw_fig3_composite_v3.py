from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
OUT_DIR = FIG_DIR / "figures"
DATA_XLSX = FIG_DIR / "全球总装机 0518.xlsx"
WORLD_SHP = ROOT / "data" / "map" / "世界国家地图.shp"

TARGET_CRS = "ESRI:54030"
SOURCE_CRS = "EPSG:4326"
MAP_CLIP = box(-179.9, -58, 179.9, 84)

MM = 1 / 25.4
FIG_WIDTH_MM = 180
FIG_HEIGHT_MM = 150

COLORS = {
    "utility": "#2F6F9F",
    "distributed": "#7BBF7A",
    "utility_light": "#B9D6EA",
    "distributed_light": "#D2E7C8",
    "total_light": "#D7D2EA",
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
    ["#2F6F9F", "#F7F4EA", "#D85F8D"],
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
SCENARIO_LABELS = [scenario.title for scenario in SCENARIOS]


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

    for index, scenario in enumerate(SCENARIOS):
        if kind == "utility":
            cap = pd.to_numeric(raw[5 + index * 2], errors="coerce")
        else:
            cap = pd.to_numeric(raw[5 + index], errors="coerce")

        df[f"{scenario.name}_capacity"] = cap
        actual = df["actual_capacity"].where(df["actual_capacity"] > 0)
        df[f"{scenario.name}_relative_change"] = (cap - df["actual_capacity"]) / actual

    return df


def load_global_data() -> pd.DataFrame:
    raw = pd.read_excel(DATA_XLSX, sheet_name="全球", header=None)

    utility = raw.iloc[1:10, [0, 5, 9]].copy()
    utility.columns = ["scenario", "utility", "actual_utility"]
    distributed = raw.iloc[13:22, [0, 5, 9]].copy()
    distributed.columns = ["scenario", "distributed", "actual_distributed"]
    total = raw.iloc[30:39, [0, 4, 5, 6]].copy()
    total.columns = ["scenario", "total", "actual_total", "ratio"]

    df = utility.merge(distributed, on="scenario").merge(total, on="scenario")
    for col in ["utility", "actual_utility", "distributed", "actual_distributed", "total", "actual_total", "ratio"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["scenario"] = pd.Categorical(df["scenario"], [s.name for s in SCENARIOS], ordered=True)
    return df.sort_values("scenario").reset_index(drop=True)


def make_total_country_data(utility_df: pd.DataFrame, distributed_df: pd.DataFrame) -> pd.DataFrame:
    utility_cols = ["country_key", "actual_capacity"] + [f"{s.name}_capacity" for s in SCENARIOS]
    distributed_cols = ["country_key", "actual_capacity"] + [f"{s.name}_capacity" for s in SCENARIOS]
    merged = utility_df[utility_cols].merge(
        distributed_df[distributed_cols],
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


def inset_ratio_ticks(vmax: float) -> list[float]:
    if vmax >= 100:
        return [-1.0, 0.0, 10.0]
    return [-1.0, 0.0, 1.0]


def format_inset_tick(value: float, _pos: int | None = None) -> str:
    if value == -1:
        return "-1"
    if value == 0:
        return "0"
    return f"{value:g}"


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
    parts = axins.violinplot(
        [clipped.to_numpy()],
        positions=[0],
        vert=False,
        widths=0.74,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body in parts["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(0.28)

    q10, q33, med, q67, q90 = np.percentile(clipped, [10, 33, 50, 67, 90])
    stats = [
        {
            "med": med,
            "q1": q33,
            "q3": q67,
            "whislo": q10,
            "whishi": q90,
            "fliers": [],
        }
    ]
    axins.bxp(
        stats,
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
    ticks = inset_ratio_ticks(vmax)
    axins.set_xticks(ticks)
    axins.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(format_inset_tick))
    axins.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    axins.set_xlabel("Relative change" if show_xlabel else "", fontsize=6, labelpad=0.2)
    axins.tick_params(axis="x", length=1.6, pad=0.4, labelsize=6, width=0.45)
    axins.spines[["top", "left", "right"]].set_visible(False)
    axins.spines["bottom"].set_linewidth(0.45)
    axins.set_facecolor((1, 1, 1, 0.78))


def scenario_columns(scenario: Scenario) -> tuple[str, str]:
    return f"{scenario.name}_capacity", f"{scenario.name}_relative_change"


def plot_map_grid(
    fig: plt.Figure,
    spec: mpl.gridspec.SubplotSpec,
    world: gpd.GeoDataFrame,
    country_df: pd.DataFrame,
    kind: str,
    panel_label: str,
    title: str,
) -> list[plt.Axes]:
    sub = spec.subgridspec(
        4,
        3,
        height_ratios=[0.16, 1, 1, 1],
        wspace=0.13,
        hspace=0.20,
    )
    axes: list[plt.Axes] = []
    bounds = world.total_bounds
    pad_x = (bounds[2] - bounds[0]) * 0.012
    pad_y = (bounds[3] - bounds[1]) * 0.04
    inset_color = {
        "utility": COLORS["utility_light"],
        "distributed": COLORS["distributed_light"],
        "total": COLORS["total_light"],
    }.get(kind, COLORS["total_light"])

    all_ratios = []
    for scenario in SCENARIOS:
        _, ratio_col = scenario_columns(scenario)
        all_ratios.append(country_df[ratio_col])
    norm, vmax = ratio_norm(pd.concat(all_ratios))

    header_ax = fig.add_subplot(sub[0, :])
    header_ax.axis("off")
    if panel_label:
        header_ax.text(
            -0.03,
            0.5,
            panel_label,
            transform=header_ax.transAxes,
            ha="left",
            va="center",
            fontsize=8,
            fontweight="bold",
            color=COLORS["text"],
        )
    header_ax.text(
        0.026,
        0.5,
        title,
        transform=header_ax.transAxes,
        ha="left",
        va="center",
        fontsize=6,
        fontweight="bold",
        color=COLORS["text"],
    )

    cax = fig.add_axes([0, 0, 0.01, 0.01])
    pos = header_ax.get_position()
    cax.set_position([pos.x0 + pos.width * 0.46, pos.y0 + pos.height * 0.70, pos.width * 0.34, pos.height * 0.18])
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=CHANGE_CMAP)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", extend="both")
    cb.outline.set_linewidth(0.35)
    cb.set_ticks(ratio_ticks(vmax))
    cb.ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(format_ratio_tick))
    cb.ax.tick_params(length=1.4, pad=0.5, width=0.35, labelsize=6)
    cb_label = "Relative total PV capacity change vs actual" if kind == "total" else "Relative PV capacity change vs actual"
    cb.set_label(cb_label, fontsize=6, labelpad=0.4)

    panel_letters = list("abcdefghi")
    for i, scenario in enumerate(SCENARIOS):
        row, col = divmod(i, 3)
        ax = fig.add_subplot(sub[row + 1, col])
        axes.append(ax)
        _, ratio_col = scenario_columns(scenario)
        merged = world.merge(country_df[["country_key", ratio_col]], on="country_key", how="left")

        ax.set_facecolor("white")
        add_graticule(ax, lon_labels=False, lat_labels=col == 0)
        world.plot(ax=ax, facecolor=COLORS["land"], edgecolor="none", linewidth=0, zorder=1, rasterized=True)
        merged.dropna(subset=[ratio_col]).plot(
            column=ratio_col,
            ax=ax,
            cmap=CHANGE_CMAP,
            norm=norm,
            linewidth=0,
            edgecolor="none",
            zorder=2,
            rasterized=True,
        )
        world.plot(ax=ax, facecolor="none", edgecolor="#FFFFFF", linewidth=0.13, zorder=3, rasterized=True)
        world.plot(ax=ax, facecolor="none", edgecolor=COLORS["border"], linewidth=0.07, zorder=4)

        plot_violin_box_inset(
            ax,
            merged[ratio_col],
            inset_color,
            norm,
            vmax,
            show_xlabel=row == 2,
        )

        ax.text(
            0.5,
            1.018,
            scenario.title,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=6,
            fontweight="bold",
            color=COLORS["text"],
            clip_on=False,
        )
        ax.text(
            -0.055,
            1.045,
            panel_letters[i],
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            fontweight="bold",
            color=COLORS["text"],
            clip_on=False,
        )
        ax.set_xlim(bounds[0] - pad_x, bounds[2] + pad_x)
        ax.set_ylim(bounds[1] - pad_y, bounds[3] + pad_y)
        ax.set_axis_off()

    return axes


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.055,
        1.045,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        fontweight="bold",
        color=COLORS["text"],
        clip_on=False,
    )


def plot_stats(ax: plt.Axes, global_df: pd.DataFrame, panel_label: str) -> None:
    x = np.arange(len(global_df))
    utility = global_df["utility"].to_numpy(dtype=float)
    distributed = global_df["distributed"].to_numpy(dtype=float)
    total = global_df["total"].to_numpy(dtype=float)
    ratio = global_df["ratio"].to_numpy(dtype=float)

    ax.bar(x, utility, width=0.68, color=COLORS["utility"], label="Utility-scale PV", zorder=3)
    ax.bar(x, distributed, width=0.68, bottom=utility, color=COLORS["distributed"], label="Distributed PV", zorder=3)

    existing_capacity = float(np.nanmedian(global_df["actual_total"]))
    irena_2030, irena_2050 = 5500.0, 14400.0
    iea_nze_2030, iea_nze_2050 = 6300.0, 20000.0
    target_lines = [
        (existing_capacity, COLORS["target_red"], (0, (2, 1.4))),
        (irena_2030, COLORS["target_blue"], (0, (1.2, 1.2))),
        (irena_2050, COLORS["target_blue"], (0, (1.2, 1.2))),
        (iea_nze_2030, COLORS["target_orange"], (0, (3, 1.5))),
        (iea_nze_2050, COLORS["target_orange"], (0, (3, 1.5))),
    ]
    for value, color, linestyle in target_lines:
        ax.axhline(value, color=color, linestyle=linestyle, linewidth=0.75, zorder=1)

    for xi, value, r in zip(x, total, ratio, strict=True):
        if np.isfinite(value) and value > 1200:
            ax.text(
                xi,
                value * 0.52,
                f"{r:.2f}",
                ha="center",
                va="center",
                fontsize=6,
                color="white",
                fontweight="bold",
                rotation=90,
                zorder=4,
            )

    ax.set_xlim(-0.7, len(x) + 1.55)
    ax.set_ylim(0, 21000)
    ax.set_yticks([0, 5000, 10000, 15000, 20000])
    ax.set_ylabel("PV Capacity (GW)", labelpad=1)
    ax.set_xticks(x)
    ax.set_xticklabels(SCENARIO_LABELS, rotation=90, ha="center", va="top")
    ax.grid(axis="y", color="#E5E2DA", linewidth=0.35, zorder=0)
    ax.tick_params(length=2, pad=1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(0.5)

    for xpos in [3.5, 5.5]:
        ax.axvline(xpos, color="#CFCBC1", linewidth=0.55, linestyle=":", zorder=1)
    group_label_transform = ax.get_xaxis_transform()
    ax.text(1.5, 1.035, "Single\nmodel", transform=group_label_transform, ha="center", va="bottom", fontsize=6, linespacing=0.86, clip_on=False)
    ax.text(4.5, 1.035, "Regional\nleader", transform=group_label_transform, ha="center", va="bottom", fontsize=6, linespacing=0.86, clip_on=False)
    ax.text(7.0, 1.035, "Cluster\nleader", transform=group_label_transform, ha="center", va="bottom", fontsize=6, linespacing=0.86, clip_on=False)

    label_box = {"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.25}
    ax.text(len(x) - 0.20, existing_capacity, "Existing\ncapacity", color=COLORS["target_red"], fontsize=6, va="center", ha="left", linespacing=0.9, bbox=label_box, clip_on=False)

    irena_x = len(x) - 0.28
    ax.text(irena_x - 0.08, irena_2030, "2030", color=COLORS["target_blue"], fontsize=6, va="center", ha="right", clip_on=False)
    ax.text(irena_x - 0.08, irena_2050, "2050", color=COLORS["target_blue"], fontsize=6, va="center", ha="right", clip_on=False)
    ax.annotate(
        "",
        xy=(irena_x, irena_2050),
        xytext=(irena_x, irena_2030),
        xycoords="data",
        textcoords="data",
        arrowprops={"arrowstyle": "<->", "color": COLORS["target_blue"], "linewidth": 0.65, "shrinkA": 0, "shrinkB": 0},
        annotation_clip=True,
    )
    ax.text(
        irena_x + 0.1,
        (irena_2030 + irena_2050) / 2,
        "IRENA 1.5°C",
        rotation=90,
        color=COLORS["target_blue"],
        fontsize=6,
        ha="center",
        va="center",
        bbox=label_box,
        clip_on=False,
    )
    iea_x = len(x) + 1.08
    ax.text(iea_x - 0.08, iea_nze_2030, "2030", color=COLORS["target_orange"], fontsize=6, va="center", ha="right", clip_on=False)
    ax.text(iea_x - 0.08, iea_nze_2050, "2050", color=COLORS["target_orange"], fontsize=6, va="center", ha="right", clip_on=False)
    ax.annotate(
        "",
        xy=(iea_x, iea_nze_2050),
        xytext=(iea_x, iea_nze_2030),
        xycoords="data",
        textcoords="data",
        arrowprops={"arrowstyle": "<->", "color": COLORS["target_orange"], "linewidth": 0.65, "shrinkA": 0, "shrinkB": 0},
        annotation_clip=False,
    )
    ax.text(
        iea_x + 0.20,
        (iea_nze_2030 + iea_nze_2050) / 2,
        "IEA Net Zero by 2050",
        rotation=90,
        color=COLORS["target_orange"],
        fontsize=6,
        ha="center",
        va="center",
        bbox=label_box,
        clip_on=False,
    )

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        ncol=1,
        frameon=False,
        handlelength=1.2,
        handletextpad=0.35,
        borderpad=0,
        labelspacing=0.2,
    )
    if panel_label:
        add_panel_label(ax, panel_label)


def make_change_stats(country_df: pd.DataFrame) -> list[dict[str, float | list[float]]]:
    stats = []
    for scenario in SCENARIOS:
        cap_col, _ = scenario_columns(scenario)
        diff = pd.to_numeric(country_df[cap_col], errors="coerce") - pd.to_numeric(country_df["actual_capacity"], errors="coerce")
        diff = diff.replace([np.inf, -np.inf], np.nan).dropna()
        stats.append(
            {
                "med": float(np.nanmedian(diff)),
                "q1": float(np.nanpercentile(diff, 33)),
                "q3": float(np.nanpercentile(diff, 67)),
                "whislo": float(np.nanpercentile(diff, 10)),
                "whishi": float(np.nanpercentile(diff, 90)),
                "mean": float(np.nanmean(diff)),
                "fliers": [],
            }
        )
    return stats


def plot_change_boxplot(
    ax: plt.Axes,
    country_df: pd.DataFrame,
    kind: str,
    panel_label: str,
    right_space: float = 0.65,
    show_mean_legend: bool = False,
) -> None:
    stats = make_change_stats(country_df)
    positions = np.arange(1, len(SCENARIOS) + 1)
    color = COLORS["utility"] if kind == "utility" else COLORS["distributed"]
    light = COLORS["utility_light"] if kind == "utility" else COLORS["distributed_light"]

    ax.axhline(0, color="#333333", linestyle=(0, (2.2, 1.4)), linewidth=0.65, zorder=1)
    bp = ax.bxp(
        stats,
        positions=positions,
        widths=0.54,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "#111111", "linewidth": 0.75},
        whiskerprops={"color": "#222222", "linewidth": 0.55},
        capprops={"color": "#222222", "linewidth": 0.55},
        boxprops={"edgecolor": "#222222", "linewidth": 0.55},
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(light)
        patch.set_alpha(0.9)

    means = [item["mean"] for item in stats]
    ax.scatter(
        positions,
        means,
        marker="o",
        s=8,
        facecolor="white",
        edgecolor=color,
        linewidth=0.65,
        zorder=5,
    )

    ax.set_yscale("symlog", linthresh=0.1)
    ax.set_xlim(0.35, len(SCENARIOS) + right_space)
    ax.set_xticks(positions)
    ax.set_xticklabels(SCENARIO_LABELS, rotation=90, ha="center", va="top")
    ax.set_ylabel("PV Capacity Change (GW)", labelpad=1)
    ax.grid(axis="y", color="#E5E2DA", linewidth=0.35, zorder=0)
    ax.tick_params(length=2, pad=1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(0.5)
    if panel_label:
        add_panel_label(ax, panel_label)
    ax.text(
        0.12,
        1.035,
        "Utility-scale PV" if kind == "utility" else "Distributed PV",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6,
        fontweight="bold",
        clip_on=False,
    )

    if show_mean_legend:
        ax.scatter([], [], marker="o", s=8, facecolor="white", edgecolor=color, linewidth=0.65, label="Mean")
        ax.legend(loc="upper right", frameon=False, handlelength=0.8, borderpad=0, labelspacing=0.2)


def add_boxplot_explainer_to_axis(ax: plt.Axes) -> None:
    bg = mpl.patches.Rectangle(
        (0.748, 0.10),
        0.245,
        0.78,
        transform=ax.transAxes,
        facecolor="white",
        edgecolor="none",
        alpha=0.86,
        zorder=20,
        clip_on=False,
    )
    ax.add_patch(bg)

    x = 0.802
    half_width = 0.025
    y10, y33, y50, y67, y90, ymean = 0.18, 0.35, 0.50, 0.67, 0.84, 0.58
    line = {"transform": ax.transAxes, "color": "#111111", "linewidth": 0.55, "zorder": 21, "clip_on": False}
    ax.plot([x, x], [y10, y33], **line)
    ax.plot([x, x], [y67, y90], **line)
    ax.plot([x - half_width, x + half_width], [y10, y10], **line)
    ax.plot([x - half_width, x + half_width], [y90, y90], **line)
    rect = mpl.patches.Rectangle(
        (x - half_width, y33),
        half_width * 2,
        y67 - y33,
        transform=ax.transAxes,
        facecolor="white",
        edgecolor="#111111",
        linewidth=0.55,
        zorder=22,
        clip_on=False,
    )
    ax.add_patch(rect)
    median_line = line | {"zorder": 24}
    ax.plot([x - half_width, x + half_width], [y50, y50], **median_line)
    ax.scatter([x], [ymean], transform=ax.transAxes, marker="o", s=10, facecolor="white", edgecolor="#111111", linewidth=0.55, zorder=23, clip_on=False)

    text_kw = {"transform": ax.transAxes, "fontsize": 6, "va": "center", "zorder": 24, "clip_on": False}
    text_x = 0.870
    ax.text(text_x, y90, "90th", ha="left", **text_kw)
    ax.text(text_x, y67, "67th", ha="left", **text_kw)
    ax.text(text_x, ymean, "Mean", ha="left", **text_kw)
    ax.text(text_x, y50, "Median", ha="left", **text_kw)
    ax.text(text_x, y33, "33rd", ha="left", **text_kw)
    ax.text(text_x, y10, "10th", ha="left", **text_kw)


def build_figure() -> None:
    set_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Fig3 data...")
    world = load_world()
    utility_df = load_country_data("utility")
    distributed_df = load_country_data("distributed")
    total_df = make_total_country_data(utility_df, distributed_df)
    global_df = load_global_data()

    fig = plt.figure(figsize=(FIG_WIDTH_MM * MM, FIG_HEIGHT_MM * MM), constrained_layout=False)

    # Use explicit figure coordinates so the visual frames align exactly.
    map_x0, map_x1 = 0.032, 0.970
    map_y0, map_y1 = 0.365, 0.975
    chart_y0, chart_h = 0.105, 0.178
    stat_bounds = [0.060, chart_y0, 0.280, chart_h]
    utility_box_bounds = [0.430, chart_y0, 0.185, chart_h]
    distributed_box_bounds = [0.700, chart_y0, 0.255, chart_h]

    map_spec = fig.add_gridspec(
        1,
        1,
        left=map_x0,
        right=map_x1,
        bottom=map_y0,
        top=map_y1,
    )[0, 0]

    print("Drawing total PV map grid...")
    plot_map_grid(fig, map_spec, world, total_df, "total", "", "Total PV")

    ax_stats = fig.add_axes(stat_bounds)
    ax_box_utility = fig.add_axes(utility_box_bounds)
    ax_box_distributed = fig.add_axes(distributed_box_bounds)

    plot_stats(ax_stats, global_df, "")
    plot_change_boxplot(ax_box_utility, utility_df, "utility", "")
    plot_change_boxplot(ax_box_distributed, distributed_df, "distributed", "", right_space=4.4)
    add_boxplot_explainer_to_axis(ax_box_distributed)

    panel_label_y = chart_y0 + chart_h + 0.030
    fig.text(map_x0 - 0.028, panel_label_y, "j", ha="left", va="top", fontsize=8, fontweight="bold", color=COLORS["text"])
    fig.text(utility_box_bounds[0] - 0.030, panel_label_y, "k", ha="left", va="top", fontsize=8, fontweight="bold", color=COLORS["text"])
    fig.text(distributed_box_bounds[0] - 0.030, panel_label_y, "l", ha="left", va="top", fontsize=8, fontweight="bold", color=COLORS["text"])

    png_path = OUT_DIR / "Fig3_composite_v3.png"
    fig.savefig(png_path, dpi=600)
    pdf_path = OUT_DIR / "Fig3_composite_v3.pdf"
    fig.savefig(pdf_path, dpi=600)
    plt.close(fig)
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    build_figure()
