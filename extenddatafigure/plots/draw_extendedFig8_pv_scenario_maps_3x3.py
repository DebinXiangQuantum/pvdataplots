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
from shapely.geometry import LineString, box

from extended_data_common import FIGURES_DIR, ROOT, mm, set_style


DATA_XLSX = ROOT / "Fig3" / "全球总装机 0518.xlsx"
WORLD_SHP = ROOT / "data" / "map" / "世界国家地图.shp"

FIG_WIDTH_MM = 180
FIG_HEIGHT_MM = 138

SCENARIO_VMAX = 1000.0

COLORS = {
    "land": "#F6F5F0",
    "border": "#C9C7BE",
    "grid": "#D7D5CE",
    "text": "#202020",
    "decrease": "#2F6F9F",
    "neutral": "#F7F4EA",
    "increase": "#D85F8D",
}

CHANGE_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "extended_fig8_signed_relative_change",
    [COLORS["decrease"], COLORS["neutral"], COLORS["increase"]],
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

PV_KIND_CONFIG = {
    "utility": {
        "sheet": "集中式",
        "label": "Utility-scale PV",
        "basename": "extendedFig8-utility-scenario-maps",
    },
    "distributed": {
        "sheet": "分布式",
        "label": "Distributed PV",
        "basename": "extendedFig8-distributed-scenario-maps",
    },
}

COUNTRY_ALIASES = {
    "UNITED STATES OF AMERICA": "UNITED STATES",
    "RUSSIA": "RUSSIAN FEDERATION",
    "VIETNAM": "VIET NAM",
    "SOUTH KOREA": "KOREA, REPUBLIC OF",
    "BRUNEI DARUSSALAM": "BRUNEI",
    "LAO PEOPLE'S DEMOCRATIC REPUBLIC": "LAOS",
    "IRAN (ISLAMIC REPUBLIC OF)": "IRAN",
    "SYRIA": "SYRIAN ARAB REPUBLIC",
    "CZECHIA": "CZECH REPUBLIC",
}


def country_key(value: object) -> str:
    key = " ".join(str(value).strip().upper().split())
    return COUNTRY_ALIASES.get(key, key)


def signed_log_transform(values: np.ndarray | pd.Series | float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.sign(arr) * np.log10(1 + np.abs(arr))


def ratio_norm() -> mcolors.TwoSlopeNorm:
    return mcolors.TwoSlopeNorm(
        vmin=float(signed_log_transform(-1.0)),
        vcenter=0.0,
        vmax=float(signed_log_transform(SCENARIO_VMAX)),
    )


def format_ratio_tick(value: float) -> str:
    if value == -1:
        return "-100%"
    if value == 0:
        return "0"
    if value == 1:
        return "+100%"
    return f"+{value:g}x"


def load_world() -> gpd.GeoDataFrame:
    world = gpd.read_file(WORLD_SHP)
    world["country_key"] = world["NAME"].map(country_key)
    world = world[world["country_key"] != "ANTARCTICA"].copy()
    world = world.clip(box(-179.9, -60, 179.9, 85))
    return world.to_crs("ESRI:54030")


def load_graticules() -> gpd.GeoDataFrame:
    lines: list[LineString] = []
    for lon in [-120, -60, 0, 60, 120]:
        lines.append(LineString([(lon, lat) for lat in np.linspace(-60, 80, 160)]))
    for lat in [-30, 0, 30, 60]:
        lines.append(LineString([(lon, lat) for lon in np.linspace(-179.9, 179.9, 320)]))
    return gpd.GeoDataFrame(geometry=lines, crs="EPSG:4326").to_crs("ESRI:54030")


def load_country_capacity(kind: str) -> pd.DataFrame:
    if kind not in PV_KIND_CONFIG:
        raise ValueError("kind must be 'utility' or 'distributed'.")

    config = PV_KIND_CONFIG[kind]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        raw = pd.read_excel(DATA_XLSX, sheet_name=config["sheet"], header=None).iloc[3:].copy()
    raw = raw[raw[0].notna()].copy()
    raw["country_key"] = raw[0].map(country_key)
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
        capacity_col = f"{scenario.name}_capacity"
        change_col = f"{scenario.name}_change"
        relative_col = f"{scenario.name}_relative_change"
        df[capacity_col] = pd.to_numeric(raw[source_col], errors="coerce")
        df[change_col] = df[capacity_col] - df["actual_capacity"]
        df[relative_col] = df[change_col] / actual
        clipped = df[relative_col].clip(lower=-1.0, upper=SCENARIO_VMAX)
        df[f"{scenario.name}_plot_value"] = signed_log_transform(clipped)

    return df.drop_duplicates("country_key").reset_index(drop=True)


def add_panel_label(fig: plt.Figure, bounds: list[float], label: str, title: str) -> None:
    top = bounds[1] + bounds[3]
    fig.text(
        bounds[0] - 0.010,
        top + 0.020,
        label,
        ha="left",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        color=COLORS["text"],
    )
    fig.text(
        bounds[0] + bounds[2] / 2,
        top + 0.019,
        title,
        ha="center",
        va="bottom",
        fontsize=6,
        fontweight="bold",
        color=COLORS["text"],
    )


def plot_map_panel(
    ax: plt.Axes,
    world: gpd.GeoDataFrame,
    graticules: gpd.GeoDataFrame,
    country_df: pd.DataFrame,
    scenario: Scenario,
    norm: mcolors.Normalize,
) -> None:
    plot_col = f"{scenario.name}_plot_value"
    merged = world.merge(country_df[["country_key", plot_col]], on="country_key", how="left")
    bounds = world.total_bounds
    pad_x = (bounds[2] - bounds[0]) * 0.010
    pad_y = (bounds[3] - bounds[1]) * 0.035

    ax.set_facecolor("white")
    ax.set_rasterization_zorder(6)
    world.plot(ax=ax, facecolor=COLORS["land"], edgecolor="none", linewidth=0, zorder=1, rasterized=True)
    merged.dropna(subset=[plot_col]).plot(
        column=plot_col,
        ax=ax,
        cmap=CHANGE_CMAP,
        norm=norm,
        edgecolor="none",
        linewidth=0,
        zorder=2,
        rasterized=True,
    )
    graticules.plot(ax=ax, color=COLORS["grid"], linewidth=0.22, alpha=0.70, zorder=3, rasterized=True)
    world.plot(ax=ax, facecolor="none", edgecolor="#FFFFFF", linewidth=0.12, zorder=4, rasterized=True)
    world.plot(ax=ax, facecolor="none", edgecolor=COLORS["border"], linewidth=0.065, zorder=5, rasterized=True)

    ax.set_xlim(bounds[0] - pad_x, bounds[2] + pad_x)
    ax.set_ylim(bounds[1] - pad_y, bounds[3] + pad_y)
    ax.set_axis_off()


def add_colorbar(fig: plt.Figure, norm: mcolors.Normalize, label: str) -> None:
    cax = fig.add_axes([0.292, 0.914, 0.416, 0.012])
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=CHANGE_CMAP)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", extend="max")
    raw_ticks = [-1, 0, 1, 1000]
    cb.set_ticks([float(signed_log_transform(tick)) for tick in raw_ticks])
    cb.set_ticklabels([])
    cb.outline.set_linewidth(0.35)
    cb.ax.tick_params(length=1.35, pad=0.7, width=0.35, labelbottom=False)
    for tick in raw_ticks:
        rel_x = float(norm(float(signed_log_transform(tick))))
        text_x = rel_x
        ha = "center"
        if tick == 0:
            text_x -= 0.014
            ha = "right"
        elif tick == 1:
            text_x += 0.014
            ha = "left"
        elif tick == 1000:
            ha = "right"
        cax.text(
            text_x,
            -0.82,
            format_ratio_tick(tick),
            transform=cax.transAxes,
            ha=ha,
            va="top",
            fontsize=6,
            color=COLORS["text"],
            clip_on=False,
        )
    fig.text(
        cax.get_position().x0 + cax.get_position().width / 2,
        cax.get_position().y0 + 0.024,
        f"Relative {label} capacity change",
        ha="center",
        va="bottom",
        fontsize=6,
        color=COLORS["text"],
    )


def map_grid_bounds() -> list[list[float]]:
    left = 0.038
    col_gap = 0.024
    col_w = (0.982 - left - 2 * col_gap) / 3
    row_h = 0.172
    row_y = [0.675, 0.422, 0.169]
    return [[left + col * (col_w + col_gap), row_y[row], col_w, row_h] for row in range(3) for col in range(3)]


def draw_scenario_map_grid(kind: str, world: gpd.GeoDataFrame, graticules: gpd.GeoDataFrame) -> None:
    config = PV_KIND_CONFIG[kind]
    country_df = load_country_capacity(kind)
    norm = ratio_norm()

    fig = plt.figure(figsize=(mm(FIG_WIDTH_MM), mm(FIG_HEIGHT_MM)), constrained_layout=False)
    fig.text(
        0.038,
        0.952,
        f"{config['label']} scenario maps",
        ha="left",
        va="top",
        fontsize=7,
        fontweight="bold",
        color=COLORS["text"],
    )
    add_colorbar(fig, norm, config["label"])

    for index, (scenario, bounds) in enumerate(zip(SCENARIOS, map_grid_bounds(), strict=True)):
        ax = fig.add_axes(bounds)
        plot_map_panel(ax, world, graticules, country_df, scenario, norm)
        add_panel_label(fig, bounds, chr(97 + index), scenario.title)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIGURES_DIR / f"{config['basename']}.pdf"
    png_path = FIGURES_DIR / f"{config['basename']}.png"
    fig.savefig(pdf_path, dpi=300)
    fig.savefig(png_path, dpi=600)
    plt.close(fig)
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


def build_figures() -> None:
    set_style(6)
    mpl.rcParams["pdf.compression"] = 9
    world = load_world()
    graticules = load_graticules()
    draw_scenario_map_grid("utility", world, graticules)
    draw_scenario_map_grid("distributed", world, graticules)


if __name__ == "__main__":
    build_figures()
