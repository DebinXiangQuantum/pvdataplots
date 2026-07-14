from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
from string import ascii_lowercase

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyogrio
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.ticker import FuncFormatter, MaxNLocator
from shapely.geometry import box


ROOT = Path(__file__).resolve().parents[2]
WORLD_SHP = ROOT / "data" / "国家边界矢量" / "World_countries.shp"
SOLAR_SHP = ROOT / "data" / "10km" / "Solar_10km.shp"
POINTS_CSV = ROOT / "Fig1" / "data" / "solar_points_cache.csv"
FIGURE_DIR = ROOT / "extenddatafigure" / "figures"
PDF_DIR = FIGURE_DIR / "extendedpdfs"

FIGURE_BASENAME = "extendedFig-key-region-pv-zoom-maps"
CAPACITY_FIELDS = ("utility", "distributed")

COLORS = {
    "land": "#f8f8f6",
    "target": "#eeeeea",
    "border": "#c9c9c6",
    "target_border": "#4f4f4d",
    "grid": "#ddddda",
    "text": "#000000",
}

PV_CMAP = LinearSegmentedColormap.from_list(
    "pv_capacity",
    ["#d9f1fb", "#79bdd6", "#f1d46a", "#e86f43", "#8c1aa3"],
)
PV_CMAP.set_bad((1, 1, 1, 0))


@dataclass(frozen=True)
class RegionSpec:
    name: str
    country_codes: tuple[str, ...]
    extent: tuple[float, float, float, float]


EUROPE_CODES = (
    "ALB",
    "AND",
    "AUT",
    "BEL",
    "BGR",
    "BIH",
    "BLR",
    "CHE",
    "CYP",
    "CZE",
    "DEU",
    "DNK",
    "ESP",
    "EST",
    "FIN",
    "FRA",
    "GBR",
    "GRC",
    "HRV",
    "HUN",
    "IRL",
    "ISL",
    "ITA",
    "LIE",
    "LTU",
    "LUX",
    "LVA",
    "MCO",
    "MDA",
    "MKD",
    "MLT",
    "MNE",
    "NLD",
    "NOR",
    "POL",
    "PRT",
    "ROU",
    "RUS",
    "SMR",
    "SRB",
    "SVK",
    "SVN",
    "SWE",
    "TUR",
    "UKR",
    "VAT",
    # Non-standard codes used by the manuscript's boundary shapefile.
    "SEB",
    "YUG",
)

REGIONS = (
    RegionSpec("China", ("CHN",), (73.0, 18.0, 135.0, 54.0)),
    RegionSpec("United States", ("USA",), (-126.0, 24.0, -66.0, 50.0)),
    RegionSpec("India", ("IND",), (67.0, 6.0, 98.0, 37.0)),
    RegionSpec("Europe", EUROPE_CODES, (-12.0, 34.0, 45.0, 72.0)),
    RegionSpec("Australia", ("AUS",), (112.0, -45.0, 155.0, -10.0)),
    RegionSpec("Japan", ("JPN",), (128.0, 29.0, 146.0, 46.0)),
    RegionSpec("Mexico", ("MEX",), (-119.0, 14.0, -86.0, 33.0)),
    RegionSpec("Chile", ("CHL",), (-77.0, -56.0, -66.0, -17.0)),
)

KIND_LABELS = {
    "utility": "Utility-scale PV",
    "distributed": "Distributed PV",
}


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
            "font.size": 6,
            "text.color": "#000000",
            "axes.labelsize": 6,
            "axes.labelcolor": "#000000",
            "axes.titlesize": 6,
            "axes.titlecolor": "#000000",
            "axes.titleweight": "normal",
            "axes.linewidth": 0.45,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "xtick.color": "#000000",
            "ytick.color": "#000000",
            "xtick.major.width": 0.4,
            "ytick.major.width": 0.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def load_data() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    world = gpd.read_file(WORLD_SHP, columns=["NAME", "SOC", "geometry"])
    world = world.dropna(subset=["geometry"]).copy()
    world["SOC"] = world["SOC"].astype(str).str.strip().str.upper()

    points = pd.read_csv(POINTS_CSV)
    required = {"lon", "lat", *CAPACITY_FIELDS}
    missing = required.difference(points.columns)
    if missing:
        raise ValueError(f"Missing columns in {POINTS_CSV}: {sorted(missing)}")

    # The Fig. 1 cache intentionally stores only coordinates and capacities.
    # Recover the original country assignment from the source DBF without
    # loading 1.5 million grid geometries. The positive-cell filter preserves
    # source order exactly, matching load_solar_points() in Fig1/draw_Fig1.py.
    attributes = pyogrio.read_dataframe(
        SOLAR_SHP,
        columns=["NAME", "jizhong_ar", "fenbu_area"],
        read_geometry=False,
    )
    attributes.columns = [column.lower() for column in attributes.columns]
    attributes = attributes[
        (attributes["jizhong_ar"] > 0) | (attributes["fenbu_area"] > 0)
    ].reset_index(drop=True)
    if len(attributes) != len(points):
        raise ValueError(
            "The Solar_10km attribute rows do not align with solar_points_cache.csv: "
            f"{len(attributes)} != {len(points)}"
        )
    expected_utility = attributes["jizhong_ar"].to_numpy(dtype=float) * 0.2 / 1e6
    expected_distributed = attributes["fenbu_area"].to_numpy(dtype=float) * 0.2 / 1e6
    if not (
        np.allclose(expected_utility, points["utility"].to_numpy(dtype=float))
        and np.allclose(expected_distributed, points["distributed"].to_numpy(dtype=float))
    ):
        raise ValueError("Solar_10km country attributes are not aligned with the Fig. 1 cache.")
    points["country_name"] = attributes["name"].astype(str).str.strip().str.upper()

    points_gdf = gpd.GeoDataFrame(
        points,
        geometry=gpd.points_from_xy(points["lon"], points["lat"]),
        crs="EPSG:4326",
    )
    return world, points_gdf


def main_figure_norms(points: gpd.GeoDataFrame) -> dict[str, LogNorm]:
    """Reproduce the global limits used by Fig. 1a/e."""
    norms: dict[str, LogNorm] = {}
    for kind in CAPACITY_FIELDS:
        values = points.loc[points[kind] > 0, kind].to_numpy(dtype=float)
        vmin = max(np.nanpercentile(values, 5), np.nanmin(values))
        vmax = np.nanpercentile(values, 99.5)
        norms[kind] = LogNorm(vmin=float(vmin), vmax=float(vmax))
    return norms


def target_rows(world: gpd.GeoDataFrame, spec: RegionSpec) -> gpd.GeoDataFrame:
    rows = world[world["SOC"].isin(spec.country_codes)].copy()
    if rows.empty:
        raise ValueError(f"No country boundaries found for {spec.name}: {spec.country_codes}")
    return rows


def region_masks(
    world: gpd.GeoDataFrame,
    points: gpd.GeoDataFrame,
) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    for spec in REGIONS:
        source_names = set(
            target_rows(world, spec)["NAME"].dropna().astype(str).str.strip().str.upper()
        )
        masks[spec.name] = points["country_name"].isin(source_names).to_numpy()
    return masks


def capacity_points(
    points: gpd.GeoDataFrame,
    mask: np.ndarray,
    kind: str,
    extent: tuple[float, float, float, float],
) -> pd.DataFrame:
    xmin, ymin, xmax, ymax = extent
    selected = points.loc[
        mask
        & (points["lon"] >= xmin - 0.15)
        & (points["lon"] <= xmax + 0.15)
        & (points["lat"] >= ymin - 0.15)
        & (points["lat"] <= ymax + 0.15)
        & (points[kind] > 0),
        ["lon", "lat", kind],
    ].copy()
    return selected.sort_values(kind)


def format_longitude(value: float, _position: int) -> str:
    rounded = int(round(value))
    if rounded == 0:
        return "0°"
    return f"{abs(rounded)}°{'W' if rounded < 0 else 'E'}"


def format_latitude(value: float, _position: int) -> str:
    rounded = int(round(value))
    if rounded == 0:
        return "0°"
    return f"{abs(rounded)}°{'S' if rounded < 0 else 'N'}"


def padded_map_extent(
    extent: tuple[float, float, float, float],
    box_aspect: float = 0.86,
) -> tuple[tuple[float, float, float, float], float]:
    """Pad a lon/lat extent to a common frame without distorting geography."""
    xmin, ymin, xmax, ymax = extent
    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2
    width = xmax - xmin
    height = ymax - ymin
    geographic_aspect = 1 / max(np.cos(np.deg2rad(center_y)), 0.25)
    current_box_aspect = geographic_aspect * height / width

    if current_box_aspect < box_aspect:
        height = box_aspect * width / geographic_aspect
    else:
        width = geographic_aspect * height / box_aspect

    padded = (
        center_x - width / 2,
        center_y - height / 2,
        center_x + width / 2,
        center_y + height / 2,
    )
    return padded, geographic_aspect


def plot_region_panel(
    ax: plt.Axes,
    world: gpd.GeoDataFrame,
    points: gpd.GeoDataFrame,
    mask: np.ndarray,
    spec: RegionSpec,
    kind: str,
    norm: LogNorm,
    panel_label: str,
) -> None:
    target = target_rows(world, spec)
    view_extent, geographic_aspect = padded_map_extent(spec.extent)
    view_box = box(*view_extent)

    local_world = world[world.geometry.intersects(view_box)].copy()
    local_world = gpd.clip(local_world, view_box, keep_geom_type=True)
    local_world["geometry"] = local_world.geometry.simplify(0.025, preserve_topology=True)

    local_target = target[target.geometry.intersects(view_box)].copy()
    local_target = gpd.clip(local_target, view_box, keep_geom_type=True)
    local_target["geometry"] = local_target.geometry.simplify(0.01, preserve_topology=True)

    local_world.plot(
        ax=ax,
        facecolor=COLORS["land"],
        edgecolor=COLORS["border"],
        linewidth=0.12,
        zorder=1,
    )
    local_target.plot(
        ax=ax,
        facecolor=COLORS["target"],
        edgecolor="none",
        zorder=2,
    )

    selected = capacity_points(points, mask, kind, spec.extent)
    if not selected.empty:
        ax.scatter(
            selected["lon"],
            selected["lat"],
            c=selected[kind],
            cmap=PV_CMAP,
            norm=norm,
            marker="s",
            s=0.72,
            linewidths=0,
            alpha=0.96,
            rasterized=True,
            zorder=3,
        )

    local_target.boundary.plot(
        ax=ax,
        color=COLORS["target_border"],
        linewidth=0.22,
        zorder=4,
    )

    xmin, ymin, xmax, ymax = view_extent
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect(geographic_aspect, adjustable="box")

    ax.xaxis.set_major_locator(MaxNLocator(nbins=3, integer=True, min_n_ticks=2))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True, min_n_ticks=2))
    ax.xaxis.set_major_formatter(FuncFormatter(format_longitude))
    ax.yaxis.set_major_formatter(FuncFormatter(format_latitude))
    ax.tick_params(length=1.6, pad=1.0, color="#666666", labelcolor="#000000", labelsize=6)
    ax.grid(color=COLORS["grid"], linewidth=0.20, linestyle=(0, (1.5, 2.0)), zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#747472")
        spine.set_linewidth(0.3)

    ax.set_title(
        f"{spec.name}\n{KIND_LABELS[kind]}",
        fontsize=6,
        color=COLORS["text"],
        pad=2.2,
        linespacing=1.12,
    )
    ax.text(
        0.018,
        0.982,
        panel_label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        fontweight="bold",
        color=COLORS["text"],
        zorder=8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.7},
    )


def add_shared_colorbar(
    fig: plt.Figure,
    position: tuple[float, float, float, float],
    kind: str,
    norm: LogNorm,
) -> None:
    cax = fig.add_axes(position)
    scalar = mpl.cm.ScalarMappable(norm=norm, cmap=PV_CMAP)
    scalar.set_array([])
    colorbar = fig.colorbar(scalar, cax=cax, orientation="horizontal")
    colorbar.outline.set_linewidth(0.3)
    colorbar.ax.xaxis.set_major_locator(mpl.ticker.LogLocator(base=10, numticks=5))
    colorbar.ax.xaxis.set_minor_locator(
        mpl.ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=100)
    )
    colorbar.ax.xaxis.set_major_formatter(mpl.ticker.LogFormatterMathtext(base=10))
    colorbar.ax.tick_params(
        which="major",
        length=1.4,
        width=0.3,
        pad=1,
        labelsize=6,
        labelcolor="#000000",
    )
    colorbar.ax.tick_params(which="minor", length=0.8, width=0.25)
    colorbar.set_label(
        f"{KIND_LABELS[kind]} capacity (GW per 10-km grid cell)",
        fontsize=6,
        color="#000000",
        labelpad=1.2,
    )


def print_region_summary(
    points: gpd.GeoDataFrame,
    masks: dict[str, np.ndarray],
) -> None:
    print("Region summary from displayed grid-cell capacity data:")
    for spec in REGIONS:
        xmin, ymin, xmax, ymax = spec.extent
        display_mask = (
            masks[spec.name]
            & (points["lon"] >= xmin)
            & (points["lon"] <= xmax)
            & (points["lat"] >= ymin)
            & (points["lat"] <= ymax)
        )
        utility = points.loc[display_mask, "utility"]
        distributed = points.loc[display_mask, "distributed"]
        print(
            f"  {spec.name:14s} "
            f"utility={utility.sum():8.2f} GW ({(utility > 0).sum():5d} cells), "
            f"distributed={distributed.sum():7.2f} GW ({(distributed > 0).sum():5d} cells)"
        )


def draw_figure() -> tuple[Path, Path, Path]:
    set_style()
    world, points = load_data()
    norms = main_figure_norms(points)
    masks = region_masks(world, points)

    width_mm = 180
    height_mm = 205
    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4), constrained_layout=False)
    # Matplotlib 3.10 rounds figsize to 0.01 inch during construction. Reset
    # the physical canvas bounds so the exported PDF is exactly 180 mm wide.
    fig.bbox_inches.set_points(
        np.array([[0.0, 0.0], [width_mm / 25.4, height_mm / 25.4]], dtype=float)
    )
    grid = fig.add_gridspec(
        4,
        4,
        left=0.052,
        right=0.985,
        bottom=0.125,
        top=0.965,
        wspace=0.22,
        hspace=0.22,
    )

    panel_index = 0
    for region_index, spec in enumerate(REGIONS):
        row = region_index // 2
        pair = region_index % 2
        for kind_index, kind in enumerate(CAPACITY_FIELDS):
            col = pair * 2 + kind_index
            ax = fig.add_subplot(grid[row, col])
            plot_region_panel(
                ax,
                world,
                points,
                masks[spec.name],
                spec,
                kind,
                norms[kind],
                ascii_lowercase[panel_index],
            )
            panel_index += 1

    add_shared_colorbar(fig, (0.075, 0.080, 0.385, 0.009), "utility", norms["utility"])
    add_shared_colorbar(fig, (0.545, 0.080, 0.385, 0.009), "distributed", norms["distributed"])

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = FIGURE_DIR / f"{FIGURE_BASENAME}.pdf"
    out_pdf_archive = PDF_DIR / f"{FIGURE_BASENAME}.pdf"
    out_png = FIGURE_DIR / f"{FIGURE_BASENAME}.png"
    fig.savefig(out_pdf, dpi=300)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    shutil.copy2(out_pdf, out_pdf_archive)

    print_region_summary(points, masks)
    print(f"Saved {out_pdf}")
    print(f"Saved {out_pdf_archive}")
    print(f"Saved {out_png}")
    return out_pdf, out_pdf_archive, out_png


if __name__ == "__main__":
    draw_figure()
