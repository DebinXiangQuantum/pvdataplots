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
from shapely.geometry import box

from extended_data_common import COLORS, FIGURES_DIR, ROOT, display_country, mm, set_style


AMBITION_XLSX = ROOT / "Fig3" / "全球雄心排名.xlsx"
GDP_XLSX = ROOT / "Fig3" / "GDP&Irradiance&capacity.xlsx"
WORLD_SHP = ROOT / "data" / "map" / "世界国家地图.shp"

AMBITION_COL = "stage1_true_minus_baseline_pct_of_baseline"
FULL_RANKING_SHEET = "installation_ambition_country_r"
GDP_COL = "2023 GDP (current US$)"

MAP_COLORS = {
    "land": "#F6F5F0",
    "border": "#C9C7BE",
    "ocean": "white",
    "missing": "#ECE9DF",
    "under": "#2F6F9F",
    "neutral": "#F7F4EA",
    "over": "#D85F8D",
}

AMBITION_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "extended_ambition_relative_pct",
    [MAP_COLORS["under"], MAP_COLORS["neutral"], MAP_COLORS["over"]],
)
AMBITION_NORM = mcolors.TwoSlopeNorm(vmin=-100, vcenter=0, vmax=100)


@dataclass(frozen=True)
class AmbitionTask:
    task: str
    sheet: str
    label: str
    color: str


TASKS = {
    "Centralized": AmbitionTask("Centralized", "Centralized", "Utility-scale PV", COLORS["utility"]),
    "Distributed": AmbitionTask("Distributed", "Distributed", "Distributed PV", COLORS["distributed"]),
}


def country_key(value: object) -> str:
    return " ".join(str(value).strip().upper().split())


def read_ambition_sheet(sheet_name: str) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        df = pd.read_excel(AMBITION_XLSX, sheet_name=sheet_name)
    df.columns = [str(col).strip() for col in df.columns]
    missing = [col for col in ["task", "region", "region_type", AMBITION_COL] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {AMBITION_XLSX.name}/{sheet_name}: {missing}")
    df["country_key"] = df["region"].map(country_key)
    df["ambition_pct"] = pd.to_numeric(df[AMBITION_COL], errors="coerce")
    return df


def load_country_ambition(task_name: str, sheet_name: str | None = None) -> pd.DataFrame:
    if task_name not in TASKS:
        raise ValueError("task_name must be 'Centralized' or 'Distributed'.")
    sheet = sheet_name or FULL_RANKING_SHEET
    df = read_ambition_sheet(sheet)
    df = df[(df["task"] == task_name) & (df["region_type"] == "COUNTRY")].copy()
    df = df.dropna(subset=["country_key", "ambition_pct"])
    df["ambition_pct_clipped"] = df["ambition_pct"].clip(-100, 100)
    return df[["region", "country_key", "ambition_pct", "ambition_pct_clipped"]].drop_duplicates("country_key")


def load_world() -> gpd.GeoDataFrame:
    world = gpd.read_file(WORLD_SHP)
    world["country_key"] = world["NAME"].map(country_key)
    world = world[world["country_key"] != "ANTARCTICA"].copy()
    world = world.clip(box(-179.9, -60, 179.9, 85))
    return world.to_crs("ESRI:54030")


def load_gdp_top20_ambition(task_name: str) -> pd.DataFrame:
    if task_name not in TASKS:
        raise ValueError("task_name must be 'Centralized' or 'Distributed'.")
    task = TASKS[task_name]

    gdp = pd.read_excel(GDP_XLSX)
    gdp.columns = [str(col).strip() for col in gdp.columns]
    if "地区" not in gdp.columns or GDP_COL not in gdp.columns:
        raise ValueError(f"Missing required GDP columns in {GDP_XLSX.name}.")
    gdp = gdp.dropna(subset=["地区"]).copy()
    gdp["country_key"] = gdp["地区"].map(country_key)
    gdp["gdp"] = pd.to_numeric(gdp[GDP_COL], errors="coerce")
    top20 = gdp.dropna(subset=["gdp"]).sort_values("gdp", ascending=False).head(20).copy()
    top20["gdp_rank"] = np.arange(1, len(top20) + 1)

    primary = load_country_ambition(task_name, task.sheet).rename(columns={"ambition_pct": "primary_ambition_pct"})
    full = load_country_ambition(task_name, FULL_RANKING_SHEET).rename(columns={"ambition_pct": "full_ambition_pct"})
    merged = top20[["country_key", "gdp", "gdp_rank"]].merge(
        primary[["country_key", "primary_ambition_pct"]],
        on="country_key",
        how="left",
    )
    merged = merged.merge(full[["country_key", "full_ambition_pct"]], on="country_key", how="left")
    merged["ambition_pct"] = merged["primary_ambition_pct"].combine_first(merged["full_ambition_pct"])
    missing = merged.loc[merged["ambition_pct"].isna(), "country_key"].tolist()
    if missing:
        raise ValueError(f"Missing {task_name} ambition values for GDP top-20 countries: {missing}")
    merged["display_country"] = merged["country_key"].map(display_country)
    return merged.sort_values("gdp_rank", ascending=True).reset_index(drop=True)


def save_figure(fig: plt.Figure, basename: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIGURES_DIR / f"{basename}.pdf"
    png_path = FIGURES_DIR / f"{basename}.png"
    fig.savefig(pdf_path, dpi=600, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


def draw_ambition_map(task_name: str, basename: str) -> None:
    task = TASKS[task_name]
    set_style(6)

    world = load_world()
    ambition = load_country_ambition(task_name)
    merged = world.merge(ambition, on="country_key", how="left")

    fig, ax = plt.subplots(figsize=(mm(178), mm(88)))
    ax.set_facecolor(MAP_COLORS["ocean"])
    world.plot(ax=ax, facecolor=MAP_COLORS["land"], edgecolor="none", linewidth=0, rasterized=True, zorder=1)
    merged.dropna(subset=["ambition_pct_clipped"]).plot(
        ax=ax,
        column="ambition_pct_clipped",
        cmap=AMBITION_CMAP,
        norm=AMBITION_NORM,
        edgecolor="none",
        linewidth=0,
        rasterized=True,
        zorder=2,
    )
    world.plot(ax=ax, facecolor="none", edgecolor="white", linewidth=0.13, rasterized=True, zorder=3)
    world.plot(ax=ax, facecolor="none", edgecolor=MAP_COLORS["border"], linewidth=0.07, zorder=4)

    bounds = world.total_bounds
    pad_x = (bounds[2] - bounds[0]) * 0.012
    pad_y = (bounds[3] - bounds[1]) * 0.035
    ax.set_xlim(bounds[0] - pad_x, bounds[2] + pad_x)
    ax.set_ylim(bounds[1] - pad_y, bounds[3] + pad_y)
    ax.set_axis_off()

    ax.text(
        0.0,
        1.012,
        f"{task.label} ambition by country",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7,
        fontweight="bold",
        color=COLORS["text"],
    )

    sm = mpl.cm.ScalarMappable(norm=AMBITION_NORM, cmap=AMBITION_CMAP)
    sm.set_array([])
    cax = fig.add_axes([0.295, 0.068, 0.410, 0.018])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", extend="both")
    cb.set_ticks([-100, -50, 0, 50, 100])
    cb.set_ticklabels(["<=-100", "-50", "0", "+50", ">=+100"])
    cb.set_label("Stage-1 ambition relative to baseline (%)", labelpad=1.8)
    cb.ax.xaxis.set_label_position("top")
    cb.outline.set_linewidth(0.35)
    cb.ax.tick_params(length=1.4, pad=1.0, width=0.35, labelsize=6)
    fig.subplots_adjust(left=0.015, right=0.985, top=0.935, bottom=0.110)
    save_figure(fig, basename)


def draw_gdp_top20_bar(task_name: str, basename: str) -> None:
    task = TASKS[task_name]
    set_style(6)
    df = load_gdp_top20_ambition(task_name)

    values = df["ambition_pct"].to_numpy(dtype=float)
    y = np.arange(len(df))
    colors = [task.color if value >= 0 else mcolors.to_rgba("#6B7F92", 0.95) for value in values]

    fig, ax = plt.subplots(figsize=(mm(178), mm(100)))
    ax.barh(y, values, height=0.62, color=colors, edgecolor="none", zorder=3)
    ax.axvline(0, color="#303030", linewidth=0.55, zorder=2)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.35, zorder=0)

    spread = max(float(np.nanmax(values) - np.nanmin(values)), 1.0)
    label_offset = max(spread * 0.015, 1.1)
    for yi, value in zip(y, values, strict=True):
        ha = "left" if value >= 0 else "right"
        x_text = value + label_offset if value >= 0 else value - label_offset
        ax.text(
            x_text,
            yi,
            f"{value:+.1f}",
            ha=ha,
            va="center",
            fontsize=5.8,
            color=COLORS["text"],
            clip_on=False,
        )

    labels = [f"{rank}. {country}" for rank, country in zip(df["gdp_rank"], df["display_country"], strict=True)]
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_title(f"GDP top 20 countries - {task.label}", loc="left", fontsize=7, fontweight="bold", pad=2)
    ax.set_xlabel("Stage-1 ambition relative to baseline (%)", labelpad=1)
    ax.set_ylabel("GDP rank and country", labelpad=1)

    min_val = float(np.nanmin(values))
    max_val = float(np.nanmax(values))
    x_min = min(0, min_val) - max(5.0, spread * 0.10)
    x_max = max(0, max_val) + max(5.0, spread * 0.16)
    ax.set_xlim(x_min, x_max)
    ax.tick_params(length=2, pad=1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(0.5)
    fig.subplots_adjust(left=0.225, right=0.965, top=0.925, bottom=0.135)
    save_figure(fig, basename)
