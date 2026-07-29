from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
FIG3_DIR = ROOT / "Fig3"
FIG5_DIR = ROOT / "Fig5"
FIGURES_DIR = FIG5_DIR / "figures"
COUNTRY_CONTINENT_CSV = FIG3_DIR / "country_continent_mapping.csv"

FIGURE_BASENAME = "Fig5_composite"
FIGURE_WIDTH_MM = 180
FIGURE_HEIGHT_MM = 140
PNG_DPI = 508
MM = 1 / 25.4


def import_script(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


fig3_base = import_script(
    FIG3_DIR / "plots" / "draw_fig3_composite.py",
    "figure5_fig3_base",
)

SCENARIOS = fig3_base.SCENARIOS
SCENARIO_LABELS = [scenario.title for scenario in SCENARIOS]
COLORS = fig3_base.COLORS
MAP_CMAP = fig3_base.CHANGE_CMAP
HEATMAP_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "figure5_signed_relative_change",
    ["#2F6F9F", "#F7F4EA", "#D85F8D"],
)


def load_country_codes() -> dict[str, str]:
    mapping = pd.read_csv(COUNTRY_CONTINENT_CSV)
    mapping["country_key"] = mapping["country"].astype(str).str.upper().str.strip()
    return mapping.drop_duplicates("country_key").set_index("country_key")[
        "country_code"
    ].to_dict()


COUNTRY_CODES = load_country_codes()


def set_style() -> None:
    fig3_base.set_style()
    mpl.rcParams.update(
        {
            "font.size": 6,
            "axes.labelsize": 6,
            "axes.titlesize": 6,
            "legend.fontsize": 6,
        }
    )


def add_panel_label(fig: plt.Figure, x: float, y: float, label: str) -> None:
    fig.text(
        x,
        y,
        label,
        ha="left",
        va="top",
        fontsize=8,
        fontweight="bold",
        color=COLORS["text"],
    )


def scenario_columns(scenario) -> tuple[str, str]:
    return f"{scenario.name}_capacity", f"{scenario.name}_relative_change"


def add_change_columns(country_df: pd.DataFrame) -> pd.DataFrame:
    out = country_df.copy()
    for scenario in SCENARIOS:
        out[f"{scenario.name}_change"] = (
            out[f"{scenario.name}_capacity"] - out["actual_capacity"]
        )
    return out


def load_all_data():
    world = fig3_base.load_world()
    utility_df = fig3_base.load_country_data("utility")
    distributed_df = fig3_base.load_country_data("distributed")
    total_df = fig3_base.make_total_country_data(utility_df, distributed_df)
    total_df = add_change_columns(total_df)
    global_df = fig3_base.load_global_data()
    return world, total_df, global_df


def plot_map_panel(
    ax: plt.Axes,
    world,
    country_df: pd.DataFrame,
    scenario_name: str,
    title: str,
    norm: mcolors.Normalize,
    vmax: float,
    *,
    show_xlabel: bool = False,
) -> None:
    scenario = next(item for item in SCENARIOS if item.name == scenario_name)
    _, ratio_col = scenario_columns(scenario)
    merged = world.merge(
        country_df[["country_key", ratio_col]],
        on="country_key",
        how="left",
    )

    bounds = world.total_bounds
    pad_x = (bounds[2] - bounds[0]) * 0.012
    pad_y = (bounds[3] - bounds[1]) * 0.045

    ax.set_facecolor("white")
    fig3_base.add_graticule(ax, lon_labels=False, lat_labels=True)
    world.plot(
        ax=ax,
        facecolor=COLORS["land"],
        edgecolor="none",
        linewidth=0,
        zorder=1,
        rasterized=True,
    )
    merged.dropna(subset=[ratio_col]).plot(
        column=ratio_col,
        ax=ax,
        cmap=MAP_CMAP,
        norm=norm,
        linewidth=0,
        edgecolor="none",
        zorder=2,
        rasterized=True,
    )
    world.plot(
        ax=ax,
        facecolor="none",
        edgecolor="#FFFFFF",
        linewidth=0.13,
        zorder=3,
        rasterized=True,
    )
    world.plot(
        ax=ax,
        facecolor="none",
        edgecolor=COLORS["border"],
        linewidth=0.07,
        zorder=4,
    )
    fig3_base.plot_violin_box_inset(
        ax,
        merged[ratio_col],
        COLORS["total_light"],
        norm,
        vmax,
        show_xlabel=show_xlabel,
    )

    ax.text(
        0.5,
        1.018,
        title,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=6,
        fontweight="bold",
        color=COLORS["text"],
        clip_on=False,
    )
    ax.set_xlim(bounds[0] - pad_x, bounds[2] + pad_x)
    ax.set_ylim(bounds[1] - pad_y, bounds[3] + pad_y)
    ax.set_axis_off()


def select_heatmap_countries(
    country_df: pd.DataFrame, count: int = 20
) -> list[str]:
    df = country_df.copy()
    capacity_cols = [f"{scenario.name}_capacity" for scenario in SCENARIOS]
    change_cols = [f"{scenario.name}_change" for scenario in SCENARIOS]
    df["max_capacity"] = df[capacity_cols].max(axis=1)
    df["max_abs_change"] = df[change_cols].abs().max(axis=1)
    df = df.sort_values(
        ["max_capacity", "actual_capacity", "max_abs_change"],
        ascending=False,
    )
    return df["country_key"].head(count).tolist()


def country_label(country: str) -> str:
    key = str(country).upper().strip()
    return COUNTRY_CODES.get(key, key[:3])


def signed_log_transform(values: np.ndarray | pd.Series) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return np.sign(array) * np.log10(1 + np.abs(array))


def signed_log_norm(values: pd.Series) -> mcolors.TwoSlopeNorm:
    clean = (
        pd.to_numeric(values, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    transformed = np.abs(signed_log_transform(clean.to_numpy()))
    max_abs = float(np.nanpercentile(transformed, 98)) if len(transformed) else 1.0
    max_abs = max(0.6, min(4.0, max_abs))
    return mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)


def format_change_tick(value: float) -> str:
    if value == -1:
        return "-100%"
    if value == 0:
        return "0"
    if value == 1:
        return "+100%"
    return f"+{value:g}x"


def plot_country_heatmap(
    fig: plt.Figure,
    ax: plt.Axes,
    country_df: pd.DataFrame,
) -> None:
    countries = select_heatmap_countries(country_df)
    indexed = country_df.set_index("country_key")
    relative = np.vstack(
        [
            indexed.loc[countries, f"{scenario.name}_relative_change"]
            for scenario in SCENARIOS
        ]
    )
    absolute_change = np.abs(
        np.vstack(
            [
                indexed.loc[countries, f"{scenario.name}_change"]
                for scenario in SCENARIOS
            ]
        )
    )
    all_relative = country_df[
        [f"{scenario.name}_relative_change" for scenario in SCENARIOS]
    ].stack()
    norm = signed_log_norm(all_relative)
    color_values = signed_log_transform(relative)
    n_scenarios, n_countries = relative.shape

    ax.pcolormesh(
        np.arange(n_countries + 1),
        np.arange(n_scenarios + 1),
        color_values,
        cmap=HEATMAP_CMAP,
        norm=norm,
        edgecolors="white",
        linewidth=0.25,
        antialiased=True,
        zorder=1,
    )
    dot_scale = max(1.0, float(np.nanpercentile(absolute_change, 95)))
    dot_sizes = np.sqrt(np.clip(absolute_change, 0, dot_scale) / dot_scale) * 34
    dot_sizes = np.where(absolute_change > 0, np.maximum(dot_sizes, 1.8), 0.0)
    xx, yy = np.meshgrid(
        np.arange(n_countries) + 0.5,
        np.arange(n_scenarios) + 0.5,
    )
    ax.scatter(
        xx.ravel(),
        yy.ravel(),
        s=dot_sizes.ravel(),
        facecolor="white",
        edgecolor="#2C2C2C",
        linewidth=0.18,
        alpha=0.94,
        zorder=2,
    )
    for y in [4, 6]:
        ax.axhline(y, color="#BDB8AA", linewidth=0.55, zorder=3)

    ax.set_xlim(0, n_countries)
    ax.set_ylim(0, n_scenarios)
    ax.invert_yaxis()
    ax.set_xticks(np.arange(n_countries) + 0.5)
    ax.set_xticklabels(
        [country_label(country) for country in countries],
        rotation=90,
        ha="center",
        va="top",
    )
    ax.set_yticks(np.arange(n_scenarios) + 0.5)
    ax.set_yticklabels(SCENARIO_LABELS)
    ax.set_ylabel("Scenario", labelpad=1.0)
    ax.tick_params(axis="both", length=0, pad=1.0)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color(COLORS["axis"])
    ax.set_facecolor(COLORS["land"])

    pos = ax.get_position()
    legend_y = pos.y1 + 0.006
    legend_h = 0.056

    cax = fig.add_axes(
        [
            pos.x0 + pos.width * 0.52,
            legend_y + legend_h * 0.56,
            pos.width * 0.44,
            0.007,
        ]
    )
    scalar = mpl.cm.ScalarMappable(norm=norm, cmap=HEATMAP_CMAP)
    colorbar = fig.colorbar(scalar, cax=cax, orientation="horizontal")
    raw_ticks = [-1, 0, 10]
    tick_positions = signed_log_transform(np.array(raw_ticks, dtype=float))
    valid = [
        index
        for index, tick in enumerate(tick_positions)
        if norm.vmin <= tick <= norm.vmax
    ]
    colorbar.set_ticks([tick_positions[index] for index in valid])
    colorbar.set_ticklabels([])
    colorbar.outline.set_linewidth(0.35)
    colorbar.ax.tick_params(
        length=1.2,
        pad=0.4,
        width=0.35,
        labelbottom=False,
    )
    for index in valid:
        relative_x = (tick_positions[index] - norm.vmin) / (norm.vmax - norm.vmin)
        horizontal_alignment = "right" if raw_ticks[index] == -1 else "center"
        colorbar.ax.text(
            relative_x,
            -1.65,
            format_change_tick(raw_ticks[index]),
            transform=colorbar.ax.transAxes,
            ha=horizontal_alignment,
            va="top",
            fontsize=6,
            color=COLORS["text"],
            clip_on=False,
        )
    fig.text(
        pos.x0 + pos.width * 0.74,
        legend_y + legend_h * 0.86,
        "Relative change",
        ha="center",
        va="bottom",
        fontsize=6,
    )

    size_ax = fig.add_axes(
        [pos.x0 + pos.width * 0.02, legend_y, pos.width * 0.34, legend_h]
    )
    size_ax.axis("off")
    size_ax.text(
        0.46,
        0.86,
        "Abs. change (GW)",
        ha="center",
        va="bottom",
        fontsize=6,
    )
    legend_dot_scale = max(dot_scale, 1000.0)
    for index, value in enumerate([100, 500, 1000]):
        x = 0.17 + index * 0.29
        size_ax.scatter(
            [x],
            [0.54],
            s=max(4, np.sqrt(value / legend_dot_scale) * 34),
            facecolor="white",
            edgecolor="#2C2C2C",
            linewidth=0.2,
        )
        size_ax.text(
            x,
            0.12,
            f"{value:g}",
            ha="center",
            va="bottom",
            fontsize=6,
        )
    size_ax.set_xlim(0, 1)
    size_ax.set_ylim(0, 1)


def plot_scenario_ranking(ax: plt.Axes, global_df: pd.DataFrame) -> None:
    df = global_df.sort_values("total", ascending=False).reset_index(drop=True)
    x = np.arange(len(df))
    ax.bar(
        x,
        df["utility"],
        width=0.62,
        color=COLORS["utility"],
        label="Utility-scale PV",
        zorder=3,
    )
    ax.bar(
        x,
        df["distributed"],
        width=0.62,
        bottom=df["utility"],
        color=COLORS["distributed"],
        label="Distributed PV",
        zorder=3,
    )
    for x_position, utility, ratio in zip(
        x, df["utility"], df["ratio"], strict=True
    ):
        if utility > 1200:
            ax.text(
                x_position,
                utility * 0.50,
                f"{ratio:.2f}",
                ha="center",
                va="center",
                fontsize=6,
                color="white",
                fontweight="bold",
                rotation=90,
                zorder=4,
            )

    existing_capacity = float(np.nanmedian(df["actual_total"]))
    irena_2030, irena_2050 = 5500.0, 14400.0
    iea_nze_2030, iea_nze_2050 = 6300.0, 20000.0
    target_lines = [
        (existing_capacity, COLORS["target_red"], (0, (2, 1.4))),
        (irena_2030, COLORS["target_blue"], (0, (1.2, 1.2))),
        (irena_2050, COLORS["target_blue"], (0, (1.2, 1.2))),
        (iea_nze_2030, COLORS["distributed"], (0, (3, 1.5))),
        (iea_nze_2050, COLORS["distributed"], (0, (3, 1.5))),
    ]
    for value, color, linestyle in target_lines:
        ax.axhline(
            value,
            color=color,
            linestyle=linestyle,
            linewidth=0.65,
            zorder=1,
        )

    target_label_box = {
        "facecolor": "white",
        "edgecolor": "none",
        "alpha": 0.78,
        "pad": 0.18,
    }
    ax.text(
        len(x) - 0.52,
        existing_capacity,
        "Existing\ncapacity",
        ha="left",
        va="center",
        fontsize=6,
        color=COLORS["target_red"],
        linespacing=0.9,
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.82,
            "pad": 0.2,
        },
        clip_on=True,
    )

    irena_x = len(x) + 0.58
    ax.annotate(
        "",
        xy=(irena_x, irena_2050),
        xytext=(irena_x, irena_2030),
        xycoords="data",
        textcoords="data",
        arrowprops={
            "arrowstyle": "<->",
            "color": COLORS["target_blue"],
            "linewidth": 0.65,
            "shrinkA": 0,
            "shrinkB": 0,
        },
        annotation_clip=False,
    )
    for value, label in [(irena_2030, "2030"), (irena_2050, "2050")]:
        ax.text(
            irena_x - 0.14,
            value,
            label,
            color=COLORS["target_blue"],
            fontsize=6,
            va="center",
            ha="right",
            bbox=target_label_box,
            clip_on=True,
        )
    ax.text(
        irena_x + 0.22,
        (irena_2030 + irena_2050) / 2,
        "IRENA 1.5°C",
        color=COLORS["target_blue"],
        fontsize=6,
        ha="center",
        va="center",
        rotation=90,
        bbox=target_label_box,
        clip_on=True,
    )

    iea_x = len(x) - 0.72
    ax.annotate(
        "",
        xy=(iea_x, iea_nze_2050),
        xytext=(iea_x, iea_nze_2030),
        xycoords="data",
        textcoords="data",
        arrowprops={
            "arrowstyle": "<->",
            "color": COLORS["distributed"],
            "linewidth": 0.65,
            "shrinkA": 0,
            "shrinkB": 0,
        },
        annotation_clip=False,
    )
    for value, label in [(iea_nze_2030, "2030"), (iea_nze_2050, "2050")]:
        ax.text(
            iea_x - 0.14,
            value,
            label,
            color=COLORS["distributed"],
            fontsize=6,
            va="center",
            ha="right",
            bbox=target_label_box,
            clip_on=True,
        )
    ax.text(
        iea_x + 0.18,
        (iea_nze_2030 + iea_nze_2050) / 2,
        "IEA Net Zero by 2050",
        color=COLORS["distributed"],
        fontsize=6,
        ha="center",
        va="center",
        rotation=90,
        bbox=target_label_box,
        clip_on=True,
    )

    ax.set_ylabel("PV Capacity (GW)", labelpad=1)
    ax.set_xticks(x)
    ax.set_xticklabels(
        df["scenario"].astype(str),
        rotation=55,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_xlim(-0.62, len(x) + 1.05)
    ax.set_ylim(0, 21000)
    ax.set_yticks([0, 5000, 10000, 15000, 20000])
    ax.grid(axis="y", color="#E5E2DA", linewidth=0.35, zorder=0)
    ax.tick_params(length=2, pad=1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(0.5)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.00, 1.045),
        ncol=2,
        frameon=False,
        handlelength=1.2,
        handletextpad=0.35,
        borderpad=0,
        columnspacing=0.8,
        labelspacing=0.2,
    )


def add_map_colorbar(
    fig: plt.Figure,
    top_map_bounds: list[float],
    norm: mcolors.Normalize,
    vmax: float,
) -> None:
    colorbar_ax = fig.add_axes(
        [
            top_map_bounds[0] + top_map_bounds[2] * 0.25,
            top_map_bounds[1] + top_map_bounds[3] + 0.030,
            top_map_bounds[2] * 0.40,
            0.008,
        ]
    )
    scalar = mpl.cm.ScalarMappable(norm=norm, cmap=MAP_CMAP)
    colorbar = fig.colorbar(
        scalar,
        cax=colorbar_ax,
        orientation="horizontal",
        extend="both",
    )
    colorbar.set_ticks(fig3_base.ratio_ticks(vmax))
    colorbar.ax.xaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(fig3_base.format_ratio_tick)
    )
    colorbar.outline.set_linewidth(0.35)
    colorbar.ax.tick_params(length=1.2, pad=0.4, width=0.35, labelsize=6)
    fig.text(
        colorbar_ax.get_position().x0 + colorbar_ax.get_position().width / 2,
        colorbar_ax.get_position().y0 + 0.016,
        "Relative total PV capacity change",
        ha="center",
        va="bottom",
        fontsize=6,
    )


def build_figure() -> None:
    set_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Figure 5 data...")
    world, total_df, global_df = load_all_data()

    selected_maps = ["EU-Global", "Subcont-Lead", "Clus20-Lead"]
    all_ratios = pd.concat(
        [total_df[f"{scenario}_relative_change"] for scenario in selected_maps]
    )
    map_norm, map_vmax = fig3_base.ratio_norm(all_ratios)

    fig = plt.figure(
        figsize=(FIGURE_WIDTH_MM * MM, FIGURE_HEIGHT_MM * MM),
        constrained_layout=False,
    )

    heatmap_bounds = [0.105, 0.570, 0.380, 0.285]
    ranking_bounds = [0.105, 0.150, 0.380, 0.315]

    right_x, right_width = 0.545, 0.425
    map_group_bottom, map_group_top = 0.055, 0.885
    map_gap = 0.023
    map_height = (map_group_top - map_group_bottom - 2 * map_gap) / 3
    map_bounds = [
        [right_x, map_group_top - map_height, right_width, map_height],
        [
            right_x,
            map_group_bottom + map_height + map_gap,
            right_width,
            map_height,
        ],
        [right_x, map_group_bottom, right_width, map_height],
    ]

    print("Drawing country heatmap...")
    heatmap_ax = fig.add_axes(heatmap_bounds)
    plot_country_heatmap(fig, heatmap_ax, total_df)

    print("Drawing scenario capacity ranking...")
    ranking_ax = fig.add_axes(ranking_bounds)
    plot_scenario_ranking(ranking_ax, global_df)

    print("Drawing selected scenario maps...")
    for index, (bounds, scenario_name) in enumerate(
        zip(map_bounds, selected_maps, strict=True)
    ):
        map_ax = fig.add_axes(bounds)
        plot_map_panel(
            map_ax,
            world,
            total_df,
            scenario_name,
            scenario_name,
            map_norm,
            map_vmax,
            show_xlabel=index == 2,
        )

    add_map_colorbar(fig, map_bounds[0], map_norm, map_vmax)

    # Reading order: heatmap, top map, middle map, ranking, bottom map.
    add_panel_label(fig, 0.035, 0.950, "a")
    add_panel_label(fig, right_x - 0.025, 0.950, "b")
    add_panel_label(
        fig,
        right_x - 0.025,
        map_bounds[1][1] + map_bounds[1][3] + 0.018,
        "c",
    )
    add_panel_label(
        fig,
        0.035,
        ranking_bounds[1] + ranking_bounds[3] + 0.025,
        "d",
    )
    add_panel_label(
        fig,
        right_x - 0.025,
        map_bounds[2][1] + map_bounds[2][3] + 0.018,
        "e",
    )

    png_path = FIGURES_DIR / f"{FIGURE_BASENAME}.png"
    pdf_path = FIGURES_DIR / f"{FIGURE_BASENAME}.pdf"
    fig.savefig(png_path, dpi=PNG_DPI)
    fig.savefig(pdf_path, dpi=600)
    plt.close(fig)

    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    build_figure()
