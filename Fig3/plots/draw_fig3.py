from __future__ import annotations

from pathlib import Path
import importlib.util
import sys
import warnings

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
FIG3_DIR = ROOT / "Fig3"
OUT_DIR = FIG3_DIR / "figures"
AMBITION_XLSX = FIG3_DIR / "全球雄心排名.xlsx"

MM = 1 / 25.4
FIG_WIDTH_MM = 180
FIG_HEIGHT_MM = 150


def import_script(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


fig3_base = import_script(FIG3_DIR / "plots" / "draw_fig3_composite_v3.py", "fig3_composite_v3_base")
fig4_heat = import_script(ROOT / "Fig4" / "plots" / "draw_fig4_capacity_heatmap_stats.py", "fig4_heatmap_base")

SCENARIOS = fig3_base.SCENARIOS
SCENARIO_LABELS = [scenario.title for scenario in SCENARIOS]
COLORS = fig3_base.COLORS
CHANGE_CMAP = fig3_base.CHANGE_CMAP


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


def add_panel_label(fig: plt.Figure, bounds: list[float], label: str) -> None:
    fig.text(
        bounds[0] - 0.020,
        bounds[1] + bounds[3] + 0.020,
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
        out[f"{scenario.name}_change"] = out[f"{scenario.name}_capacity"] - out["actual_capacity"]
    return out


def load_all_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    world = fig3_base.load_world()
    utility_df = fig3_base.load_country_data("utility")
    distributed_df = fig3_base.load_country_data("distributed")
    total_df = fig3_base.make_total_country_data(utility_df, distributed_df)
    total_df = add_change_columns(total_df)
    global_df = fig3_base.load_global_data()
    return world, utility_df, distributed_df, total_df, global_df


def plot_map_panel(
    ax: plt.Axes,
    world,
    country_df: pd.DataFrame,
    scenario_name: str,
    title: str,
    norm: mcolors.Normalize,
    vmax: float,
    show_lat_labels: bool = True,
    show_xlabel: bool = False,
) -> None:
    scenario = next(item for item in SCENARIOS if item.name == scenario_name)
    _, ratio_col = scenario_columns(scenario)
    merged = world.merge(country_df[["country_key", ratio_col]], on="country_key", how="left")

    bounds = world.total_bounds
    pad_x = (bounds[2] - bounds[0]) * 0.012
    pad_y = (bounds[3] - bounds[1]) * 0.045

    ax.set_facecolor("white")
    fig3_base.add_graticule(ax, lon_labels=False, lat_labels=show_lat_labels)
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


def load_ambition(task: str, top_n: int = 10) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        raw = pd.read_excel(AMBITION_XLSX, sheet_name="installation_ambition_country_r")
    df = raw[(raw["task"] == task) & (raw["region_type"] == "COUNTRY")].copy()
    df["stage1_rank"] = pd.to_numeric(df["rank_stage1_total_desc"], errors="coerce")
    df["stage1_pct"] = pd.to_numeric(df["stage1_true_minus_baseline_pct_of_baseline"], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["stage1_rank", "stage1_pct"])
    df = df.sort_values("stage1_rank", ascending=True).head(top_n)
    # barh renders the first row at the bottom, so ascending data displays as descending top-to-bottom.
    return df.sort_values("stage1_pct", ascending=True).reset_index(drop=True)


def country_label(country: str) -> str:
    return fig4_heat.COUNTRY_CODE.get(str(country).upper(), str(country)[:3].upper())


def plot_ambition_bar(ax: plt.Axes, task: str, title: str, color: str) -> None:
    df = load_ambition(task)
    y = np.arange(len(df))
    values = df["stage1_pct"].to_numpy(dtype=float)
    max_value = float(np.nanmax(values))
    min_value = float(np.nanmin(values))
    label_offset = max((max_value - min(0, min_value)) * 0.018, 0.35)
    ax.barh(y, values, color=color, height=0.58, zorder=3)
    for yi, value in zip(y, values, strict=True):
        ha = "left" if value >= 0 else "right"
        x_text = value + label_offset if value >= 0 else value - label_offset
        ax.text(
            x_text,
            yi,
            f"{value:.1f}",
            ha=ha,
            va="center",
            fontsize=6,
            color=COLORS["text"],
            clip_on=False,
        )
    ax.set_yticks(y)
    ax.set_yticklabels([country_label(country) for country in df["region"]])
    ax.set_title(title, fontsize=6, fontweight="bold", pad=2, linespacing=0.92)
    ax.set_xlabel("Change from baseline (%)", labelpad=0.7)
    if min_value < 0:
        ax.axvline(0, color="#555555", linewidth=0.45, zorder=2)
    ax.grid(axis="x", color="#E5E2DA", linewidth=0.35, zorder=0)
    ax.tick_params(length=2, pad=1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(0.5)
    ax.set_xlim(min(0, min_value * 1.12), max_value * 1.24)


def select_heatmap_countries(country_df: pd.DataFrame, count: int = 20) -> list[str]:
    df = country_df.copy()
    cap_cols = [f"{scenario.name}_capacity" for scenario in SCENARIOS]
    change_cols = [f"{scenario.name}_change" for scenario in SCENARIOS]
    df["max_capacity"] = df[cap_cols].max(axis=1)
    df["max_abs_change"] = df[change_cols].abs().max(axis=1)
    df = df.sort_values(["max_capacity", "actual_capacity", "max_abs_change"], ascending=False)
    return df["country_key"].head(count).tolist()


def signed_log_transform(values: np.ndarray | pd.Series) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.sign(arr) * np.log10(1 + np.abs(arr))


def signed_log_norm(values: pd.Series) -> tuple[mcolors.TwoSlopeNorm, float]:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        max_abs = 1.0
    else:
        transformed = np.abs(signed_log_transform(clean.to_numpy()))
        max_abs = float(np.nanpercentile(transformed, 98))
        max_abs = max(0.6, min(4.0, max_abs))
    return mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs), max_abs


def format_change_tick(value: float) -> str:
    if value == -1:
        return "-1x"
    if value == 0:
        return "0"
    if value == 1:
        return "+100%"
    return f"+{value:g}x"


def plot_country_heatmap(fig: plt.Figure, ax: plt.Axes, country_df: pd.DataFrame) -> None:
    countries = select_heatmap_countries(country_df)
    rel = np.vstack([country_df.set_index("country_key").loc[countries, f"{scenario.name}_relative_change"] for scenario in SCENARIOS])
    abs_change = np.abs(np.vstack([country_df.set_index("country_key").loc[countries, f"{scenario.name}_change"] for scenario in SCENARIOS]))
    norm, _ = signed_log_norm(country_df[[f"{scenario.name}_relative_change" for scenario in SCENARIOS]].stack())
    color_values = signed_log_transform(rel)
    n_scenarios, n_countries = rel.shape

    ax.pcolormesh(
        np.arange(n_countries + 1),
        np.arange(n_scenarios + 1),
        color_values,
        cmap=fig4_heat.CHANGE_CMAP,
        norm=norm,
        edgecolors="white",
        linewidth=0.25,
        antialiased=True,
        zorder=1,
    )
    dot_scale = max(1.0, float(np.nanpercentile(abs_change, 95)))
    dot_sizes = np.sqrt(np.clip(abs_change, 0, dot_scale) / dot_scale) * 34
    dot_sizes = np.where(abs_change > 0, np.maximum(dot_sizes, 1.8), 0.0)
    xx, yy = np.meshgrid(np.arange(n_countries) + 0.5, np.arange(n_scenarios) + 0.5)
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
    ax.set_xticklabels([country_label(country) for country in countries], rotation=90, ha="center", va="top")
    ax.set_yticks(np.arange(n_scenarios) + 0.5)
    ax.set_yticklabels(SCENARIO_LABELS)
    ax.set_title("")
    ax.set_ylabel("Scenario", labelpad=1.0)
    ax.tick_params(axis="both", length=0, pad=1.0)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color(COLORS["axis"])
    ax.set_facecolor(COLORS["land"])

    pos = ax.get_position()
    legend_y = pos.y1 + 0.006
    legend_h = 0.052

    cax = fig.add_axes([pos.x0 + pos.width * 0.52, legend_y + legend_h * 0.56, pos.width * 0.44, 0.007])
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=fig4_heat.CHANGE_CMAP)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    raw_ticks = [-1, 0, 10]
    tick_positions = signed_log_transform(np.array(raw_ticks, dtype=float))
    valid = [i for i, tick in enumerate(tick_positions) if norm.vmin <= tick <= norm.vmax]
    cb.set_ticks([tick_positions[i] for i in valid])
    cb.set_ticklabels([])
    cb.outline.set_linewidth(0.35)
    cb.ax.tick_params(length=1.2, pad=0.4, width=0.35, labelbottom=False)
    for i in valid:
        rel_x = (tick_positions[i] - norm.vmin) / (norm.vmax - norm.vmin)
        ha = "right" if raw_ticks[i] == -1 else "center"
        cb.ax.text(
            rel_x,
            -1.65,
            format_change_tick(raw_ticks[i]),
            transform=cb.ax.transAxes,
            ha=ha,
            va="top",
            fontsize=6,
            color=COLORS["text"],
            clip_on=False,
        )
    fig.text(pos.x0 + pos.width * 0.74, legend_y + legend_h * 0.86, "Relative change", ha="center", va="bottom", fontsize=6)

    size_ax = fig.add_axes([pos.x0 + pos.width * 0.02, legend_y, pos.width * 0.34, legend_h])
    size_ax.axis("off")
    size_ax.text(0.46, 0.86, "Abs. change (GW)", ha="center", va="bottom", fontsize=6)
    legend_dot_scale = max(dot_scale, 1000.0)
    for xi, value in enumerate([100, 500, 1000]):
        x = 0.17 + xi * 0.29
        size_ax.scatter(
            [x],
            [0.54],
            s=max(4, np.sqrt(value / legend_dot_scale) * 34),
            facecolor="white",
            edgecolor="#2C2C2C",
            linewidth=0.2,
        )
        size_ax.text(x, 0.12, f"{value:g}", ha="center", va="bottom", fontsize=6)
    size_ax.set_xlim(0, 1)
    size_ax.set_ylim(0, 1)


def plot_scenario_ranking(ax: plt.Axes, global_df: pd.DataFrame) -> None:
    df = global_df.sort_values("total", ascending=False).reset_index(drop=True)
    x = np.arange(len(df))
    ax.bar(x, df["utility"], width=0.62, color=COLORS["utility"], label="Utility-scale PV", zorder=3)
    ax.bar(x, df["distributed"], width=0.62, bottom=df["utility"], color=COLORS["distributed"], label="Distributed PV", zorder=3)
    for xi, utility, ratio in zip(x, df["utility"], df["ratio"], strict=True):
        if utility > 1200:
            ax.text(
                xi,
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
        (iea_nze_2030, COLORS["target_orange"], (0, (3, 1.5))),
        (iea_nze_2050, COLORS["target_orange"], (0, (3, 1.5))),
    ]
    for value, color, linestyle in target_lines:
        ax.axhline(value, color=color, linestyle=linestyle, linewidth=0.65, zorder=1)

    target_label_box = {"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.18}
    ax.text(
        len(x) - 0.52,
        existing_capacity,
        "Existing\ncapacity",
        ha="left",
        va="center",
        fontsize=6,
        color=COLORS["target_red"],
        linespacing=0.9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.2},
        clip_on=True,
    )
    irena_x = len(x) + 0.58
    ax.annotate(
        "",
        xy=(irena_x, irena_2050),
        xytext=(irena_x, irena_2030),
        xycoords="data",
        textcoords="data",
        arrowprops={"arrowstyle": "<->", "color": COLORS["target_blue"], "linewidth": 0.65, "shrinkA": 0, "shrinkB": 0},
        annotation_clip=False,
    )
    ax.text(
        irena_x - 0.14,
        irena_2030,
        "2030",
        color=COLORS["target_blue"],
        fontsize=6,
        va="center",
        ha="right",
        bbox=target_label_box,
        clip_on=True,
    )
    ax.text(
        irena_x - 0.14,
        irena_2050,
        "2050",
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
        arrowprops={"arrowstyle": "<->", "color": COLORS["target_orange"], "linewidth": 0.65, "shrinkA": 0, "shrinkB": 0},
        annotation_clip=False,
    )
    ax.text(
        iea_x - 0.14,
        iea_nze_2030,
        "2030",
        color=COLORS["target_orange"],
        fontsize=6,
        va="center",
        ha="right",
        bbox=target_label_box,
        clip_on=True,
    )
    ax.text(
        iea_x - 0.14,
        iea_nze_2050,
        "2050",
        color=COLORS["target_orange"],
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
        color=COLORS["target_orange"],
        fontsize=6,
        ha="center",
        va="center",
        rotation=90,
        bbox=target_label_box,
        clip_on=True,
    )
    ax.set_title("")
    ax.set_ylabel("PV Capacity (GW)", labelpad=1)
    ax.set_xticks(x)
    ax.set_xticklabels(df["scenario"].astype(str), rotation=55, ha="right", rotation_mode="anchor")
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


def add_map_colorbar(fig: plt.Figure, bounds: list[float], norm: mcolors.Normalize, vmax: float) -> None:
    cax = fig.add_axes([bounds[0] + bounds[2] * 0.25, bounds[1] + bounds[3] + 0.050, bounds[2] * 0.40, 0.008])
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=CHANGE_CMAP)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", extend="both")
    cb.set_ticks(fig3_base.ratio_ticks(vmax))
    cb.ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fig3_base.format_ratio_tick))
    cb.outline.set_linewidth(0.35)
    cb.ax.tick_params(length=1.2, pad=0.4, width=0.35, labelsize=6)
    fig.text(
        cax.get_position().x0 + cax.get_position().width / 2,
        cax.get_position().y0 + 0.016,
        "Relative total PV capacity change",
        ha="center",
        va="bottom",
        fontsize=6,
    )


def build_figure() -> None:
    set_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Fig3 data...")
    world, _utility_df, _distributed_df, total_df, global_df = load_all_data()

    selected_maps = ["EU-Global", "Subcont-Lead", "Clus20-Lead"]
    all_ratios = pd.concat([total_df[f"{scenario}_relative_change"] for scenario in selected_maps])
    map_norm, map_vmax = fig3_base.ratio_norm(all_ratios)

    fig = plt.figure(figsize=(FIG_WIDTH_MM * MM, FIG_HEIGHT_MM * MM), constrained_layout=False)

    left_x, left_w = 0.055, 0.425
    stats_x = 0.105
    stats_w = left_x + left_w - stats_x
    right_x, right_w = 0.540, 0.405
    stats_gap = 0.032
    ambition_w = (left_w - stats_gap) / 2
    map_bounds = [
        [right_x, 0.650, right_w, 0.200],
        [right_x, 0.405, right_w, 0.200],
        [right_x, 0.150, right_w, 0.200],
    ]
    ambition_c_bounds = [left_x, 0.775, ambition_w, 0.160]
    ambition_d_bounds = [left_x + ambition_w + stats_gap, 0.775, ambition_w, 0.160]
    heatmap_bounds = [stats_x, 0.420, stats_w, 0.215]
    ranking_bounds = [stats_x, 0.180, stats_w, 0.180]

    # fig.text(left_x, 0.932, "Total PV scenario maps", ha="left", va="top", fontsize=6, fontweight="bold", color=COLORS["text"])
    add_map_colorbar(fig, map_bounds[0], map_norm, map_vmax)

    print("Drawing selected scenario maps...")
    map_titles = ["EU-Global", "Subcont-Lead", "Clus20-Lead"]
    map_labels = ["c", "e", "g"]
    for index, (bounds, scenario_name, title) in enumerate(zip(map_bounds, selected_maps, map_titles, strict=True)):
        ax = fig.add_axes(bounds)
        plot_map_panel(ax, world, total_df, scenario_name, title, map_norm, map_vmax, show_lat_labels=True, show_xlabel=index == 2)
        add_panel_label(fig, bounds, map_labels[index])

    print("Drawing ambition index bars...")
    ax_ambition_c = fig.add_axes(ambition_c_bounds)
    plot_ambition_bar(ax_ambition_c, "Centralized", "Stage-1 ambition\nUtility-scale PV", COLORS["utility"])
    add_panel_label(fig, ambition_c_bounds, "a")

    ax_ambition_d = fig.add_axes(ambition_d_bounds)
    plot_ambition_bar(ax_ambition_d, "Distributed", "Stage-1 ambition\nDistributed PV", COLORS["distributed"])
    add_panel_label(fig, ambition_d_bounds, "b")

    print("Drawing country heatmap...")
    ax_heatmap = fig.add_axes(heatmap_bounds)
    plot_country_heatmap(fig, ax_heatmap, total_df)
    fig.text(
        left_x - 0.020,
        heatmap_bounds[1] + heatmap_bounds[3] + 0.060,
        "d",
        ha="left",
        va="top",
        fontsize=8,
        fontweight="bold",
        color=COLORS["text"],
    )

    print("Drawing scenario ranking...")
    ax_ranking = fig.add_axes(ranking_bounds)
    plot_scenario_ranking(ax_ranking, global_df)
    fig.text(
        left_x - 0.020,
        ranking_bounds[1] + ranking_bounds[3] + 0.020,
        "f",
        ha="left",
        va="top",
        fontsize=8,
        fontweight="bold",
        color=COLORS["text"],
    )

    png_path = OUT_DIR / "Fig3_composite_v4.png"
    pdf_path = OUT_DIR / "Fig3_composite_v4.pdf"
    fig.savefig(png_path, dpi=600)
    fig.savefig(pdf_path, dpi=600)
    plt.close(fig)
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    build_figure()
