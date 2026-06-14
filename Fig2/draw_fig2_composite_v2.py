from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from fig2_common import (
    COLORS,
    COUNTRY_LABEL_POSITIONS,
    EXPORT_DIR,
    LAYOUT_COUNTRIES,
    add_panel_label,
    add_size_legend,
    format_tick,
    load_distribution_data,
    load_scatter_data,
    load_solar_map_data,
    load_weighted_irradiance_targets,
    load_world_projected,
    mm,
    nice_upper,
    normalize_country_key,
    plot_capacity_scatter,
    plot_radiation_installation_map,
    set_style,
    smooth_xy,
)


FIG_WIDTH_MM = 180
FIG_HEIGHT_MM = 150


def add_layout_axes(fig: plt.Figure) -> dict[str, plt.Axes]:
    def bounds(x: float, y: float, w: float, h: float) -> list[float]:
        return [x / FIG_WIDTH_MM, y / FIG_HEIGHT_MM, w / FIG_WIDTH_MM, h / FIG_HEIGHT_MM]

    axes: dict[str, plt.Axes] = {}
    axes["map"] = fig.add_axes(bounds(37.5, 80, 100, 60))
    axes["global"] = fig.add_axes(bounds(45, 60, 90, 15))
    axes["scatter_utility"] = fig.add_axes(bounds(7.5, 14, 69.5, 30))
    axes["scatter_distributed"] = fig.add_axes(bounds(92.5, 14, 72, 30))

    country_y = [129, 106, 83, 60]
    for idx, y in enumerate(country_y):
        axes[f"country_{idx}"] = fig.add_axes(bounds(7.5, y, 23.5, 15))
    for idx, y in enumerate(country_y, start=4):
        axes[f"country_{idx}"] = fig.add_axes(bounds(149, y, 23.5, 15))
    return axes


def split_scale(utility_max: float, distributed_max: float, lower_fraction: float = 0.58) -> tuple[float, float, float]:
    utility_top = nice_upper(utility_max)
    distributed_top = nice_upper(distributed_max)
    if distributed_top <= 0:
        return utility_top, 1.0, 1.0
    distributed_scale = max(1.0, utility_top * lower_fraction / distributed_top)
    return utility_top, distributed_top, distributed_scale


def set_split_axis(ax: plt.Axes, utility_top: float, distributed_top: float, distributed_scale: float) -> None:
    ax.set_ylim(-distributed_top * distributed_scale * 1.08, utility_top * 1.08)
    ticks = [-distributed_top * distributed_scale, 0, utility_top]
    ticklabels = [format_tick(distributed_top), "0", format_tick(utility_top)]
    ax.set_yticks(ticks)
    labels = ax.set_yticklabels(ticklabels)
    if labels:
        labels[0].set_color(COLORS["distributed"])
        labels[-1].set_color(COLORS["utility"])


def plot_split_curves(
    ax: plt.Axes,
    x: np.ndarray,
    utility: np.ndarray,
    distributed: np.ndarray,
    points: int = 260,
    line_width: float = 0.72,
    close_to_zero: bool = False,
) -> None:
    smooth = smooth_xy_closed_to_zero if close_to_zero else smooth_xy
    xs_u, ys_u = smooth(x, utility, points=points)
    xs_d, ys_d = smooth(x, distributed, points=points)
    utility_top, distributed_top, distributed_scale = split_scale(
        float(np.nanmax(ys_u)) if len(ys_u) else 0,
        float(np.nanmax(ys_d)) if len(ys_d) else 0,
    )

    if len(xs_u):
        ax.fill_between(xs_u, 0, ys_u, color=COLORS["utility"], alpha=0.16, linewidth=0)
        ax.plot(xs_u, ys_u, color=COLORS["utility"], linewidth=line_width, label="Utility-scale PV")
    if len(xs_d):
        ys_d_plot = -ys_d * distributed_scale
        ax.fill_between(xs_d, 0, ys_d_plot, color=COLORS["distributed"], alpha=0.18, linewidth=0)
        ax.plot(xs_d, ys_d_plot, color=COLORS["distributed"], linewidth=line_width, label="Distributed PV")

    ax.axhline(0, color="#333333", linewidth=0.45)
    set_split_axis(ax, utility_top, distributed_top, distributed_scale)
    ax.set_xlim(2500, 9800)
    ax.set_xticks([3000, 6000, 9000])
    ax.grid(axis="x", color="#e4e4e4", linewidth=0.32)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(0.45)


def smooth_xy_closed_to_zero(x: np.ndarray, y: np.ndarray, points: int = 260) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if not valid.any():
        return np.array([]), np.array([])

    x_grid = np.sort(np.unique(x[np.isfinite(x)]))
    x_valid = x[valid]
    y_valid = y[valid]
    x_min = np.nanmin(x_valid)
    x_max = np.nanmax(x_valid)
    pad_x: list[float] = []

    left_candidates = x_grid[x_grid < x_min]
    if len(left_candidates):
        pad_x.append(float(left_candidates[-1]))

    right_candidates = x_grid[x_grid > x_max]
    if len(right_candidates):
        pad_x.append(float(right_candidates[0]))

    if pad_x:
        x_valid = np.concatenate([x_valid, np.asarray(pad_x, dtype=float)])
        y_valid = np.concatenate([y_valid, np.zeros(len(pad_x), dtype=float)])

    return smooth_xy(x_valid, y_valid, points=points)


def add_weighted_irradiance_lines(
    ax: plt.Axes,
    values: tuple[float, float] | None,
) -> None:
    if values is None:
        return

    ymin, ymax = ax.get_ylim()
    utility_value, distributed_value = values
    line_specs = [
        (utility_value, 0, ymax * 0.94, COLORS["utility"]),
        (distributed_value, ymin * 0.94, 0, COLORS["distributed"]),
    ]
    for x_value, y0, y1, color in line_specs:
        if not np.isfinite(x_value):
            continue
        (line,) = ax.plot(
            [x_value, x_value],
            [y0, y1],
            color=color,
            linewidth=0.65,
            linestyle=(0, (2.0, 1.4)),
            alpha=0.95,
            zorder=7,
            solid_capstyle="butt",
        )
        line.set_path_effects([])


def add_country_weighted_irradiance_lines(
    ax: plt.Axes,
    country: str,
    targets: dict[str, tuple[float, float]],
) -> None:
    add_weighted_irradiance_lines(ax, targets.get(normalize_country_key(country)))


def calculate_weighted_irradiance(
    irradiance: np.ndarray,
    capacity: np.ndarray,
) -> float:
    mask = np.isfinite(irradiance) & np.isfinite(capacity) & (capacity > 0)
    if not mask.any():
        return float("nan")
    return float(np.average(irradiance[mask], weights=capacity[mask]))


def plot_country_distribution_v2(
    ax: plt.Axes,
    country: str,
    distribution_df,
    country_cols: dict[str, tuple[str, str]],
    weighted_irradiance: dict[str, tuple[float, float]],
) -> None:
    x = distribution_df["光照"].to_numpy(dtype=float)
    utility_col, distributed_col = country_cols[country]
    plot_split_curves(
        ax,
        x,
        distribution_df[utility_col].to_numpy(dtype=float),
        distribution_df[distributed_col].to_numpy(dtype=float),
        points=260,
        line_width=0.72,
        close_to_zero=True,
    )
    add_country_weighted_irradiance_lines(ax, country, weighted_irradiance)

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
    ax.set_xlabel("Annual solar irradiance\n(MJ m$^{-2}$ yr$^{-1}$)", labelpad=0.4, fontsize=6)
    ax.set_ylabel("PV capacity\n(GW)", labelpad=0.3, fontsize=6)
    ax.yaxis.set_label_coords(-0.13, 0.5)
    ax.tick_params(axis="x", length=1.8, pad=0.5, labelsize=6)
    ax.tick_params(axis="y", length=1.8, pad=0.5, labelsize=6)


def plot_global_distribution_v2(
    ax: plt.Axes,
    distribution_df,
    panel_label: str | None = None,
) -> None:
    x = distribution_df["光照"].to_numpy(dtype=float)
    plot_split_curves(
        ax,
        x,
        distribution_df["集中式"].to_numpy(dtype=float),
        distribution_df["分布式"].to_numpy(dtype=float),
        points=360,
        line_width=0.85,
    )
    global_weighted_irradiance = (
        calculate_weighted_irradiance(x, distribution_df["集中式"].to_numpy(dtype=float)),
        calculate_weighted_irradiance(x, distribution_df["分布式"].to_numpy(dtype=float)),
    )
    add_weighted_irradiance_lines(ax, global_weighted_irradiance)
    ax.set_xlabel("Annual solar irradiance\n(MJ m$^{-2}$ yr$^{-1}$)", labelpad=0.4)
    ax.set_ylabel("Global PV capacity\n(GW)", labelpad=0.5)
    ax.tick_params(length=2, pad=1, labelsize=6)
    ax.legend(
        handles=[
            Line2D([0], [0], color=COLORS["utility"], linewidth=0.85, label="Utility-scale PV"),
            Line2D([0], [0], color=COLORS["distributed"], linewidth=0.85, label="Distributed PV"),
            Line2D([0], [0], color="#666666", linewidth=0.65, linestyle=(0, (2.0, 1.4)), label="Capacity-weighted mean"),
        ],
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.02),
        ncol=3,
        handlelength=1.4,
        columnspacing=0.65,
        fontsize=6,
        borderaxespad=0,
    )
    if panel_label:
        add_panel_label(ax, panel_label, x=-0.103, y=1.34)


def main() -> None:
    set_style(6)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Fig2 data...")
    world = load_world_projected()
    solar_df = load_solar_map_data()
    distribution_df, country_cols = load_distribution_data()
    weighted_irradiance = load_weighted_irradiance_targets()
    utility_scatter = load_scatter_data("utility")
    distributed_scatter = load_scatter_data("distributed")

    fig = plt.figure(figsize=(1, 1), constrained_layout=False)
    fig.set_size_inches(mm(FIG_WIDTH_MM), mm(FIG_HEIGHT_MM), forward=False)
    axes = add_layout_axes(fig)

    print("Drawing radiation and installation map...")
    plot_radiation_installation_map(fig, axes["map"], world, solar_df, panel_label="a")

    print("Drawing split linear national radiation distributions...")
    country_panel_labels = list("bcdefghi")
    for idx, country in enumerate(LAYOUT_COUNTRIES):
        plot_country_distribution_v2(
            axes[f"country_{idx}"],
            country,
            distribution_df,
            country_cols,
            weighted_irradiance,
        )
        add_panel_label(axes[f"country_{idx}"], country_panel_labels[idx], x=-0.262, y=1.13)

    print("Drawing split linear global radiation distribution...")
    plot_global_distribution_v2(axes["global"], distribution_df, panel_label="j")

    print("Drawing capacity scatter plots...")
    plot_capacity_scatter(axes["scatter_utility"], utility_scatter, "Utility-scale PV", panel_label="k")
    size_legend = add_size_legend(
        axes["scatter_utility"],
        (1, 30, 300),
        loc="lower left",
        bbox_to_anchor=(0.08, 0.04),
    )
    axes["scatter_utility"].add_artist(size_legend)
    plot_capacity_scatter(
        axes["scatter_distributed"],
        distributed_scatter,
        "Distributed PV",
        panel_label="l",
        colorbar_x=1.054,
        ylabel_x=-0.018,
        show_ylabel=False,
    )
    size_legend = add_size_legend(
        axes["scatter_distributed"],
        (0.1, 5, 25),
        loc="lower left",
        bbox_to_anchor=(0.08, 0.04),
    )
    axes["scatter_distributed"].add_artist(size_legend)

    out_pdf = EXPORT_DIR / "Fig2_composite_v2.pdf"
    out_png = EXPORT_DIR / "Fig2_composite_v2.png"
    fig.savefig(out_pdf, dpi=600)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
