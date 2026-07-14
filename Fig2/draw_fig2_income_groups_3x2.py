from __future__ import annotations

from collections import OrderedDict

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from fig2_common import COLORS, EXPORT_DIR, add_panel_label, load_scatter_data, marker_sizes, mm, set_style


FIG_WIDTH_MM = 180
FIG_HEIGHT_MM = 92

# GDP per capita groups requested for the Fig. 2 comparison.
INCOME_GROUPS = OrderedDict(
    [
        ("Low income", ("< US$5,000", "#0072b2")),
        ("Middle income", ("US$5,000-20,000", "#E6A45C")),
        ("High income", ("> US$20,000", "#d74b9b")),
    ]
)


def assign_income_groups(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.copy()
    income = grouped["gdp_pc_raw"].to_numpy(dtype=float)
    grouped["income_group_3"] = pd.Categorical(
        np.select(
            [
                income < 5_000,
                (income >= 5_000) & (income <= 20_000),
                income > 20_000,
            ],
            list(INCOME_GROUPS),
            default=None,
        ),
        categories=list(INCOME_GROUPS),
        ordered=True,
    )
    return grouped.dropna(subset=["income_group_3"])


def draw_size_key(
    ax: plt.Axes,
    values: tuple[float, float, float],
    capacity_reference: pd.Series,
    pv_type: str,
) -> None:
    sizes = marker_sizes(pd.Series(values), reference_capacity=capacity_reference)
    y_positions = np.array([0.63, 0.37, 0.12])
    ax.scatter(
        np.full(len(values), 0.24),
        y_positions,
        s=sizes,
        color="#3b3b3b",
        edgecolors="none",
        zorder=2,
    )
    ax.text(0.0, 0.97, pv_type, ha="left", va="top", fontsize=6, fontweight="bold")
    ax.text(0.0, 0.81, "PV capacity (GW)", ha="left", va="top", fontsize=6)
    for y, value in zip(y_positions, values, strict=True):
        ax.text(0.48, y, f"{value:g}", ha="left", va="center", fontsize=6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()


def plot_group_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    group_name: str,
    color: str,
    y_limits: tuple[float, float],
    panel_label: str,
    show_ylabel: bool,
    show_xlabels: bool,
) -> None:
    subset = data[data["income_group_3"] == group_name]
    ax.scatter(
        subset["annual_irradiance"],
        subset["PVCapPerCapita"],
        s=subset["marker_size"],
        color=color,
        alpha=0.88,
        edgecolors="white",
        linewidths=0.32,
        zorder=3,
    )
    ax.axhline(0, color="#8a8a8a", linewidth=0.55, zorder=1)
    ax.grid(axis="y", color="#e7e7e7", linewidth=0.35, zorder=0)
    ax.set_xlim(2_500, 9_000)
    ax.set_xticks([3_000, 6_000, 9_000])
    ax.set_ylim(*y_limits)
    ax.tick_params(length=2, pad=1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(0.5)
    add_panel_label(ax, panel_label, x=0.02, y=0.97, fontsize=8)
    ax.text(
        0.98,
        0.95,
        f"n = {len(subset)}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6,
        color="#555555",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.35, "alpha": 0.85},
        zorder=5,
    )
    if show_ylabel:
        ax.set_ylabel("PV capacity per capita\n(log$_{10}$ MW per 10,000 people)", labelpad=2)
    else:
        ax.tick_params(axis="y", left=False, labelleft=False)
    if not show_xlabels:
        ax.tick_params(labelbottom=False)


def y_limits(data: pd.DataFrame) -> tuple[float, float]:
    values = data["PVCapPerCapita"].to_numpy(dtype=float)
    span = float(np.nanmax(values) - np.nanmin(values))
    padding = max(span * 0.10, 0.2)
    return float(np.nanmin(values) - padding), float(np.nanmax(values) + padding)


def main() -> None:
    set_style(6)
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
        }
    )
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    datasets = {
        "Utility-scale PV": assign_income_groups(load_scatter_data("utility")),
        "Distributed PV": assign_income_groups(load_scatter_data("distributed")),
    }
    for data in datasets.values():
        data["marker_size"] = marker_sizes(data["capacity_gw"], reference_capacity=data["capacity_gw"])

    titles = list(datasets)
    group_items = list(INCOME_GROUPS.items())
    fig = plt.figure(figsize=(mm(FIG_WIDTH_MM), mm(FIG_HEIGHT_MM)), constrained_layout=False)

    # Exact geometry keeps the scientific comparison grid visually disciplined at 180 mm.
    left_mm = 19.0
    panel_width_mm = 44.0
    column_gap_mm = 4.0
    panel_height_mm = 29.0
    top_row_y_mm = 48.0
    bottom_row_y_mm = 12.0
    key_x_mm = 163.0
    key_width_mm = 12.0
    axes: dict[tuple[int, int], plt.Axes] = {}
    for row, y in enumerate((top_row_y_mm, bottom_row_y_mm)):
        for col in range(3):
            x = left_mm + col * (panel_width_mm + column_gap_mm)
            axes[row, col] = fig.add_axes(
                [x / FIG_WIDTH_MM, y / FIG_HEIGHT_MM, panel_width_mm / FIG_WIDTH_MM, panel_height_mm / FIG_HEIGHT_MM]
            )

    for col, (group_name, (range_label, color)) in enumerate(group_items):
        center_x = (left_mm + col * (panel_width_mm + column_gap_mm) + panel_width_mm / 2) / FIG_WIDTH_MM
        fig.text(
            center_x,
            87 / FIG_HEIGHT_MM,
            group_name,
            ha="center",
            va="bottom",
            fontsize=6,
            fontweight="bold",
            color=COLORS["text"],
        )
        fig.add_artist(
            Line2D(
                [center_x - 0.045, center_x + 0.045],
                [85.2 / FIG_HEIGHT_MM, 85.2 / FIG_HEIGHT_MM],
                transform=fig.transFigure,
                color=color,
                linewidth=1.8,
                solid_capstyle="butt",
            )
        )
        fig.text(
            center_x,
            81.7 / FIG_HEIGHT_MM,
            range_label,
            ha="center",
            va="bottom",
            fontsize=6,
            color=COLORS["text"],
        )

    row_heading_specs = [
        ("Utility-scale PV", 78.8),
        ("Distributed PV", 43.0),
    ]
    for heading, y in row_heading_specs:
        fig.text(
            left_mm / FIG_WIDTH_MM,
            y / FIG_HEIGHT_MM,
            heading,
            ha="left",
            va="bottom",
            fontsize=6,
            fontweight="bold",
            color=COLORS["text"],
        )

    for row, title in enumerate(titles):
        data = datasets[title]
        limits = y_limits(data)
        for col, (group_name, (_, color)) in enumerate(group_items):
            plot_group_panel(
                axes[row, col],
                data,
                group_name,
                color,
                limits,
                panel_label=chr(ord("a") + row * 3 + col),
                show_ylabel=col == 0,
                show_xlabels=row == 1,
            )

    grid_center_x = (left_mm + (3 * panel_width_mm + 2 * column_gap_mm) / 2) / FIG_WIDTH_MM
    fig.text(
        grid_center_x,
        0.030,
        "Annual solar irradiance (MJ m$^{-2}$ yr$^{-1}$)",
        ha="center",
        va="bottom",
        fontsize=6,
        color=COLORS["text"],
    )

    utility_key = fig.add_axes([key_x_mm / FIG_WIDTH_MM, 55 / FIG_HEIGHT_MM, key_width_mm / FIG_WIDTH_MM, 22 / FIG_HEIGHT_MM])
    distributed_key = fig.add_axes([key_x_mm / FIG_WIDTH_MM, 19 / FIG_HEIGHT_MM, key_width_mm / FIG_WIDTH_MM, 22 / FIG_HEIGHT_MM])
    draw_size_key(utility_key, (1, 30, 300), datasets["Utility-scale PV"]["capacity_gw"], "Utility-scale")
    draw_size_key(distributed_key, (0.1, 5, 25), datasets["Distributed PV"]["capacity_gw"], "Distributed")

    out_pdf = EXPORT_DIR / "Fig2_income_groups_3x2.pdf"
    out_png = EXPORT_DIR / "Fig2_income_groups_3x2.png"
    fig.savefig(out_pdf, dpi=600)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
