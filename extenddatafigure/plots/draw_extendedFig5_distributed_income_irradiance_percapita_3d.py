from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects as pe
from matplotlib.lines import Line2D

from extended_data_common import (
    COLORS,
    FIGURES_DIR,
    INCOME_COLORS,
    capacity_marker_sizes,
    capacity_size_vmax,
    get_pv_kind_config,
    load_pv_irradiance_income_data,
    mm,
    set_style,
)


KIND = "distributed"
FIGURE_BASENAME = "extendedFig5-distributed-income-irradiance-percapita-3d"
SIZE_LEGEND_VALUES = (0.1, 5, 25)
LABEL_COUNTRIES = {
    "CHINA",
    "UNITED STATES",
    "AUSTRALIA",
    "MALTA",
}
LABEL_OFFSETS = {
    "CHINA": (-70, -0.02, -0.03),
    "UNITED STATES": (-70, -0.02, -0.03),
    "AUSTRALIA": (-80, 0.02, 0.08),
    "MALTA": (110, 0.02, -0.04),
}


def tick_label_from_log(value: float) -> str:
    actual = 10**value
    if actual >= 1000:
        return f"{actual / 1000:g}k"
    if actual >= 1:
        return f"{actual:g}"
    return f"{actual:g}"


def style_3d_axes(ax) -> None:
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor("#d7d7d7")
        axis._axinfo["grid"]["color"] = (0.88, 0.88, 0.88, 1.0)
        axis._axinfo["grid"]["linewidth"] = 0.35
        axis._axinfo["axisline"]["linewidth"] = 0.45
    ax.tick_params(pad=0, labelsize=6, length=2)


def add_country_labels(ax, df: pd.DataFrame) -> None:
    for _, row in df[df["country"].isin(LABEL_COUNTRIES)].iterrows():
        dx, dy, dz = LABEL_OFFSETS.get(row["country"], (0, 0, 0))
        text = ax.text(
            row["weighted_irradiance_mj_m2"] + dx,
            row["log_gdp_per_capita"] + dy,
            row["log_pv_per_capita"] + dz,
            row["display_country"],
            fontsize=5.2,
            color=COLORS["text"],
            zorder=9,
        )
        text.set_path_effects([pe.withStroke(linewidth=1.1, foreground="white")])


def draw_income_irradiance_3d() -> None:
    set_style(6)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    config = get_pv_kind_config(KIND)
    df = load_pv_irradiance_income_data(KIND, require_positive_capacity=True)
    df["log_gdp_per_capita"] = np.log10(df["gdp_per_capita_usd"])
    df["log_pv_per_capita"] = np.log10(df["pv_per_capita_mw_per_10k"])
    size_vmax = capacity_size_vmax(df["capacity_gw"])

    fig = plt.figure(figsize=(mm(178), mm(122)))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_proj_type("persp", focal_length=0.92)

    for group, color in INCOME_COLORS.items():
        sub = df[df["income_group"] == group]
        if sub.empty:
            continue
        sizes = capacity_marker_sizes(sub["capacity_gw"], vmax=size_vmax, min_size=15, max_size=95)
        ax.scatter(
            sub["weighted_irradiance_mj_m2"],
            sub["log_gdp_per_capita"],
            sub["log_pv_per_capita"],
            s=sizes,
            color=color,
            alpha=0.88,
            edgecolors="white",
            linewidths=0.28,
            depthshade=False,
            zorder=5,
        )

    add_country_labels(ax, df)
    style_3d_axes(ax)
    ax.view_init(elev=22, azim=-57)
    ax.set_box_aspect((1.34, 1.0, 0.82))

    ax.set_xlim(2700, 9000)
    ax.set_xticks([3000, 5000, 7000, 9000])
    ax.set_xlabel("Weighted mean annual solar irradiance\n(MJ m$^{-2}$ yr$^{-1}$)", labelpad=5)

    ax.set_ylim(np.log10(350), np.log10(160000))
    ax.set_yticks(np.log10([1000, 10000, 100000]))
    ax.set_yticklabels(["1k", "10k", "100k"])
    ax.set_ylabel("GDP per capita\n(current USD, log scale)", labelpad=6)

    z_min = max(np.floor(df["log_pv_per_capita"].min()), -5)
    z_max = np.ceil(df["log_pv_per_capita"].max() * 2) / 2
    ax.set_zlim(z_min, z_max)
    z_ticks = [tick for tick in [-4, -3, -2, -1, 0, 1] if z_min <= tick <= z_max]
    ax.set_zticks(z_ticks)
    ax.set_zticklabels([tick_label_from_log(tick) for tick in z_ticks])
    ax.set_zlabel("PV capacity per capita\n(MW per 10,000 people, log scale)", labelpad=6)

    fig.text(0.035, 0.975, config["label"], ha="left", va="top", fontsize=7, fontweight="bold", color=COLORS["text"])

    income_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markeredgecolor="none", markersize=4, label=group)
        for group, color in INCOME_COLORS.items()
    ]
    income_legend = fig.legend(
        handles=income_handles,
        title="Income level",
        loc="upper right",
        bbox_to_anchor=(0.965, 0.88),
        frameon=False,
        borderpad=0,
        labelspacing=0.35,
        title_fontsize=6,
    )
    income_legend._legend_box.align = "left"

    legend_sizes = capacity_marker_sizes(pd.Series(SIZE_LEGEND_VALUES), vmax=size_vmax, min_size=15, max_size=95)
    size_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#2b2b2b",
            markeredgecolor="white",
            markeredgewidth=0.28,
            markersize=size**0.5,
            label=f"{value:g}",
        )
        for value, size in zip(SIZE_LEGEND_VALUES, legend_sizes, strict=True)
    ]
    size_legend = fig.legend(
        handles=size_handles,
        title="PV capacity (GW)",
        loc="upper right",
        bbox_to_anchor=(0.965, 0.54),
        frameon=False,
        borderpad=0,
        labelspacing=0.42,
        title_fontsize=6,
    )
    size_legend._legend_box.align = "left"

    fig.subplots_adjust(left=0.01, right=0.80, top=0.94, bottom=0.03)
    pdf_path = FIGURES_DIR / f"{FIGURE_BASENAME}.pdf"
    png_path = FIGURES_DIR / f"{FIGURE_BASENAME}.png"
    fig.savefig(pdf_path, dpi=600)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


if __name__ == "__main__":
    draw_income_irradiance_3d()
