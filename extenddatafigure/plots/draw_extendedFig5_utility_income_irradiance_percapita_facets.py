from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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


KIND = "utility"
FIGURE_BASENAME = "extendedFig5-utility-income-irradiance-percapita-facets"
SIZE_LEGEND_VALUES = (1, 30, 300)
LABEL_COUNTRIES = {
    "CHINA",
    "UNITED STATES",
    "INDIA",
    "GERMANY",
    "JAPAN",
    "AUSTRALIA",
    "CHILE",
    "UNITED ARAB EMIRATES",
    "QATAR",
}
LABEL_OFFSETS = {
    "CHINA": (-10, 8, "right", "bottom"),
    "UNITED STATES": (-10, -9, "right", "top"),
    "INDIA": (7, 7, "left", "bottom"),
    "GERMANY": (7, 7, "left", "bottom"),
    "JAPAN": (8, -8, "left", "top"),
    "AUSTRALIA": (-10, 9, "right", "bottom"),
    "CHILE": (-9, 8, "right", "bottom"),
    "UNITED ARAB EMIRATES": (8, -8, "left", "top"),
    "QATAR": (7, 7, "left", "bottom"),
}


def add_country_labels(ax: plt.Axes, sub: pd.DataFrame) -> None:
    for _, row in sub[sub["country"].isin(LABEL_COUNTRIES)].iterrows():
        dx, dy, ha, va = LABEL_OFFSETS.get(row["country"], (7, 7, "left", "bottom"))
        text = ax.annotate(
            row["display_country"],
            xy=(row["weighted_irradiance_mj_m2"], row["pv_per_capita_mw_per_10k"]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=5.4,
            color=COLORS["text"],
            arrowprops=dict(arrowstyle="-", color=COLORS["text"], linewidth=0.28, shrinkA=1, shrinkB=2),
            zorder=7,
        )
        text.set_path_effects([pe.withStroke(linewidth=1.2, foreground="white")])


def draw_income_facets() -> None:
    set_style(6)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    config = get_pv_kind_config(KIND)
    df = load_pv_irradiance_income_data(KIND, require_positive_capacity=True)
    size_vmax = capacity_size_vmax(df["capacity_gw"])

    y_values = df["pv_per_capita_mw_per_10k"]
    y_min = max(y_values.min() * 0.58, 1e-5)
    y_max = y_values.max() * 1.85

    fig, axes = plt.subplots(2, 2, figsize=(mm(178), mm(118)), sharex=True, sharey=True)
    axes_flat = axes.ravel()

    for ax, (group, color) in zip(axes_flat, INCOME_COLORS.items(), strict=True):
        sub = df[df["income_group"] == group].copy()
        if not sub.empty:
            sizes = capacity_marker_sizes(sub["capacity_gw"], vmax=size_vmax)
            ax.scatter(
                sub["weighted_irradiance_mj_m2"],
                sub["pv_per_capita_mw_per_10k"],
                s=sizes,
                color=color,
                alpha=0.88,
                edgecolors="white",
                linewidths=0.28,
                zorder=3,
            )
            add_country_labels(ax, sub)

        ax.set_title(f"{group} income (n={len(sub)})", loc="left", fontweight="bold", pad=2)
        ax.set_yscale("log")
        ax.set_xlim(2700, 9000)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks([3000, 5000, 7000, 9000])
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10, subs=(1.0,)))
        ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=(2.0, 5.0)))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _: f"{value:g}"))
        ax.grid(which="major", color=COLORS["grid"], linewidth=0.35, zorder=0)
        ax.grid(which="minor", axis="y", color=COLORS["grid"], linewidth=0.22, alpha=0.52, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=2, pad=1)
        ax.tick_params(axis="y", which="minor", length=1.2, width=0.35)

    for ax in axes[1, :]:
        ax.set_xlabel("Weighted mean annual solar irradiance\n(MJ m$^{-2}$ yr$^{-1}$)", labelpad=1)
    for ax in axes[:, 0]:
        ax.set_ylabel("PV capacity per capita\n(MW per 10,000 people, log scale)", labelpad=1)

    fig.text(0.035, 0.985, config["label"], ha="left", va="top", fontsize=7, fontweight="bold", color=COLORS["text"])

    legend_sizes = capacity_marker_sizes(pd.Series(SIZE_LEGEND_VALUES), vmax=size_vmax)
    handles = [
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
    legend = fig.legend(
        handles=handles,
        title="PV capacity (GW)",
        loc="lower center",
        bbox_to_anchor=(0.54, 0.012),
        frameon=False,
        ncol=len(handles),
        handletextpad=0.5,
        columnspacing=0.9,
        title_fontsize=6,
    )
    legend._legend_box.align = "left"

    fig.subplots_adjust(left=0.075, right=0.975, top=0.93, bottom=0.17, wspace=0.17, hspace=0.28)
    pdf_path = FIGURES_DIR / f"{FIGURE_BASENAME}.pdf"
    png_path = FIGURES_DIR / f"{FIGURE_BASENAME}.png"
    fig.savefig(pdf_path, dpi=600, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


if __name__ == "__main__":
    draw_income_facets()
