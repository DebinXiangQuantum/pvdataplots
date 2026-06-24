from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from matplotlib import patheffects as pe
from matplotlib.lines import Line2D

from extended_data_common import (
    COLORS,
    FIGURES_DIR,
    INCOME6_BINS,
    INCOME6_COLORS,
    capacity_marker_sizes,
    capacity_size_vmax,
    get_pv_kind_config,
    load_pv_irradiance_income_data,
    mm,
    set_style,
)


KIND = "utility"
FIGURE_BASENAME = "extendedFig5-utility-income6-irradiance-percapita-linear-scatter"
SIZE_LEGEND_VALUES = (1, 30, 300)
LABEL_COUNTRIES = {
    "CHINA",
    "UNITED STATES",
    "INDIA",
    "GERMANY",
    "JAPAN",
    "SPAIN",
    "AUSTRALIA",
    "CHILE",
    "UNITED ARAB EMIRATES",
    "QATAR",
    "DENMARK",
    "BULGARIA",
    "CZECH REPUBLIC",
    "UKRAINE",
    "UNITED KINGDOM",
    "ITALY",
    "MEXICO",
    "TURKEY",
    "BRAZIL",
}
LABEL_OFFSETS = {
    "CHINA": (-13, 9, "right", "bottom"),
    "UNITED STATES": (8, -10, "left", "top"),
    "INDIA": (8, 8, "left", "bottom"),
    "GERMANY": (7, 7, "left", "bottom"),
    "JAPAN": (-9, -9, "right", "top"),
    "SPAIN": (8, 13, "left", "bottom"),
    "AUSTRALIA": (-12, 10, "right", "bottom"),
    "CHILE": (-10, 9, "right", "bottom"),
    "UNITED ARAB EMIRATES": (8, -8, "left", "top"),
    "QATAR": (7, 7, "left", "bottom"),
    "DENMARK": (-10, 8, "right", "bottom"),
    "BULGARIA": (-12, -10, "right", "top"),
    "CZECH REPUBLIC": (-9, -8, "right", "top"),
    "UKRAINE": (-9, 7, "right", "bottom"),
    "UNITED KINGDOM": (7, -8, "left", "top"),
    "ITALY": (10, -8, "left", "top"),
    "MEXICO": (7, 7, "left", "bottom"),
    "TURKEY": (-9, -8, "right", "top"),
    "BRAZIL": (7, -8, "left", "top"),
}


def add_country_labels(ax: plt.Axes, df: pd.DataFrame) -> None:
    for _, row in df[df["country"].isin(LABEL_COUNTRIES)].iterrows():
        dx, dy, ha, va = LABEL_OFFSETS.get(row["country"], (7, 7, "left", "bottom"))
        text = ax.annotate(
            row["display_country"],
            xy=(row["weighted_irradiance_mj_m2"], row["pv_per_capita_mw_per_10k"]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=6,
            color=COLORS["text"],
            arrowprops=dict(arrowstyle="-", color=COLORS["text"], linewidth=0.28, shrinkA=1, shrinkB=2),
            zorder=7,
        )
        text.set_path_effects([pe.withStroke(linewidth=1.2, foreground="white")])


def add_legends(ax: plt.Axes, size_vmax: float) -> None:
    income_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.28,
            markersize=4.0,
            label=label,
        )
        for label, color in INCOME6_COLORS.items()
    ]
    income_legend = ax.legend(
        handles=income_handles,
        title="GDP per capita (USD)",
        loc="upper left",
        bbox_to_anchor=(1.018, 1.0),
        frameon=False,
        borderaxespad=0,
        labelspacing=0.36,
        handletextpad=0.45,
        title_fontsize=6,
    )
    income_legend._legend_box.align = "left"
    ax.add_artist(income_legend)

    legend_sizes = capacity_marker_sizes(pd.Series(SIZE_LEGEND_VALUES), vmax=size_vmax)
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
    size_legend = ax.legend(
        handles=size_handles,
        title="PV capacity (GW)",
        loc="lower left",
        bbox_to_anchor=(1.018, 0.02),
        frameon=False,
        borderaxespad=0,
        labelspacing=0.42,
        handletextpad=0.55,
        title_fontsize=6,
    )
    size_legend._legend_box.align = "left"


def draw_income6_linear_scatter() -> None:
    set_style(6)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    config = get_pv_kind_config(KIND)
    df = load_pv_irradiance_income_data(KIND, require_positive_capacity=True)
    df["income6_group"] = pd.cut(
        df["gdp_per_capita_usd"],
        bins=INCOME6_BINS,
        labels=list(INCOME6_COLORS.keys()),
        right=False,
    ).astype(str)
    size_vmax = capacity_size_vmax(df["capacity_gw"])

    fig, ax = plt.subplots(figsize=(mm(178), mm(98)))
    for group, color in INCOME6_COLORS.items():
        sub = df[df["income6_group"] == group]
        if sub.empty:
            continue
        ax.scatter(
            sub["weighted_irradiance_mj_m2"],
            sub["pv_per_capita_mw_per_10k"],
            s=capacity_marker_sizes(sub["capacity_gw"], vmax=size_vmax),
            color=color,
            alpha=0.88,
            edgecolors="white",
            linewidths=0.28,
            zorder=3,
        )

    add_country_labels(ax, df)
    add_legends(ax, size_vmax)

    ax.set_title(config["label"], loc="left", fontweight="bold", pad=2)
    ax.set_xlim(2700, 9000)
    ax.set_ylim(0, df["pv_per_capita_mw_per_10k"].max() * 1.18)
    ax.set_xticks([3000, 5000, 7000, 9000])
    ax.yaxis.set_major_locator(mticker.MaxNLocator(5))
    ax.set_xlabel("Weighted mean annual solar irradiance\n(MJ m$^{-2}$ yr$^{-1}$)", labelpad=1)
    ax.set_ylabel("PV capacity per capita\n(MW per 10,000 people)", labelpad=1)
    ax.grid(color=COLORS["grid"], linewidth=0.35, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2, pad=1)

    fig.subplots_adjust(left=0.08, right=0.78, top=0.92, bottom=0.16)
    pdf_path = FIGURES_DIR / f"{FIGURE_BASENAME}.pdf"
    png_path = FIGURES_DIR / f"{FIGURE_BASENAME}.png"
    fig.savefig(pdf_path, dpi=600)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


if __name__ == "__main__":
    draw_income6_linear_scatter()
