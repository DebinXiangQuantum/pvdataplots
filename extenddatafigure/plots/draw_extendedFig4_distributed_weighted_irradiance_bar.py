from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects as pe

from extended_data_common import (
    COLORS,
    FIGURES_DIR,
    get_pv_kind_config,
    load_pv_irradiance_income_data,
    mm,
    set_style,
)


KIND = "distributed"
FIGURE_BASENAME = "extendedFig4-distributed-weighted-irradiance"
FOCUS_COUNTRIES = {
    "CHINA",
    "UNITED STATES",
    "GERMANY",
    "JAPAN",
    "AUSTRALIA",
    "MALTA",
    "NETHERLANDS",
}


def annotate_countries(ax: plt.Axes, plot_df, selected: set[str]) -> None:
    for _, row in plot_df[plot_df["country"].isin(selected)].iterrows():
        text = ax.annotate(
            row["display_country"],
            xy=(row["rank"], row["weighted_irradiance_mj_m2"]),
            xytext=(0, -5),
            textcoords="offset points",
            ha="center",
            va="top",
            rotation=90,
            fontsize=5.3,
            color=COLORS["text"],
            arrowprops=None,
            clip_on=True,
            zorder=7,
        )
        text.set_path_effects([pe.withStroke(linewidth=1.2, foreground="white")])


def draw_weighted_irradiance_bar() -> None:
    set_style(6)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    config = get_pv_kind_config(KIND)

    df = load_pv_irradiance_income_data(KIND)
    plot_df = df.sort_values("weighted_irradiance_mj_m2", ascending=False).reset_index(drop=True)
    plot_df["rank"] = np.arange(1, len(plot_df) + 1)

    selected = set(plot_df.head(1)["country"]) | FOCUS_COUNTRIES
    selected &= set(plot_df["country"])

    base_color = config["color"]
    colors = [mcolors.to_rgba(base_color, 0.42)] * len(plot_df)
    for idx, row in plot_df.iterrows():
        if row["country"] in selected:
            colors[idx] = mcolors.to_rgba(base_color, 0.88)

    fig, ax = plt.subplots(figsize=(mm(178), mm(78)))
    ax.bar(
        plot_df["rank"],
        plot_df["weighted_irradiance_mj_m2"],
        width=0.82,
        color=colors,
        edgecolor="none",
        zorder=3,
    )

    cap_mask = plot_df["capacity_gw"] > 0
    global_mean = np.average(
        plot_df.loc[cap_mask, "weighted_irradiance_mj_m2"],
        weights=plot_df.loc[cap_mask, "capacity_gw"],
    )
    ax.axhline(global_mean, color="#222222", linewidth=0.68, linestyle=(0, (3.0, 1.7)), zorder=5)
    ax.text(
        len(plot_df) + 0.6,
        global_mean,
        f"Installed-capacity weighted mean: {global_mean:.0f}",
        ha="right",
        va="bottom",
        fontsize=5.7,
        color=COLORS["text"],
    )

    ax.set_title(config["label"], loc="left", fontweight="bold", pad=2)
    ax.set_xlabel("Country rank by weighted mean irradiance", labelpad=1)
    ax.set_ylabel("Weighted mean annual solar irradiance\n(MJ m$^{-2}$ yr$^{-1}$)", labelpad=1)
    ax.set_xlim(0.2, len(plot_df) + 0.8)
    ax.set_ylim(0, max(9200, plot_df["weighted_irradiance_mj_m2"].max() * 1.08))
    ax.set_xticks([1, 25, 50, 75, 100, 125, 150, len(plot_df)])
    ax.set_xticklabels(["1", "25", "50", "75", "100", "125", "150", str(len(plot_df))])
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.35, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", length=2, pad=1)
    ax.tick_params(axis="y", length=2, pad=1)

    annotate_countries(ax, plot_df, selected)

    pdf_path = FIGURES_DIR / f"{FIGURE_BASENAME}.pdf"
    png_path = FIGURES_DIR / f"{FIGURE_BASENAME}.png"
    fig.savefig(pdf_path, dpi=600, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


if __name__ == "__main__":
    draw_weighted_irradiance_bar()
