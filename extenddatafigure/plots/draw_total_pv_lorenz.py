from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from extended_data_common import (
    COLORS,
    FIGURES_DIR,
    GDP_PC_CMAP,
    gini_from_sorted_values,
    load_total_pv_gdp_data,
    mm,
    set_style,
)


def plot_total_pv_lorenz(
    ax: plt.Axes,
    fig: plt.Figure,
    colorbar_ax: plt.Axes | None = None,
) -> None:
    df = load_total_pv_gdp_data()
    df_var = df.sort_values("total_pv_gw").reset_index(drop=True)
    n = len(df_var)

    cumulative_region, cumulative_pv, gini_pv = gini_from_sorted_values(df_var["total_pv_gw"].to_numpy())
    _, cumulative_gdp, gini_gdp = gini_from_sorted_values(df.sort_values("gdp_2024_usd")["gdp_2024_usd"].to_numpy())

    y_at_90 = np.interp(90, cumulative_region, cumulative_pv)
    y_gap = 100 - y_at_90

    ax.set_facecolor("white")

    ax.fill_between(cumulative_region, cumulative_region, cumulative_pv, color="#c0c0c0", alpha=0.16, zorder=1)

    norm = mcolors.LogNorm(
        vmin=float(df["gdp_per_capita_2024_usd"].min()),
        vmax=float(df["gdp_per_capita_2024_usd"].max()),
    )
    for i in range(n):
        x1, x2 = cumulative_region[i], cumulative_region[i + 1]
        y1, y2 = cumulative_pv[i], cumulative_pv[i + 1]
        color = GDP_PC_CMAP(norm(float(df_var["gdp_per_capita_2024_usd"].iloc[i])))
        ax.fill_between([x1, x2], 0, [y1, y2], color=color, edgecolor="none", zorder=2)

    ax.plot([0, 100], [0, 100], color="#111111", linewidth=1.0, zorder=3)
    ax.plot(
        cumulative_region,
        cumulative_pv,
        color=COLORS["total"],
        linewidth=1.15,
        zorder=5,
        label=f"Total PV (Gini={gini_pv:.2f})",
    )
    ax.plot(
        cumulative_region,
        cumulative_gdp,
        color=COLORS["gdp"],
        linewidth=1.0,
        linestyle="--",
        zorder=4,
        label=f"2024 GDP (Gini={gini_gdp:.2f})",
    )

    ax.axvline(90, color="#1e90ff", linestyle="--", linewidth=0.55, zorder=6)
    ax.annotate(
        "",
        xy=(89, y_at_90),
        xytext=(89, 100),
        arrowprops=dict(arrowstyle="<->, widthB=1.2, lengthB=0.3", color="#111111", lw=0.55),
    )
    ax.annotate(
        "",
        xy=(90.25, 101.9),
        xytext=(99.75, 101.9),
        arrowprops=dict(arrowstyle="<->", color="#111111", lw=0.75, mutation_scale=7),
        clip_on=False,
        annotation_clip=False,
        zorder=7,
    )
    ax.plot([90, 90], [100, 101.45], color="#111111", linewidth=0.65, clip_on=False, zorder=7)
    ax.plot([100, 100], [100, 101.45], color="#111111", linewidth=0.65, clip_on=False, zorder=7)
    ax.text(76.5, (y_at_90 + 100) / 2, f"{y_gap:.0f}%", ha="left", va="center", fontweight="bold")
    ax.text(95, 102.7, "10%", ha="center", va="bottom", fontweight="bold", clip_on=False)
    ax.text(34, 40, "Perfect equality", rotation=46, ha="center", va="center", alpha=0.7)
    ax.text(52, 10, "Lorenz curve", rotation=12, ha="center", va="center", alpha=0.7)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Cumulative region percentage (%)", labelpad=1)
    ax.set_ylabel("Cumulative percentage (%)", labelpad=1)
    ax.tick_params(length=2, pad=1)
    ax.legend(
        loc="upper left",
        borderaxespad=0.45,
        frameon=False,
        handlelength=1.6,
        labelspacing=0.35,
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.55)

    if colorbar_ax is None:
        divider = make_axes_locatable(ax)
        colorbar_ax = divider.append_axes("right", size="4.6%", pad=0.055)
    sm = plt.cm.ScalarMappable(cmap=GDP_PC_CMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=colorbar_ax)
    cbar.outline.set_linewidth(0.35)
    cbar.ax.tick_params(length=1.5, pad=1, width=0.35, labelsize=6)
    cbar.set_label("GDP per capita\n(constant 2015 US$)", fontsize=6, labelpad=1)


def draw_total_pv_lorenz() -> None:
    set_style(6)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(mm(88), mm(70)))
    plot_total_pv_lorenz(ax, fig)

    pdf_path = FIGURES_DIR / "extendedFig1-panelA.pdf"
    png_path = FIGURES_DIR / "extendedFig1-panelA.png"
    fig.savefig(pdf_path, dpi=600, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


if __name__ == "__main__":
    draw_total_pv_lorenz()
