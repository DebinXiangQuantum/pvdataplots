from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from extended_data_common import (
    COLORS,
    FIGURES_DIR,
    GDP_PC_CMAP,
    annotate_label,
    format_si,
    load_total_pv_gdp_data,
    mm,
    set_style,
)


LABEL_OFFSETS = {
    "CHINA": (-17, 16, "right", "bottom"),
    "UNITED STATES": (10, 3, "left", "center"),
    "INDIA": (12, 6, "left", "center"),
    "GERMANY": (12, -9, "left", "top"),
    "JAPAN": (-16, 11, "right", "bottom"),
    "CHILE": (-14, 18, "right", "bottom"),
    "AUSTRALIA": (-14, 11, "right", "bottom"),
    "UNITED ARAB EMIRATES": (15, -18, "left", "top"),
    "LUXEMBOURG": (17, 13, "left", "bottom"),
    "UKRAINE": (-12, 18, "right", "bottom"),
    "MOLDOVA": (-15, 12, "right", "bottom"),
    "HONDURAS": (-13, 12, "right", "bottom"),
    "GUYANA": (12, 12, "left", "bottom"),
    "EQUATORIAL GUINEA": (-12, 0, "right", "center"),
    "PAPUA NEW GUINEA": (12, 8, "left", "bottom"),
    "ICELAND": (-9, -7, "right", "top"),
    "MAURITANIA": (-14, 3, "right", "center"),
    "MACEDONIA,THE FORMER YUGOSLAV REPUBLIC OF": (14, 17, "left", "bottom"),
    "NAMIBIA": (22, -8, "left", "top"),
}

SCATTER_LABELS = {
    "EQUATORIAL GUINEA": "Eq. Guinea",
    "PAPUA NEW GUINEA": "Papua N.G.",
    "MACEDONIA,THE FORMER YUGOSLAV REPUBLIC OF": "N. Macedonia",
}

LABEL_COUNTRIES = [
    "CHINA",
    "UNITED STATES",
    "INDIA",
    "GERMANY",
    "JAPAN",
    "CHILE",
    "AUSTRALIA",
    "UNITED ARAB EMIRATES",
    "LUXEMBOURG",
    "UKRAINE",
    "MOLDOVA",
    "HONDURAS",
    "GUYANA",
    "EQUATORIAL GUINEA",
    "PAPUA NEW GUINEA",
    "ICELAND",
    "MAURITANIA",
    "MACEDONIA,THE FORMER YUGOSLAV REPUBLIC OF",
    "NAMIBIA",
]


def marker_sizes(values: pd.Series) -> np.ndarray:
    raw = np.sqrt(np.clip(values.to_numpy(dtype=float), 0, None))
    vmax = np.nanpercentile(raw[raw > 0], 98) if np.any(raw > 0) else 1
    return np.interp(np.clip(raw, 0, vmax), [0, vmax], [10, 78])


def draw_total_pv_gdp_scatter() -> None:
    set_style(6)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_total_pv_gdp_data()
    df["log_gdp"] = np.log10(df["gdp_2024_usd"])
    df["log_total_pv"] = np.log10(df["total_pv_gw"])

    norm = mcolors.LogNorm(
        vmin=float(df["gdp_per_capita_2024_usd"].min()),
        vmax=float(df["gdp_per_capita_2024_usd"].max()),
    )
    sizes = marker_sizes(df["pv_per_capita_mw_per_10k"])

    fig, ax = plt.subplots(figsize=(mm(98), mm(72)))
    scatter = ax.scatter(
        df["log_gdp"],
        df["log_total_pv"],
        s=sizes,
        c=df["gdp_per_capita_2024_usd"],
        cmap=GDP_PC_CMAP,
        norm=norm,
        alpha=0.9,
        edgecolors="white",
        linewidths=0.28,
        zorder=3,
    )

    fit = np.polyfit(df["log_gdp"], df["log_total_pv"], 1)
    xfit = np.linspace(df["log_gdp"].min(), df["log_gdp"].max(), 200)
    yfit = fit[0] * xfit + fit[1]
    corr = np.corrcoef(df["log_gdp"], df["log_total_pv"])[0, 1]
    df["fit"] = fit[0] * df["log_gdp"] + fit[1]
    df["fit_residual"] = df["log_total_pv"] - df["fit"]
    ax.plot(xfit, yfit, color="#222222", linewidth=0.75, zorder=2)
    ax.text(
        0.03,
        0.97,
        f"log-log fit: slope={fit[0]:.2f}, r={corr:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6,
        color=COLORS["text"],
    )

    label_rows = df[df["country"].isin(LABEL_COUNTRIES)].copy()
    label_rows["_label_order"] = label_rows["country"].map({country: idx for idx, country in enumerate(LABEL_COUNTRIES)})
    label_rows = label_rows.sort_values("_label_order").reset_index(drop=True)
    for _, row in label_rows.iterrows():
        dx, dy, ha, va = LABEL_OFFSETS.get(row["country"], (7, 6, "left", "bottom"))
        label = SCATTER_LABELS.get(row["country"], row["display_country"])
        annotate_label(ax, row["log_gdp"], row["log_total_pv"], label, (dx, dy), ha=ha, va=va)

    ax.set_xlabel("GDP (log$_{10}$ constant 2015 USD)", labelpad=1)
    ax.set_ylabel("Total PV capacity (log$_{10}$ GW)", labelpad=1)
    ax.grid(color=COLORS["grid"], linewidth=0.35, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2, pad=1)

    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _: f"$10^{{{int(value)}}}$"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _: f"$10^{{{int(value)}}}$"))

    x_pad = (df["log_gdp"].max() - df["log_gdp"].min()) * 0.07
    y_pad = (df["log_total_pv"].max() - df["log_total_pv"].min()) * 0.10
    ax.set_xlim(df["log_gdp"].min() - x_pad, df["log_gdp"].max() + x_pad)
    ax.set_ylim(df["log_total_pv"].min() - y_pad, df["log_total_pv"].max() + y_pad)

    cax = ax.inset_axes([1.065, 0.03, 0.023, 0.78], transform=ax.transAxes)
    cbar = fig.colorbar(scatter, cax=cax)
    cbar.outline.set_linewidth(0.35)
    cbar.ax.tick_params(length=1.4, pad=1, width=0.35, labelsize=6)
    cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _: format_si(value)))
    cbar.set_label("GDP per capita\n(constant 2015 USD)", fontsize=6, labelpad=1)

    legend_values = np.array([1, 5, 15], dtype=float)
    legend_sizes = marker_sizes(pd.Series(legend_values))
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#303030",
            markeredgecolor="white",
            markeredgewidth=0.28,
            markersize=np.sqrt(size),
            label=f"{value:g}",
        )
        for value, size in zip(legend_values, legend_sizes, strict=True)
    ]
    size_legend = ax.legend(
        handles=handles,
        title="PV per capita\n(MW per 10,000)",
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(0.99, 0.02),
        borderpad=0,
        handletextpad=0.55,
        labelspacing=0.45,
        title_fontsize=6,
    )
    size_legend._legend_box.align = "left"

    pdf_path = FIGURES_DIR / "extendedFig3.pdf"
    png_path = FIGURES_DIR / "extendedFig3.png"
    fig.savefig(pdf_path, dpi=600, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


if __name__ == "__main__":
    draw_total_pv_gdp_scatter()
