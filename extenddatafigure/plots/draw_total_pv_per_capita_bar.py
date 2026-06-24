from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from extended_data_common import (
    COLORS,
    FIGURES_DIR,
    load_total_pv_per_capita_data,
    mm,
    set_style,
)


TOP_N = 30


def draw_total_pv_per_capita_bar() -> None:
    set_style(6)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_total_pv_per_capita_data()
    plot_df = df.sort_values("pv_per_capita_mw_per_10k", ascending=False).head(TOP_N).reset_index(drop=True)

    x = np.arange(len(plot_df))
    values = plot_df["pv_per_capita_mw_per_10k"].to_numpy(dtype=float)
    colors = [COLORS["total_bar"]] * len(plot_df)

    fig, ax = plt.subplots(figsize=(mm(178), mm(72)))
    ax.bar(x, values, width=0.72, color=colors, edgecolor="none", zorder=3)

    ymax = values.max() * 1.22
    ax.set_ylim(0, ymax)
    for xi, value in zip(x, values, strict=True):
        label = f"{value:.1f}" if value >= 1 else f"{value:.2f}"
        ax.text(xi, value + ymax * 0.015, label, ha="center", va="bottom", fontsize=5.5, rotation=90)

    labels = plot_df["display_country"].tolist()
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=42, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Total PV capacity per capita\n(MW per 10,000 people)", labelpad=1)
    ax.set_xlabel("")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.35, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=1)
    ax.tick_params(axis="y", length=2, pad=1)
    ax.margins(x=0.006)

    pdf_path = FIGURES_DIR / "extendedFig2.pdf"
    png_path = FIGURES_DIR / "extendedFig2.png"
    fig.savefig(pdf_path, dpi=600, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


if __name__ == "__main__":
    draw_total_pv_per_capita_bar()
