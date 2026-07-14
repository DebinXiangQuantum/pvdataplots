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


def plot_total_pv_capacity_barh(ax: plt.Axes) -> None:
    df = load_total_pv_per_capita_data()
    plot_df = df.sort_values("total_pv_gw", ascending=False).head(TOP_N).reset_index(drop=True)

    y = np.arange(len(plot_df))
    values = plot_df["total_pv_gw"].to_numpy(dtype=float)
    colors = [COLORS["total_bar"]] * len(plot_df)

    ax.barh(y, values, height=0.68, color=colors, edgecolor="none", zorder=3)

    xmax = values.max() * 1.18
    ax.set_xlim(0, xmax)
    for yi, value in zip(y, values, strict=True):
        label = f"{value:.1f}" if value >= 10 else f"{value:.2f}"
        ax.text(value + xmax * 0.012, yi, label, ha="left", va="center", fontsize=6)

    labels = plot_df["display_country"].tolist()
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Total PV capacity (GW)", labelpad=2)
    ax.set_ylabel("")
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.35, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", length=2, pad=1)
    ax.tick_params(axis="y", length=0, pad=2)


def draw_total_pv_capacity_bar() -> None:
    set_style(6)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    legacy_pdf_dir = FIGURES_DIR / "extendedpdfs"
    legacy_pdf_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(mm(92), mm(112)))
    plot_total_pv_capacity_barh(ax)

    pdf_path = FIGURES_DIR / "extendedFig1-panelB.pdf"
    png_path = FIGURES_DIR / "extendedFig1-panelB.png"
    legacy_pdf_path = legacy_pdf_dir / "extendedFig2.pdf"
    legacy_png_path = FIGURES_DIR / "extendedFig2.png"
    fig.savefig(pdf_path, dpi=600, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(legacy_pdf_path, dpi=600, bbox_inches="tight")
    fig.savefig(legacy_png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")
    print(f"Saved {legacy_pdf_path}")
    print(f"Saved {legacy_png_path}")


if __name__ == "__main__":
    draw_total_pv_capacity_bar()
