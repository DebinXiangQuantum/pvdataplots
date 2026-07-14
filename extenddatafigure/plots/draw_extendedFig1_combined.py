from __future__ import annotations

import matplotlib.pyplot as plt

from draw_total_pv_lorenz import plot_total_pv_lorenz
from draw_total_pv_per_capita_bar import plot_total_pv_capacity_barh
from extended_data_common import FIGURES_DIR, mm, set_style


def draw_extended_figure_1_combined() -> None:
    set_style(6)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    legacy_pdf_dir = FIGURES_DIR / "extendedpdfs"
    legacy_pdf_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(mm(180), mm(96)))
    # pyplot rounds the initial interactive canvas to whole display pixels;
    # reset the physical size explicitly so vector/raster exports remain exact.
    fig.set_size_inches(mm(180), mm(96), forward=False)
    outer_grid = fig.add_gridspec(
        1,
        2,
        left=0.070,
        right=0.985,
        bottom=0.120,
        top=0.860,
        width_ratios=[1.08, 0.92],
        wspace=0.34,
    )
    panel_a_grid = outer_grid[0].subgridspec(
        1,
        2,
        width_ratios=[1.0, 0.05],
        wspace=0.09,
    )
    ax_a = fig.add_subplot(panel_a_grid[0])
    colorbar_ax = fig.add_subplot(panel_a_grid[1])
    ax_b = fig.add_subplot(outer_grid[1])

    plot_total_pv_lorenz(ax_a, fig, colorbar_ax=colorbar_ax)
    plot_total_pv_capacity_barh(ax_b)

    fig.text(0.012, 0.890, "A", ha="left", va="top", fontsize=6, fontweight="bold")
    fig.text(0.560, 0.890, "B", ha="left", va="top", fontsize=6, fontweight="bold")

    pdf_path = FIGURES_DIR / "extendedFig1.pdf"
    png_path = FIGURES_DIR / "extendedFig1.png"
    legacy_pdf_path = legacy_pdf_dir / "extendedFig1.pdf"
    fig.savefig(pdf_path, dpi=600)
    # 508 dpi equals exactly 20 pixels per millimetre: 180 mm -> 3600 px.
    fig.savefig(png_path, dpi=508)
    fig.savefig(legacy_pdf_path, dpi=600)
    plt.close(fig)
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")
    print(f"Saved {legacy_pdf_path}")


if __name__ == "__main__":
    draw_extended_figure_1_combined()
