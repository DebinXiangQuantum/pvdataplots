from __future__ import annotations

import matplotlib.pyplot as plt

from ambition_common import add_deployment_rate_colorbar, plot_deployment_rate_map
from extended_data_common import COLORS, FIGURES_DIR, mm, set_style


FIGURE_BASENAME = "extendedFig6"
FIGURE_WIDTH_MM = 180
FIGURE_HEIGHT_MM = 190
TITLE_X = 0.040
PANEL_LABEL_X = 0.015
PANEL_LABEL_FONT_SIZE = 8
PANEL_TEXT_OFFSET = 0.004
TOP_MAP_TOP_MM = 169
MAP_GAP_MM = 6


def draw_extended_figure_6_combined() -> None:
    set_style(6)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    legacy_pdf_dir = FIGURES_DIR / "extendedpdfs"
    legacy_pdf_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(mm(FIGURE_WIDTH_MM), mm(FIGURE_HEIGHT_MM)))
    # Keep a stable physical canvas so the PNG uses the same 20 px/mm export
    # convention as the other combined extended figures.
    fig.set_size_inches(mm(FIGURE_WIDTH_MM), mm(FIGURE_HEIGHT_MM), forward=False)
    grid = fig.add_gridspec(
        2,
        1,
        left=TITLE_X,
        right=0.985,
        bottom=0.055,
        top=0.895,
        hspace=0.070,
    )
    axes = [fig.add_subplot(grid[index, 0]) for index in range(2)]

    panels = []
    for panel_label, task_name, ax in zip(("a", "b"), ("Centralized", "Distributed"), axes, strict=True):
        task = plot_deployment_rate_map(ax, task_name)
        panels.append((panel_label, task.label, ax))

    # Map axes may be resized to preserve the projection's aspect ratio. Place
    # the panels by their measured size so the visual map gap remains compact.
    fig.canvas.draw()
    map_height = axes[0].get_position().height
    top_map_top = TOP_MAP_TOP_MM / FIGURE_HEIGHT_MM
    map_gap = MAP_GAP_MM / FIGURE_HEIGHT_MM
    top_map_y = top_map_top - map_height
    bottom_map_y = top_map_y - map_gap - map_height
    for ax, map_y in zip(axes, (top_map_y, bottom_map_y), strict=True):
        ax.set_position([TITLE_X, map_y, 0.985 - TITLE_X, map_height])

    # Position labels in figure coordinates so both rows share exact alignment.
    fig.canvas.draw()
    for panel_label, task_label, ax in panels:
        label_y = ax.get_position().y1 + PANEL_TEXT_OFFSET
        fig.text(
            PANEL_LABEL_X,
            label_y,
            panel_label,
            ha="left",
            va="bottom",
            fontsize=PANEL_LABEL_FONT_SIZE,
            fontweight="bold",
            color=COLORS["text"],
        )
        fig.text(
            TITLE_X,
            label_y,
            f"{task_label} deployment realization rate by country",
            ha="left",
            va="bottom",
            fontsize=6,
            fontweight="bold",
            color=COLORS["text"],
        )

    cax = fig.add_axes([0.295, 0.948, 0.410, 0.012])
    add_deployment_rate_colorbar(fig, cax)

    png_path = FIGURES_DIR / f"{FIGURE_BASENAME}.png"
    legacy_pdf_path = legacy_pdf_dir / f"{FIGURE_BASENAME}.pdf"
    fig.savefig(png_path, dpi=508)
    fig.savefig(legacy_pdf_path, dpi=600)
    plt.close(fig)
    print(f"Saved {png_path}")
    print(f"Saved {legacy_pdf_path}")


if __name__ == "__main__":
    draw_extended_figure_6_combined()
