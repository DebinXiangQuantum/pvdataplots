from __future__ import annotations

import matplotlib.pyplot as plt

from ambition_common import plot_gdp_top20_bar
from extended_data_common import COLORS, FIGURES_DIR, mm, set_style


FIGURE_BASENAME = "extendedFig7"
FIGURE_WIDTH_MM = 180
FIGURE_HEIGHT_MM = 100
X_LIMITS = (0.0, 200.0)
TITLE_FONT_SIZE = 6
PANEL_LABEL_FONT_SIZE = 8
PANEL_LABEL_OFFSET = 0.030


def draw_extended_figure_7_combined() -> None:
    set_style(6)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    legacy_pdf_dir = FIGURES_DIR / "extendedpdfs"
    legacy_pdf_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(mm(FIGURE_WIDTH_MM), mm(FIGURE_HEIGHT_MM)))
    # Export at 20 px/mm, matching the existing combined extended figures.
    fig.set_size_inches(mm(FIGURE_WIDTH_MM), mm(FIGURE_HEIGHT_MM), forward=False)
    grid = fig.add_gridspec(
        1,
        2,
        left=0.155,
        right=0.985,
        bottom=0.170,
        top=0.840,
        wspace=0.130,
    )
    utility_ax = fig.add_subplot(grid[0, 0])
    distributed_ax = fig.add_subplot(grid[0, 1])

    utility_task = plot_gdp_top20_bar(
        utility_ax,
        "Centralized",
        xlim=X_LIMITS,
        value_label_fontsize=6,
    )
    distributed_task = plot_gdp_top20_bar(
        distributed_ax,
        "Distributed",
        xlim=X_LIMITS,
        show_ylabels=False,
        value_label_fontsize=6,
    )

    for ax in (utility_ax, distributed_ax):
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([0, 50, 100, 150, 200])

    fig.canvas.draw()
    title_y = utility_ax.get_position().y1 + 0.014
    utility_x = utility_ax.get_position().x0
    distributed_x = distributed_ax.get_position().x0
    fig.text(
        utility_x - PANEL_LABEL_OFFSET,
        title_y,
        "a",
        ha="left",
        va="bottom",
        fontsize=PANEL_LABEL_FONT_SIZE,
        fontweight="bold",
        color=COLORS["text"],
    )
    fig.text(
        utility_x,
        title_y,
        f"GDP top 20 countries - {utility_task.label}",
        ha="left",
        va="bottom",
        fontsize=TITLE_FONT_SIZE,
        fontweight="bold",
        color=COLORS["text"],
    )
    fig.text(
        distributed_x - PANEL_LABEL_OFFSET,
        title_y,
        "b",
        ha="left",
        va="bottom",
        fontsize=PANEL_LABEL_FONT_SIZE,
        fontweight="bold",
        color=COLORS["text"],
    )
    fig.text(
        distributed_x,
        title_y,
        f"GDP top 20 countries - {distributed_task.label}",
        ha="left",
        va="bottom",
        fontsize=TITLE_FONT_SIZE,
        fontweight="bold",
        color=COLORS["text"],
    )
    fig.text(
        (utility_ax.get_position().x0 + distributed_ax.get_position().x1) / 2,
        0.10,
        "Deployment realization rate (%)",
        ha="center",
        va="center",
        fontsize=6,
        color=COLORS["text"],
    )

    png_path = FIGURES_DIR / f"{FIGURE_BASENAME}.png"
    legacy_pdf_path = legacy_pdf_dir / f"{FIGURE_BASENAME}.pdf"
    fig.savefig(png_path, dpi=508)
    fig.savefig(legacy_pdf_path, dpi=600)
    plt.close(fig)
    print(f"Saved {png_path}")
    print(f"Saved {legacy_pdf_path}")


if __name__ == "__main__":
    draw_extended_figure_7_combined()
