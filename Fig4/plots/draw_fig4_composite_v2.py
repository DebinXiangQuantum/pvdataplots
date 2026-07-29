from __future__ import annotations

import matplotlib.pyplot as plt

from draw_fig4_composite import (
    COLORS,
    FIGURE_WIDTH_MM,
    FIGURES_DIR,
    FONT_SIZE,
    PNG_DPI,
    X_LIMITS,
    load_deployment_ring_data,
    mm,
    plot_deployment_ring,
    plot_gdp_top20_bar,
    set_style,
)


FIGURE_BASENAME = "Fig4_composite_v2"
FIGURE_HEIGHT_MM = 185
PANEL_LABEL_FONT_SIZE = 8

RING_LEFT = 0.020
RING_WIDTH = 0.620
RING_HEIGHT = RING_WIDTH * FIGURE_WIDTH_MM / FIGURE_HEIGHT_MM

RIGHT_COLUMN_LEFT = 0.675
BAR_AXES_LEFT = 0.795
BAR_AXES_RIGHT = 0.985
BAR_AXES_WIDTH = BAR_AXES_RIGHT - BAR_AXES_LEFT
BAR_AXES_HEIGHT = 0.415
UTILITY_BOTTOM = 0.505
DISTRIBUTED_BOTTOM = 0.065

STACKED_PANELS_CENTER = (
    DISTRIBUTED_BOTTOM + UTILITY_BOTTOM + BAR_AXES_HEIGHT
) / 2
RING_BOTTOM = STACKED_PANELS_CENTER - RING_HEIGHT / 2
RING_BOUNDS = [RING_LEFT, RING_BOTTOM, RING_WIDTH, RING_HEIGHT]


def add_stacked_bar_panels(fig: plt.Figure) -> tuple[plt.Axes, plt.Axes]:
    utility_ax = fig.add_axes(
        [BAR_AXES_LEFT, UTILITY_BOTTOM, BAR_AXES_WIDTH, BAR_AXES_HEIGHT]
    )
    distributed_ax = fig.add_axes(
        [BAR_AXES_LEFT, DISTRIBUTED_BOTTOM, BAR_AXES_WIDTH, BAR_AXES_HEIGHT]
    )

    plot_gdp_top20_bar(
        utility_ax,
        "Centralized",
        xlim=X_LIMITS,
        value_label_fontsize=FONT_SIZE,
    )
    plot_gdp_top20_bar(
        distributed_ax,
        "Distributed",
        xlim=X_LIMITS,
        value_label_fontsize=FONT_SIZE,
    )

    for ax, panel_label in [(utility_ax, "b"), (distributed_ax, "c")]:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([0, 50, 100, 150, 200])

        position = ax.get_position()
        fig.text(
            RIGHT_COLUMN_LEFT,
            position.y1,
            panel_label,
            ha="left",
            va="top",
            fontsize=PANEL_LABEL_FONT_SIZE,
            fontweight="bold",
            color=COLORS["text"],
        )

    distributed_position = distributed_ax.get_position()
    fig.text(
        (distributed_position.x0 + distributed_position.x1) / 2,
        distributed_position.y0 - 0.035,
        "Deployment realization rate (%)",
        ha="center",
        va="center",
        fontsize=FONT_SIZE,
        color=COLORS["text"],
    )

    return utility_ax, distributed_ax


def draw_figure_4_v2() -> None:
    set_style(FONT_SIZE)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    ring_df = load_deployment_ring_data()
    fig = plt.figure(
        figsize=(mm(FIGURE_WIDTH_MM), mm(FIGURE_HEIGHT_MM)), dpi=PNG_DPI
    )

    ring_ax = fig.add_axes(RING_BOUNDS, projection="polar")
    plot_deployment_ring(ring_ax, ring_df, legend_layout="below")
    fig.text(
        RING_LEFT,
        UTILITY_BOTTOM + BAR_AXES_HEIGHT,
        "a",
        ha="left",
        va="top",
        fontsize=PANEL_LABEL_FONT_SIZE,
        fontweight="bold",
        color=COLORS["text"],
    )

    add_stacked_bar_panels(fig)

    png_path = FIGURES_DIR / f"{FIGURE_BASENAME}.png"
    pdf_path = FIGURES_DIR / f"{FIGURE_BASENAME}.pdf"
    fig.savefig(png_path, dpi=PNG_DPI)
    fig.savefig(pdf_path, dpi=600)
    plt.close(fig)

    print(f"Ring countries: {len(ring_df)}")
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    draw_figure_4_v2()
