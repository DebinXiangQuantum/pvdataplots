from __future__ import annotations

from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXTENDED_PLOTS_DIR = ROOT / "extenddatafigure" / "plots"
if str(EXTENDED_PLOTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXTENDED_PLOTS_DIR))

from ambition_common import (  # noqa: E402
    country_key,
    load_country_deployment_rate,
    plot_gdp_top20_bar,
)
from extended_data_common import COLORS, mm, set_style  # noqa: E402


FIG4_DIR = ROOT / "Fig4"
FIGURES_DIR = FIG4_DIR / "figures"
CONTINENT_CSV = ROOT / "Fig3" / "country_continent_mapping.csv"

FIGURE_BASENAME = "Fig4_composite"
FIGURE_WIDTH_MM = 180
FIGURE_HEIGHT_MM = 185
PNG_DPI = 508
X_LIMITS = (0.0, 200.0)

FONT_SIZE = 6

CONTINENT_ORDER = [
    "Africa",
    "Asia",
    "Europe",
    "North America",
    "South America",
    "Oceania",
]
CONTINENT_COLORS = {
    "Africa": "#00a087",
    "Asia": "#2878b8",
    "Europe": "#7e62a3",
    "North America": "#d74b9b",
    "South America": "#e98b69",
    "Oceania": "#d9a441",
}


def load_deployment_ring_data() -> pd.DataFrame:
    utility = load_country_deployment_rate("Centralized")[
        ["region", "country_key", "deployment_rate"]
    ].rename(columns={"deployment_rate": "utility"})
    distributed = load_country_deployment_rate("Distributed")[
        ["country_key", "deployment_rate"]
    ].rename(columns={"deployment_rate": "distributed"})

    mapping = pd.read_csv(CONTINENT_CSV)
    mapping["country_key"] = mapping["country"].map(country_key)
    mapping = mapping[["country_key", "country_code", "continent"]].drop_duplicates(
        "country_key"
    )

    # Retain only paired country observations; a missing task is not interpreted as zero.
    df = utility.merge(distributed, on="country_key", how="inner", validate="one_to_one")
    df = df.merge(mapping, on="country_key", how="left", validate="one_to_one")
    missing_mask = df[["continent", "country_code"]].isna().any(axis=1)
    if missing_mask.any():
        missing = df.loc[missing_mask, "region"].tolist()
        raise ValueError(f"Missing continent metadata for: {missing}")

    unknown_continents = sorted(set(df["continent"]) - set(CONTINENT_ORDER))
    if unknown_continents:
        raise ValueError(f"Unexpected continent labels: {unknown_continents}")

    # The outer utility-scale-PV track defines the clockwise country order.
    return df.sort_values(
        ["utility", "distributed", "country_code"], ascending=True
    ).reset_index(drop=True)


def plot_deployment_ring(
    ax: plt.Axes, df: pd.DataFrame, *, legend_layout: str = "right"
) -> None:
    n = len(df)
    gap = np.deg2rad(18)
    arc_start = gap / 2
    arc_end = 2 * np.pi - gap / 2
    theta = np.linspace(arc_start, arc_end, n)
    spacing = (arc_end - arc_start) / (n - 1)
    width = spacing * 0.94
    country_label_radii = np.resize(np.array([1.92, 2.22]), n)
    track_max = {
        "utility": max(100.0, float(np.ceil(df["utility"].max() / 50.0) * 50.0)),
        "distributed": max(
            100.0, float(np.ceil(df["distributed"].max() / 50.0) * 50.0)
        ),
    }

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 2.30)
    ax.set_axis_off()

    for angle, label_radius in zip(theta, country_label_radii, strict=True):
        ax.plot(
            [angle, angle],
            [0.52, label_radius - 0.04],
            color="#D8D5CC",
            linewidth=0.12,
            zorder=0,
        )

    inner_bottom, inner_height = 0.54, 0.24
    for angle, continent in zip(theta, df["continent"], strict=True):
        ax.bar(
            angle,
            inner_height,
            width=width * 0.96,
            bottom=inner_bottom,
            color=CONTINENT_COLORS[continent],
            edgecolor="white",
            linewidth=0.10,
            align="center",
            zorder=1,
        )

    tracks = [
        ("distributed", 0.88, COLORS["distributed"], "Distributed PV"),
        ("utility", 1.40, COLORS["utility"], "Utility-scale PV"),
    ]
    track_span = 0.43
    arc = np.linspace(arc_start, arc_end, 500)
    for column, baseline, color, _label in tracks:
        rates = df[column].to_numpy(dtype=float)
        scale_max = track_max[column]
        radii = baseline + track_span * np.clip(rates, 0, scale_max) / scale_max
        ax.plot(
            arc,
            np.full_like(arc, baseline),
            color="#AFAAA0",
            linewidth=0.35,
            zorder=1,
        )
        ax.plot(
            arc,
            np.full_like(arc, baseline + track_span),
            color="#D8D5CC",
            linewidth=0.28,
            zorder=1,
        )
        reference_r = baseline + track_span * min(100.0, scale_max) / scale_max
        ax.plot(
            arc,
            np.full_like(arc, reference_r),
            color=color,
            linewidth=0.30,
            linestyle=(0, (1.0, 1.2)),
            alpha=0.55,
            zorder=1,
        )
        for angle, radius in zip(theta, radii, strict=True):
            ax.plot(
                [angle, angle],
                [baseline, radius],
                color=color,
                linewidth=0.38,
                solid_capstyle="round",
                zorder=2,
            )
        ax.scatter(theta, radii, s=1.4, facecolor=color, edgecolor="none", zorder=3)

    # Place both radial scales inside the deliberate gap at the top of the ring.
    scale_guides = [
        (
            gap * 0.42,
            0.88,
            COLORS["distributed"],
            track_max["distributed"],
            -0.030,
            "right",
        ),
        (
            -gap * 0.42,
            1.40,
            COLORS["utility"],
            track_max["utility"],
            0.030,
            "left",
        ),
    ]
    for scale_angle, baseline, color, scale_max, text_offset, ha in scale_guides:
        ax.plot(
            [scale_angle, scale_angle],
            [baseline, baseline + track_span],
            color=color,
            linewidth=0.32,
            zorder=4,
        )
        text_angle = scale_angle + text_offset
        for radius, value in [(baseline, 0), (baseline + track_span, scale_max)]:
            ax.plot(
                [scale_angle - 0.018, scale_angle + 0.018],
                [radius, radius],
                color=color,
                linewidth=0.32,
                zorder=4,
            )
            ax.text(
                text_angle,
                radius,
                f"{value:.0f}",
                ha=ha,
                va="center",
                fontsize=FONT_SIZE,
                color=color,
                fontweight="bold",
                zorder=5,
            )

    for angle, label_radius, code in zip(
        theta, country_label_radii, df["country_code"], strict=True
    ):
        angle_deg = np.degrees(angle)
        rotation = 90 - angle_deg
        ha = "left"
        if 90 < angle_deg < 270:
            rotation += 180
            ha = "right"
        ax.text(
            angle,
            label_radius,
            code,
            rotation=rotation,
            rotation_mode="anchor",
            ha=ha,
            va="center",
            fontsize=FONT_SIZE,
            color=COLORS["text"],
            clip_on=False,
        )

    ax.text(
        0,
        0,
        "Deployment\nrealization rate\n(%)",
        ha="center",
        va="center",
        fontsize=FONT_SIZE,
        fontweight="bold",
        linespacing=0.92,
        color=COLORS["text"],
    )

    if legend_layout == "right":
        data_legend_position = {
            "loc": "upper left",
            "bbox_to_anchor": (1.10, 0.86),
            "ncol": 1,
        }
        continent_legend_position = {
            "loc": "upper left",
            "bbox_to_anchor": (1.10, 0.58),
            "ncol": 1,
        }
    elif legend_layout == "below":
        data_legend_position = {
            "loc": "upper center",
            "bbox_to_anchor": (0.50, -0.10),
            "ncol": 2,
        }
        continent_legend_position = {
            "loc": "upper center",
            "bbox_to_anchor": (0.50, -0.19),
            "ncol": 3,
        }
    else:
        raise ValueError(f"Unsupported legend layout: {legend_layout}")

    data_handles = [
        mpl.lines.Line2D(
            [],
            [],
            color=COLORS["utility"],
            marker="o",
            markerfacecolor=COLORS["utility"],
            markeredgewidth=0,
            markersize=2.4,
            linewidth=0.7,
            label="Utility-scale PV",
        ),
        mpl.lines.Line2D(
            [],
            [],
            color=COLORS["distributed"],
            marker="o",
            markerfacecolor=COLORS["distributed"],
            markeredgewidth=0,
            markersize=2.4,
            linewidth=0.7,
            label="Distributed PV",
        ),
    ]
    legend_data = ax.legend(
        handles=data_handles,
        frameon=False,
        handlelength=1.2,
        handletextpad=0.35,
        borderpad=0,
        labelspacing=0.35,
        columnspacing=1.1,
        fontsize=FONT_SIZE,
        **data_legend_position,
    )
    ax.add_artist(legend_data)

    continent_handles = [
        mpl.patches.Patch(
            facecolor=CONTINENT_COLORS[name], edgecolor="none", label=name
        )
        for name in CONTINENT_ORDER
    ]
    ax.legend(
        handles=continent_handles,
        frameon=False,
        handlelength=0.65,
        handleheight=0.65,
        handletextpad=0.25,
        labelspacing=0.30,
        columnspacing=0.9,
        borderpad=0,
        fontsize=FONT_SIZE,
        **continent_legend_position,
    )


def add_bar_panels(fig: plt.Figure) -> tuple[plt.Axes, plt.Axes]:
    grid = fig.add_gridspec(
        1,
        2,
        left=0.155,
        right=0.985,
        bottom=0.080,
        top=0.480,
        wspace=0.130,
    )
    utility_ax = fig.add_subplot(grid[0, 0])
    distributed_ax = fig.add_subplot(grid[0, 1])

    utility_task = plot_gdp_top20_bar(
        utility_ax,
        "Centralized",
        xlim=X_LIMITS,
        value_label_fontsize=FONT_SIZE,
    )
    distributed_task = plot_gdp_top20_bar(
        distributed_ax,
        "Distributed",
        xlim=X_LIMITS,
        show_ylabels=False,
        value_label_fontsize=FONT_SIZE,
    )

    for ax in (utility_ax, distributed_ax):
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([0, 50, 100, 150, 200])

    fig.canvas.draw()
    title_y = utility_ax.get_position().y1 + 0.012
    utility_x = utility_ax.get_position().x0
    distributed_x = distributed_ax.get_position().x0
    for x, label, task in [
        (utility_x, "b", utility_task),
        (distributed_x, "c", distributed_task),
    ]:
        fig.text(
            x - 0.030,
            title_y,
            label,
            ha="left",
            va="bottom",
            fontsize=FONT_SIZE,
            fontweight="bold",
            color=COLORS["text"],
        )
        fig.text(
            x,
            title_y,
            f"GDP top 20 countries - {task.label}",
            ha="left",
            va="bottom",
            fontsize=FONT_SIZE,
            fontweight="bold",
            color=COLORS["text"],
        )

    fig.text(
        (utility_ax.get_position().x0 + distributed_ax.get_position().x1) / 2,
        0.035,
        "Deployment realization rate (%)",
        ha="center",
        va="center",
        fontsize=FONT_SIZE,
        color=COLORS["text"],
    )
    return utility_ax, distributed_ax


def draw_figure_4() -> None:
    set_style(FONT_SIZE)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    ring_df = load_deployment_ring_data()
    fig = plt.figure(figsize=(mm(FIGURE_WIDTH_MM), mm(FIGURE_HEIGHT_MM)))

    ring_bounds = [0.200, 0.545, 0.450, 0.438]
    ring_ax = fig.add_axes(ring_bounds, projection="polar")
    plot_deployment_ring(ring_ax, ring_df)
    fig.text(
        ring_bounds[0] - 0.040,
        ring_bounds[1] + ring_bounds[3] - 0.004,
        "a",
        ha="left",
        va="top",
        fontsize=FONT_SIZE,
        fontweight="bold",
        color=COLORS["text"],
    )

    add_bar_panels(fig)

    png_path = FIGURES_DIR / f"{FIGURE_BASENAME}.png"
    pdf_path = FIGURES_DIR / f"{FIGURE_BASENAME}.pdf"
    fig.savefig(png_path, dpi=PNG_DPI)
    fig.savefig(pdf_path, dpi=600)
    plt.close(fig)

    print(f"Ring countries: {len(ring_df)}")
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    draw_figure_4()
