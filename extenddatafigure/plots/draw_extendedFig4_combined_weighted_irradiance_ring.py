from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from extended_data_common import (
    COLORS,
    FIGURES_DIR,
    load_pv_irradiance_income_data,
    mm,
    set_style,
)


FIGURE_BASENAME = "extendedFig4-combined-weighted-irradiance-ring"
VALUE_MAX = 9_000.0
GAP_DEGREES = 20.0

RING_CONFIG = {
    "utility": {
        "label": "Utility-scale PV",
        "color": COLORS["utility"],
        "base": 1.49,
    },
    "distributed": {
        "label": "Distributed PV",
        "color": COLORS["distributed"],
        "base": 0.75,
    },
}
RING_HEIGHT = 0.60

FOCUS_COUNTRIES = {
    "AUSTRALIA",
    "CHILE",
    "CHINA",
    "GERMANY",
    "INDIA",
    "JAPAN",
    "MALTA",
    "NAMIBIA",
    "NETHERLANDS",
    "UNITED STATES",
}

SHORT_COUNTRY_LABELS = {
    "BOSNIA AND HERZEGOVINA": "Bosnia & Herz.",
    "CENTRAL AFRICAN REPUBLIC": "Central African Rep.",
    "CONGO": "Congo Rep.",
    "CONGO,THE DEMOCRATIC REPUBLIC OF THE": "DR Congo",
    "KOREA,DEMOCRATIC PEOPLE'S REPUBLIC OF": "North Korea",
    "KOREA,DEMOCRATIC PEOPLE’S REPUBLIC OF": "North Korea",
    "KOREA, REPUBLIC OF": "South Korea",
    "MACEDONIA,THE FORMER YUGOSLAV REPUBLIC OF": "North Macedonia",
    "MICRONESIA,FEDERATED STATES OF": "Micronesia",
    "MOLDOVA, REPUBLIC OF": "Moldova",
    "PALESTINE, STATE OF": "Palestine",
    "SAINT VINCENT AND THE GRENADINES": "St Vincent & Gren.",
    "SRILANKA": "Sri Lanka",
    "SYRIAN ARAB REPUBLIC": "Syria",
    "TANZANIA, UNITED REPUBLIC OF": "Tanzania",
    "VENEZUELA, BOLIVARIAN REPUBLIC OF": "Venezuela",
    "VIRGIN ISLANDS,U.S.": "U.S. Virgin Is.",
}


def load_combined_data() -> pd.DataFrame:
    utility = load_pv_irradiance_income_data("utility")[
        ["country", "display_country", "weighted_irradiance_mj_m2", "capacity_gw"]
    ].rename(
        columns={
            "weighted_irradiance_mj_m2": "utility_irradiance",
            "capacity_gw": "utility_capacity_gw",
        }
    )
    distributed = load_pv_irradiance_income_data("distributed")[
        ["country", "weighted_irradiance_mj_m2", "capacity_gw"]
    ].rename(
        columns={
            "weighted_irradiance_mj_m2": "distributed_irradiance",
            "capacity_gw": "distributed_capacity_gw",
        }
    )

    combined = utility.merge(distributed, on="country", how="inner", validate="one_to_one")
    if len(combined) != len(utility) or len(combined) != len(distributed):
        raise ValueError("Utility-scale and distributed PV country sets do not match.")

    combined["paired_mean_irradiance"] = combined[
        ["utility_irradiance", "distributed_irradiance"]
    ].mean(axis=1)
    combined = combined.sort_values(
        ["paired_mean_irradiance", "country"], ascending=[False, True]
    ).reset_index(drop=True)
    combined["short_country"] = combined.apply(
        lambda row: SHORT_COUNTRY_LABELS.get(row["country"], row["display_country"]),
        axis=1,
    )
    return combined


def installed_capacity_weighted_mean(df: pd.DataFrame, kind: str) -> float:
    value_col = f"{kind}_irradiance"
    capacity_col = f"{kind}_capacity_gw"
    mask = df[capacity_col] > 0
    return float(np.average(df.loc[mask, value_col], weights=df.loc[mask, capacity_col]))


def radial_height(values: pd.Series | np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.clip(values / VALUE_MAX, 0, 1) * RING_HEIGHT


def label_rotation(theta: float) -> tuple[float, str]:
    screen_angle = 90.0 - np.degrees(theta)
    if screen_angle < -90.0:
        return screen_angle + 180.0, "right"
    if screen_angle > 90.0:
        return screen_angle - 180.0, "right"
    return screen_angle, "left"


def add_country_labels(ax: plt.Axes, df: pd.DataFrame, theta: np.ndarray) -> None:
    label_radius = RING_CONFIG["utility"]["base"] + RING_HEIGHT + 0.075
    guide_inner = RING_CONFIG["utility"]["base"] + RING_HEIGHT + 0.012

    for angle, row in zip(theta, df.itertuples(index=False), strict=True):
        rotation, horizontal_alignment = label_rotation(angle)
        is_focus = row.country in FOCUS_COUNTRIES

        ax.plot(
            [angle, angle],
            [guide_inner, label_radius - 0.015],
            color="#c8cbd0" if not is_focus else "#7b7f86",
            linewidth=0.22 if not is_focus else 0.38,
            zorder=2,
        )
        ax.text(
            angle,
            label_radius,
            row.short_country,
            rotation=rotation,
            rotation_mode="anchor",
            ha=horizontal_alignment,
            va="center",
            fontsize=6,
            fontweight="normal" if not is_focus else "bold",
            color="#5f6368" if not is_focus else COLORS["text"],
            clip_on=False,
            zorder=7,
        )


def add_start_radial_axes(ax: plt.Axes, start_theta: float) -> None:
    tick_values = (0, 3_000, 6_000, 9_000)
    tick_labels = ("0", "3,000", "6,000", "9,000")
    tick_angles = start_theta + np.deg2rad([-0.7, 0.7])

    for kind in ("distributed", "utility"):
        config = RING_CONFIG[kind]
        base = config["base"]
        color = config["color"]

        ax.plot(
            [start_theta, start_theta],
            [base, base + RING_HEIGHT],
            color=color,
            linewidth=0.62,
            zorder=8,
        )
        for value, label in zip(tick_values, tick_labels, strict=True):
            radius = base + radial_height([value])[0]
            ax.plot(
                tick_angles,
                [radius, radius],
                color=color,
                linewidth=0.62,
                zorder=8,
            )
            ax.annotate(
                label,
                xy=(start_theta, radius),
                xytext=(-4, 0),
                textcoords="offset points",
                ha="right",
                va="center",
                fontsize=6,
                color="#55585d",
                annotation_clip=False,
                zorder=9,
            )


def draw_ring_figure() -> None:
    set_style(6)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pdf_dir = FIGURES_DIR / "extendedpdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    df = load_combined_data()
    n_countries = len(df)
    gap = np.deg2rad(GAP_DEGREES)
    theta_edges = np.linspace(0, 2 * np.pi - gap, n_countries + 1)
    theta = (theta_edges[:-1] + theta_edges[1:]) / 2
    bar_width = np.diff(theta_edges) * 0.93

    fig = plt.figure(figsize=(mm(180), mm(180)))
    ax = fig.add_axes([0.015, 0.015, 0.97, 0.97], projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 2.88)
    ax.set_axis_off()

    for kind in ("distributed", "utility"):
        config = RING_CONFIG[kind]
        base = config["base"]
        values = df[f"{kind}_irradiance"]

        ax.bar(
            theta,
            np.full(n_countries, RING_HEIGHT),
            width=bar_width,
            bottom=base,
            color="#f1f2f4",
            edgecolor="white",
            linewidth=0.18,
            align="center",
            zorder=1,
        )
        ax.bar(
            theta,
            radial_height(values),
            width=bar_width,
            bottom=base,
            color=config["color"],
            edgecolor="white",
            linewidth=0.18,
            alpha=0.88,
            align="center",
            zorder=3,
        )

        weighted_mean = installed_capacity_weighted_mean(df, kind)
        mean_radius = base + radial_height([weighted_mean])[0]
        mean_theta = np.linspace(0, 2 * np.pi - gap, 700)
        ax.plot(
            mean_theta,
            np.full_like(mean_theta, mean_radius),
            color="#202124",
            linewidth=0.52,
            linestyle=(0, (2.2, 1.5)),
            zorder=5,
        )

    separator_theta = np.linspace(0, 2 * np.pi - gap, 700)
    for radius in (
        RING_CONFIG["distributed"]["base"],
        RING_CONFIG["distributed"]["base"] + RING_HEIGHT,
        RING_CONFIG["utility"]["base"],
        RING_CONFIG["utility"]["base"] + RING_HEIGHT,
    ):
        ax.plot(
            separator_theta,
            np.full_like(separator_theta, radius),
            color="#d7d9dd",
            linewidth=0.35,
            zorder=4,
        )

    add_country_labels(ax, df, theta)
    add_start_radial_axes(ax, theta_edges[0])

    utility_mean = installed_capacity_weighted_mean(df, "utility")
    distributed_mean = installed_capacity_weighted_mean(df, "distributed")
    legend_handles = [
        Patch(
            facecolor=RING_CONFIG["utility"]["color"],
            edgecolor="none",
            alpha=0.88,
            label=f"Utility-scale PV ",
        ),
        Patch(
            facecolor=RING_CONFIG["distributed"]["color"],
            edgecolor="none",
            alpha=0.88,
            label=f"Distributed PV",
        ),
        Line2D(
            [0],
            [0],
            color="#202124",
            linewidth=0.6,
            linestyle=(0, (2.2, 1.5)),
            label="Installed-capacity weighted mean",
        ),
    ]

    ax.text(
        0.5,
        0.535,
        "PV-weighted solar irradiance",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=6,
        fontweight="bold",
        color=COLORS["text"],
    )
    # ax.text(
    #     0.5,
    #     0.535,
    #     "Weighted mean annual solar irradiance\n(MJ m$^{-2}$ yr$^{-1}$)",
    #     transform=ax.transAxes,
    #     ha="center",
    #     va="center",
    #     fontsize=4.8,
    #     linespacing=1.2,
    #     color="#55585d",
    # )
    ax.legend(
        handles=legend_handles,
        loc="center",
        bbox_to_anchor=(0.5, 0.475),
        frameon=False,
        fontsize=6,
        handlelength=1.5,
        handletextpad=0.55,
        labelspacing=0.35,
        borderaxespad=0,
    )

    png_path = FIGURES_DIR / f"{FIGURE_BASENAME}.png"
    pdf_path = pdf_dir / f"{FIGURE_BASENAME}.pdf"
    fig.savefig(png_path, dpi=600, facecolor="white")
    fig.savefig(pdf_path, dpi=600, facecolor="white")
    plt.close(fig)
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    draw_ring_figure()
