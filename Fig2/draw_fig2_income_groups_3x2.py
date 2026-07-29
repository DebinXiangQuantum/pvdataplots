from __future__ import annotations

from collections import OrderedDict

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox

from fig2_common import COLORS, EXPORT_DIR, add_panel_label, load_scatter_data, marker_sizes, mm, set_style


FIG_WIDTH_MM = 180
FIG_HEIGHT_MM = 92
COUNTRY_LABEL_FONT_SIZE = 6
COUNTRY_LABEL_COUNT = 8

# GDP per capita groups requested for the Fig. 2 comparison.
INCOME_GROUPS = OrderedDict(
    [
        ("Low income", ("< US$5,000", "#0072b2")),
        ("Middle income", ("US$5,000-20,000", "#E6A45C")),
        ("High income", ("> US$20,000", "#d74b9b")),
    ]
)


def assign_income_groups(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.copy()
    income = grouped["gdp_pc_raw"].to_numpy(dtype=float)
    grouped["income_group_3"] = pd.Categorical(
        np.select(
            [
                income < 5_000,
                (income >= 5_000) & (income <= 20_000),
                income > 20_000,
            ],
            list(INCOME_GROUPS),
            default=None,
        ),
        categories=list(INCOME_GROUPS),
        ordered=True,
    )
    return grouped.dropna(subset=["income_group_3"])


def select_special_countries(subset: pd.DataFrame) -> pd.DataFrame:
    """Select all primary extremes, then fill with second-ranked extremes."""
    if subset.empty:
        return subset.copy()

    ranked = subset.copy()
    x = ranked["annual_irradiance"].to_numpy(dtype=float)
    y = ranked["PVCapPerCapita"].to_numpy(dtype=float)
    if len(ranked) > 1 and not np.allclose(x, x[0]):
        slope, intercept = np.polyfit(x, y, 1)
        ranked["_trend_residual"] = y - (slope * x + intercept)
    else:
        ranked["_trend_residual"] = y - np.nanmedian(y)

    ranking_rules = [
        ("capacity_gw", False),
        ("_trend_residual", False),
        ("_trend_residual", True),
        ("annual_irradiance", True),
        ("annual_irradiance", False),
        ("PVCapPerCapita", False),
        ("PVCapPerCapita", True),
    ]
    rankings = [
        ranked.sort_values(column, ascending=ascending).index.tolist()
        for column, ascending in ranking_rules
    ]
    chosen: list[int] = []
    target = min(COUNTRY_LABEL_COUNT, len(ranked))
    rank_position = 0
    while len(chosen) < target:
        for ranking in rankings:
            idx = ranking[rank_position]
            if idx not in chosen:
                chosen.append(idx)
            if len(chosen) >= target:
                break
        rank_position += 1
    return ranked.loc[chosen]


def _country_label(row: pd.Series) -> str:
    short_name = row.get("ShortName")
    if pd.notna(short_name) and str(short_name).strip():
        return str(short_name).strip().upper()
    return str(row["Nation"])


def _marker_bboxes(ax: plt.Axes, subset: pd.DataFrame, dpi: float) -> list[Bbox]:
    points = ax.transData.transform(
        subset[["annual_irradiance", "PVCapPerCapita"]].to_numpy(dtype=float)
    )
    # Matplotlib scatter sizes are marker areas in pt^2. The extra padding also
    # protects the white marker edge and the highlight ring around labelled points.
    radii = (np.sqrt(subset["marker_size"].to_numpy(dtype=float)) / 2 + 2.2) * dpi / 72
    return [
        Bbox.from_extents(x - radius, y - radius, x + radius, y + radius)
        for (x, y), radius in zip(points, radii, strict=True)
    ]


def _annotation_offsets() -> list[tuple[float, float]]:
    offsets: list[tuple[float, float]] = []
    directions = [
        (1.0, 0.72),
        (-1.0, 0.72),
        (1.0, -0.72),
        (-1.0, -0.72),
        (0.0, 1.0),
        (0.0, -1.0),
        (1.0, 0.0),
        (-1.0, 0.0),
    ]
    for distance in (7, 11, 16, 23, 32, 43, 54):
        offsets.extend((distance * dx, distance * dy) for dx, dy in directions)
    return offsets


def _text_alignment(dx: float, dy: float) -> tuple[str, str]:
    ha = "center" if np.isclose(dx, 0) else ("left" if dx > 0 else "right")
    va = "center" if np.isclose(dy, 0) else ("bottom" if dy > 0 else "top")
    return ha, va


def _candidate_text_bbox(
    anchor: np.ndarray,
    width: float,
    height: float,
    dx: float,
    dy: float,
    dpi: float,
) -> tuple[Bbox, str, str]:
    ha, va = _text_alignment(dx, dy)
    x, y = anchor + np.asarray([dx, dy]) * dpi / 72
    if ha == "left":
        x0, x1 = x, x + width
    elif ha == "right":
        x0, x1 = x - width, x
    else:
        x0, x1 = x - width / 2, x + width / 2
    if va == "bottom":
        y0, y1 = y, y + height
    elif va == "top":
        y0, y1 = y - height, y
    else:
        y0, y1 = y - height / 2, y + height / 2

    # Slightly more conservative than the visible 0.08-em white label pad.
    pad = 1.8 * dpi / 72
    return Bbox.from_extents(x0 - pad, y0 - pad, x1 + pad, y1 + pad), ha, va


def annotate_special_countries(
    fig: plt.Figure,
    ax: plt.Axes,
    subset: pd.DataFrame,
) -> list[mpl.text.Annotation]:
    selected = select_special_countries(subset)
    if selected.empty:
        return []

    renderer = fig.canvas.get_renderer()
    dpi = fig.dpi
    font = FontProperties(family="Arial", size=COUNTRY_LABEL_FONT_SIZE)
    point_boxes = _marker_bboxes(ax, subset, dpi)
    axes_box = ax.get_window_extent(renderer)
    boundary_pad = 0.8 * dpi / 72
    fixed_boxes = [
        text.get_window_extent(renderer).padded(0.6 * dpi / 72)
        for text in ax.texts
        if text.get_visible()
    ]
    annotations: list[mpl.text.Annotation] = []

    # ISO-3 labels keep the visual density low and the leader lines short.
    selected = selected.assign(_display_label=selected.apply(_country_label, axis=1))
    label_specs: list[tuple[pd.Series, str, list[tuple[float, float, str, str, Bbox]]]] = []
    for _, row in selected.iterrows():
        label = str(row["_display_label"])
        width, height, _ = renderer.get_text_width_height_descent(label, font, ismath=False)
        anchor = ax.transData.transform([row["annual_irradiance"], row["PVCapPerCapita"]])
        candidates: list[tuple[float, float, str, str, Bbox]] = []
        for dx, dy in _annotation_offsets():
            candidate, ha, va = _candidate_text_bbox(anchor, width, height, dx, dy, dpi)
            inside_axes = (
                candidate.x0 >= axes_box.x0 + boundary_pad
                and candidate.x1 <= axes_box.x1 - boundary_pad
                and candidate.y0 >= axes_box.y0 + boundary_pad
                and candidate.y1 <= axes_box.y1 - boundary_pad
            )
            if not inside_axes:
                continue
            if any(candidate.overlaps(box) for box in point_boxes):
                continue
            if any(candidate.overlaps(box) for box in fixed_boxes):
                continue
            candidates.append((dx, dy, ha, va, candidate))

        if not candidates:
            raise RuntimeError(f"No collision-free country-label candidates found for {label}")
        label_specs.append((row, label, candidates))

    placements: dict[int, tuple[float, float, str, str, Bbox]] = {}
    search_states = 0

    def solve(unplaced: tuple[int, ...], occupied: list[Bbox]) -> bool:
        nonlocal search_states
        search_states += 1
        if search_states > 200_000:
            return False
        if not unplaced:
            return True

        compatible: list[tuple[int, list[tuple[float, float, str, str, Bbox]]]] = []
        for spec_idx in unplaced:
            options = [
                candidate
                for candidate in label_specs[spec_idx][2]
                if not any(candidate[4].overlaps(box) for box in occupied)
            ]
            if not options:
                return False
            compatible.append((spec_idx, options))

        # Place the most constrained label first, then prefer the shortest leader.
        spec_idx, options = min(compatible, key=lambda item: len(item[1]))
        remaining = tuple(idx for idx in unplaced if idx != spec_idx)
        for candidate in options:
            placements[spec_idx] = candidate
            if solve(remaining, [*occupied, candidate[4]]):
                return True
        placements.pop(spec_idx, None)
        return False

    if not solve(tuple(range(len(label_specs))), []):
        raise RuntimeError("No global collision-free layout found for all special-country labels")

    for spec_idx, (row, label, _) in enumerate(label_specs):
        placement = placements[spec_idx]
        dx, dy, ha, va, candidate = placement
        marker_size = float(row["marker_size"])
        ax.scatter(
            [row["annual_irradiance"]],
            [row["PVCapPerCapita"]],
            s=[marker_size + 14],
            facecolors="none",
            edgecolors="#333333",
            linewidths=0.45,
            zorder=5,
        )
        leader = ax.annotate(
            "",
            xy=(row["annual_irradiance"], row["PVCapPerCapita"]),
            xytext=(dx, dy),
            textcoords="offset points",
            arrowprops={
                "arrowstyle": "-",
                "color": "#555555",
                "linewidth": 0.35,
                "shrinkA": 1.2,
                "shrinkB": max(np.sqrt(marker_size) / 2, 2.0),
            },
            annotation_clip=False,
            zorder=2.5,
        )
        leader.set_gid("country-leader")
        annotation = ax.annotate(
            label,
            xy=(row["annual_irradiance"], row["PVCapPerCapita"]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=COUNTRY_LABEL_FONT_SIZE,
            color=COLORS["text"],
            bbox={
                "boxstyle": "square,pad=0.08",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.90,
            },
            annotation_clip=False,
            zorder=6,
        )
        annotation.set_gid("country-label")
        annotations.append(annotation)
    return annotations


def validate_country_label_layout(
    fig: plt.Figure,
    panels: dict[plt.Axes, tuple[pd.DataFrame, list[mpl.text.Annotation]]],
) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    errors: list[str] = []
    for ax, (subset, annotations) in panels.items():
        axes_box = ax.get_window_extent(renderer)
        point_boxes = _marker_bboxes(ax, subset, fig.dpi)
        label_boxes = [annotation.get_bbox_patch().get_window_extent(renderer) for annotation in annotations]
        fixed_text_boxes = [
            text.get_window_extent(renderer)
            for text in ax.texts
            if text.get_visible() and text.get_gid() not in {"country-label", "country-leader"}
        ]
        for idx, label_box in enumerate(label_boxes):
            if (
                label_box.x0 < axes_box.x0
                or label_box.x1 > axes_box.x1
                or label_box.y0 < axes_box.y0
                or label_box.y1 > axes_box.y1
            ):
                errors.append("country label outside panel boundary")
            if any(label_box.overlaps(point_box) for point_box in point_boxes):
                errors.append("country label overlaps a scatter point")
            if any(label_box.overlaps(text_box) for text_box in fixed_text_boxes):
                errors.append("country label overlaps panel text")
            if any(label_box.overlaps(other) for other in label_boxes[idx + 1 :]):
                errors.append("country labels overlap each other")
    if errors:
        raise RuntimeError("; ".join(sorted(set(errors))))


def draw_size_key(
    ax: plt.Axes,
    values: tuple[float, float, float],
    capacity_reference: pd.Series,
    pv_type: str,
) -> None:
    sizes = marker_sizes(pd.Series(values), reference_capacity=capacity_reference)
    y_positions = np.array([0.63, 0.37, 0.12])
    ax.scatter(
        np.full(len(values), 0.24),
        y_positions,
        s=sizes,
        color="#3b3b3b",
        edgecolors="none",
        zorder=2,
    )
    ax.text(0.0, 0.97, pv_type, ha="left", va="top", fontsize=6, fontweight="bold")
    ax.text(0.0, 0.81, "PV capacity (GW)", ha="left", va="top", fontsize=6)
    for y, value in zip(y_positions, values, strict=True):
        ax.text(0.48, y, f"{value:g}", ha="left", va="center", fontsize=6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()


def plot_group_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    group_name: str,
    color: str,
    y_limits: tuple[float, float],
    panel_label: str,
    show_ylabel: bool,
    show_xlabels: bool,
) -> None:
    subset = data[data["income_group_3"] == group_name]
    ax.scatter(
        subset["annual_irradiance"],
        subset["PVCapPerCapita"],
        s=subset["marker_size"],
        color=color,
        alpha=0.88,
        edgecolors="white",
        linewidths=0.32,
        zorder=3,
    )
    ax.axhline(0, color="#8a8a8a", linewidth=0.55, zorder=1)
    ax.grid(axis="y", color="#e7e7e7", linewidth=0.35, zorder=0)
    ax.set_xlim(2_300, 9_300)
    ax.set_xticks([3_000, 6_000, 9_000])
    ax.set_ylim(*y_limits)
    ax.tick_params(length=2, pad=1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(0.5)
    add_panel_label(ax, panel_label, x=0.02, y=0.97, fontsize=8)
    # ax.text(
    #     0.98,
    #     0.95,
    #     f"n = {len(subset)}",
    #     transform=ax.transAxes,
    #     ha="right",
    #     va="top",
    #     fontsize=6,
    #     color="#555555",
    #     bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.35, "alpha": 0.85},
    #     zorder=5,
    # )
    if show_ylabel:
        ax.set_ylabel("PV capacity per capita\n(log$_{10}$ MW per 10,000 people)", labelpad=2)
    else:
        ax.tick_params(axis="y", left=False, labelleft=False)
    if not show_xlabels:
        ax.tick_params(labelbottom=False)


def y_limits(data: pd.DataFrame) -> tuple[float, float]:
    values = data["PVCapPerCapita"].to_numpy(dtype=float)
    span = float(np.nanmax(values) - np.nanmin(values))
    bottom_padding = max(span * 0.18, 0.3)
    # The upper band is intentionally larger: most high-adoption countries sit
    # near the top of the range, and this keeps their direct labels close by.
    top_padding = max(span * 0.24, 0.45)
    return float(np.nanmin(values) - bottom_padding), float(np.nanmax(values) + top_padding)


def main() -> None:
    set_style(6)
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
        }
    )
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    datasets = {
        "Utility-scale PV": assign_income_groups(load_scatter_data("utility")),
        "Distributed PV": assign_income_groups(load_scatter_data("distributed")),
    }
    for data in datasets.values():
        data["marker_size"] = marker_sizes(data["capacity_gw"], reference_capacity=data["capacity_gw"])

    titles = list(datasets)
    group_items = list(INCOME_GROUPS.items())
    # Use the final PNG resolution for collision-aware annotation placement.
    fig = plt.figure(figsize=(mm(FIG_WIDTH_MM), mm(FIG_HEIGHT_MM)), dpi=300, constrained_layout=False)

    # Exact geometry keeps the scientific comparison grid visually disciplined at 180 mm.
    left_mm = 19.0
    panel_width_mm = 44.0
    column_gap_mm = 4.0
    panel_height_mm = 29.0
    top_row_y_mm = 48.0
    bottom_row_y_mm = 12.0
    key_x_mm = 163.0
    key_width_mm = 12.0
    axes: dict[tuple[int, int], plt.Axes] = {}
    for row, y in enumerate((top_row_y_mm, bottom_row_y_mm)):
        for col in range(3):
            x = left_mm + col * (panel_width_mm + column_gap_mm)
            axes[row, col] = fig.add_axes(
                [x / FIG_WIDTH_MM, y / FIG_HEIGHT_MM, panel_width_mm / FIG_WIDTH_MM, panel_height_mm / FIG_HEIGHT_MM]
            )

    for col, (group_name, (range_label, color)) in enumerate(group_items):
        center_x = (left_mm + col * (panel_width_mm + column_gap_mm) + panel_width_mm / 2) / FIG_WIDTH_MM
        fig.text(
            center_x,
            87 / FIG_HEIGHT_MM,
            group_name,
            ha="center",
            va="bottom",
            fontsize=6,
            fontweight="bold",
            color=COLORS["text"],
        )
        fig.add_artist(
            Line2D(
                [center_x - 0.045, center_x + 0.045],
                [85.2 / FIG_HEIGHT_MM, 85.2 / FIG_HEIGHT_MM],
                transform=fig.transFigure,
                color=color,
                linewidth=1.8,
                solid_capstyle="butt",
            )
        )
        fig.text(
            center_x,
            81.7 / FIG_HEIGHT_MM,
            range_label,
            ha="center",
            va="bottom",
            fontsize=6,
            color=COLORS["text"],
        )

    row_heading_specs = [
        ("Utility-scale PV", 78.8),
        ("Distributed PV", 43.0),
    ]
    for heading, y in row_heading_specs:
        fig.text(
            left_mm / FIG_WIDTH_MM,
            y / FIG_HEIGHT_MM,
            heading,
            ha="left",
            va="bottom",
            fontsize=6,
            fontweight="bold",
            color=COLORS["text"],
        )

    for row, title in enumerate(titles):
        data = datasets[title]
        limits = y_limits(data)
        for col, (group_name, (_, color)) in enumerate(group_items):
            plot_group_panel(
                axes[row, col],
                data,
                group_name,
                color,
                limits,
                panel_label=chr(ord("a") + row * 3 + col),
                show_ylabel=col == 0,
                show_xlabels=row == 1,
            )

    grid_center_x = (left_mm + (3 * panel_width_mm + 2 * column_gap_mm) / 2) / FIG_WIDTH_MM
    fig.text(
        grid_center_x,
        0.030,
        "Annual solar irradiance (MJ m$^{-2}$ yr$^{-1}$)",
        ha="center",
        va="bottom",
        fontsize=6,
        color=COLORS["text"],
    )

    utility_key = fig.add_axes([key_x_mm / FIG_WIDTH_MM, 55 / FIG_HEIGHT_MM, key_width_mm / FIG_WIDTH_MM, 22 / FIG_HEIGHT_MM])
    distributed_key = fig.add_axes([key_x_mm / FIG_WIDTH_MM, 19 / FIG_HEIGHT_MM, key_width_mm / FIG_WIDTH_MM, 22 / FIG_HEIGHT_MM])
    draw_size_key(utility_key, (1, 30, 300), datasets["Utility-scale PV"]["capacity_gw"], "Utility-scale")
    draw_size_key(distributed_key, (0.1, 5, 25), datasets["Distributed PV"]["capacity_gw"], "Distributed")

    # Place labels only after all fixed panel text exists, so panel letters and
    # sample-size labels participate in the collision checks.
    fig.canvas.draw()
    annotated_panels: dict[plt.Axes, tuple[pd.DataFrame, list[mpl.text.Annotation]]] = {}
    for row, title in enumerate(titles):
        data = datasets[title]
        for col, (group_name, _) in enumerate(group_items):
            subset = data[data["income_group_3"] == group_name]
            annotations = annotate_special_countries(fig, axes[row, col], subset)
            annotated_panels[axes[row, col]] = (subset, annotations)
    validate_country_label_layout(fig, annotated_panels)

    out_pdf = EXPORT_DIR / "Fig3_income_groups_3x2.pdf"
    out_png = EXPORT_DIR / "Fig3_income_groups_3x2.png"
    fig.savefig(out_pdf, dpi=600)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
