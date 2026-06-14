from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
FIG3_DIR = ROOT / "Fig3"
FIG4_DIR = ROOT / "Fig4"
DATA_XLSX = FIG3_DIR / "全球总装机 0518.xlsx"
OUT_DIR = FIG4_DIR / "figures"

MM = 1 / 25.4
FIG_WIDTH_MM = 180
FIG_HEIGHT_MM = 75

COLORS = {
    "utility": "#2F6F9F",
    "distributed": "#7BBF7A",
    "land": "#F6F5F0",
    "border": "#C9C7BE",
    "grid": "#D7D5CE",
    "axis": "#4F4F4F",
    "text": "#202020",
    "increase": "#D85F8D",
    "decrease": "#2F6F9F",
    "neutral": "#F7F4EA",
    "actual": "#D64F4B",
}

CHANGE_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "fig4_signed_relative_change",
    [COLORS["decrease"], COLORS["neutral"], COLORS["increase"]],
)


@dataclass(frozen=True)
class Scenario:
    name: str
    title: str
    group: str


SCENARIOS = [
    Scenario("CHN-Global", "CHN-Global", "Single-model"),
    Scenario("USA-Global", "USA-Global", "Single-model"),
    Scenario("DEU-Global", "DEU-Global", "Single-model"),
    Scenario("EU-Global", "EU-Global", "Single-model"),
    Scenario("Cont-Lead", "Cont-Lead", "Regional-leader"),
    Scenario("Subcont-Lead", "Subcont-Lead", "Regional-leader"),
    Scenario("Clus5-Lead", "Clus5-Lead", "Cluster-leader"),
    Scenario("Clus10-Lead", "Clus10-Lead", "Cluster-leader"),
    Scenario("Clus20-Lead", "Clus20-Lead", "Cluster-leader"),
]
SCENARIO_LABELS = [scenario.title for scenario in SCENARIOS]

COUNTRY_CODE = {
    "ARGENTINA": "ARG",
    "AUSTRALIA": "AUS",
    "AUSTRIA": "AUT",
    "BANGLADESH": "BGD",
    "BRAZIL": "BRA",
    "CANADA": "CAN",
    "CANARIAS": "CNR",
    "CHILE": "CHL",
    "CHINA": "CHN",
    "COLOMBIA": "COL",
    "CZECH REPUBLIC": "CZE",
    "EGYPT": "EGY",
    "FINLAND": "FIN",
    "FRANCE": "FRA",
    "GERMANY": "DEU",
    "INDIA": "IND",
    "INDONESIA": "IDN",
    "IRAN": "IRN",
    "IRAQ": "IRQ",
    "ITALY": "ITA",
    "JAPAN": "JPN",
    "KAZAKHSTAN": "KAZ",
    "KOREA, REPUBLIC OF": "KOR",
    "MALAYSIA": "MYS",
    "MEXICO": "MEX",
    "NORWAY": "NOR",
    "PAKISTAN": "PAK",
    "POLAND": "POL",
    "RUSSIAN FEDERATION": "RUS",
    "SAUDI ARABIA": "SAU",
    "SOUTH AFRICA": "ZAF",
    "SPAIN": "ESP",
    "SUDAN": "SDN",
    "SWEDEN": "SWE",
    "THAILAND": "THA",
    "TURKEY": "TUR",
    "UKRAINE": "UKR",
    "UNITED ARAB EMIRATES": "ARE",
    "UNITED KINGDOM": "GBR",
    "UNITED STATES": "USA",
    "VIET NAM": "VNM",
}


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 6,
            "axes.labelsize": 6,
            "axes.titlesize": 6,
            "axes.linewidth": 0.5,
            "axes.edgecolor": COLORS["axis"],
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.fontsize": 6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def country_key(values: pd.Series) -> pd.Series:
    return values.astype(str).str.upper().str.strip()


def load_country_data(kind: str) -> pd.DataFrame:
    if kind not in {"utility", "distributed"}:
        raise ValueError("kind must be 'utility' or 'distributed'")

    sheet = "集中式" if kind == "utility" else "分布式"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        raw = pd.read_excel(DATA_XLSX, sheet_name=sheet, header=None).iloc[3:].copy()
    raw = raw[raw[0].notna()].copy()
    raw["country_key"] = country_key(raw[0])
    raw = raw[~raw["country_key"].isin({"TOTAL", "NAN"})].copy()

    df = pd.DataFrame(
        {
            "country_key": raw["country_key"],
            "actual_capacity": pd.to_numeric(raw[2], errors="coerce"),
        }
    )

    for index, scenario in enumerate(SCENARIOS):
        if kind == "utility":
            cap = pd.to_numeric(raw[5 + index * 2], errors="coerce")
        else:
            cap = pd.to_numeric(raw[5 + index], errors="coerce")

        df[f"{scenario.name}_capacity"] = cap

    return df


def make_total_country_data(utility_df: pd.DataFrame, distributed_df: pd.DataFrame) -> pd.DataFrame:
    utility_cols = ["country_key", "actual_capacity"] + [f"{s.name}_capacity" for s in SCENARIOS]
    distributed_cols = ["country_key", "actual_capacity"] + [f"{s.name}_capacity" for s in SCENARIOS]
    merged = utility_df[utility_cols].merge(
        distributed_df[distributed_cols],
        on="country_key",
        how="outer",
        suffixes=("_utility", "_distributed"),
    )

    out = pd.DataFrame({"country_key": merged["country_key"]})
    out["actual_capacity"] = merged[["actual_capacity_utility", "actual_capacity_distributed"]].sum(axis=1, min_count=1)
    actual = out["actual_capacity"].where(out["actual_capacity"] > 0)

    for scenario in SCENARIOS:
        cap_cols = [f"{scenario.name}_capacity_utility", f"{scenario.name}_capacity_distributed"]
        out[f"{scenario.name}_capacity"] = merged[cap_cols].sum(axis=1, min_count=1)
        out[f"{scenario.name}_change"] = out[f"{scenario.name}_capacity"] - out["actual_capacity"]
        out[f"{scenario.name}_relative_change"] = out[f"{scenario.name}_change"] / actual

    return out


def capacity_columns() -> list[str]:
    return [f"{scenario.name}_capacity" for scenario in SCENARIOS]


def change_columns() -> list[str]:
    return [f"{scenario.name}_change" for scenario in SCENARIOS]


def relative_columns() -> list[str]:
    return [f"{scenario.name}_relative_change" for scenario in SCENARIOS]


def country_label(country: str) -> str:
    return COUNTRY_CODE.get(country, country[:3].upper())


def signed_log_transform(values: np.ndarray | pd.Series) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.sign(arr) * np.log10(1 + np.abs(arr))


def signed_log_norm(values: pd.Series) -> tuple[mcolors.TwoSlopeNorm, float]:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        max_abs = 1.0
    else:
        transformed = np.abs(signed_log_transform(clean.to_numpy()))
        max_abs = float(np.nanpercentile(transformed, 98))
        max_abs = max(0.6, min(4.0, max_abs))
    return mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs), max_abs


def color_for_relative(relative: float, norm: mcolors.Normalize) -> tuple[float, float, float, float]:
    if not np.isfinite(relative):
        return mcolors.to_rgba("#ECE9DF")
    return CHANGE_CMAP(norm(signed_log_transform(np.array([relative]))[0]))


def format_change_tick(value: float) -> str:
    if value == -1:
        return "-100%"
    if value == -0.5:
        return "-50%"
    if value == 0:
        return "0"
    if value == 1:
        return "+100%"
    return f"+{value:g}x"


def draw_panel_label(fig: plt.Figure, x: float, y: float, label: str) -> None:
    fig.text(
        x,
        y,
        label,
        ha="left",
        va="top",
        fontsize=8,
        fontweight="bold",
        color=COLORS["text"],
    )


def add_shared_guides(
    fig: plt.Figure,
    norm: mcolors.Normalize,
    heatmap_bounds: list[float],
    dot_scale: float,
    dot_ref_values: list[float],
) -> None:
    legend_y = heatmap_bounds[1] + heatmap_bounds[3] + 0.020
    size_group = [
        heatmap_bounds[0] + heatmap_bounds[2] * 0.210,
        legend_y,
        heatmap_bounds[2] * 0.255,
        0.110,
    ]
    color_group = [
        heatmap_bounds[0] + heatmap_bounds[2] * 0.520,
        legend_y,
        heatmap_bounds[2] * 0.340,
        0.110,
    ]

    colorbar_ax = fig.add_axes(
        [
            color_group[0],
            color_group[1] + color_group[3] * 0.76,
            color_group[2],
            0.010,
        ]
    )
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=CHANGE_CMAP)
    cb = fig.colorbar(sm, cax=colorbar_ax, orientation="horizontal")
    raw_ticks = [-1, 0, 10, 100]
    tick_positions = signed_log_transform(np.array(raw_ticks, dtype=float))
    valid = [i for i, tick in enumerate(tick_positions) if norm.vmin <= tick <= norm.vmax]
    cb.set_ticks([tick_positions[i] for i in valid])
    cb.set_ticklabels([format_change_tick(raw_ticks[i]) for i in valid])
    labels = cb.ax.get_xticklabels()
    if len(labels) >= 2:
        labels[0].set_ha("right")
        labels[1].set_ha("left")
    cb.outline.set_linewidth(0.35)
    cb.ax.tick_params(length=1.4, pad=0.5, width=0.35, labelsize=6)
    cb.set_label("")
    fig.text(
        color_group[0] + color_group[2] / 2,
        color_group[1] + color_group[3] * 0.04,
        "Relative change vs actual",
        ha="center",
        va="bottom",
        fontsize=6,
        color=COLORS["text"],
    )

    size_ax = fig.add_axes(size_group)
    size_ax.axis("off")
    size_ax.text(
        0.5,
        0.04,
        "Absolute change (GW)",
        ha="center",
        va="bottom",
        fontsize=6,
        color=COLORS["text"],
    )
    x_positions = np.linspace(0.28, 0.72, len(dot_ref_values))
    for i, value in enumerate(dot_ref_values):
        x = float(x_positions[i])
        size_ax.scatter(
            [x],
            [0.82],
            s=max(4, np.sqrt(value / dot_scale) * 42),
            facecolor="white",
            edgecolor="#3A3A3A",
            linewidth=0.35,
            zorder=3,
        )
        size_ax.text(x, 0.52, f"{value:g}", ha="center", va="center", fontsize=6, color=COLORS["text"])
    size_ax.set_xlim(0, 1)
    size_ax.set_ylim(0, 1)


def plot_heatmap(
    fig: plt.Figure,
    ax: plt.Axes,
    country_df: pd.DataFrame,
    countries: list[str],
    norm: mcolors.Normalize,
) -> float:
    rel = np.vstack([country_df.set_index("country_key").loc[countries, f"{scenario.name}_relative_change"] for scenario in SCENARIOS])
    abs_change = np.abs(
        np.vstack([country_df.set_index("country_key").loc[countries, f"{scenario.name}_change"] for scenario in SCENARIOS])
    )
    color_values = signed_log_transform(rel)

    n_scenarios, n_countries = rel.shape
    x_edges = np.arange(n_countries + 1)
    y_edges = np.arange(n_scenarios + 1)

    ax.pcolormesh(
        x_edges,
        y_edges,
        color_values,
        cmap=CHANGE_CMAP,
        norm=norm,
        edgecolors="white",
        linewidth=0.28,
        antialiased=True,
        zorder=1,
    )

    dot_scale = max(1.0, float(np.nanpercentile(abs_change, 95)))
    dot_sizes = np.sqrt(np.clip(abs_change, 0, dot_scale) / dot_scale) * 42
    dot_sizes = np.where(abs_change > 0, np.maximum(dot_sizes, 2.0), 0.0)
    xx, yy = np.meshgrid(np.arange(n_countries) + 0.5, np.arange(n_scenarios) + 0.5)
    ax.scatter(
        xx.ravel(),
        yy.ravel(),
        s=dot_sizes.ravel(),
        facecolor="white",
        edgecolor="#2C2C2C",
        linewidth=0.20,
        alpha=0.94,
        zorder=2,
    )

    for y in [4, 6]:
        ax.axhline(y, color="#BDB8AA", linewidth=0.65, zorder=3)

    ax.set_xlim(0, n_countries)
    ax.set_ylim(0, n_scenarios)
    ax.invert_yaxis()
    ax.set_xticks(np.arange(n_countries) + 0.5)
    ax.set_xticklabels([country_label(country) for country in countries], rotation=90, ha="center", va="top")
    ax.set_yticks(np.arange(n_scenarios) + 0.5)
    ax.set_yticklabels(SCENARIO_LABELS)
    ax.set_xlabel("Country", labelpad=1.2)
    ax.set_ylabel("Scenario", labelpad=1.2)
    ax.tick_params(axis="both", length=0, pad=1.2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color(COLORS["axis"])
    ax.set_facecolor(COLORS["land"])

    return dot_scale


def aggregate_bar_segments(country_df: pd.DataFrame, countries: list[str]) -> pd.DataFrame:
    key = country_df.set_index("country_key")
    selected = key.loc[countries].copy()
    others = key.drop(index=countries, errors="ignore")
    others_row: dict[str, float | str] = {"country_key": "Others", "actual_capacity": others["actual_capacity"].sum()}
    actual = float(others_row["actual_capacity"])
    for scenario in SCENARIOS:
        cap = float(others[f"{scenario.name}_capacity"].sum())
        others_row[f"{scenario.name}_capacity"] = cap
        others_row[f"{scenario.name}_change"] = cap - actual
        others_row[f"{scenario.name}_relative_change"] = (cap - actual) / actual if actual > 0 else np.nan

    selected = selected.reset_index()
    return pd.concat([selected, pd.DataFrame([others_row])], ignore_index=True)


def plot_stacked_totals(
    ax: plt.Axes,
    country_df: pd.DataFrame,
    countries: list[str],
    norm: mcolors.Normalize,
) -> None:
    segments = aggregate_bar_segments(country_df, countries)
    labels = [country_label(country) if country != "Others" else "Others" for country in segments["country_key"]]
    y = np.arange(len(SCENARIOS))
    bar_height = 0.66
    totals = [float(country_df[f"{scenario.name}_capacity"].sum()) for scenario in SCENARIOS]
    actual_total = float(country_df["actual_capacity"].sum())
    max_total = max(totals) if totals else 1.0
    label_min_width = max_total * 0.115

    for row_index, scenario in enumerate(SCENARIOS):
        left = 0.0
        for _, segment in segments.iterrows():
            width = float(segment[f"{scenario.name}_capacity"])
            if not np.isfinite(width) or width <= 0:
                continue
            rel = float(segment[f"{scenario.name}_relative_change"])
            color = color_for_relative(rel, norm)
            ax.barh(
                row_index,
                width,
                left=left,
                height=bar_height,
                color=color,
                edgecolor="white",
                linewidth=0.25,
                zorder=3,
            )
            if width >= label_min_width:
                label = labels[int(segment.name)]
                rgb = np.array(color[:3])
                luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
                text_color = "white" if luminance < 0.55 else COLORS["text"]
                ax.text(
                    left + width / 2,
                    row_index,
                    label,
                    ha="center",
                    va="center",
                    fontsize=6,
                    color=text_color,
                    clip_on=True,
                    zorder=4,
                )
            left += width

    for yline in [3.5, 5.5]:
        ax.axhline(yline, color="#BDB8AA", linewidth=0.65, zorder=1)

    ax.axvline(actual_total, color=COLORS["actual"], linestyle=(0, (2.2, 1.4)), linewidth=0.75, zorder=5)
    ax.text(
        actual_total,
        1.045,
        "Actual total\nPV capacity",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=6,
        linespacing=0.88,
        color=COLORS["actual"],
        clip_on=False,
    )

    ax.set_xlim(0, max_total * 1.02)
    ax.set_ylim(-0.5, len(SCENARIOS) - 0.5)
    ax.invert_yaxis()
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("PV capacity (GW)", labelpad=1.2)
    ax.set_xticks([0, 4000, 8000, 12000])
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda value, _: f"{value/1000:g}k" if value >= 1000 else f"{value:g}"))
    ax.grid(axis="x", color="#E5E2DA", linewidth=0.35, zorder=0)
    ax.tick_params(axis="x", length=2, pad=1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(0.5)


def select_countries(country_df: pd.DataFrame, count: int) -> list[str]:
    df = country_df.copy()
    df["max_capacity"] = df[capacity_columns()].max(axis=1)
    df["max_abs_change"] = df[change_columns()].abs().max(axis=1)
    df = df.sort_values(["max_capacity", "actual_capacity", "max_abs_change"], ascending=False)
    return df["country_key"].head(count).tolist()


def build_figure() -> None:
    set_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading country-level PV capacity data...")
    utility_df = load_country_data("utility")
    distributed_df = load_country_data("distributed")
    country_df = make_total_country_data(utility_df, distributed_df)

    heatmap_countries = select_countries(country_df, 38)
    bar_countries = select_countries(country_df, 6)
    all_relative = country_df[relative_columns()].stack()
    norm, _ = signed_log_norm(all_relative)

    fig = plt.figure(figsize=(FIG_WIDTH_MM * MM, FIG_HEIGHT_MM * MM), constrained_layout=False)

    heatmap_bounds = [0.078, 0.310, 0.700, 0.397]
    bar_bounds = [0.818, 0.310, 0.155, 0.397]
    ax_heatmap = fig.add_axes(heatmap_bounds)
    ax_bar = fig.add_axes(bar_bounds)

    print("Drawing heatmap and total stacked statistics...")
    dot_scale = plot_heatmap(fig, ax_heatmap, country_df, heatmap_countries, norm)
    plot_stacked_totals(ax_bar, country_df, bar_countries, norm)

    add_shared_guides(
        fig,
        norm,
        heatmap_bounds,
        dot_scale,
        dot_ref_values=[100, 500, 1000],
    )

    panel_label_y = heatmap_bounds[1] + heatmap_bounds[3] + 0.130
    draw_panel_label(fig, heatmap_bounds[0] - 0.013, panel_label_y, "a")
    draw_panel_label(fig, bar_bounds[0] - 0.014, panel_label_y, "b")

    png_path = OUT_DIR / "Fig4_capacity_heatmap_stats.png"
    pdf_path = OUT_DIR / "Fig4_capacity_heatmap_stats.pdf"
    fig.savefig(png_path, dpi=600)
    fig.savefig(pdf_path, dpi=600)
    plt.close(fig)
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    build_figure()
