from __future__ import annotations

import matplotlib.pyplot as plt

from fig2_common import (
    EXPORT_DIR,
    LAYOUT_COUNTRIES,
    add_panel_label,
    add_size_legend,
    load_distribution_data,
    load_scatter_data,
    load_solar_map_data,
    load_world_projected,
    mm,
    plot_capacity_scatter,
    plot_country_distribution,
    plot_global_distribution,
    plot_radiation_installation_map,
    set_style,
)


FIG_WIDTH_MM = 180
FIG_HEIGHT_MM = 150


def add_layout_axes(fig: plt.Figure) -> dict[str, plt.Axes]:
    def bounds(x: float, y: float, w: float, h: float) -> list[float]:
        return [x / FIG_WIDTH_MM, y / FIG_HEIGHT_MM, w / FIG_WIDTH_MM, h / FIG_HEIGHT_MM]

    axes: dict[str, plt.Axes] = {}
    axes["map"] = fig.add_axes(bounds(37.5, 80, 100, 60))
    axes["global"] = fig.add_axes(bounds(45, 60, 90, 15))
    axes["scatter_utility"] = fig.add_axes(bounds(7.5, 14, 69.5, 30))
    axes["scatter_distributed"] = fig.add_axes(bounds(92.5, 14, 72, 30))

    country_y = [129, 106, 83, 60]
    for idx, y in enumerate(country_y):
        axes[f"country_{idx}"] = fig.add_axes(bounds(7.5, y, 23.5, 15))
    for idx, y in enumerate(country_y, start=4):
        axes[f"country_{idx}"] = fig.add_axes(bounds(149, y, 23.5, 15))
    return axes


def main() -> None:
    set_style(6)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Fig2 data...")
    world = load_world_projected()
    solar_df = load_solar_map_data()
    distribution_df, country_cols = load_distribution_data()
    utility_scatter = load_scatter_data("utility")
    distributed_scatter = load_scatter_data("distributed")

    fig = plt.figure(figsize=(1, 1), constrained_layout=False)
    fig.set_size_inches(mm(FIG_WIDTH_MM), mm(FIG_HEIGHT_MM), forward=False)
    axes = add_layout_axes(fig)

    print("Drawing radiation and installation map...")
    plot_radiation_installation_map(fig, axes["map"], world, solar_df, panel_label="a")

    print("Drawing national radiation distributions...")
    country_panel_labels = list("bcdefghi")
    for idx, country in enumerate(LAYOUT_COUNTRIES):
        plot_country_distribution(
            axes[f"country_{idx}"],
            country,
            distribution_df,
            country_cols,
        )
        add_panel_label(axes[f"country_{idx}"], country_panel_labels[idx], x=-0.262, y=1.13)

    print("Drawing global radiation distribution...")
    plot_global_distribution(axes["global"], distribution_df, panel_label="j", compact=True)

    print("Drawing capacity scatter plots...")
    plot_capacity_scatter(axes["scatter_utility"], utility_scatter, "Utility-scale PV", panel_label="k")
    size_legend = add_size_legend(
        axes["scatter_utility"],
        (1, 30, 300),
        loc="lower left",
        bbox_to_anchor=(0.08, 0.04),
    )
    axes["scatter_utility"].add_artist(size_legend)
    plot_capacity_scatter(
        axes["scatter_distributed"],
        distributed_scatter,
        "Distributed PV",
        panel_label="l",
        colorbar_x=1.054,
        ylabel_x=-0.018,
        show_ylabel=False,
    )
    size_legend = add_size_legend(
        axes["scatter_distributed"],
        (0.1, 5, 25),
        loc="lower left",
        bbox_to_anchor=(0.08, 0.04),
    )
    axes["scatter_distributed"].add_artist(size_legend)

    out_pdf = EXPORT_DIR / "Fig2_composite.pdf"
    out_png = EXPORT_DIR / "Fig2_composite.png"
    fig.savefig(out_pdf, dpi=600)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
