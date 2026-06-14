from __future__ import annotations

import matplotlib.pyplot as plt

from fig2_common import (
    EXPORT_DIR,
    add_income_legend,
    add_size_legend,
    load_scatter_data,
    mm,
    plot_capacity_scatter,
    set_style,
)


def export_scatter_plots() -> None:
    set_style(6)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    configs = [
        ("utility", "Utility-scale PV", "集中式"),
        ("distributed", "Distributed PV", "分布式"),
    ]
    for kind, title, stem in configs:
        df = load_scatter_data(kind)
        fig, ax = plt.subplots(figsize=(mm(70), mm(52)))
        plot_capacity_scatter(ax, df, title)
        size_legend = add_size_legend(ax, (1, 30, 300) if kind == "utility" else (0.1, 5, 25))
        ax.add_artist(size_legend)
        add_income_legend(ax)
        pdf_path = EXPORT_DIR / f"{stem}.pdf"
        png_path = EXPORT_DIR / f"{stem}.png"
        fig.savefig(pdf_path, dpi=600, bbox_inches="tight", transparent=True)
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Exported {pdf_path}")
        print(f"Exported {png_path}")


if __name__ == "__main__":
    export_scatter_plots()
