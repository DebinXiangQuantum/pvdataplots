from __future__ import annotations

import matplotlib.pyplot as plt

from fig2_common import (
    COUNTRY_COLUMN_ORDER,
    FIG_DIR,
    load_distribution_data,
    mm,
    plot_country_distribution,
    set_style,
)


def export_distribution_plots(output_dir=FIG_DIR / "exported_plots" / "nationsFig2") -> None:
    set_style(6)
    output_dir.mkdir(parents=True, exist_ok=True)
    distribution_df, country_cols = load_distribution_data()

    for country in COUNTRY_COLUMN_ORDER:
        if country not in country_cols:
            print(f"Skipping {country}: missing paired distribution columns.")
            continue
        fig, ax = plt.subplots(figsize=(mm(26), mm(16)))
        plot_country_distribution(ax, country, distribution_df, country_cols, show_xlabel=True)
        save_path = output_dir / f"{country}_distribution.pdf"
        fig.savefig(save_path, dpi=600, bbox_inches="tight", transparent=True)
        plt.close(fig)
        print(f"Exported {save_path}")


if __name__ == "__main__":
    export_distribution_plots()
