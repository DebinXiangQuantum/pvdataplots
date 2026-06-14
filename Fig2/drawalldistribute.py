from __future__ import annotations

import matplotlib.pyplot as plt

from fig2_common import (
    FIG_DIR,
    load_distribution_data,
    mm,
    plot_global_distribution,
    set_style,
)


def export_distribution_plot(output_dir=FIG_DIR / "exported_plots" / "nationsFig2") -> None:
    set_style(6)
    output_dir.mkdir(parents=True, exist_ok=True)
    distribution_df, _ = load_distribution_data()

    fig, ax = plt.subplots(figsize=(mm(72), mm(18)))
    plot_global_distribution(ax, distribution_df)
    save_path = output_dir / "all_distributionshort.pdf"
    fig.savefig(save_path, dpi=600, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"Exported {save_path}")


if __name__ == "__main__":
    export_distribution_plot()
