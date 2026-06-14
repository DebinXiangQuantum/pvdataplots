from __future__ import annotations

import matplotlib.pyplot as plt

from fig2_common import (
    EXPORT_DIR,
    load_solar_map_data,
    load_world_projected,
    mm,
    plot_radiation_installation_map,
    set_style,
)


def main() -> None:
    set_style(6)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading projected world and solar grid data...")
    world = load_world_projected()
    solar_df = load_solar_map_data()

    fig, ax = plt.subplots(figsize=(mm(128), mm(62)))
    print("Drawing graticule radiation map with PV installation overlay...")
    plot_radiation_installation_map(fig, ax, world, solar_df)
    fig.savefig(EXPORT_DIR / "solar_pv_combined_map.pdf", dpi=600, bbox_inches="tight")
    fig.savefig(EXPORT_DIR / "solar_pv_combined_map.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved solar_pv_combined_map.pdf and solar_pv_combined_map.png")


if __name__ == "__main__":
    main()
