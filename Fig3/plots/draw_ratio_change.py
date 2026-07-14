import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from shapely.geometry import box
import matplotlib as mpl
import os

# --- 1. Nature Standard Setup ---
mm_to_inch = 1 / 25.4
nature_double_col_width = 180 * mm_to_inch

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.size'] = 6
mpl.rcParams['axes.linewidth'] = 0.5

# --- 2. Data Preparation ---
scenarios = [
    "情景1", "情景2", "情景3", "情景4",
    "情景5一级", "情景5二级",
    "情景6一级", "情景6二级",
    "情景7一级", "情景7二级",
    "情景8一级", "情景8二级",
    "情景9一级", "情景9二级"
]

def load_data_split(mode='distributed'):
    """mode: 'distributed' or 'centralized'"""
    all_data = None
    for scenario in scenarios:
        fenbu_path = f"data/counterfactual_csv_data_fenbu_merged/{scenario}_merged.csv"
        jizhong_path = f"data/counterfactual_csv_data_jizhong_adjusted/{scenario}_50.csv"
        
        if not os.path.exists(fenbu_path) or not os.path.exists(jizhong_path):
            print(f"Warning: {scenario} missing files")
            continue
            
        df_f = pd.read_csv(fenbu_path)
        df_j = pd.read_csv(jizhong_path)
        
        if mode == 'distributed':
            agg = df_f.groupby('国家/地区')[['真实基础装机量 (Actual)', '反事实外推装机量 (Counterfactual)']].sum().reset_index()
            agg.columns = ['国家/地区', 'Actual', 'Scenario']
        else:
            agg = df_j.groupby('国家/地区')[['Actual', '反事实外推装机量_校准后']].sum().reset_index()
            agg.columns = ['国家/地区', 'Actual', 'Scenario']
            
        agg['Actual'] = agg['Actual'] / 1e6 # Convert to GW
        agg['Scenario'] = agg['Scenario'] / 1e6
        
        scenario_col = f"Scenario_{scenario}"
        if all_data is None:
            all_data = agg.copy()
            all_data.columns = ['国家/地区', 'Actual_Total', scenario_col]
        else:
            all_data = pd.merge(all_data, agg[['国家/地区', 'Scenario']], on='国家/地区', how='outer')
            all_data.rename(columns={'Scenario': scenario_col}, inplace=True)
            
    return all_data

# --- 3. GeoData Loading ---
print("Loading and projecting shapefile...")
world_path = "data/国家边界矢量/World_countries.shp"
world_gdf = gpd.read_file(world_path)
world_gdf['NAME_UPPER'] = world_gdf['NAME'].str.upper()
name_mapping = {
    'UNITED STATES OF AMERICA': 'UNITED STATES',
    'RUSSIAN FEDERATION': 'RUSSIA',
    'VIET NAM': 'VIETNAM',
    'KOREA, REPUBLIC OF': 'SOUTH KOREA',
    'BRUNEI DARUSSALAM': 'BRUNEI',
    'LAO PEOPLE\'S DEMOCRATIC REPUBLIC': 'LAOS',
    'IRAN (ISLAMIC REPUBLIC OF)': 'IRAN',
    'SYRIAN ARAB REPUBLIC': 'SYRIA',
    'CZECH REPUBLIC': 'CZECHIA',
}
world_gdf['NAME_UPPER'] = world_gdf['NAME_UPPER'].replace(name_mapping)
# Projection
target_crs = "ESRI:54030" # Robinson
clean_bbox = box(-179.9, -60, 179.9, 85) # Clip South to hide Antarctica
world_gdf = world_gdf[world_gdf['NAME_UPPER'] != 'ANTARCTICA']
world_gdf = world_gdf.clip(clean_bbox).to_crs(target_crs)

def plot_grid(df_all, mode_name, output_path):
    print(f"Plotting grid for {mode_name}...")
    
    # Calculate Change
    for s in scenarios:
        df_all[f"Change_{s}"] = df_all[f"Scenario_{s}"] - df_all['Actual_Total']

    # 14 Scenarios (No Actual plot)
    plot_items = [f"Change_{s}" for s in scenarios]
    scenario_items = [f"Scenario_{s}" for s in scenarios]
    labels = [
        "S1", "S2", "S3", "S4",
        "S5-I", "S5-II",
        "S6-I", "S6-II",
        "S7-I", "S7-II",
        "S8-I", "S8-II",
        "S9-I", "S9-II"
    ]

    fig, axes = plt.subplots(4, 4, figsize=(nature_double_col_width, nature_double_col_width * 0.65))
    axes = axes.flatten()

    # Color scale for maps - SymLogNorm to handle negative and positive differences
    # Find global min and max change
    vmax_val = df_all[plot_items].max().max()
    vmin_val = df_all[plot_items].min().min()
    max_abs = max(abs(vmax_val), abs(vmin_val))
    if max_abs == 0:
        max_abs = 1

    # Using blue to red gradient
    norm = mcolors.SymLogNorm(linthresh=0.1, vmin=-max_abs, vmax=max_abs, base=10)
    cmap = plt.cm.RdBu_r

    for i in range(14):
        col = plot_items[i]
        ax = axes[i]
        merged_gdf = world_gdf.merge(df_all, left_on='NAME_UPPER', right_on='国家/地区', how='left')
        
        world_gdf.plot(ax=ax, color='#F5F5F5', linewidth=0.1, edgecolor='#CCCCCC')
        merged_gdf.dropna(subset=[col]).plot(
            column=col, ax=ax, norm=norm, cmap=cmap, linewidth=0, edgecolor='none'
        )
        
        # Subplot labels
        ax.text(0.02, 1.1, f"({chr(97+i)}) {labels[i]}", transform=ax.transAxes, 
                fontsize=7, fontweight='bold', va='top')
        ax.axis('off')

    # Turn off axes 14 and 15
    axes[14].axis('off')
    axes[15].axis('off')

    # --- Boxplot: Change relative to Actual (Scenario - Actual) ---
    ax_box = fig.add_subplot(4, 4, (15, 16))
    
    # Calculate differences: Scenario - Actual for each country
    box_data_list = [(df_all[col] - df_all['Actual_Total']).dropna() for col in scenario_items]
    
    # Pre-calculate stats for bxp
    stats = []
    for d in box_data_list:
        stats.append({
            'med': np.median(d),
            'q1': np.percentile(d, 33),
            'q3': np.percentile(d, 67),
            'whislo': np.percentile(d, 10),
            'whishi': np.percentile(d, 90),
            'mean': np.mean(d),
            'fliers': [] # We don't want to show outliers
        })
    
    # Horizontal line at 0 change
    # ax_box.axhline(0, color='black', linestyle='--', linewidth=0.5, zorder=1)
    ## but at the last box 
    
    # Boxplot Style using bxp for custom percentiles
    bp = ax_box.bxp(stats, positions=range(1, 15), showfliers=False, 
                    patch_artist=True, widths=0.6,
                    showmeans=False,
                    medianprops={"color": "black", "linewidth": 0.8, "linestyle": "-"},
                    whiskerprops={"color": "black", "linewidth": 0.6},
                    capprops={"color": "black", "linewidth": 0.6}) 

    # Plot Mean as black-bordered white circle
    means = [s['mean'] for s in stats]
    ax_box.scatter(range(1, 15), means, marker='o', s=10, facecolor='white', 
                   edgecolor='black', linewidth=0.6, zorder=10, label='Mean')

    # Colors for boxes
    colors = plt.cm.tab20(np.linspace(0, 1, 14))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
        patch.set_linewidth(0.5)

    ax_box.set_xticks(range(1, 15))
    ax_box.set_xticklabels(labels, rotation=90)
    ax_box.set_ylabel(f'Capacity change (GW)')
    # Symmetric log scale for changes
    ax_box.set_yscale('symlog', linthresh=0.1) 
    ax_box.tick_params(axis='both', which='both', labelsize=6, length=2)
    
    # Leave space on the right for notation
    ax_box.set_xlim(0, 18)
    
    ax_box.spines['top'].set_visible(False)
    ax_box.spines['right'].set_visible(False)
    ax_box.text(-0.18, 1.15, f"({chr(97+14)}) PV capacity change", transform=ax_box.transAxes, 
                fontsize=7, fontweight='bold', va='top')

    # --- Add Boxplot Notation (Legend) ---
    not_x = 0.84  # Move to the right empty space
    not_y = 0.2
    leg_w = 0.03
    leg_h = 0.8
    not_lw = 0.6
    ax_box.axhline(0, xmin=0, xmax=not_x-0.03, color='black', linestyle='--', linewidth=0.5, zorder=1)
    # Whiskers
    ax_box.plot([not_x, not_x], [not_y + leg_h*0.1, not_y + leg_h*0.25], transform=ax_box.transAxes, color='black', lw=not_lw)
    ax_box.plot([not_x, not_x], [not_y + leg_h*0.75, not_y + leg_h*0.9], transform=ax_box.transAxes, color='black', lw=not_lw)
    # Caps
    ax_box.plot([not_x-leg_w*0.3, not_x+leg_w*0.3], [not_y + leg_h*0.1, not_y + leg_h*0.1], transform=ax_box.transAxes, color='black', lw=not_lw)
    ax_box.plot([not_x-leg_w*0.3, not_x+leg_w*0.3], [not_y + leg_h*0.9, not_y + leg_h*0.9], transform=ax_box.transAxes, color='black', lw=not_lw)
    # Box
    rect = mpl.patches.Rectangle((not_x - leg_w/2, not_y + leg_h*0.25), leg_w, leg_h*0.5, 
                                 linewidth=not_lw, edgecolor='black', facecolor='white', 
                                 transform=ax_box.transAxes, zorder=10)
    ax_box.add_patch(rect)
    # Median
    ax_box.plot([not_x - leg_w/2, not_x + leg_w/2], [not_y + leg_h*0.5, not_y + leg_h*0.5], 
                transform=ax_box.transAxes, color='black', lw=0.8, zorder=11)
    # Mean
    ax_box.scatter([not_x], [not_y + leg_h*0.6], marker='o', s=6, facecolor='white', 
                   edgecolor='black', linewidth=not_lw, transform=ax_box.transAxes, zorder=12)
    
    # Labels for notation
    ax_box.text(not_x + 0.025, not_y + leg_h*0.9, "90th percentile", transform=ax_box.transAxes, fontsize=5, va='center')
    ax_box.text(not_x + 0.025, not_y + leg_h*0.75, "67th percentile", transform=ax_box.transAxes, fontsize=5, va='center')
    ax_box.text(not_x + 0.025, not_y + leg_h*0.6, "Mean", transform=ax_box.transAxes, fontsize=5, va='center')
    ax_box.text(not_x + 0.025, not_y + leg_h*0.5, "Median", transform=ax_box.transAxes, fontsize=5, va='center')
    ax_box.text(not_x + 0.025, not_y + leg_h*0.25, "33rd percentile", transform=ax_box.transAxes, fontsize=5, va='center')
    ax_box.text(not_x + 0.025, not_y + leg_h*0.1, "10th percentile", transform=ax_box.transAxes, fontsize=5, va='center')

    # Force a draw to ensure positions are updated before getting them
    plt.subplots_adjust(bottom=0.2, top=0.98, left=0.05, right=0.98, hspace=-0.05, wspace=0.05)
    fig.canvas.draw()


    # Get positions of the subplots for alignment
    pos14 = axes[14].get_position()
    pos15 = axes[15].get_position()
    pos10 = axes[10].get_position()
    pos11 = axes[11].get_position()
    
    ## 设置 box plot 的位置 和宽度高度，使其与上方的地图对齐，并且占满14和15的空间
    # Set ax_box position to align with axes above and span 14 and 15
    # Add an offset to x0 to align the y-axis with the actual map in (k)
    offset_x = 0.07
    offset_y = 0.02
    new_x0 = pos10.x0 + offset_x
    new_width = pos11.x1 - new_x0 - 0.03
    ax_box.set_position([new_x0, pos14.y0-offset_y, new_width, pos14.height-offset_y])

    # Colorbar centered below subplots 12, 13
    pos12 = axes[12].get_position()
    pos13 = axes[13].get_position()
    
    cb_width = (pos13.x1 - pos12.x0) * 0.8
    cb_left = pos12.x0 + ((pos13.x1 - pos12.x0) - cb_width) / 2
    cb_bottom = pos12.y0 - 0.02
    
    cax = fig.add_axes([cb_left, cb_bottom, cb_width, 0.015]) 
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, cax=cax, orientation='horizontal')
    if mode_name.lower() == 'distributed':
        cb.set_label(f'Distributed PV capacity change (GW)')
    else:
        cb.set_label(f'Utility-scale PV capacity change (GW)')
    # cb.set_label('Δ Capacity (GW)')
    cb.ax.tick_params(labelsize=6, length=2)
    cb.outline.set_linewidth(0.5)

    plt.savefig(output_path+'.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_path+'.png', dpi=300, bbox_inches='tight')
    print(f"Saved to {output_path}")
    plt.close()

# Main execution
df_distributed = load_data_split('distributed')
plot_grid(df_distributed, 'Distributed', 'Fig3/figures/distributed_ratio_change')

df_centralized = load_data_split('centralized')
plot_grid(df_centralized, 'Centralized', 'Fig3/figures/centralized_ratio_change')
