import geopandas as gpd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from shapely.geometry import box

# ================= 1. Nature 出版级全局设置 =================
# 尺寸转换 (mm -> inch)
mm_to_inch = 1 / 25.4
nature_width_mm = 160
fig_width = nature_width_mm * mm_to_inch
# 宽高比：Robinson 投影通常宽:高约 2:1，考虑到柱状图空间，设为 0.6
fig_height = fig_width * 0.4 

# 字体设置 (严苛模式)
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial'] # 强制 Arial
mpl.rcParams['font.size'] = 6               # 基准字号 6pt
mpl.rcParams['axes.linewidth'] = 0.5        # 边框线宽 0.5pt (符合 Nature 细线要求)
mpl.rcParams['xtick.major.width'] = 0.5
mpl.rcParams['ytick.major.width'] = 0.5
mpl.rcParams['xtick.labelsize'] = 6
mpl.rcParams['ytick.labelsize'] = 6
mpl.rcParams['legend.fontsize'] = 6
mpl.rcParams['pdf.fonttype'] = 42           # 保证文本可编辑 (非轮廓)

# ================= 2. 核心参数 =================
world_shp = "data/国家边界矢量/World_countries.shp"
solar_shp = r"data/10km/Solar_10km.shp"
excel_path = r"Fig1/excel/barchartFig1.xlsx"
# 绘图模式: "分布式" / "集中式" / "总量"
PVtype = "分布式"   
plot_map_bundary= False
grid_size = 400  
target_crs = "ESRI:54030" # Robinson 投影

# --- 颜色定义 (Colorblind friendly where possible) ---
color_dist = "#F4A582"  # 分布式 (暖橙)
color_util = "#92C5DE"  # 集中式 (冷蓝)
color_single = "#41b6c4" 

# --- 关键：统计图布局坐标 (Left, Bottom, Width, Height) ---
# 坐标系为 Figure 坐标 (0~1)，针对 Robinson 投影的空白区域进行了微调
if PVtype == "分布式":
    chart_positions = {
        # 北美：左上角 (太平洋东北部)
        "North America": [0.09, 0.60, 0.12, 0.3],
        # 南美：左下角 (太平洋东南部)
        "South America": [0.18, 0.23, 0.12, 0.18],
        # 欧洲：中上方 (北大西洋)
        "Europe":        [0.36, 0.68, 0.12, 0.25],
        # 非洲：中下方 (南大西洋或几内亚湾)
        "Africa":        [0.42, 0.15, 0.12, 0.18],
        # 亚洲：右上角 (俄罗斯/北太平洋)
        "Asia":          [0.84, 0.65, 0.12, 0.23],
        # 大洋洲：右下角 (南太平洋)
        "Oceania":       [0.63, 0.15, 0.10, 0.29]
    }
else:
    chart_positions = {
        # 北美：左上角 (太平洋东北部)
        "North America": [0.09, 0.60, 0.12, 0.25],
        # 南美：左下角 (太平洋东南部)
        "South America": [0.18, 0.23, 0.12, 0.18],
        # 欧洲：中上方 (北大西洋)
        "Europe":        [0.36, 0.68, 0.12, 0.25],
        # 非洲：中下方 (南大西洋或几内亚湾)
        "Africa":        [0.42, 0.15, 0.12, 0.18],
        # 亚洲：右上角 (俄罗斯/北太平洋)
        "Asia":          [0.84, 0.65, 0.12, 0.23],
        # 大洋洲：右下角 (南太平洋)
        "Oceania":       [0.63, 0.15, 0.10, 0.29]
    }

# ================= 3. 数据处理 =================
print("正在读取并处理数据...")

# A. 读取 Excel 并清洗
def load_chart_data(path, pv_type):
    xls = pd.ExcelFile(path)
    df_dist = pd.read_excel(xls, 'DistributedPV-GW')
    df_util = pd.read_excel(xls, 'Utility-scalePV-GW')
    
    # 列名标准化
    for df in [df_dist, df_util]:
        df.columns = [c.strip().capitalize() for c in df.columns] 
        df['Nation'] = df['Nation'].str.title() # 巴西: BRAZIL -> Brazil

    if pv_type == "总量":
        # 合并用于堆叠图
        df_merged = pd.merge(df_dist, df_util, on=['Nation', 'Region'], suffixes=('_dist', '_util'), how='outer')
        df_merged.fillna(0, inplace=True)
        df_merged['Value_total'] = df_merged['Value_dist'] + df_merged['Value_util']
        df_merged.sort_values('Value_total', ascending=False, inplace=True)
        return df_merged, True
    elif pv_type == "分布式":
        df_dist['Value'] = df_dist['Value']
        return df_dist.sort_values('Value', ascending=False), False
    else:
        df_util['Value'] = df_util['Value']
        return df_util.sort_values('Value', ascending=False), False

chart_df, is_stacked = load_chart_data(excel_path, PVtype)

# B. 读取矢量地图
world_map = gpd.read_file(world_shp)
gdf = gpd.read_file(solar_shp)
gdf.columns = [c.lower() for c in gdf.columns]

# C. 字段匹配
field_map = {
    "集中式": ["jizhong_area", "jizhong_ar", "jizhong"],
    "分布式": ["fenbu_area", "fenbu"],
    "总量":   ["total_area", "total"]
}
candidates = field_map.get(PVtype, [])
value_field = next((f for f in candidates if f in gdf.columns), None)
if not value_field: raise ValueError(f"字段未找到: {candidates}")

output_filename = f"Nature_Global_{PVtype}_Final.pdf"

# D. 投影处理 (修复 180度横线问题 + 椭圆计算)
bbox = box(-180, -58, 180, 90) # 移除了 -90 到 -58 之间的南极区域
world_map = world_map.clip(bbox).to_crs(target_crs)

# 提取光伏质心并投影
gdf = gdf[gdf[value_field] > 0].copy()
gdf_proj = gdf.to_crs(target_crs)
gdf_points = gdf_proj.copy()
gdf_points['geometry'] = gdf_proj.geometry.centroid

x = gdf_points.geometry.x
y = gdf_points.geometry.y
C = gdf_points[value_field]*0.2/1e6

# ================= 4. 绘图主程序 =================
fig = plt.figure(figsize=(fig_width, fig_height)) # 严格设定 180mm 宽

# 主地图轴 (稍微留边给 Colorbar)
ax_map = fig.add_axes([0.01, 0.05, 0.98, 0.94]) 

# ================= 修正后的 Section A =================
# --- A. 绘制背景（椭圆边界）---
if plot_map_bundary:
    # 1. 创建基础的全球矩形（WGS84）
    geometry = box(-180, -90, 180, 90)
    boundary = gpd.GeoDataFrame(geometry=[geometry], crs="EPSG:4326")

    # 2. 【关键步骤】致密化（Segmentize）
    # 在投影前，每隔 1 度加一个点。
    # 只有加了足够多的点，投影后的边缘才会呈现出圆滑的弧度。
    # 注意：segmentize 需要 geopandas >= 0.13.0
    if hasattr(boundary, "segmentize"):
        boundary = boundary.segmentize(max_segment_length=1.0)
    else:
        # 兼容旧版本 GeoPandas 的写法 (手动插值)
        from shapely.geometry import Polygon, LineString
        import numpy as np
        
        # 手动生成高密度的点围成一圈
        lons = np.concatenate([
            np.linspace(-180, 180, 181),       # 下边缘
            np.linspace(180, 180, 91),         # 右边缘
            np.linspace(180, -180, 181),       # 上边缘
            np.linspace(-180, -180, 91)        # 左边缘
        ])
        lats = np.concatenate([
            np.linspace(-90, -90, 181),        # 下边缘
            np.linspace(-90, 90, 91),          # 右边缘
            np.linspace(90, 90, 181),          # 上边缘
            np.linspace(90, -90, 91)           # 左边缘
        ])
        boundary = gpd.GeoDataFrame(
            geometry=[Polygon(zip(lons, lats))], 
            crs="EPSG:4326"
        )

    # 3. 投影转换
    boundary = boundary.to_crs(target_crs)

    # 4. 绘图
    # 绘制浅灰色背景填充
    boundary.plot(ax=ax_map, facecolor="#f0f0f0", edgecolor="none", zorder=0)
    # 绘制深灰色边界线（盖在最上面）
    boundary.plot(ax=ax_map, facecolor="none", edgecolor="#404040", linewidth=0.1, zorder=3)



# --- Step 2: 世界底图 ---
world_map.plot(ax=ax_map, facecolor="#e0e0e0", edgecolor="white", linewidth=0.3, zorder=1)

# --- Step 3: 六边形热力图 ---
# 动态计算 Vmax (避免单个超大值导致整体颜色过浅)
vmax_val = np.nanpercentile(C, 99) * 5 
hb = ax_map.hexbin(
    x, y, C=C, gridsize=grid_size, cmap="YlOrRd",
    norm=LogNorm(vmin=np.nanpercentile(C, 5), vmax=vmax_val),
    reduce_C_function=np.sum, linewidths=0, mincnt=1, zorder=2
)
ax_map.set_axis_off()

# --- Step 4: 添加统计柱状图 (悬浮在对应位置) ---
print("正在绘制叠加图表...")
for region, pos in chart_positions.items():
    # 数据筛选 (取前5)
    reg_data = chart_df[chart_df['Region'] == region].head(5)
    if reg_data.empty: continue
    
    # 建立子图坐标系
    ax_sub = fig.add_axes(pos)
    
    nations = reg_data['Nation'].tolist()
    x_pos = np.arange(len(nations))
    
    if is_stacked:
        v_u = reg_data['Value_util'].values
        v_d = reg_data['Value_dist'].values
        ax_sub.bar(x_pos, v_u, color=color_util, width=0.7, label='Utility-scale')
        ax_sub.bar(x_pos, v_d, bottom=v_u, color=color_dist, width=0.7, label='Distributed')
        max_h = max(v_u + v_d)
    else:
        v = reg_data['Value'].values
        ax_sub.bar(x_pos, v, color=color_single, width=0.7)
        max_h = max(v)

    # --- 精细化调整样式 (Nature Style) ---
    ax_sub.set_title(region, fontweight='bold', pad=3)
    ax_sub.set_xlim(-0.6, len(nations)-0.4)
    ax_sub.set_ylim(0, max_h * 1.25) # 顶部留空写字
    
    # 隐藏边框，只保留左轴
    ax_sub.spines['top'].set_visible(False)
    ax_sub.spines['right'].set_visible(False)
    # ax_sub.spines['bottom'].set_visible(False)
    # ax_sub.spines['left'].set_linewidth(0.5)

    
    
    # X轴完全隐藏
    ax_sub.set_xticks([])
    # 设置ax background color
    # ax_sub.set_facecolor('lightgray')
    ax_sub.set_facecolor('none')
    # Y轴刻度设置
    ax_sub.tick_params(axis='y',  width=0.5, length=2, pad=1)
    
    # 标注国家名 (放在柱子上方或内部)
    for i, nation in enumerate(nations):
        # 统一放在柱子底部上方一点，垂直显示
        label_y = max_h * 0.05
        ax_sub.text(i, label_y, nation, rotation=90, 
                   ha='center', va='bottom',  zorder=5)

# --- Step 5: 全局组件 ---

# 色标 (放在最右边，旋转90度)
cax = fig.add_axes([0.98, 0.15, 0.012, 0.7])
cb = plt.colorbar(hb, cax=cax, orientation="vertical")
if PVtype == "集中式" :
    cb.set_label("Installed Capacity of Utility-scale PV (GW)")
elif PVtype == "分布式" :
    cb.set_label("Installed Capacity of Distributed PV (GW)")
cb.ax.tick_params(length=2, width=0.5)

# 图例 (仅总量模式需要)
if is_stacked:
    import matplotlib.patches as mpatches
    p1 = mpatches.Patch(color=color_dist, label='Distributed')
    p2 = mpatches.Patch(color=color_util, label='Utility-scale')
    # 放在左下角或合适位置
    fig.legend(handles=[p1, p2], loc='lower left', bbox_to_anchor=(0.02, 0.05),
               frameon=False, ncol=1, title="PV Type")

# ================= 5. 保存输出 =================
print(f"输出文件: {output_filename}")
# Nature 要求 300-600 dpi
plt.savefig(output_filename, dpi=300, bbox_inches='tight', pad_inches=0.05)
# plt.savefig(output_filename.replace(".pdf", ".svg"), dpi=300, bbox_inches='tight')
plt.close()
print("绘图完成！")
