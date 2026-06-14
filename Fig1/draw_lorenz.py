import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

# 基础设置
mm_to_inch = 1 / 25.4
nature_width_mm = 180
fig_width = nature_width_mm * mm_to_inch
fig_height = fig_width * 0.3 # 强制要求的 0.3 比例

# 字体设置 (减小基准字号以适应极其扁平的布局)
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']
mpl.rcParams['font.size'] = 6
mpl.rcParams['xtick.labelsize'] = 6
mpl.rcParams['ytick.labelsize'] = 6
mpl.rcParams['legend.fontsize'] = 6
mpl.rcParams['axes.linewidth'] = 0.5
mpl.rcParams['xtick.major.width'] = 0.5
mpl.rcParams['ytick.major.width'] = 0.5
mpl.rcParams['pdf.fonttype'] = 42

# 读取数据
df = pd.read_excel('Fig1/excel/lorendata.xlsx')

def plot_lorenz(fig, ax, df, var_col, gdp_col, gdp_pc_col, label):
    valid_df = df[[var_col, gdp_col, gdp_pc_col]].dropna()
    
    # ---------------------------------------------------------
    # 1. 计算 Lorenz 曲线 (PV 变量)
    # ---------------------------------------------------------
    df_var = valid_df.sort_values(by=var_col).reset_index(drop=True)
    n = len(df_var)
    cum_pop = np.linspace(0, 100, n + 1)
    vals = df_var[var_col].values
    cum_vals = np.concatenate([[0], np.cumsum(vals)])
    cum_vals_pct = cum_vals / cum_vals[-1] * 100
    
    try:
        area_lorenz_var = np.trapezoid(cum_vals_pct/100, cum_pop/100)
    except AttributeError:
        area_lorenz_var = np.trapz(cum_vals_pct/100, cum_pop/100)
    gini_var = (0.5 - area_lorenz_var) / 0.5
    
    # ---------------------------------------------------------
    # 2. 计算 Lorenz 曲线 (2023 GDP)
    # ---------------------------------------------------------
    df_gdp = valid_df.sort_values(by=gdp_col).reset_index(drop=True)
    gdp_vals = df_gdp[gdp_col].values
    cum_gdp = np.concatenate([[0], np.cumsum(gdp_vals)])
    cum_gdp_pct = cum_gdp / cum_gdp[-1] * 100
    
    try:
        area_lorenz_gdp = np.trapezoid(cum_gdp_pct/100, cum_pop/100)
    except AttributeError:
        area_lorenz_gdp = np.trapz(cum_gdp_pct/100, cum_pop/100)
    gini_gdp = (0.5 - area_lorenz_gdp) / 0.5

    # 计算 10% Gap 标注所需数据 (针对 PV 曲线)
    y_at_90 = np.interp(90, cum_pop, cum_vals_pct)
    y_gap = 100 - y_at_90

    # ---------------------------------------------------------
    # 3. 绘图与填充
    # ---------------------------------------------------------
    # 填充 PV 曲线与对角线之间的区域
    ax.fill_between(cum_pop, cum_pop, cum_vals_pct, color='#c0c0c0', alpha=0.8, zorder=1)
    
    # GDP per capita 着色条 (位于最底层)
    cmap = plt.get_cmap('RdPu')
    norm = mcolors.LogNorm(vmin=valid_df[gdp_pc_col].min(), vmax=valid_df[gdp_pc_col].max())
    
    for i in range(n):
        x1, x2 = cum_pop[i], cum_pop[i+1]
        y1, y2 = cum_vals_pct[i], cum_vals_pct[i+1]
        c_val = df_var[gdp_pc_col].iloc[i]
        color = cmap(norm(c_val))
        ax.fill_between([x1, x2], 0, [y1, y2], color=color, alpha=1.0, edgecolor='none', zorder=2)
        
    ax.plot([0, 100], [0, 100], color='black', linewidth=1, zorder=3) # Perfect Equality
    
    # 绘制两条 Lorenz 曲线
    var_name = 'Utility-scale PV' if 'Utility' in var_col else 'Distributed PV'
    ax.plot(cum_pop, cum_vals_pct, color='#4b0082', linewidth=1, zorder=5, label=f'{var_name} (Gini={gini_var:.2f})')
    ax.plot(cum_pop, cum_gdp_pct, color='#ff8c00', linewidth=1, linestyle='--', zorder=4, label=f'2023 GDP (Gini={gini_gdp:.2f})')
    
    # ---------------------------------------------------------
    # 4. 10% Gap 标注
    # ---------------------------------------------------------
    ax.axvline(90, color='#1e90ff', linestyle='--', linewidth=0.5, zorder=6)
    ax.annotate('', xy=(89, y_at_90), xytext=(89, 100),
                arrowprops=dict(arrowstyle='<->, widthB=1.2, lengthB=0.3', color='black', lw=0.6))
    ax.text(78, (y_at_90 + 100)/2, f'{y_gap:.0f}%', ha='left', va='center', fontweight='bold')
    ax.text(95, 100.5, '10%', ha='center', va='bottom', fontweight='bold')
    # ---------------------------------------------------------
    # 独立 Colorbar
    # ---------------------------------------------------------
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label('GDP per capita\n(current US$)')
    cbar.ax.tick_params()

    # ---------------------------------------------------------
    # 文本注释与图例
    # ---------------------------------------------------------
    ax.text(34, 40, 'Perfect equality', rotation=32, ha='center', va='center', alpha=0.7)
    ax.text(64, 10, 'Lorenz curve', rotation=12, ha='center', va='center', alpha=0.7)
    # 子图标签 (a) (b)
    # ax.text(-0.12, 1.05, label, transform=ax.transAxes,  fontweight='bold', va='top')
    
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel('Cumulative region percentage (%)', )
    ax.set_ylabel('Cumulative percentage (%)', )
    ax.tick_params(axis='both')
    
    # 优化图例位置，避免遮挡曲线
    ax.legend(loc='upper left', frameon=False, edgecolor='none', facecolor='white', framealpha=0.8)
    
    return norm, cmap

# 绘图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height))

plot_lorenz(fig, ax1, df, 'Utility-scale PV GW', '2023 GDP (current US$)', '2023 GDP per capita (current US$) 排序', 'c')
plot_lorenz(fig, ax2, df, 'Distributed PV GW', '2023 GDP (current US$)', '2023 GDP per capita (current US$) 排序', 'd')

# 调整布局：bottom 增加到 0.28 以给轴标签留空间，top 保持紧凑
plt.subplots_adjust(left=0.08, right=0.92, bottom=0.28, top=0.88, wspace=0.6)
plt.savefig('Fig1/lorenz_curve.pdf', dpi=300)
print("Plot saved to Fig1/lorenz_curve.pdf")
