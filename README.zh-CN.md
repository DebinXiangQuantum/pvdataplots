# PVplotHub 数据下载与绘图说明

[English README](README.md)

本仓库保存光伏绘图代码与小型图表输入文件。大型空间矢量数据和反事实情景 CSV 不提交到 Git，而是通过外部数据归档下载到仓库根目录的 `data/`。

## 推荐托管方式

本地 `data/` 目前约为 5.4 GiB，包含约 1.1 GiB 的 10 km 光照矢量数据和约 4.2 GiB 的集中式反事实 CSV，不能直接放进 GitHub。

推荐把每个正式数据版本发布到 [Zenodo](https://zenodo.org/)。它适合学术数据归档，可生成 DOI、保留不可变的已发布版本；其[每条记录默认配额为 50 GB](https://help.zenodo.org/docs/deposit/manage-quota/)，足以容纳此数据集，[官方也建议多文件上传时使用 ZIP](https://help.zenodo.org/docs/deposit/manage-files/)。仓库只保存 ZIP 的公开下载 URL 和 SHA-256，不保存数据本体。

可选平台包括：Hugging Face Hub（适合频繁迭代的公开数据集，但[免费公开存储是 best-effort](https://huggingface.co/docs/hub/en/storage-limits)，更适合作为镜像）、Kaggle Datasets（适合面向 Notebook 的公开复用）、OSF（适合项目资料协作，但[存储规则更复杂](https://help.osf.io/article/137-osf-storage-caps)）和 Google Drive/OneDrive（适合私有协作）。论文或代码发布版本应优先使用 Zenodo，工作过程中的私有数据可另放共享盘。

## 使用者：下载数据并运行绘图

所有命令都在仓库根目录执行。需要 Python 3.14 或更高版本，以及 [uv](https://docs.astral.sh/uv/)。

```bash
uv sync
uv run python scripts/fetch_data.py
uv run python scripts/fetch_data.py --check
```

下载器支持断点续传，会用 SHA-256 校验 ZIP，再安全解压出 `data/`。首次下载和解压请预留至少 15 GiB 的空闲磁盘空间；使用 `--force` 覆盖已有数据时，还要保留旧 `data/` 的空间，当前版本建议约 22 GiB。中断下载会保留在 `.cache/data/<archive>.part`，再次执行同一命令即可续传。

目前 [`config/data-source.json`](config/data-source.json) 中保留的是占位符；数据发布者完成下节配置前，下载命令不会成功。这是为了避免仓库中出现一个失效或错误的数据链接。

数据校验通过后，可运行以下主要绘图脚本：

```bash
# 主图
uv run python Fig1/draw_Fig1.py
uv run python Fig2/draw_fig2_composite_v2.py
uv run python Fig3/plots/draw_scenarios.py
uv run python Fig3/plots/draw_ratio_change.py
uv run python Fig4/plots/draw_fig4_capacity_heatmap_stats.py

# 部分扩展图
uv run python extenddatafigure/plots/draw_extendedFig1_combined.py
uv run python extenddatafigure/plots/draw_extendedFig4_combined_weighted_irradiance_ring.py
uv run python extenddatafigure/plots/draw_extendedFig6_combined.py
uv run python extenddatafigure/plots/draw_extendedFig7_combined.py
uv run python extenddatafigure/plots/draw_extendedFig8_pv_scenario_maps_3x3.py
uv run python extenddatafigure/plots/draw_key_region_pv_zoom_maps.py
```

输出位置分别为：Fig1 的 `Fig1/PDFs/` 和 `Fig1/`，Fig2 的 `Fig2/exported_plots/`，Fig3 的 `Fig3/figures/`，Fig4 的 `Fig4/figures/`，以及扩展图的 `extenddatafigure/figures/`。这些 PNG/PDF 输出已被 Git 忽略。

## 数据发布者：配置 Zenodo

1. 在本地把当前 `data/` 打包。此命令只读取数据，不修改数据目录。

   ```bash
   uv run python scripts/package_data.py \
     --version v1.0.0 \
     --output releases/pvplothub-data-v1.0.0.zip
   ```

2. 在 Zenodo 创建一个新上传记录，上传 ZIP，并填写标题、作者、描述、许可证和数据引用信息后发布。发布后 Zenodo 会为该版本分配 DOI；论文和配置应使用这个版本 DOI。
3. 从已发布记录复制 ZIP 的直接下载地址，形式通常如下：

   ```text
   https://zenodo.org/records/<record-id>/files/pvplothub-data-v1.0.0.zip?download=1
   ```

4. 将 `package_data.py` 最后打印的 JSON 内容填写到 [`config/data-source.json`](config/data-source.json)：替换版本 DOI、URL、文件名、压缩包大小和 SHA-256。该 JSON 是公开配置，不含账号或密钥，应与代码一起提交。
5. 在一个全新的克隆目录中执行“使用者”部分的三条命令，确认下载、校验、解压都成功后再分享 GitHub 仓库。

`config/data-source.json` 中的必需文件清单会在打包前与解压后校验，避免已发布数据缺少某个绘图所需文件。不要把 `data/`、生成的 ZIP 或 `.cache/` 加入 Git；它们已经在 [`.gitignore`](.gitignore) 中忽略。

## 数据更新规则

每次数据变化都应视为一个新版本：重新打包、在 Zenodo 发布新版本、更新 `config/data-source.json` 中的版本 DOI、URL、大小与 SHA-256，并将该配置和依赖它的代码一起提交。这样旧版代码始终能对应到旧版数据，结果可以复现。
