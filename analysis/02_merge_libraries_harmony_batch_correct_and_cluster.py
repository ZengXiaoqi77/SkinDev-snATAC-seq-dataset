#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import argparse
import logging
import warnings
from glob import glob
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd

# 设置 matplotlib 后台，服务器上无图形界面时需要
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import snapatac2 as snap
import anndata

# ========== 0. 基础设置 ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 阶段颜色，用于绘图
STAGE_COLORS = {
    "E16.5": "#E69F00",
    "P0": "#56B4E9",
    "P5": "#009E73",
    "Adult": "#CC79A7",
}


def save_snap_figure(fig, path, dpi=150):
    """
    兼容保存 SnapATAC2 返回的 figure。
    SnapATAC2 根据版本/环境可能返回:
      - matplotlib Figure
      - plotly Figure
      - IPython.core.display.Image (bytes 数据)
      - 其他类型 (尝试直接写 bytes)
    """
    if fig is None:
        logger.warning(f"  figure 为 None，跳过保存: {path}")
        return
    try:
        # matplotlib Figure
        if hasattr(fig, "savefig"):
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"  已保存 (matplotlib): {path}")
            return
        # plotly Figure
        if hasattr(fig, "write_image"):
            fig.write_image(path, scale=2)
            logger.info(f"  已保存 (plotly): {path}")
            return
        # IPython Image 或其他 bytes-like
        fig_type = type(fig).__name__
        if fig_type == "Image":
            # IPython.display.Image: data 可能是 bytes 或 base64 str
            data = getattr(fig, "data", None)
            if data is not None:
                if isinstance(data, str):
                    import base64
                    data = base64.b64decode(data)
                with open(path, "wb") as f:
                    f.write(data)
                logger.info(f"  已保存 (IPython Image): {path}")
                return
        # 兜底: 尝试直接写 bytes
        if hasattr(fig, "__bytes__") or isinstance(fig, bytes):
            data = fig if isinstance(fig, bytes) else bytes(fig)
            with open(path, "wb") as f:
                f.write(data)
            logger.info(f"  已保存 (bytes): {path}")
            return
        logger.warning(f"  未知的 figure 类型 {type(fig)} ({fig_type})，跳过保存: {path}")
    except Exception as e:
        logger.warning(f"  保存 figure 失败: {e}")


# ========== 1. 解析命令行参数 ==========
parser = argparse.ArgumentParser(
    description="Step 1: 数据合并与批次校正 (SnapATAC2)",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
示例:
    python 02_merge_libraries_harmony_batch_correct_and_cluster.py \
        --indir /path/to/qc_output/h5ad \
        --outdir /path/to/integration_output \
        --n-features 500000 \
        --n-jobs 8
    """,
)
parser.add_argument(
    "--indir",
    type=str,
    required=True,
    help="输入目录，包含 *_filtered.h5ad",
)
parser.add_argument(
    "--outdir",
    type=str,
    required=True,
    help="输出目录",
)
parser.add_argument(
    "--n-features",
    type=int,
    default=500000,
    help="选择的特征数 (默认: 500000)",
)
parser.add_argument(
    "--n-jobs",
    type=int,
    default=8,
    help="并行线程数 (默认: 8)",
)
parser.add_argument(
    "--skip-obs-names-fix",
    action="store_true",
    help="跳过 obs_names 前缀修复（如果已预先处理过）",
)
parser.add_argument(
    "--export-adata",
    action="store_true",
    help="额外导出普通 AnnData (.h5ad) 用于下游分析",
)

args = parser.parse_args()

logger.info("=" * 60)
logger.info("Step 1: 数据合并与批次校正")
logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info("=" * 60)
logger.info(f"输入目录: {args.indir}")
logger.info(f"输出目录: {args.outdir}")
logger.info(f"N features: {args.n_features}")
logger.info(f"N jobs: {args.n_jobs}")


# ========== 2. 收集所有 filtered h5ad 文件 ==========
logger.info("\n[Step 1/10] 收集样本...")

pattern = os.path.join(args.indir, "*_filtered.h5ad")
h5ad_files = sorted(glob(pattern))

if not h5ad_files:
    logger.error(f"在 {args.indir} 下未找到 *_filtered.h5ad 文件")
    sys.exit(1)

logger.info(f"  找到 {len(h5ad_files)} 个样本:")
h5ads_pairs = []
for path in h5ad_files:
    sample_name = Path(path).stem.replace("_filtered", "")
    h5ads_pairs.append((sample_name, path))
    logger.info(f"    {sample_name} -> {path}")


# ========== 3. 创建输出目录 ==========
h5ads_dir = os.path.join(args.outdir, "h5ads")
h5ad_dir = os.path.join(args.outdir, "h5ad")
fig_dir = os.path.join(args.outdir, "figures")
report_dir = os.path.join(args.outdir, "reports")
for d in [h5ads_dir, h5ad_dir, fig_dir, report_dir]:
    os.makedirs(d, exist_ok=True)


# ========== 4. 修复 obs_names（确保唯一且持久化） ==========
logger.info("\n[Step 2/10] 修复细胞 barcode 唯一性...")

if not args.skip_obs_names_fix:
    for sample_name, h5ad_path in h5ads_pairs:
        try:
            adata = snap.read(h5ad_path, backed=None)
            original_names = list(adata.obs_names)
            new_names = [f"{sample_name}:{bc}" for bc in original_names]
            adata.obs_names = new_names
            adata.write_h5ad(h5ad_path)
            logger.info(f"  {sample_name}: {len(new_names)} cells, barcode 已添加前缀")
            adata = None  # 释放内存
        except Exception as e:
            logger.warning(f"  {sample_name} 修复 barcode 失败: {e}")
else:
    logger.info("  跳过 obs_names 修复（--skip-obs-names-fix）")


# ========== 5. 合并为 AnnDataSet ==========
logger.info("\n[Step 3/10] 合并为 AnnDataSet...")

merged_path = os.path.join(h5ads_dir, "merged.h5ads")
data = snap.AnnDataSet(
    adatas=h5ads_pairs,
    filename=merged_path,
)
logger.info(f"  合并完成: {data.n_obs} cells x {data.n_vars} features")
logger.info(f"  保存路径: {merged_path}")


# ========== 6. 验证 barcode 唯一性 ==========
logger.info("\n[Step 4/10] 验证细胞 barcode 唯一性...")

unique_ids = list(data.obs_names)
if len(unique_ids) != len(set(unique_ids)):
    dup_count = len(unique_ids) - len(set(unique_ids))
    logger.error(f"  发现 {dup_count} 个重复 barcode! 请检查 --skip-obs-names-fix 参数")
    sys.exit(1)
else:
    logger.info(f"  所有 {data.n_obs} 个细胞 barcode 已唯一化")


# ========== 7. 复制元信息到 AnnDataSet 级别 ==========
logger.info("\n[Step 5/10] 复制元信息 (metadata)...")

"""
说明: snap.AnnDataSet 合并后，只有 .obs['sample'] 会自动创建。
其他列（如 Age, n_fragment, tsse）保存在各个子样本中，
不会自动提升到 AnnDataSet 级别。
如果不复制，后续 color='Age' 会报错。
参考: SnapATAC2 作者 kai zhang 在 GitHub issue #230 的回复
"""

# 先读取第一个样本，看看有哪些 obs 列
try:
    first_adata = snap.read(h5ads_pairs[0][1], backed=None)
    obs_columns = list(first_adata.obs.columns)
    first_adata = None
    logger.info(f"  检测到 obs 列: {obs_columns}")
except Exception as e:
    logger.warning(f"  读取第一个样本失败: {e}")
    obs_columns = []

# 关键列：确保复制到合并后的数据集
key_columns = ["Age", "SampleIndex", "n_fragment", "tsse", "RawDataPath"]
for k in key_columns:
    if k not in obs_columns:
        obs_columns.append(k)

# 逐个尝试复制（更安全的方式：逐个读取子数据集拼接）
copied = []
failed = []
for key in obs_columns:
    # 如果已经存在则跳过
    try:
        _ = data.obs[key]
        continue
    except Exception:
        pass

    try:
        vals = []
        for sample_name, h5ad_path in h5ads_pairs:
            ad = snap.read(h5ad_path, backed=None)
            if key in ad.obs.columns:
                vals.extend(list(ad.obs[key]))
            else:
                vals.extend([None] * ad.n_obs)
            ad = None
        data.obs[key] = vals
        copied.append(key)
    except Exception as e:
        failed.append(f"{key}({e})")

if copied:
    logger.info(f"  成功复制: {copied}")
if failed:
    logger.warning(f"  复制失败: {failed}")

# 如果 Age 列不存在，从样本名推断
age_exists = False
try:
    _ = data.obs["Age"]
    age_exists = True
except Exception:
    pass

if not age_exists:
    logger.warning("  'Age' 列不存在，尝试从样本名推断...")
    ages = []
    for s in list(data.obs["sample"]):
        s_str = str(s)
        if "E16.5" in s_str:
            ages.append("E16.5")
        elif "P5" in s_str:
            ages.append("P5")
        elif "P0" in s_str:
            ages.append("P0")
        elif "Adult" in s_str:
            ages.append("Adult")
        else:
            ages.append("Unknown")
    data.obs["Age"] = ages
    logger.info(f"  Age 分布: {dict(Counter(ages))}")
else:
    logger.info(f"  Age 分布: {dict(Counter(data.obs['Age']))}")


# ========== 8. 特征选择 ==========
logger.info(f"\n[Step 6/10] 特征选择 (n_features={args.n_features})...")
snap.pp.select_features(data, n_features=args.n_features)
logger.info(f"  选中 {data.n_vars} 个特征")


# ========== 9. Spectral 降维 ==========
logger.info("\n[Step 7/10] Spectral embedding...")
snap.tl.spectral(data)
logger.info("  Spectral 完成")


# ========== 10. UMAP (未校正) ==========
logger.info("\n[Step 8/10] UMAP (未校正)...")
snap.tl.umap(data, n_comps=2)

for color_by in ["Age", "sample"]:
    try:
        fig = snap.pl.umap(data, color=color_by, interactive=False)
        out_path = os.path.join(fig_dir, f"UMAP_uncorrected_by_{color_by}.png")
        save_snap_figure(fig, out_path)
    except Exception as e:
        logger.error(f"  UMAP (uncorrected, {color_by}) 绘图失败: {e}")


# ========== 11. MNN 批次校正 (默认推荐) + 聚类 ==========
logger.info("\n[Step 9/10] MNN 批次校正 + 聚类...")
try:
    snap.pp.mnc_correct(data, batch="sample", key_added="X_spectral_mnn", n_jobs=args.n_jobs)
    logger.info("  MNN 校正完成")

    # KNN 图 + Leiden 聚类
    logger.info("  KNN + Leiden 聚类 (MNN)...")
    snap.pp.knn(data, use_rep="X_spectral_mnn")
    snap.tl.leiden(data, key_added="leiden_mnn")
    logger.info(f"  Leiden 聚类完成，共 {len(set(data.obs['leiden_mnn']))} 个 cluster")

    # UMAP
    snap.tl.umap(data, n_comps=2, use_rep="X_spectral_mnn")

    for color_by in ["Age", "sample", "leiden_mnn"]:
        try:
            fig = snap.pl.umap(data, color=color_by, interactive=False)
            out_path = os.path.join(fig_dir, f"UMAP_mnn_corrected_by_{color_by}.png")
            save_snap_figure(fig, out_path)
        except Exception as e:
            logger.error(f"  UMAP (MNN, {color_by}) 绘图失败: {e}")

except Exception as e:
    logger.error(f"  MNN 校正失败: {e}")


# ========== 12. Harmony 批次校正 (可选对比) + 聚类 ==========
logger.info("\n[Step 10/10] Harmony 批次校正 + 聚类...")
try:
    snap.pp.harmony(data, batch="sample", key_added="X_spectral_harmony", n_jobs=args.n_jobs)
    logger.info("  Harmony 校正完成")

    # KNN 图 + Leiden 聚类
    logger.info("  KNN + Leiden 聚类 (Harmony)...")
    snap.pp.knn(data, use_rep="X_spectral_harmony")
    snap.tl.leiden(data, key_added="leiden_harmony")
    logger.info(f"  Leiden 聚类完成，共 {len(set(data.obs['leiden_harmony']))} 个 cluster")

    # UMAP
    snap.tl.umap(data, n_comps=2, use_rep="X_spectral_harmony")

    for color_by in ["Age", "sample", "leiden_harmony"]:
        try:
            fig = snap.pl.umap(data, color=color_by, interactive=False)
            out_path = os.path.join(fig_dir, f"UMAP_harmony_corrected_by_{color_by}.png")
            save_snap_figure(fig, out_path)
        except Exception as e:
            logger.error(f"  UMAP (Harmony, {color_by}) 绘图失败: {e}")

except Exception as e:
    logger.error(f"  Harmony 校正失败: {e}")


# ========== 13. 保存汇总报告 ==========
logger.info("\n[保存] 生成汇总报告...")

summary_records = []
for sample_name, h5ad_path in h5ads_pairs:
    try:
        ad = snap.read(h5ad_path, backed=None)
        record = {
            "sample": sample_name,
            "n_cells": ad.n_obs,
            "n_features": ad.n_vars,
            "median_tsse": float(ad.obs["tsse"].median()) if "tsse" in ad.obs.columns else np.nan,
            "median_n_fragment": float(ad.obs["n_fragment"].median()) if "n_fragment" in ad.obs.columns else np.nan,
        }
        summary_records.append(record)
        ad = None
    except Exception as e:
        logger.warning(f"  读取 {sample_name} 汇总信息失败: {e}")

if summary_records:
    summary_df = pd.DataFrame(summary_records)
    summary_path = os.path.join(report_dir, "merge_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"  汇总表已保存: {summary_path}")
    logger.info(f"\n{summary_df.to_string(index=False)}")


# ========== 14. 导出普通 AnnData (可选) ==========
if args.export_adata:
    logger.info("\n[可选] 导出普通 AnnData...")
    try:
        adata_merged = data.to_adata()
        adata_path = os.path.join(h5ad_dir, "merged.h5ad")
        adata_merged.write_h5ad(adata_path)
        logger.info(f"  已导出: {adata_path} ({adata_merged.n_obs} cells x {adata_merged.n_vars} features)")
    except Exception as e:
        logger.warning(f"  导出 AnnData 失败: {e}")


# ========== 15. 关闭并保存 ==========
logger.info("\n[保存] 关闭 AnnDataSet...")
try:
    data.close()
    logger.info(f"  AnnDataSet 已关闭: {merged_path}")
except AttributeError:
    logger.info("  当前版本无需显式 close()")
except Exception as e:
    logger.warning(f"  关闭时警告: {e}")

logger.info(f"\n{'='*60}")
logger.info("Step 1 完成!")
logger.info(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info("=" * 60)
logger.info("\n输出文件:")
logger.info(f"  AnnDataSet: {merged_path}")
logger.info(f"  Figures:    {fig_dir}")
logger.info(f"  Reports:    {report_dir}")
logger.info("\n下一步: Step 2 聚类与细胞类型注释")
logger.info("  推荐用 MNN 校正后的结果: obs['leiden_mnn']")
logger.info("=" * 60)
