#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import logging
import warnings
import gzip
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# 设置 matplotlib 后台，防止在服务器上报错
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import snapatac2 as snap

# ========== 0. 基础设置 ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 默认 QC 阈值
DEFAULT_MIN_TSSE = 4
DEFAULT_MIN_FRAGMENTS = 1000
DEFAULT_MAX_FRAGMENTS = 50000
DEFAULT_N_FEATURES = 500000

# 阶段颜色，用于绘图
STAGE_COLORS = {
    "E16.5": "#E69F00",
    "P0": "#56B4E9",
    "P5": "#009E73",
    "Adult": "#CC79A7",
}


# ========== 1. 解析命令行参数 ==========
parser = argparse.ArgumentParser(
    description="Step 0: 单样本 QC 与预处理 (SnapATAC2)",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
示例:
    python 01_import_fragments_filter_nuclei_select_features_remove_doublets.py \\
        --sample-info /path/to/library_metadata.csv \\
        --outdir /path/to/qc_output \\
        --min-tsse 4 \\
        --n-features 500000
    """,
)
parser.add_argument(
    "--sample-info",
    type=str,
    required=True,
    help="样本信息表路径 (.xlsx 或 .csv)",
)
parser.add_argument(
    "--outdir",
    type=str,
    required=True,
    help="输出目录",
)
parser.add_argument(
    "--fragment-dir",
    type=str,
    default=None,
    help=("Optional directory containing released fragment files. When set, "
          "the basename in fragments_path is resolved inside this directory."),
)
parser.add_argument(
    "--gtf",
    type=str,
    required=True,
    help="GTF 注释文件路径",
)
parser.add_argument(
    "--chrom-sizes",
    type=str,
    required=True,
    help="染色体大小文件路径",
)
parser.add_argument(
    "--min-tsse",
    type=float,
    default=DEFAULT_MIN_TSSE,
    help=f"TSS enrichment 过滤阈值 (默认: {DEFAULT_MIN_TSSE})",
)
parser.add_argument(
    "--min-frags",
    type=int,
    default=DEFAULT_MIN_FRAGMENTS,
    help=f"最小片段数 (默认: {DEFAULT_MIN_FRAGMENTS})",
)
parser.add_argument(
    "--max-frags",
    type=int,
    default=DEFAULT_MAX_FRAGMENTS,
    help=f"最大片段数 (默认: {DEFAULT_MAX_FRAGMENTS})",
)
parser.add_argument(
    "--n-features",
    type=int,
    default=DEFAULT_N_FEATURES,
    help=f"选择的特征数 (默认: {DEFAULT_N_FEATURES})",
)
parser.add_argument(
    "--samples",
    type=str,
    default=None,
    help="指定处理的样本名，逗号分隔 (默认处理全部)",
)
parser.add_argument(
    "--skip-existing",
    action="store_true",
    help="跳过已存在 filtered.h5ad 的样本",
)
parser.add_argument(
    "--force-chr",
    type=str,
    default="auto",
    choices=["auto", "yes", "no"],
    help="强制指定染色体 'chr' 前缀: auto=自动检测, yes=强制添加chr, no=强制移除chr",
)

args = parser.parse_args()

logger.info("=" * 60)
logger.info("Step 0: 单样本 QC 与预处理")
logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info("=" * 60)
logger.info(f"样本信息: {args.sample_info}")
logger.info(f"输出目录: {args.outdir}")
logger.info(f"GTF: {args.gtf}")
logger.info(f"Chrom sizes: {args.chrom_sizes}")
logger.info(f"Min TSSE: {args.min_tsse}")
logger.info(f"Min Frags: {args.min_frags}")
logger.info(f"Max Frags: {args.max_frags}")
logger.info(f"N features: {args.n_features}")


# ========== 2. 读取样本信息 ==========
sample_info_path = Path(args.sample_info)

# 自动判断文件类型（有些 .csv 实际是 xlsx）
is_excel = False
try:
    with open(sample_info_path, "rb") as f:
        header = f.read(4)
        if header == b"PK\x03\x04":
            is_excel = True
except Exception:
    pass

if is_excel or sample_info_path.suffix in [".xlsx", ".xls"]:
    logger.info(f"读取 Excel: {sample_info_path}")
    sample_df = pd.read_excel(sample_info_path)
else:
    logger.info(f"读取 CSV: {sample_info_path}")
    for encoding in ["utf-8", "gbk", "latin1"]:
        try:
            sample_df = pd.read_csv(sample_info_path, encoding=encoding)
            logger.info(f"  CSV 编码: {encoding}")
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"无法解析样本信息表的编码: {sample_info_path}")

if "in_final_dataset" in sample_df.columns:
    sample_df = sample_df[
        sample_df["in_final_dataset"].astype(str).str.lower().eq("yes")
    ].copy()
if sample_df.astype(str).apply(
    lambda column: column.str.contains("AR1", regex=False).any()
).any():
    raise ValueError("废弃的 AR1 样本出现在最终输入表中")

# 检查必需列
required_cols = ["library_id", "developmental_stage", "fragments_path"]
for col in required_cols:
    if col not in sample_df.columns:
        raise ValueError(f"样本信息表缺少必需列: {col}")

# 如果用户指定了样本，进行筛选
if args.samples:
    target_samples = [s.strip() for s in args.samples.split(",")]
    sample_df = sample_df[sample_df["library_id"].isin(target_samples)]
    logger.info(f"筛选后处理 {len(sample_df)} 个指定样本")
else:
    logger.info(f"共 {len(sample_df)} 个样本待处理")


# ========== 3. 读取染色体大小文件 ==========
logger.info("读取染色体大小文件...")
chrom_sizes_df = pd.read_csv(args.chrom_sizes, sep="\t", header=None, index_col=0)
chrom_sizes_df = chrom_sizes_df.iloc[:22, :]  # 大鼠保留前22条常染色体
logger.info(f"  共 {len(chrom_sizes_df)} 条染色体")


# ========== 4. 逐个样本处理 ==========
all_reports = []
failed_samples = []

for idx, row in sample_df.iterrows():
    sample_name = row["library_id"]
    fragment_path = row["fragments_path"]
    if args.fragment_dir is not None:
        fragment_path = str(Path(args.fragment_dir) / Path(fragment_path).name)
    age = row["developmental_stage"]
    sample_index = row.get("animal_id", "")

    logger.info(f"\n{'='*60}")
    logger.info(f"处理样本: {sample_name} (Age: {age})")
    logger.info(f"Fragment 文件: {fragment_path}")
    logger.info(f"{'='*60}")

    # 检查 fragment 文件是否存在
    if not os.path.exists(fragment_path):
        logger.error(f"Fragment 文件不存在，跳过: {fragment_path}")
        failed_samples.append(sample_name)
        continue

    # 如果指定了 skip-existing，检查是否已处理过
    if args.skip_existing:
        filtered_check = os.path.join(
            args.outdir, "01.h5ad", sample_name, f"{sample_name}_filtered.h5ad"
        )
        if os.path.exists(filtered_check):
            logger.info(f"  已存在，跳过: {filtered_check}")
            continue

    # 创建输出目录
    sample_h5ad_dir = os.path.join(args.outdir, "01.h5ad", sample_name)
    fig_dir = os.path.join(args.outdir, "03.figure", "Fig2_QC", "per_sample")
    report_dir = os.path.join(args.outdir, "02.result", "QC_reports")
    for d in [sample_h5ad_dir, fig_dir, report_dir]:
        os.makedirs(d, exist_ok=True)

    try:
        # ---- 4.1 检测染色体 'chr' 前缀 ----
        has_chr = False
        if args.force_chr == "yes":
            has_chr = True
            logger.info("  用户指定: 添加 chr 前缀")
        elif args.force_chr == "no":
            has_chr = False
            logger.info("  用户指定: 不添加 chr 前缀")
        else:
            # 自动检测：读取 fragment 文件前 1000 行
            try:
                with gzip.open(fragment_path, "rt") as f:
                    for i, line in enumerate(f):
                        if i >= 1000:
                            break
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        parts = line.split("\t")
                        if len(parts) >= 4:
                            chrom = parts[0]
                            if chrom.startswith("chr"):
                                has_chr = True
                                logger.info(f"  自动检测到 chr 前缀 (如 {chrom})")
                            else:
                                has_chr = False
                                logger.info(f"  自动检测到无 chr 前缀 (如 {chrom})")
                            break
            except Exception as e:
                logger.warning(f"  自动检测失败: {e}，默认无 chr 前缀")

        # 根据检测结果准备 chrom_sizes 字典
        chrom_sizes = chrom_sizes_df.copy()
        chrom_sizes.index = chrom_sizes.index.astype(str).str.replace("chr", "", regex=False)
        if has_chr:
            chrom_sizes.index = "chr" + chrom_sizes.index
        chrom_sizes_dict = chrom_sizes.to_dict()[1]

        # ---- 4.2 导入数据 ----
        logger.info("[1/9] 导入 fragment 文件...")
        data = snap.pp.import_data(
            fragment_path,
            chrom_sizes=chrom_sizes_dict,
            sorted_by_barcode=False,
        )
        logger.info(f"  原始数据: {data.n_obs} cells x {data.n_vars} features")

        # 添加样本元信息
        data.obs["sample"] = sample_name
        data.obs["Age"] = age
        data.obs["SampleIndex"] = sample_index
        data.obs["RawDataPath"] = fragment_path

        # ---- 4.3 计算 TSS Enrichment ----
        logger.info("[2/9] 计算 TSS enrichment...")
        snap.metrics.tsse(data, gene_anno=args.gtf)
        median_tsse = float(data.obs["tsse"].median())
        logger.info(f"  TSS enrichment: median={median_tsse:.2f}, mean={data.obs['tsse'].mean():.2f}")

        # 若 TSS 异常低，尝试调整染色体命名
        if median_tsse < 2.0:
            logger.warning(f"  TSS 异常低 ({median_tsse:.2f})，尝试调整染色体命名...")
            try:
                ref_seqs = data.uns["reference_sequences"]["reference_seq_name"].copy()
                if str(ref_seqs[0]).startswith("chr"):
                    data.uns["reference_sequences"]["reference_seq_name"] = ref_seqs.str.replace("chr", "", regex=False)
                else:
                    data.uns["reference_sequences"]["reference_seq_name"] = "chr" + ref_seqs
                snap.metrics.tsse(data, gene_anno=args.gtf)
                new_median = float(data.obs["tsse"].median())
                logger.info(f"  调整后: median={new_median:.2f}")
                if new_median < median_tsse:
                    logger.warning("  调整无效，恢复原始设置")
                    data.uns["reference_sequences"]["reference_seq_name"] = ref_seqs
                    snap.metrics.tsse(data, gene_anno=args.gtf)
            except Exception as e:
                logger.warning(f"  调整失败: {e}")

        # 记录过滤前的 QC 指标
        n_cells_raw = data.n_obs
        median_fragments_raw = float(data.obs["n_fragment"].median())
        mean_fragments_raw = float(data.obs["n_fragment"].mean())
        median_tsse_raw = float(data.obs["tsse"].median())
        mean_tsse_raw = float(data.obs["tsse"].mean())
        min_tsse_raw = float(data.obs["tsse"].min())
        max_tsse_raw = float(data.obs["tsse"].max())

        # ---- 4.4 绘制 QC 图（过滤前） ----
        logger.info("  绘制 QC 图...")

        # 准备绘图数据
        log10_nFrags = np.log10(data.obs["n_fragment"].values)
        tsse_values = data.obs["tsse"].values
        color = STAGE_COLORS.get(age, "gray")

        # 图1: log10(nFrags) 小提琴图
        fig, ax = plt.subplots(figsize=(5, 5))
        parts = ax.violinplot([log10_nFrags], positions=[1], showmeans=True, showmedians=True)
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        ax.set_xticks([1])
        ax.set_xticklabels([sample_name], rotation=45, ha="right")
        ax.set_ylabel("log10(Number of Fragments)")
        ax.set_title(f"{sample_name}\nlog10(nFrags)")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"{sample_name}_violin_log10nFrags.png"), dpi=150, bbox_inches="tight")
        plt.close()

        # 图2: TSS enrichment 小提琴图
        fig, ax = plt.subplots(figsize=(5, 5))
        parts = ax.violinplot([tsse_values], positions=[1], showmeans=True, showmedians=True)
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        ax.axhline(args.min_tsse, color="red", linestyle="--", label=f"cutoff={args.min_tsse}")
        ax.set_xticks([1])
        ax.set_xticklabels([sample_name], rotation=45, ha="right")
        ax.set_ylabel("TSS Enrichment")
        ax.set_title(f"{sample_name}\nTSS Enrichment")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"{sample_name}_violin_TSSE.png"), dpi=150, bbox_inches="tight")
        plt.close()

        # 图3: TSS Enrichment vs log10(nFrags) 散点密度图
        fig, ax = plt.subplots(figsize=(6, 5.5))
        h = ax.hist2d(log10_nFrags, tsse_values, bins=100, cmap="YlOrRd", cmin=1)
        ax.axhline(args.min_tsse, color="blue", linestyle="--", linewidth=1.5, label=f"TSS cutoff={args.min_tsse}")
        ax.set_xlabel("log10(Number of Fragments)")
        ax.set_ylabel("TSS Enrichment")
        ax.set_title(f"{sample_name}\nTSS Enrichment vs log10(nFrags)")
        plt.colorbar(h[3], ax=ax, label="Cell density")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"{sample_name}_scatter_TSSE_vs_nFrags.png"), dpi=150, bbox_inches="tight")
        plt.close()

        # 图4: SnapATAC2 官方 TSS profile（插入谱）
        try:
            fig = snap.pl.tsse(data, interactive=False, show=False)
            fig.savefig(os.path.join(fig_dir, f"{sample_name}_TSS_profile.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info("  TSS profile 已保存")
        except Exception as e:
            logger.warning(f"  TSS profile 绘图失败: {e}")

        # 图5: 片段大小分布
        try:
            fig = snap.pl.frag_size_distr(data, interactive=False, show=False)
            fig.savefig(os.path.join(fig_dir, f"{sample_name}_frag_size_distr.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info("  Fragment size distribution 已保存")
        except Exception as e:
            logger.warning(f"  Fragment size distribution 绘图失败: {e}")

        # ---- 4.5 过滤低质量细胞 ----
        logger.info(f"[3/9] 过滤细胞 (min_tsse={args.min_tsse}, min_frags={args.min_frags}, max_frags={args.max_frags})...")
        n_before_filter = data.n_obs

        # 先按 TSS 过滤（snap 内置）
        snap.pp.filter_cells(data, min_tsse=args.min_tsse)

        # 再按 fragment 数过滤（手动）
        mask = (data.obs["n_fragment"] >= args.min_frags) & (data.obs["n_fragment"] <= args.max_frags)
        data = data[mask, :]

        n_after_filter = data.n_obs
        logger.info(f"  过滤后细胞数: {n_after_filter} (去除了 {n_before_filter - n_after_filter})")

        if data.n_obs == 0:
            logger.error("  过滤后无细胞剩余，跳过该样本")
            failed_samples.append(sample_name)
            continue

        # ---- 4.6 构建 Tile Matrix ----
        logger.info("[4/9] 添加 tile matrix (500bp bins)...")
        snap.pp.add_tile_matrix(data)
        logger.info(f"  Tile matrix: {data.n_obs} cells x {data.n_vars} tiles")

        # ---- 4.7 特征选择 ----
        logger.info(f"[5/9] 选择特征 (n_features={args.n_features})...")
        snap.pp.select_features(data, n_features=args.n_features)
        logger.info(f"  选中 {data.n_vars} 个特征")

        # ---- 4.8 Spectral 降维 ----
        logger.info("[6/9] Spectral embedding...")
        snap.tl.spectral(data)
        logger.info("  Spectral 完成")

        # ---- 4.9 UMAP 降维 ----
        logger.info("[7/9] UMAP embedding...")
        snap.tl.umap(data)
        logger.info("  UMAP 完成")

        # ---- 4.10 保存双细胞检测前数据 ----
        logger.info("[8/9] 保存 noscrublet 数据...")
        noscrublet_path = os.path.join(sample_h5ad_dir, f"{sample_name}_noscrublet.h5ad")
        data.write_h5ad(noscrublet_path)
        logger.info(f"  已保存: {noscrublet_path}")

        # ---- 4.11 双细胞检测与过滤 ----
        logger.info("[9/9] Scrublet 双细胞检测...")
        n_before_doublet = data.n_obs
        snap.pp.scrublet(data)

        doublet_score_median = float(data.obs["doublet_score"].median())
        doublet_probability_median = float(data.obs["doublet_probability"].median())

        snap.pp.filter_doublets(data)
        n_after_doublet = data.n_obs
        n_doublets = n_before_doublet - n_after_doublet

        logger.info(f"  双细胞去除: {n_doublets} 个 ({n_doublets/n_before_doublet*100:.1f}%)")
        logger.info(f"  最终细胞数: {n_after_doublet}")

        # 保存过滤后数据
        filtered_path = os.path.join(sample_h5ad_dir, f"{sample_name}_filtered.h5ad")
        data.write_h5ad(filtered_path)
        logger.info(f"  已保存: {filtered_path}")

        # ---- 4.12 汇总 QC 指标 ----
        report = {
            "sample": sample_name,
            "age": age,
            "sample_index": sample_index,
            "raw_data_path": fragment_path,
            "n_cells_raw": n_cells_raw,
            "n_cells_filtered": n_after_doublet,
            "cell_retention_rate_pct": round(n_after_doublet / n_cells_raw * 100, 2) if n_cells_raw > 0 else 0,
            "median_fragments_raw": median_fragments_raw,
            "mean_fragments_raw": mean_fragments_raw,
            "median_tsse_raw": median_tsse_raw,
            "mean_tsse_raw": mean_tsse_raw,
            "min_tsse_raw": min_tsse_raw,
            "max_tsse_raw": max_tsse_raw,
            "median_fragments_filtered": float(data.obs["n_fragment"].median()),
            "mean_fragments_filtered": float(data.obs["n_fragment"].mean()),
            "median_tsse_filtered": float(data.obs["tsse"].median()),
            "mean_tsse_filtered": float(data.obs["tsse"].mean()),
            "min_tsse_filtered": float(data.obs["tsse"].min()),
            "max_tsse_filtered": float(data.obs["tsse"].max()),
            "n_doublets_removed": n_doublets,
            "doublet_rate_pct": round(n_doublets / n_before_doublet * 100, 2) if n_before_doublet > 0 else 0,
            "doublet_score_median": doublet_score_median,
            "doublet_probability_median": doublet_probability_median,
            "filtered_h5ad": filtered_path,
        }
        all_reports.append(report)
        logger.info(f"✅ {sample_name} 处理完成")

    except Exception as e:
        logger.error(f"❌ {sample_name} 处理失败: {e}", exc_info=True)
        failed_samples.append(sample_name)
        continue


# ========== 5. 保存汇总报告 ==========
if all_reports:
    summary_df = pd.DataFrame(all_reports)
    summary_path = os.path.join(args.outdir, "02.result", "QC_reports", "qc_summary.csv")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    summary_df.to_csv(summary_path, index=False)

    logger.info(f"\n{'='*60}")
    logger.info("QC 汇总报告")
    logger.info(f"{'='*60}")
    logger.info(f"成功样本: {len(all_reports)} / {len(sample_df)}")
    logger.info(f"总过滤后细胞数: {summary_df['n_cells_filtered'].sum():,.0f}")
    logger.info(f"\n各样本概览:")
    logger.info(
        f"\n{summary_df[['sample', 'age', 'n_cells_raw', 'n_cells_filtered', 'cell_retention_rate_pct', 'median_tsse_raw', 'doublet_rate_pct']].to_string(index=False)}"
    )
    logger.info(f"\nQC 汇总表已保存: {summary_path}")

if failed_samples:
    logger.warning(f"\n失败样本 ({len(failed_samples)}): {', '.join(failed_samples)}")

logger.info(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info("=" * 60)
