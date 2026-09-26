# -*- coding: utf-8 -*-
"""
p 值排序版 motif 热图
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

FIGURE3_STYLE_DIR = Path(__file__).resolve().parents[1] / "figure3"
if str(FIGURE3_STYLE_DIR) not in sys.path:
    sys.path.insert(0, str(FIGURE3_STYLE_DIR))

from figure3_style import CELL_TYPE_DISPLAY_NAMES

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input", required=True, type=Path,
                    help="All cell-type motif enrichment results CSV.")
parser.add_argument("--output-prefix", required=True, type=Path)
parser.add_argument("--selection-output", type=Path, default=None)
parser.add_argument("--matrix-output", type=Path, default=None)
args = parser.parse_args()
if not args.input.is_file() or args.input.stat().st_size == 0:
    raise FileNotFoundError(args.input)

# Style-only inheritance from the reference Figure 5a. Anchor colours were
# sampled from its published colour bar: pale grey -> blue -> purple -> black.
REFERENCE_FIG5A_CMAP = LinearSegmentedColormap.from_list(
    "reference_fig5a_blue_purple_black",
    [
        "#d4dade", "#93b7db", "#6198cd", "#4e7bba", "#5269ae",
        "#5d56a3", "#63479b", "#6a3391", "#5f2872", "#361b3d", "#0f050d",
    ],
    N=256,
)

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)

ALPHA = 0.05
LOG2FC_MIN = 0.0
TOP_N = 10
DISPLAY_CAP = 50.0
# 只保留 Top5 的细胞类型
TOP_N_SMALL = 5
SMALL_TYPES = {
    "FB_DP", "FB_DS", "FB_fascia", "FB_papi", "FB_reti",
    "Immu_B", "Immu_Macr", "Immu_Neu", "Immu_T_NK",
    "Endo", "MELAN", "Muscle", "Neuron", "Pericyte", "SCH",
}

CELL_TYPE_ORDER = [
    "KC_basal", "KC_sbasal", "KC_ORS", "KC_placode",
    "KC_HFSC1", "KC_HFSC2", "KC_IRS1", "KC_IRS2",
    "FB_DP", "FB_DS", "FB_fascia", "FB_papi", "FB_reti",
    "Immu_B", "Immu_Macr", "Immu_Neu", "Immu_T_NK",
    "Endo", "MELAN", "Muscle", "Pericyte", "SCH",
]

# ---------------- 谱系白名单（大写基因符号） ----------------
KC = {
    "TP63", "TP73", "TP53", "GATA3", "GRHL1", "GRHL2", "GRHL3", "KLF4", "KLF5",
    "CEBPA", "CEBPB", "CEBPD", "CEBPG", "JUN", "JUNB", "JUND", "FOS", "FOSB",
    "FOSL1", "FOSL2", "ATF3", "ATF4", "IRF6", "POU2F3", "POU3F1", "POU3F3",
    "POU2F1", "POU2F2", "POU3F4", "POU5F1", "POU6F2", "SOX2", "SOX9", "SOX4",
    "SOX11", "MAF", "MAFB", "RBPJ", "RUNX1", "RUNX2", "RUNX3", "ZNF750", "HOPX",
    "LEF1", "TCF7", "TCF7L1", "TCF7L2", "SOX18", "LHX2", "LHX5", "LHX6",
    "HOXD13", "HOXA5", "HOXA3", "HOXB1", "HOXB6",
    "HOXB7", "HOXD3", "GSX2", "MEOX2", "NFKB1", "NFKB2", "REL", "RELA",
    "RELB", "PAX3", "PAX4", "PAX6", "MSX1", "MSX2", "ALX4", "HNF1A", "HNF1B",
    "PITX1", "PITX2", "PITX3", "SIX1", "SIX4", "DLX1", "DLX2", "DLX3", "DLX5",
    "DLX6", "SHOX", "ETS1", "ELF3", "NFATC1", "NFATC2",
}

FB = {
    "TCF21", "TWIST1", "TWIST2", "MEIS1", "MEIS2", "PBX1", "PBX2", "PBX3",
    "PRRX1", "PRRX2", "SOX4", "SOX5", "SOX6", "SOX9", "SOX11", "SOX7", "SOX17",
    "STAT1", "STAT2", "STAT3", "STAT4", "STAT5A", "STAT5B", "STAT6", "GATA4",
    "GATA6", "MEF2A", "MEF2C", "MEF2D", "NFIL3", "HLF", "ONECUT1", "ONECUT2",
    "ONECUT3", "FOXC1", "FOXC2", "BARX1", "BARX2", "ISL1", "LHX3", "LHX9",
    "PHOX2A", "PHOX2B", "LMX1B", "JUN", "JUNB", "JUND", "FOS", "FOSB", "FOSL1",
    "FOSL2", "ATF3", "ATF4", "CEBPA", "CEBPB", "CEBPD", "CEBPG", "BACH1",
    "BACH2", "ALX1", "ALX3", "ALX4", "RUNX1", "RUNX2", "RUNX3", "HNF1A", "HNF1B",
}

Immu = {
    "SPI1", "SPIB", "SPIC", "IRF1", "IRF4", "IRF5", "IRF8", "TBX21", "EOMES",
    "GATA3", "RORC", "BCL11A", "BCL11B", "EBF1", "PAX5", "TCF3", "TCF4", "TCF12",
    "RUNX1", "RUNX3", "ETS1", "ETV6", "ETV7", "ELF1", "ELF2", "ELF3", "ELF4",
    "ELF5", "ERG", "FLI1", "GABPA", "EHF", "KLF2", "KLF3", "KLF4", "NFATC1",
    "NFATC2", "FOXP3", "FOXO1", "FOXO3", "ZBTB16", "BATF", "BATF3", "MAF", "MAFB",
    "ID2", "AHR", "SMAD3", "SMAD4", "POU2F2", "JUN", "JUNB", "JUND", "FOS", "FOSB",
    "FOSL1", "FOSL2", "STAT1", "STAT3", "STAT5A", "STAT5B", "STAT4", "CEBPA",
    "CEBPB", "CEBPD", "CEBPE", "CEBPG", "TBX2", "TBX3",
}

Endo = {
    "ERG", "FLI1", "ETV2", "ETS1", "ETS2", "ELF1", "ELF2", "ELF3", "ELF4",
    "GABPA", "ETV6", "EHF", "SOX7", "SOX17", "SOX18", "GATA2", "NR2F2", "PROX1",
    "KLF2", "KLF4", "FOXC1", "FOXC2", "TAL1", "LYL1", "SMAD1", "SMAD5", "HEY1",
    "HEY2",
}

MELAN = {
    "MITF", "PAX3", "SOX10", "SOX11", "TFAP2A", "TFAP2B", "TFAP2C", "TFAP2E",
    "NR4A1", "NR4A2", "USF1", "USF2", "OTX2", "LEF1", "ZEB2",
}

Muscle = {
    "MYOD1", "MYOG", "MYF5", "MYF6", "MEF2A", "MEF2C", "MEF2D", "PAX3", "PAX7",
    "SIX1", "SIX4", "TEAD1", "TEAD3", "TEAD4", "SRF", "ASCL1", "ASCL2",
}

Neuron = {
    "NEUROD1", "NEUROD2", "NEUROD6", "NEUROG1", "NEUROG2", "ASCL1", "SOX2",
    "SOX4", "SOX11", "POU3F2", "POU3F3", "POU4F1", "PBX1", "MEIS1", "LHX2",
    "LHX5", "LHX6", "LHX3", "DLX1", "DLX2", "DLX5", "DLX6", "GSX2", "ONECUT1",
    "ONECUT2", "NR4A2", "SIX3", "SIX6", "ZEB1", "ZEB2", "DMRT3", "NEUROD4",
    "ISL1", "SOX14", "ZIC1", "ZIC2", "EBF3", "GATA2", "TP73", "MXI1", "NFYC",
}

Pericyte = {
    "NR2F1", "NR2F2", "NR2F6", "PPARA", "PPARD", "PPARG", "RXRA", "RXRB", "RXRG",
    "THRA", "THRB", "RBPJ", "HES1", "HEY1", "HEY2", "FOXD1", "FOXF1", "EBF1",
    "EBF2", "EBF3", "EBF4", "TEAD1", "TEAD3", "TEAD4", "SOX9", "SOX11", "MEIS2",
    "PBX1", "PBX2",
}

SCH = {
    "SOX10", "SOX9", "SOX2", "SOX4", "SOX6", "SOX11", "EGR2", "NR4A1", "NR4A2",
    "TEAD3", "TEAD4", "NFATC4", "POU3F1", "FOXC1", "ZEB2", "LHX2", "PAX3",
    "POU4F1",
}

LINEAGE_WHITELIST = {
    "KC": KC, "FB": FB, "Immu": Immu, "Endo": Endo, "MELAN": MELAN,
    "Muscle": Muscle, "Neuron": Neuron, "Pericyte": Pericyte, "SCH": SCH,
}


def lineage_of(cell_type):
    for key in ("KC", "FB", "Immu"):
        if cell_type.startswith(key + "_"):
            return key
    return cell_type


args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
if args.selection_output is not None:
    args.selection_output.parent.mkdir(parents=True, exist_ok=True)
if args.matrix_output is not None:
    args.matrix_output.parent.mkdir(parents=True, exist_ok=True)

# ---------------- 读数据 ----------------
all_df = pd.read_csv(args.input)


def tf_label(row):
    name = row["name"]
    if name is not None and str(name).strip() not in ("", "nan", "None"):
        return str(name).strip().upper()
    rid = str(row["id"])
    return rid.split(" ", 1)[1].strip().upper() if " " in rid else rid.upper()


all_df["tf"] = all_df.apply(tf_label, axis=1)

# 去复合 motif（含 "::"）
all_df = all_df[~all_df["tf"].str.contains("::", na=False)]

# 按谱系白名单过滤
all_df["lineage"] = all_df["cell_type"].map(lineage_of)


def keep(row):
    return row["tf"] in LINEAGE_WHITELIST.get(row["lineage"], set())


all_df = all_df[all_df.apply(keep, axis=1)]

# 同一 cell_type 同一 tf 名多条 motif：保留 adj p 最小的一条
all_df = all_df.sort_values(
    ["cell_type", "tf", "adjusted p-value", "log2(fold change)"],
    ascending=[True, True, True, False],
)
all_df = all_df.drop_duplicates(["cell_type", "tf"], keep="first")

# ---------------- 显著 & 每类选 top ----------------
sig = all_df[
    (all_df["adjusted p-value"] < ALPHA) & (all_df["log2(fold change)"] > LOG2FC_MIN)
].copy()


def minus_log10_padj(p):
    if p <= 0:
        p = 1e-300
    return min(-np.log10(p), DISPLAY_CAP)


sig["minus_log10_padj"] = sig["adjusted p-value"].map(minus_log10_padj)

selected = []
for ct in CELL_TYPE_ORDER:
    ct_df = sig[sig["cell_type"] == ct].sort_values(
        ["adjusted p-value", "log2(fold change)"],
        ascending=[True, False],
    )
    n_top = TOP_N_SMALL if ct in SMALL_TYPES else TOP_N
    chosen = ct_df.head(n_top).copy()
    chosen["rank"] = np.arange(1, len(chosen) + 1)
    selected.append(chosen)

selected_df = pd.concat(selected, ignore_index=True)
selected_tfs = selected_df["tf"].drop_duplicates().tolist()

# Display order only: place FOSL1 immediately before FOSL2.
if "FOSL1" in selected_tfs and "FOSL2" in selected_tfs:
    selected_tfs.remove("FOSL1")
    selected_tfs.insert(selected_tfs.index("FOSL2"), "FOSL1")

if args.selection_output is not None:
    selected_df.to_csv(args.selection_output, index=False)
print(f"选中 motif 数(unique tf): {len(selected_tfs)}", flush=True)

# ---------------- 构建热图矩阵 ----------------
def plot_val(row):
    if (row["adjusted p-value"] < ALPHA) and (row["log2(fold change)"] > LOG2FC_MIN):
        return minus_log10_padj(row["adjusted p-value"])
    return 0.0


sig["minus_log10_padj_plot"] = sig.apply(plot_val, axis=1)

mat = sig[sig["tf"].isin(selected_tfs)].pivot_table(
    index="cell_type", columns="tf", values="minus_log10_padj_plot",
    aggfunc="max", fill_value=0,
)
mat = mat.reindex(index=CELL_TYPE_ORDER, columns=selected_tfs).fillna(0)

if args.matrix_output is not None:
    mat.to_csv(args.matrix_output)
print("热图矩阵已构建", flush=True)

# ---------------- 绘制热图 ----------------
n_cols = len(selected_tfs)
fig_w = max(12, 0.32 * n_cols)
fig_h = max(6, 0.42 * len(CELL_TYPE_ORDER))

vals = mat.values[mat.values > 0]
vmax = np.nanpercentile(vals, 95) if vals.size else 2
vmax = max(vmax, 2)

missing_display_names = [
    cell_type for cell_type in mat.index
    if cell_type not in CELL_TYPE_DISPLAY_NAMES
]
if missing_display_names:
    raise ValueError(
        "Figure 3 display-name mapping is missing cell types: "
        f"{missing_display_names}"
    )
display_labels = [CELL_TYPE_DISPLAY_NAMES[cell_type] for cell_type in mat.index]

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
fig.subplots_adjust(left=0.06, right=0.995, bottom=0.24, top=0.88)
cbar_ax = fig.add_axes([0.78, 0.925, 0.18, 0.022])
sns.heatmap(
    mat,
    cmap=REFERENCE_FIG5A_CMAP,
    vmin=0,
    vmax=vmax,
    linewidths=0.3,
    linecolor="lightgray",
    cbar_ax=cbar_ax,
    cbar_kws={"orientation": "horizontal"},
    yticklabels=display_labels,
    ax=ax,
)
ax.set_xlabel("")
ax.set_ylabel("")
plt.xticks(rotation=90, ha="right", fontsize=8)
plt.yticks(rotation=0, fontsize=9)
cbar_ax.invert_xaxis()
cbar_ax.tick_params(axis="x", labelsize=8, length=2, pad=1)
cbar_ax.set_title(r"$-\log_{10}$(adjusted $P$ value)", fontsize=8, pad=2)

plt.savefig(args.output_prefix.with_suffix(".pdf"), bbox_inches="tight")
plt.savefig(args.output_prefix.with_suffix(".svg"), bbox_inches="tight")
plt.savefig(args.output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
print(
    f"热图已保存: {args.output_prefix}.pdf/.png",
    flush=True,
)
print(f"vmax(95分位) = {vmax:.1f}", flush=True)
