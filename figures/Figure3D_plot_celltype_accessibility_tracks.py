#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Side-by-side peak tracks from CPM bigWig coverage —— 复现参考图峰值 + 保留自定义风格。

与参考脚本一致(决定峰值): 用 bigWig 覆盖度; 窗口为 TSS 上游 2 kb、下游 1 kb;
y 轴按每个基因列的全细胞类型最大值统一缩放(per-column)。
自定义风格: 细胞顺序/配色与 UMAP 一致; 基因结构按链向红蓝; 整列分格; 基因名在框下; 可编辑字体。

数据(本地): peak_plot/data/{bigwig/*.bw, Rattus_norvegicus.mRatBN7.2.dna.chr.gtf.gz, merged_peaks.bed}
"""
import argparse
import gzip
import os
import re
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "font.size": 7,
})
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

from figure3_style import (
    CELL_TYPE_COLORS,
    TRACK_CELL_TYPE_ORDER,
    TRACK_GENE_ORDER,
    display_name,
)

parser = argparse.ArgumentParser(description="Plot Figure 3D from cell-type CPM bigWig files.")
parser.add_argument("--bigwig-dir", required=True, type=Path)
parser.add_argument("--gtf", required=True, type=Path)
parser.add_argument("--peaks", required=True, type=Path)
parser.add_argument("--output-dir", required=True, type=Path)
parser.add_argument("--up", type=int, default=2000)
parser.add_argument("--down", type=int, default=1000)
args = parser.parse_args()
BWDIR = args.bigwig_dir
GTF = args.gtf
PEAKS = args.peaks
OUTPUT_DIR = args.output_dir
UP = args.up
DOWN = args.down
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
for path in (BWDIR, GTF, PEAKS):
    if not path.exists():
        raise FileNotFoundError(path)

# Imported after CLI/input validation so ``--help`` works without pyBigWig.
import pyBigWig

# ---- 参数 ----
# 窗口锚定在 TSS, 链向感知: +链 [TSS-UP, TSS+DOWN]; -链 [TSS-DOWN, TSS+UP]
# 正式图固定按基因列归一：同一基因列中的全部细胞类型共用一个 y 轴。
TAG = f"_u{UP}d{DOWN}" if (UP, DOWN) != (2000, 1000) else ""

ROW_ORDER = list(TRACK_CELL_TYPE_ORDER)
# 基因↔细胞对应集中在 figure3_style.py；顺序与 ROW_ORDER 1:1 对齐。
GENE_ORDER = list(TRACK_GENE_ORDER)
STRAND_COLOR = {"+":"#8B0000","-":"#16356B"}       # 正链深红 / 负链深蓝

def darken(hexc,f=0.6):
    h=hexc.lstrip("#"); r,g,b=(int(h[i:i+2],16) for i in (0,2,4))
    return (r*f/255.,g*f/255.,b*f/255.)

# ---------- 基因模型 (GTF) ----------
@dataclass
class GeneModel:
    chrom:str; start:int; end:int; strand:str
    exons:list; cds_min:int; cds_max:int
    @property
    def tss(self): return self.start if self.strand=="+" else self.end

def parse_attrs(a):
    return {k:v.strip() for k,v in re.findall(r'([A-Za-z0-9_]+)\s+"?([^";]+)"?', a)}

def load_gene_models(gtf, want):
    rows={}; ex=defaultdict(list); cds=defaultdict(list)
    with gzip.open(gtf,"rt") as fh:
        for line in fh:
            if line.startswith("#"): continue
            f=line.rstrip("\n").split("\t")
            if len(f)<9: continue
            chrom,_,feat,s,e,_,strand,_,attr=f
            at=parse_attrs(attr); gn=at.get("gene_name") or at.get("gene_id")
            if gn not in want: continue
            s,e=int(s),int(e)
            if feat=="gene": rows[gn]=(chrom,s,e,strand)
            elif feat=="exon": ex[gn].append((s,e))
            elif feat=="CDS": cds[gn].append((s,e))
    models={}
    for g,(chrom,s,e,strand) in rows.items():
        cc=cds.get(g,[])
        models[g]=GeneModel(chrom,s,e,strand,sorted(set(ex.get(g,[]))),
                            (min(a for a,_ in cc) if cc else 0),
                            (max(b for _,b in cc) if cc else -1))
    return models

def norm_chrom(chrom, bw):
    for c in (chrom, chrom.replace("chr",""), f"chr{chrom.replace('chr','')}"):
        if c in bw.chroms(): return c
    return None

def read_sig(bw, chrom, start, end):
    c=norm_chrom(chrom,bw)
    if c is None: return None
    e=min(end, bw.chroms(c)); s=max(0,start)
    if e<=s: return None
    v=bw.values(c,s,e)
    if v is None or len(v)==0: return None
    return np.array([0 if x is None or np.isnan(x) else x for x in v],dtype=float)

def load_peaks(path):
    df=pd.read_csv(path,sep="\t",header=None).iloc[:,:3]
    df.columns=["chr","start","end"]; df["chr"]=df["chr"].astype(str)
    return df

# ---------- 主流程 ----------
def main():
    models=load_gene_models(GTF,set(GENE_ORDER))
    genes=[g for g in GENE_ORDER if g in models]
    miss=[g for g in GENE_ORDER if g not in models]
    if miss: print("[warn] 未在GTF找到:",miss)
    peaks_df=load_peaks(PEAKS)

    bws={ct:(pyBigWig.open(os.path.join(BWDIR,f"{ct}.bw"))
             if os.path.exists(os.path.join(BWDIR,f"{ct}.bw")) else None) for ct in ROW_ORDER}

    per={}
    for g in genes:
        m=models[g]; tss=m.tss
        # 染色体长度(用于夹取窗口)
        cl=None
        for ct in ROW_ORDER:
            if bws[ct]:
                c=norm_chrom(m.chrom,bws[ct])
                if c: cl=bws[ct].chroms(c); break
        # TSS 锚定 (链向感知): 上游 UP, 下游 DOWN
        if m.strand=="+": start,end = tss-UP, tss+DOWN
        else:             start,end = tss-DOWN, tss+UP
        start=max(0,start)
        if cl and end>cl: end=cl
        sigs={ct:(read_sig(bws[ct],m.chrom,start,end) if bws[ct] else None) for ct in ROW_ORDER}
        gmax=max([float(np.nanmax(s)) for s in sigs.values() if s is not None and s.size]+[0.0])
        gmax=max(gmax*1.05,0.1)                   # 每基因列统一 y 轴; 少留白->峰更满
        rp=peaks_df[(peaks_df.chr==m.chrom)&(peaks_df.start<end)&(peaks_df.end>start)]
        per[g]=dict(m=m,start=int(start),end=int(end),sigs=sigs,gmax=gmax,peaks=rp)
    for h in bws.values():
        if h: h.close()

    # 诊断: Corin 各细胞类型峰值
    if "Corin" in per:
        d=per["Corin"]; tops=sorted(((ct,(0 if d["sigs"][ct] is None else float(np.nanmax(d["sigs"][ct])))) for ct in ROW_ORDER),key=lambda x:-x[1])[:4]
        print("[check] Corin gmax=%.3f 前4:"%d["gmax"], [(c,round(v,3)) for c,v in tops])

    print("[norm] mode=column; one shared y-axis scale per gene locus")

    nC=len(ROW_ORDER); nG=len(genes)
    fig=plt.figure(figsize=(0.9*nG+1.7, 0.34*nC+2.8))     # 等宽列 (不设 width_ratios)
    gs=GridSpec(nC+2,nG,figure=fig,height_ratios=[1]*nC+[1.6,0.5],
                wspace=0.0,hspace=0.0,left=0.075,right=0.996,top=0.95,bottom=0.12)
    tax={}; gax={}; pax={}; box_lw=0.45
    cell_border_color="#B8B8B8"
    for ci,g in enumerate(genes):
        d=per[g]; m=d["m"]; start=d["start"]; end=d["end"]; gmax=d["gmax"]
        for ri,ct in enumerate(ROW_ORDER):
            ax=fig.add_subplot(gs[ri,ci]); tax[(ri,ci)]=ax; col=CELL_TYPE_COLORS[ct]
            sig=d["sigs"][ct]
            if sig is not None and sig.size and float(np.nanmax(sig))>0:
                x=np.linspace(start,end,len(sig))
                ax.fill_between(x,0,sig,color=col,lw=0,zorder=2)
                ax.plot(x,sig,color=darken(col),lw=0.5,zorder=3)
            ax.set_xlim(start,end); ax.set_ylim(0,gmax)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_visible(False)
            # 相邻 track 共用浅灰细分隔线，避免黑色粗框压过信号。
            ax.axhline(0,color=cell_border_color,lw=box_lw,zorder=4)
            if ci==0: ax.set_ylabel(display_name(ct),rotation=0,ha="right",va="center",
                                    fontsize=8.5,color=darken(col,0.85),fontweight="bold")
        # 基因模型 (链向红蓝, 全外显子; CDS 粗 / UTR 细)
        axg=fig.add_subplot(gs[nC,ci]); axg.set_xlim(start,end); axg.set_ylim(-1,1); axg.axis("off"); gax[ci]=axg
        gc=STRAND_COLOR[m.strand]
        # 主干线画满窗口内的整段基因(含内含子), 不管是否有 peak/外显子触及
        gb0,gb1=max(m.start,start),min(m.end,end)
        if gb1>gb0:
            axg.plot([gb0,gb1],[0,0],color=gc,lw=1.0,zorder=1)
            for xa in np.linspace(gb0,gb1,8)[1:-1]:
                axg.plot(xa,0,marker=(">" if m.strand=="+" else "<"),color=gc,markersize=3.0,zorder=2)
        for s,e in m.exons:                       # 外显子统一方块 (UTR 与 CDS 合并为一种粗细)
            s2,e2=max(s,start),min(e,end)
            if e2>s2: axg.add_patch(Rectangle((s2,-0.62),e2-s2,1.24,color=gc,lw=0,zorder=3))
        # peak 条带
        axp=fig.add_subplot(gs[nC+1,ci]); axp.set_xlim(start,end); axp.set_ylim(0,1); axp.axis("off"); pax[ci]=axp
        for _,pk in d["peaks"].iterrows():
            ps,pe=max(int(pk["start"]),start),min(int(pk["end"]),end)
            if pe>ps: axp.add_patch(Rectangle((ps,0.28),pe-ps,0.44,color="#9A9A9A",lw=0))
        if ci==0: axp.text(-0.03,0.5,"Peaks",transform=axp.transAxes,ha="right",va="center",fontsize=8)

    fig.suptitle("Gene list side-by-side tracks (CPM coverage)",fontsize=14,fontweight="bold",y=0.985)
    fig.canvas.draw()
    for ci in range(nG):
        pt=tax[(0,ci)].get_position(); gp=gax[ci].get_position(); pp=pax[ci].get_position()
        x0,x1=pt.x0,pt.x1
        fig.add_artist(Rectangle((x0,pp.y0),x1-x0,pt.y1-pp.y0,fill=False,
                                 edgecolor=cell_border_color,lw=box_lw,zorder=10))
        for yy in (gp.y1,pp.y1):
            fig.add_artist(Line2D([x0,x1],[yy,yy],color=cell_border_color,
                                  lw=box_lw,zorder=10))
        fig.text((x0+x1)/2,pp.y0-0.006,genes[ci],rotation=45,rotation_mode="anchor",
                 ha="right",va="top",fontsize=9,fontstyle="italic",
                 fontweight="normal",color="#222")

    fig.savefig(OUTPUT_DIR / f"Figure3D_celltype_accessibility_tracks{TAG}.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"Figure3D_celltype_accessibility_tracks{TAG}.svg", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"Figure3D_celltype_accessibility_tracks{TAG}.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"Figure3D_celltype_accessibility_tracks{TAG}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {OUTPUT_DIR}/Figure3D_celltype_accessibility_tracks{TAG}.pdf/.png  (UP={UP} DOWN={DOWN})")

if __name__=="__main__":
    main()
