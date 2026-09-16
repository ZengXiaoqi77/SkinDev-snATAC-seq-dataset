#!/usr/bin/env python3
"""Frozen display, ordering, colour, and metric contracts for Figure 3.

Raw annotation values remain unchanged in metadata and source-data files.  The
display-name mapping controls publication graphics and explicit display-label
columns, while preserving the raw identifiers needed for reproducibility.
"""

from __future__ import annotations

from types import MappingProxyType


# One full order for all panels, frozen to the author-specified sequence.
CELL_TYPE_ORDER = (
    "Endo",
    "Pericyte",
    "Neuron",
    "SCH",
    "MELAN",
    "Muscle",
    "Immu_Macr",
    "Immu_Neu",
    "Immu_T_NK",
    "Immu_B",
    "FB_DP",
    "FB_DS",
    "FB_papi",
    "FB_reti",
    "FB_fascia",
    "KC_basal",
    "KC_sbasal",
    "KC_HFSC1",
    "KC_HFSC2",
    "KC_IRS1",
    "KC_IRS2",
    "KC_ORS",
    "KC_placode",
)

# Figure 3D intentionally omits Neuron, as confirmed by the author.  Deriving
# this order from CELL_TYPE_ORDER prevents all other order drift.
TRACK_CELL_TYPE_ORDER = tuple(
    cell_type for cell_type in CELL_TYPE_ORDER if cell_type != "Neuron"
)

CELL_TYPE_DISPLAY_NAMES = MappingProxyType(
    {
        "Endo": "ENDO",
        "Pericyte": "PER",
        "Neuron": "Neuron",
        "SCH": "SCH",
        "MELAN": "MELAN",
        "Muscle": "MUS",
        "Immu_Macr": "MACR",
        "Immu_Neu": "NEU",
        "Immu_T_NK": "T/NK",
        "Immu_B": "B",
        "FB_DP": "DP",
        "FB_DS": "DS",
        "FB_papi": "papiFB",
        "FB_reti": "retiFB",
        "FB_fascia": "fascFB",
        "KC_basal": "basalKC",
        "KC_sbasal": "sbasalKC",
        "KC_HFSC1": "HFSC1",
        "KC_HFSC2": "HFSC2",
        "KC_IRS1": "IRS1",
        "KC_IRS2": "IRS2",
        "KC_ORS": "ORS",
        "KC_placode": "PC",
    }
)

# Existing palette is frozen to preserve continuity with the approved
# annotation and composition plots.
CELL_TYPE_COLORS = MappingProxyType(
    {
        "Endo": "#809693",
        "Pericyte": "#8FB0FF",
        "Neuron": "#004D43",
        "SCH": "#DDEFFF",
        "MELAN": "#5A0007",
        "Muscle": "#1B4400",
        "Immu_Macr": "#7B4F4B",
        "Immu_Neu": "#A1C299",
        "Immu_T_NK": "#B79762",
        "Immu_B": "#997D87",
        "FB_DP": "#E6194B",
        "FB_DS": "#F58231",
        "FB_papi": "#F032E6",
        "FB_reti": "#FF4500",
        "FB_fascia": "#FFE119",
        "KC_basal": "#7FFF00",
        "KC_sbasal": "#8A2BE2",
        "KC_HFSC1": "#3CB44B",
        "KC_HFSC2": "#42D4F4",
        "KC_IRS1": "#4363D8",
        "KC_IRS2": "#911EB4",
        "KC_ORS": "#1E90FF",
        "KC_placode": "#00CED1",
    }
)

# Figure 3D uses one representative locus per displayed cell type.  Keeping
# this relation keyed by cell type makes the gene columns follow any frozen row
# ordering automatically and preserves the intended diagonal correspondence.
TRACK_MARKER_GENE_BY_CELL_TYPE = MappingProxyType(
    {
        "Endo": "Vwf",
        "Pericyte": "Des",
        "SCH": "Sox10",
        "MELAN": "Tyrp1",
        "Muscle": "Myod1",
        "Immu_Macr": "Cd68",
        "Immu_Neu": "Cxcl2",
        "Immu_T_NK": "Nkg7",
        "Immu_B": "Cd79a",
        "FB_DP": "Corin",
        "FB_DS": "Grem2",
        "FB_papi": "Crabp1",
        "FB_reti": "Mgp",
        "FB_fascia": "Adam12",
        "KC_basal": "Krt15",
        "KC_sbasal": "Krt10",
        "KC_HFSC1": "Sox9",
        "KC_HFSC2": "Lhx2",
        "KC_IRS1": "Dsg4",
        "KC_IRS2": "Krt73",
        "KC_ORS": "Krt75",
        "KC_placode": "Edar",
    }
)
TRACK_GENE_ORDER = tuple(
    TRACK_MARKER_GENE_BY_CELL_TYPE[cell_type]
    for cell_type in TRACK_CELL_TYPE_ORDER
)

STAGE_ORDER = ("E16.5", "P0", "P5", "Adult")

CELL_TYPE_LEGEND_TITLE = "Cell type"
GENE_ACTIVITY_COLORBAR_LABEL = "Scaled gene activity"
STAGE_COMPOSITION_XLABEL = "Proportion of retained nuclei"
GO_XLABEL = r"$-\log_{10}$(adjusted $P$ value)"
KC_PLACODE_COLOR = CELL_TYPE_COLORS["KC_placode"]

# These strings are the frozen statistical/normalization meanings behind the
# panels.  Plotting scripts implement these definitions; figure legends should
# paraphrase them faithfully rather than introducing a different denominator.
PANEL_METRIC_DEFINITIONS = MappingProxyType(
    {
        "Figure3A": (
            "UMAP coordinates and anno_0627 labels are read from the frozen "
            "annotated nucleus metadata; no quantitative summary is computed."
        ),
        "Figure3B": (
            "ATAC-derived gene activity for Col1a1, Krt17, Cdh5 and Pdgfrb "
            "is displayed on the frozen UMAP coordinates. Values are shown "
            "as continuous accessibility-derived activity scores and are not "
            "RNA expression measurements."
        ),
        "Figure3C": (
            "For each marker gene, the arithmetic mean ATAC-derived gene "
            "activity is calculated within each cell type and min-max scaled "
            "across the 23 cell types to the interval 0-1. Values are "
            "comparable across cell types within a gene, not as absolute "
            "activity across different genes."
        ),
        "Figure3D": (
            "CPM-normalized bigWig coverage is shown in a strand-aware window "
            "from 2 kb upstream to 1 kb downstream of each transcription start "
            "site. Cell types share one y-axis scale within each locus; y-axis "
            "scales differ among loci."
        ),
        "Figure3E": (
            "The ten Gene Ontology Biological Process terms with the smallest "
            "stored adjusted P values for PC (raw annotation KC_placode) are displayed as "
            "-log10(adjusted P value). The panel supports annotation and is not "
            "used to claim a new mechanism."
        ),
        "Figure3F": (
            "Within each developmental stage, retained nuclei are pooled across "
            "included libraries and the count for each cell type is divided by "
            "the total retained nuclei for that stage. The proportions are "
            "descriptive and are not replicate-aware estimates of tissue cell "
            "abundance."
        ),
    }
)


def display_name(raw_cell_type: str) -> str:
    """Return the frozen publication label for one raw annotation value."""

    return CELL_TYPE_DISPLAY_NAMES[raw_cell_type]


def validate_contract() -> None:
    """Fail early if future edits make the frozen mappings inconsistent."""

    order = set(CELL_TYPE_ORDER)
    if tuple(CELL_TYPE_DISPLAY_NAMES) != CELL_TYPE_ORDER:
        raise RuntimeError("Display-name keys do not match CELL_TYPE_ORDER")
    if tuple(CELL_TYPE_COLORS) != CELL_TYPE_ORDER:
        raise RuntimeError("Colour keys do not match CELL_TYPE_ORDER")
    if len(CELL_TYPE_ORDER) != len(order):
        raise RuntimeError("CELL_TYPE_ORDER contains duplicate values")
    if len(set(CELL_TYPE_DISPLAY_NAMES.values())) != len(CELL_TYPE_ORDER):
        raise RuntimeError("Publication display names must be unique")
    if tuple(TRACK_MARKER_GENE_BY_CELL_TYPE) != TRACK_CELL_TYPE_ORDER:
        raise RuntimeError("Track marker-gene keys do not match Figure 3D order")
    if len(set(TRACK_GENE_ORDER)) != len(TRACK_GENE_ORDER):
        raise RuntimeError("Figure 3D marker genes must be unique")


validate_contract()
