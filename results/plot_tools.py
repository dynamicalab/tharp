from pathlib import Path
import math
import os

import matplotlib as mpl
import numpy as np


project_dir = Path(__file__).resolve().parent.parent

alg_colors = {"GD": "C2", "RGD": "C3", "RAdam": "C4", "D-Mercator": "C0", "D-Mercator*": "C1"}
def alg_to_label(name):
    if "dmercator" in name:
        return "D-Mercator"
    elif "radam" in name:
        return "RAdam"
    elif "rgd" in name:
        return "RGD"
    elif "gd" in name:
        return "GD"
    return "?"


def pt_to_inch(pt):
    return pt * 0.0138888889


def cm_to_inch(cm):
    return cm * 0.3937007874


column_width = pt_to_inch(246) # For revtex
text_width = pt_to_inch(510) # For SI
# column_width = cm_to_inch(9) # For Comm. Phys.

cmap = mpl.colormaps["magma_r"]
def color_gradient(length: int):
    return cmap((np.arange(length) + .5)/length)


def figure_dir():
    fig_dir = project_dir.joinpath("figures")
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)
    return fig_dir


def cache_dir():
    cache_dir = figure_dir().joinpath("cache")
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    return cache_dir


def format_ticks(denom: int):
    if denom <= 0:
        raise ValueError("Invalid denominator for axis formatter.")

    tick_values, tick_labels = [], []
    for num in range(0, denom+1):
        tick_values.append(num/denom*np.pi)
        gcd = math.gcd(num, denom)
        tick_labels.append(format_frac(num, abs(num//gcd), denom//gcd))

    return tick_values, tick_labels

def format_frac(sign, num, denom):
    if num == 0:
        return "$0$"
    if num == 1 and denom == 1:
        return f"${'-' if sign<0 else ''}\\pi$"
    return f"${'-' if sign<0 else ''}\\frac{{{num if num != 1 else ''}\\pi}}{{{denom}}}$"


def label_subplots_letter(axes, fmt=r'{{\sffamily\bfseries {}}}', xshift=-0.25, yshift=0.25, fontsize="large", fontweight="heavy", **kwargs):
    """
    To apply after tight_layout.
    """
    label = 97 # Ascii for 'a'
    for ax in axes.ravel():
        ax.text(
            xshift, 1.+yshift, fmt.format(chr(label)),
            transform=ax.transAxes,
            fontsize=fontsize,
            fontweight=fontweight,
            **kwargs,
            va="top",
            ha="left",
        )
        label += 1


class Labeloffset():
    def __init__(self, ax, offset, axis="y"):
        self.axis = {"y":ax.yaxis, "x":ax.xaxis}[axis]
        self.offset = offset
        ax.callbacks.connect(axis+'lim_changed', self.update)
        ax.figure.canvas.draw()
        self.ax = ax
        self.update(None)

    def update(self, lim):
        offset = self.axis.get_offset_text()
        offset.set_visible(False)
        s = offset.get_text()
        self.ax.text(
            *self.offset, s,
            transform=self.ax.transAxes,
            fontsize=offset.get_fontsize(),
            fontfamily=offset.get_family(),
        )

methods_markers = {
    "mdl": "o",
    "ferra": "s",
    "almagro": "^",
    "jankowski": "d",
}

dataset_cite_keys = {
    "CElegans-C"  : "ahn_2006_WiringCostOrganization",
    "Human1-C"    : "ahn_2006_WiringCostOrganization",
    "Human2-C"    : "ahn_2006_WiringCostOrganization",
    "Human3-C"    : "ahn_2006_WiringCostOrganization",
    "Human4-C"    : "ahn_2006_WiringCostOrganization",
    "Human5-C"    : "ahn_2006_WiringCostOrganization",
    "Human6-C"    : "hagmann_2008_MappingStructuralCore",
    "Macaque-C"   : "harriger_2012_RichClubOrganization",
    "Mouse-C"     : "oh_2014_MesoscaleConnectomeMouse",
    "Drosophila-G": "stark_2006_BioGRIDGeneralRepository",
    "Human-M"     : "serrano_2012_UncoveringHiddenGeometry",
    "Human1-P"    : "chang_2015_BRENDA2015Exciting",
    "Human2-P"    : "chang_2015_BRENDA2015Exciting",
    "Facebook-H"  : "rozemberczki_2021_MultiScaleAttributedNode",
    "Wiki-H"      : "heaberlin_2016_EvolutionWikipediasNorm",
    "HepTh-CIT"   : "gehrke_2003_Overview2003KDD",
    "Jazz-CA"     : "gleiser_2003_CommunityStructureJazz",
    "AstroPh-CA"  : "leskovec_2007_GraphEvolutionDensification",
    "CondMat-CA"  : "leskovec_2007_GraphEvolutionDensification",
    "Geom-CA"     : "beebe__NelsonBeebesBibliographies",
    "GrQc-CA"     : "leskovec_2007_GraphEvolutionDensification",
    "HepTh-CA"    : "leskovec_2007_GraphEvolutionDensification",
    "Bible-CO"    : "harrison__BibleCrossReferences",
    "Music-CO"    : "serra_2012_MeasuringEvolutionContemporary",
    "Friends-OFF" : "moody_2001_PeerInfluenceGroups",
    "EUEmail"     : "yin_2017_LocalHigherOrderGraph",
    "EnronEmail"  : "klimt_2004_EnronCorpusNew",
    "URVEmail"    : "guimera_2003_SelfsimilarCommunityStructure",
    "Friends-ON"  : "kunegis_2013_KONECTKoblenzNetwork",
    "PGP-Trust"   : "boguna_2004_ModelsSocialNetworks",
    "Internet"    : "claffy_2009_InternetMappingArt",
    "USCommute"   : "grady_2012_RobustClassificationSalient",
    "Cargoships"  : "kaluza_2010_ComplexNetworkGlobal",
    "AMiner_citation_network_2009": "ferramarcus_2026_ChordlessCycleFiltrations",
    "Atlas_of_Economic_Complexity_export_network_N774": "ferramarcus_2026_ChordlessCycleFiltrations",
    "Atlas_of_Economic_Complexity_export_network_N866": "ferramarcus_2026_ChordlessCycleFiltrations",
    "Binary_interactomes_homo_sapiens_2012": "ferramarcus_2026_ChordlessCycleFiltrations",
    "Binary_interactomes_musculus_mouse_2012_N537": "ferramarcus_2026_ChordlessCycleFiltrations",
    "Binary_interactomes_musculus_mouse_2012_N705": "ferramarcus_2026_ChordlessCycleFiltrations",
    "Binary_interactomes_musculus_mouse_2012_N890": "ferramarcus_2026_ChordlessCycleFiltrations",
    "Binary_interactomes_various_species_2012": "ferramarcus_2026_ChordlessCycleFiltrations",
    "C_elegans_interactomes_2009": "ferramarcus_2026_ChordlessCycleFiltrations",
    "C_elegans_interactomes_genetic_2009": "ferramarcus_2026_ChordlessCycleFiltrations",
    "C_elegans_interactomes_scaffold_2009": "ferramarcus_2026_ChordlessCycleFiltrations",
    "C_elegans_interactomes_WI-2004_2009": "ferramarcus_2026_ChordlessCycleFiltrations",
    "C_elegans_interactomes_WI-2007_2009": "ferramarcus_2026_ChordlessCycleFiltrations",
    "C_elegans_posterior_nervous_system_male": "ferramarcus_2026_ChordlessCycleFiltrations",
    "European_airline_network": "ferramarcus_2026_ChordlessCycleFiltrations",
    "FAA_Preferred_Routes_2010": "ferramarcus_2026_ChordlessCycleFiltrations",
    "Garfields_citation_networks_2001": "ferramarcus_2026_ChordlessCycleFiltrations",
    "Guava_library_dependencies_2012": "ferramarcus_2026_ChordlessCycleFiltrations",
    "India_bus_routes_2016_N1009": "ferramarcus_2026_ChordlessCycleFiltrations",
    "India_bus_routes_2016_N1087": "ferramarcus_2026_ChordlessCycleFiltrations",
    "India_bus_routes_2016_N1103": "ferramarcus_2026_ChordlessCycleFiltrations",
    "India_bus_routes_2016_N1554": "ferramarcus_2026_ChordlessCycleFiltrations",
    "Ito_core_yeast_interactome_2001": "ferramarcus_2026_ChordlessCycleFiltrations",
    "Javax_Java_Jung_AppEngine_dependencies_2010_N1031": "ferramarcus_2026_ChordlessCycleFiltrations",
    "Javax_Java_Jung_AppEngine_dependencies_2010_N435": "ferramarcus_2026_ChordlessCycleFiltrations",
    "London_Transport_Network": "ferramarcus_2026_ChordlessCycleFiltrations",
    "Malaria_var_DBLa_HVR_networks_N291": "ferramarcus_2026_ChordlessCycleFiltrations",
    "Malaria_var_DBLa_HVR_networks_N298": "ferramarcus_2026_ChordlessCycleFiltrations",
    "Reguly_yeast_interactome_2006": "ferramarcus_2026_ChordlessCycleFiltrations",
    "Rogets_Thesaurus_1879": "ferramarcus_2026_ChordlessCycleFiltrations",
    "Software_function-callgraphs_AbiWord_2002": "ferramarcus_2026_ChordlessCycleFiltrations",
    "Software_function-callgraphs_VTK_2002": "ferramarcus_2026_ChordlessCycleFiltrations",
    "Uetz_screen_yeast_interactome_2000": "ferramarcus_2026_ChordlessCycleFiltrations",
    "US_airport_network_top_500_2002": "ferramarcus_2026_ChordlessCycleFiltrations",
    "WebKB_graphs_1998_N269": "ferramarcus_2026_ChordlessCycleFiltrations",
    "WebKB_graphs_1998_N280": "ferramarcus_2026_ChordlessCycleFiltrations",
    "WebKB_graphs_1998_N343": "ferramarcus_2026_ChordlessCycleFiltrations",
    "World_subways_2009_N209": "ferramarcus_2026_ChordlessCycleFiltrations",
    "World_subways_2009_N217": "ferramarcus_2026_ChordlessCycleFiltrations",
    "World_subways_2009_N266": "ferramarcus_2026_ChordlessCycleFiltrations",
    "World_subways_2009_N299": "ferramarcus_2026_ChordlessCycleFiltrations",
    "World_subways_2009_N392": "ferramarcus_2026_ChordlessCycleFiltrations",
    "World_subways_2009_N433": "ferramarcus_2026_ChordlessCycleFiltrations",
    "Yu_yeast_interactomes_2008_N1647": "ferramarcus_2026_ChordlessCycleFiltrations",
    "Yu_yeast_interactomes_2008_N964": "ferramarcus_2026_ChordlessCycleFiltrations",
    "Add-health": "moody_2001_PeerInfluenceGroups",
    "CElegans_2019": "cook_2019_WholeanimalConnectomesBoth",
    "FAO-apples": "dedomenico_2015_StructuralReducibilityMultilayer",
    "Foxglove": "jackson_2017_TopologicalAnalysisMulticellular",
    "OpenFlights": "patokallio_2023_OpenFlightsAirportAirline",
    "Polbooks": "pasternak_1970_FourUnpublishedLetters",
}

datasets_labels = {
    "CElegans-C": "CEL",
    "Human1-C": "HU1",
    "Human2-C": "HU2",
    "Human3-C": "HU3",
    "Human4-C": "HU4",
    "Human5-C": "HU5",
    "Human6-C": "HU6",
    "Macaque-C": "MAC",
    "Mouse-C": "MOU",
    "Drosophila-G": "DRO",
    "Human-M": "HUM",
    "Human1-P": "HP1",
    "Human2-P": "HP2",
    "Facebook-H": "FBH",
    "Wiki-H": "WIK",
    "HepTh-CIT": "HCI",
    "Jazz-CA": "JAZ",
    "AstroPh-CA": "AST",
    "CondMat-CA": "CMA",
    "Geom-CA": "GEO",
    "GrQc-CA": "GRQ",
    "HepTh-CA": "HCA",
    "Bible-CO": "BIB",
    "Music-CO": "MUS",
    "Friends-OFF": "FOF",
    "EUEmail": "EUE",
    "EnronEmail": "ENE",
    "URVEmail": "URE",
    "Friends-ON": "FON",
    "PGP-Trust": "PGP",
    "Internet": "INT",
    "USCommute": "USC",
    "Cargoships": "CSH",
}
