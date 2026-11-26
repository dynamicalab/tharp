import os
import json
import itertools
import re

from matplotlib.lines import Line2D
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from plot_tools import figure_dir, column_width, datasets_labels, dataset_cite_keys, methods_markers


almagro_dims = {
    "CElegans-C"  : 4,
    "Human1-C"    : 4,
    "Human2-C"    : 4,
    "Human3-C"    : 3,
    "Human4-C"    : 3,
    "Human5-C"    : 3,
    "Human6-C"    : 4,
    "Macaque-C"   : 3,
    "Mouse-C"     : 2,
    "Drosophila-G": 9,
    "Human-M"     : 4,
    "Human1-P"    : 4,
    "Human2-P"    : 2,
    "Facebook-H"  : 2,
    "Wiki-H"      : 6,
    "HepTh-CIT"   : 5,
    "Jazz-CA"     : 3,
    "AstroPh-CA"  : 4,
    "CondMat-CA"  : 4,
    "Geom-CA"     : 4,
    "GrQc-CA"     : 2,
    "HepTh-CA"    : 5,
    "Bible-CO"    : 6,
    "Music-CO"    : 7,
    "Friends-OFF" : 10,
    "EUEmail"     : 4,
    "EnronEmail"  : 9,
    "URVEmail"    : 7,
    "Friends-ON"  : 7,
    "PGP-Trust"   : 2,
    "Internet"    : 8,
    "USCommute"   : 5,
    "Cargoships"  : 4,
}

jankowski_dims = {
    "Add-health": 3,
    "CElegans_2019": 3,
    "FAO-apples": 3,
    "Foxglove": 4,
    "OpenFlights": 2,
    "Polbooks": 3,
}

ferra_dims = {
    "Human-M": 4,
    "Jazz-CA": 3,
    "Bible-CO": 11,
    "Friends-OFF": 11,
    "EUEmail": 3,
    "Friends-ON": 8,
    "Cargoships": 6,
    "AMiner_citation_network_2009": 4,
    "Atlas_of_Economic_Complexity_export_network_N774": 6,
    "Atlas_of_Economic_Complexity_export_network_N866": 5,
    "Binary_interactomes_homo_sapiens_2012": 2,
    "Binary_interactomes_musculus_mouse_2012_N537": 2,
    "Binary_interactomes_musculus_mouse_2012_N705": 2,
    "Binary_interactomes_musculus_mouse_2012_N890": 5,
    "Binary_interactomes_various_species_2012": 2,
    "C_elegans_interactomes_2009": 2,
    "C_elegans_interactomes_genetic_2009": 6,
    "C_elegans_interactomes_scaffold_2009": 2,
    "C_elegans_interactomes_WI-2004_2009": 2,
    "C_elegans_interactomes_WI-2007_2009": 2,
    "C_elegans_posterior_nervous_system_male": 3,
    "European_airline_network": 2,
    "FAA_Preferred_Routes_2010": 2,
    "Garfields_citation_networks_2001": 2,
    "Guava_library_dependencies_2012": 2,
    "India_bus_routes_2016_N1009": 2,
    "India_bus_routes_2016_N1087": 4,
    "India_bus_routes_2016_N1103": 3,
    "India_bus_routes_2016_N1554": 3,
    "Ito_core_yeast_interactome_2001": 2,
    "Javax_Java_Jung_AppEngine_dependencies_2010_N1031": 2,
    "Javax_Java_Jung_AppEngine_dependencies_2010_N435": 2,
    "London_Transport_Network": 2,
    "Malaria_var_DBLa_HVR_networks_N291": 3,
    "Malaria_var_DBLa_HVR_networks_N298": 7,
    "Reguly_yeast_interactome_2006": 3,
    "Rogets_Thesaurus_1879": 10,
    "Software_function-callgraphs_AbiWord_2002": 2,
    "Software_function-callgraphs_VTK_2002": 2,
    "Uetz_screen_yeast_interactome_2000": 2,
    "US_airport_network_top_500_2002": 2,
    "WebKB_graphs_1998_N269": 2,
    "WebKB_graphs_1998_N280": 2,
    "WebKB_graphs_1998_N343": 2,
    "World_subways_2009_N209": 2,
    "World_subways_2009_N217": 11,
    "World_subways_2009_N266": 11,
    "World_subways_2009_N299": 11,
    "World_subways_2009_N392": 11,
    "World_subways_2009_N433": 2,
    "Yu_yeast_interactomes_2008_N1647": 2,
    "Yu_yeast_interactomes_2008_N964": 2,
}

def add_noise(xs, span):
    xnoise = np.random.normal(scale=span)
    return xs + xnoise


output_dir = ".data/almagro_processed"
fig, ax = plt.subplots(figsize=(column_width, 0.6*column_width))

table = "Dataset & $|\\mathcal{V}|$ & $|e|$ & MDL & Almagro 2022 & Ferrà Marcús 2026\\\\\n\\hline\n"

np.random.seed(42)

color1 = "#967bc0"
color2 = "#ec9a29"
color3 = "#1A936F"
all_almagro = []
all_ferra = []
all_jankowski = []

datasets = list(set(ferra_dims).union(set(almagro_dims)).union(set(jankowski_dims)))
datasets.sort()
for filename in datasets:
    if filename in almagro_dims:
        output_dir = ".data/almagro_processed"
    elif filename in ferra_dims:
        output_dir = ".data/ferra_processed"
    else:
        output_dir = ".data/jankowski_processed"

    fit_file = os.path.join(output_dir, filename) + "-radam-fits.json"
    using_sradam = not os.path.isfile(fit_file)
    if using_sradam:
        # print(f"Using SRAdam fits for {filename}.")
        fit_file = os.path.join(output_dir, filename) + "-sradam-fits.json"

    if not os.path.isfile(fit_file):
        print("missing", filename)
        continue

    with open(fit_file, "r") as file_stream:
        data = json.load(file_stream)
        num_edges = np.loadtxt(os.path.join(output_dir, filename) + ".txt").shape[0]
        n = np.asarray(data["fits"][next(iter(data["fits"]))]["pos"]).shape[0]
        dls = data.get("description length")
    if dls is None:
        print("missing", filename)
        continue
    if isinstance(dls[next(iter(dls))], float):
        continue

    # Tolerance of 0.01 bit per edge
    acceptable_dl_threshold = 0.01*num_edges * np.log(2)

    mdl_dim = int(min(dls, key=lambda k: dls[k][0]))
    acceptable_dims = [int(d) for d in dls if dls[d][0] < dls[str(mdl_dim)][0] + acceptable_dl_threshold]
    min_dim, max_dim  = min(acceptable_dims), max(acceptable_dims)

    dataset_name = filename.replace('_', ' ')
    if m := re.match(r".+(N\d+)", dataset_name): # Remove NXXX suffix.
        dataset_name = dataset_name[:-len(m[1])]
    dataset_name.strip()

    table += f"{dataset_name}{'*' if using_sradam else ''}~\\cite{{{dataset_cite_keys[filename]}}}"
    table += f"& {n:,} & {num_edges:,}"
    if min_dim != max_dim:
        table += f"&{mdl_dim} [{min_dim}, {max_dim}]"
    else:
        table += f"&{mdl_dim}"

    style = {"alpha": 0.4, "color": "k"}
    style["color"] = color1

    def error_bar(ax, x, ymin, y, ymax, padding_up=0.23, padding_down=0.23, **style):
        if y!=ymin:
            ax.plot([x]*2, [ymin, y-padding_down], lw=0.5, **style)
        if y!=ymax:
            ax.plot([x]*2, [y+padding_up, ymax], lw=0.5, **style)

    if d:=almagro_dims.get(filename):
        xpos = add_noise(d, 0.1)
        ax.scatter(xpos, mdl_dim, s=10, marker=methods_markers["almagro"], linewidths=0.75, facecolor="none", **style)
        error_bar(ax, xpos, min_dim, mdl_dim,  max_dim, padding_up=0.15, **style)
        table += f"&{d}"
        all_almagro.append((mdl_dim, d))
    else:
        table += f"&-"

    style["color"] = color2
    if d:=ferra_dims.get(filename):
        xpos = add_noise(d, 0.1)
        ax.scatter(xpos, mdl_dim, s=10, marker=methods_markers["ferra"], linewidths=0.75, facecolor="none", **style)
        error_bar(ax, xpos, min_dim, mdl_dim,  max_dim, **style)
        table += f"&{d}"
        all_ferra.append((mdl_dim, d))
    else:
        table += f"&-"

    style["color"] = color3
    if d:=jankowski_dims.get(filename):
        xpos = add_noise(d, 0.1)
        ax.scatter(xpos, mdl_dim, s=10, marker=methods_markers["jankowski"], linewidths=0.75, facecolor="none", **style)
        error_bar(ax, xpos, min_dim, mdl_dim,  max_dim, **style)
        table += f"&{d}"
        all_jankowski.append((mdl_dim, d))
    else:
        table += f"&-"

    table += "\\\\\n"
res = stats.pearsonr([x[0] for x in all_almagro], [x[1] for x in all_almagro])
print("Pearson r Almagro:", res.statistic, "pvalue=", res.pvalue)
res = stats.pearsonr([x[0] for x in all_ferra], [x[1] for x in all_ferra])
print("Pearson r Ferra  :", res.statistic, "pvalue=", res.pvalue)
res = stats.pearsonr([x[0] for x in all_jankowski], [x[1] for x in all_jankowski])
print("Pearson r Jankowski  :", res.statistic, "pvalue=", res.pvalue)
print()
print(table)

ax.plot([2, 12], [2, 12], ls="--", color="#CCCCCC", zorder=0)

ax.set_ylabel("MDL dimension")
ax.set_xlabel("Alternative prediction")
# ax.set_aspect("equal")

ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))

handles = [Line2D([], [], marker=methods_markers[x], ls="none", color=c, markerfacecolor="none")
           for x, c in zip(["almagro", "ferra", "jankowski"], [color1, color2, color3])]
labels = ["Almagro", "Ferrà Marcús", "Jankowski"]
fig.legend(
    handles, labels,
    loc='lower center',
    bbox_to_anchor=(0.5, 0),
    ncol=len(labels),
    handletextpad=0.2,
    frameon=False,
)
fig.tight_layout()
fig.subplots_adjust(bottom=0.35)
fig.savefig(os.path.join(figure_dir(), "method_comparison.pdf"), bbox_inches="tight")
plt.show()
