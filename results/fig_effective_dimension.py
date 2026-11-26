import json
import os
import re
from collections import defaultdict

from matplotlib.lines import Line2D
import numpy as np
import jax
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from plot_tools import color_gradient, column_width, figure_dir, datasets_labels, methods_markers


jax.config.update('jax_enable_x64', True) # enable double precision floating-point


fig, ax = plt.subplots(figsize=(2*column_width, .8*column_width))
fig_rho, ax_rho = plt.subplots(figsize=(column_width, .6*column_width))


dmercator_color = "gray"
original_color = "k"

dims = np.arange(2, 8)

colors = color_gradient(len(dims))

# For rho(n) figure
dim_label_pos = {2:200, 3: 400, 4: 800, 5: 5000}

ax.set_xlim(80, 4e4)
for og_dim, color in zip(dims, colors):
    output_dir = f".output/various_n/{og_dim}d"
    inferred_dims = defaultdict(list)
    inferred_rhos = defaultdict(list)
    for filename in os.listdir(output_dir):
        if not (m := re.match(r"graph(\d+)_\d+.txt", filename)):
            continue
        n = int(m[1])

        fit_file = os.path.join(output_dir, filename[:-len(".txt")]+"-radam-fits.json")
        if not os.path.isfile(fit_file):
            if n < 15000:
                print(f"Skipping {os.path.join(output_dir, filename)}: fit file not found.")
            continue

        with open(fit_file, "r") as file_stream:
            data = json.load(file_stream)
            dls = data.get("description length")
        if dls is None:
            print(f"Skipping {os.path.join(output_dir, filename)}: DL not computed.")
            continue

        mdl_dim = min(dls, key=lambda k: dls[k][0])
        inferred_dims[n].append(int(mdl_dim))
        inferred_rhos[n].append(data["fits"][mdl_dim]["rho"])

    ns = list(map(int, inferred_dims.keys()))
    ns.sort()

    los, meds, his = [], [], []
    for n in ns:
        lo, med, hi = np.percentile(inferred_dims[n], [25, 50, 75])
        los.append(lo)
        meds.append(med)
        his.append(hi)
    med_rhos = [np.median(inferred_rhos[n]) for n in ns]


    label = f"$D^*\\!=\\!{og_dim}$"
    stroke = pe.withStroke(linewidth=0.2 if int(og_dim) == 2 else 0., foreground="#CCCCCC")
    ax.text(1.6e4, og_dim+0.15, label, color=color, fontsize="small", ha="center",
        path_effects=[stroke])
    if int(og_dim) <= 5:
        n_label = dim_label_pos[int(og_dim)]
        ax_rho.text(
            n_label, med_rhos[ns.index(n_label)], label,
            color=color, fontsize="small", ha="right", va="center",
            path_effects=[stroke])
    else:
        ax_rho.text(
            8000, 0.4 * (int(og_dim) - 5), label,
            color=color, fontsize="small", ha="center", va="center")

    ax.axhline(og_dim, color=color, alpha=0.3, ls="--", zorder=0)
    ax.plot(ns, meds, color=color, marker=".", markersize=2)
    ax_rho.plot(ns, med_rhos, marker=".", markersize=2, color=color)
    # ax.fill_between(ns, los, his, color=color, alpha=0.3)


# dataset_params = defaultdict(dict)
# dataset_params["Human2-C"] = {"va": "bottom"}
    # "Human3-C": {"xshift": 6.},
    # "Human6-C": {"va": "bottom"},
    # "Macaque-C": {"va": "bottom"},
    # "Mouse-C": {"xshift": -6.},
    # "Human2-P": {"va": "bottom", "ha": "right"},
    # "Facebook-H": {"va": "bottom"},
    # "Jazz-CA": {"ha": "right"},
    # "Friends-OFF": {"va": "bottom"},
    # "URVEmail": {"va": "bottom", "ha": "left"},
    # "PGP-Trust": {"va": "bottom"},
    # "Internet": {"va": "bottom"},

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


jankowski_dims = {
    "Add-health": 3,
    "CElegans_2019": 3,
    "FAO-apples": 3,
    "Foxglove": 4,
    "OpenFlights": 2,
    "Polbooks": 3,
}


def add_noise(xs, span):
    xnoise = np.random.normal(scale=span)
    return xs + xnoise


datasets = list(set(ferra_dims).union(set(almagro_dims)).union(set(jankowski_dims)))
for filename in datasets:
    if filename in almagro_dims:
        output_dir = ".data/almagro_processed"
    elif filename in ferra_dims:
        output_dir = ".data/ferra_processed"
    else:
        output_dir = ".data/jankowski_processed"

    fit_file = os.path.join(output_dir, filename) + "-radam-fits.json"
    suffix = ""
    if not os.path.isfile(fit_file):
        fit_file = os.path.join(output_dir, filename) + "-sradam-fits.json"
        suffix = "*"

    if not os.path.isfile(fit_file):
        print("missing", filename)
        continue

    with open(fit_file, "r") as file_stream:
        data = json.load(file_stream)
        n = np.asarray(data["fits"][next(iter(data["fits"]))]["pos"]).shape[0]
        dls = data.get("description length")
    if dls is None:
        print("missing", filename)
        continue

    mdl_dim = int(min(dls, key=lambda k: dls[k][0]))
    alpha = 0.2

    ax.scatter(add_noise(n, 0.1*np.log10(n)), add_noise(mdl_dim, 0.03), marker=methods_markers["mdl"], facecolor="none", linewidths=0.75, s=10, alpha=alpha, color="k", zorder=4)
    if d:=almagro_dims.get(filename):
        ax.scatter(add_noise(n, 0.1*np.log10(n)), add_noise(d, 0.03), s=10, marker=methods_markers["almagro"], facecolor="none", linewidths=0.75, alpha=alpha, color="k", zorder=4)
    if d:=ferra_dims.get(filename):
        ax.scatter(add_noise(n, 0.1*np.log10(n)), add_noise(d, 0.03), s=10, marker=methods_markers["ferra"], facecolor="none", linewidths=0.75, alpha=alpha, color="k", zorder=4)
    if d:=jankowski_dims.get(filename):
        ax.scatter(add_noise(n, 0.1*np.log10(n)), add_noise(d, 0.03), s=10, marker=methods_markers["jankowski"], facecolor="none", linewidths=0.75, alpha=alpha, color="k", zorder=4)

    # settings = dataset_params[filename]
    # va = settings.get("va", "top")
    # ha = settings.get("ha", "center")
    # ax.annotate(
        # datasets_labels[filename] + suffix,
        # xy=(n, mdl_dim + (0.25 if va=="top" else -0.3)),
        # xytext=(settings.get("xshift", 0), 0),      # 5 points to the right
        # textcoords="offset points",
        # va="center", # Handle "top" and "bottom" as shifts
        # ha=ha,
        # fontsize=4,
        # zorder=5
    # )
handles = [Line2D([], [], marker=methods_markers[x], ls="none", color="k", markerfacecolor="none")
           for x in ["mdl", "almagro", "ferra", "jankowski"]]
labels = ["MDL", "Almagro", "Ferrà Marcús", "Jankowski"]
# ax.legend(handles, labels)#, loc="upper left")
fig.legend(
    handles, labels,
    loc='lower center',
    bbox_to_anchor=(0.5, 0),
    ncol=len(labels),
    handletextpad=0.2,
)

ax.set_xlabel("Number of vertices")
ax.set_ylabel("Predicted dimension")
ax.set_xscale("log")

fig.tight_layout()
fig.subplots_adjust(bottom=0.25)

fig.savefig(str(figure_dir().joinpath("effective_dimension.pdf")), dpi=500)

ax_rho.set_xlabel("Number of vertices")
ax_rho.set_ylabel("$\\hat{\\rho}$")
ax_rho.set_xscale("log")
fig_rho.tight_layout()
fig_rho.savefig(str(figure_dir().joinpath("rho_vs_n.pdf")))

plt.show()
