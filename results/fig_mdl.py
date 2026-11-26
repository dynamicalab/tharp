from argparse import ArgumentParser
from collections import defaultdict
import json
import os
import pickle

import numpy as np
from matplotlib import pyplot as plt
import rustworkx as rx
import jax
import jax.numpy as jnp

from mdl_hyperbolic.models import hyperbolic_loglikelihood, prior_logdensity
from plot_tools import color_gradient, column_width, figure_dir, alg_colors, label_subplots_letter


jax.config.update('jax_enable_x64', True) # enable double precision floating-point


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("graph", choices=["comm", "unif"])
    args = parser.parse_args()

    if args.graph == "comm":
        path = ".output/community_case_study/"
        base_name = path + "synth_communities"
    else:
        path = ".output/unif_case_study/"
        base_name = path + "graph1000_4D"

    graph_file = base_name + ".txt"
    og_fit = base_name + "-parameters.json"
    fit_file_format = base_name + "-{}-fits.json"
    steps_dir_format = base_name + "_steps_{}/{}"
    # dmercator_file = base_name + "-dmercator-fits.json"

    edges = jnp.asarray(np.loadtxt(graph_file, dtype=np.int32))
    graph = rx.PyGraph.read_edge_list(graph_file, deliminator=" ", comment="#", multigraph=False, allow_self_loops=False)
    n = graph.num_nodes()

    def parse_fit_file(file_name):
        with open(file_name, "r") as file_stream:
            data = json.load(file_stream)
            fits = data["fits"]
            for d in fits:
                fits[d]["pos"] = jnp.asarray(fits[d]["pos"])
                if "description length" not in data or data["description length"].get(d) is None:
                    raise ValueError(f"Description length unavailable or outdated for {file_name}. Compute it before running this script.")
                fits[d]["dl"] = data["description length"][d][0]
        return fits

    fig, axes = plt.subplots(2, 2, sharex="col", figsize=(column_width, 0.8*column_width))


    with open(og_fit, "r") as file_stream:
        og_params = json.load(file_stream)
        og_params["pos"] = jnp.asarray(og_params["pos"])
        og_params["rho"] = og_params["r"]
        og_params["nu"] = 5.
        og_dim = og_params["pos"].shape[1]
    og_loglikelihood = hyperbolic_loglikelihood(edges=edges, pos=og_params["pos"], beta=og_params["beta"], r=og_params["r"])
    og_logposterior = og_loglikelihood + prior_logdensity(pos=og_params["pos"], beta=og_params["beta"], r=og_params["r"], rho=og_params["rho"], nu=og_params["nu"])
    if "description length" not in og_params:
        raise ValueError(f"Description length is not available in \"{og_fit}\". Run compute_dl.py first.")
    og_dl = og_params["description length"][str(og_dim)][0]


    algs = ["GD", "RGD", "RAdam"]
    n_algs = len(algs)
    for i, alg_name in enumerate(algs):
        first_pass = i == 0

        fits = parse_fit_file(fit_file_format.format(alg_name.lower()))

        props = defaultdict(dict)
        for dim, params in fits.items():
            d = int(dim)
            pos, beta, r, rho, nu = (params[p] for p in ["pos", "beta", "r", "rho", "nu"])
            props["loglikelihood"][d] = hyperbolic_loglikelihood(edges=edges, pos=pos, beta=beta, r=r)
            props["logposterior"][d] = props["loglikelihood"][d] + prior_logdensity(pos=pos, beta=beta, r=r, rho=rho, nu=nu)
            props["dl"][d] = params["dl"]
        dims = list(map(int, fits.keys()))
        dims.sort()
        ax_iter = iter(axes.ravel())

        color = alg_colors[alg_name]
        style = {"color": color, "marker":".", "markersize": 3, "label": alg_name}
        og_style = {"marker": "*", "color": "k", "zorder": 4, "label": "Original*", "edgecolor": "none"}

        # Log-likelihood
        ax = next(ax_iter)
        if first_pass:
            ax.scatter(og_dim, og_loglikelihood, **og_style)
        ax.plot(dims, [props["loglikelihood"][d] for d in dims], **style)
        ax.set_ylabel("Log-likelihood")
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

        # Log-posterior
        ax = next(ax_iter)
        ax.plot(dims, [props["logposterior"][d] for d in dims], **style)
        if first_pass:
            ax.scatter(og_dim, og_logposterior, **og_style)
        ax.set_ylabel("Log-posterior")
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

        # DL
        ax = next(ax_iter)
        ax.plot(dims, [props["dl"][d] for d in dims], **style)
        mdl_dim = min(props["dl"], key=props["dl"].get)
        if n_algs == 0:
            shift = 0
        else:
            shift = 0.2*( 2*i/(len(algs)-1) - 1 )
        ax.axvline(shift+mdl_dim, alpha=0.7, color=color, ls="--")
        if first_pass:
            ax.scatter(og_dim, og_dl, **og_style)
        ax.set_ylabel("Description length")
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

        # Rho
        ax = next(ax_iter)
        ax.plot(dims, [fits[str(d)]["rho"] for d in dims], **style)
        if first_pass:
            ax.scatter(og_dim, og_params["r"], **og_style)
        ax.set_ylabel("$\\hat{\\rho}$")


    for ax in axes.ravel():
        ticks = [min(dims), int(0.5*(min(dims)+max(dims))), max(dims)]
        ax.set_xticks(ticks, list(map(str, ticks)))

    for ax in axes[-1, :]:
        ax.set_xlabel("Dimension")

    # Rho vs iteration
    # dims = np.arange(2, 5)
    # for color, dim in zip(color_gradient(len(dims)), dims):
        # steps_dir = steps_dir_format.format("sradam", str(dim))
        # rho_values = {}
        # for file in os.listdir(steps_dir):
            # iteration = int(file[len("embedding_"):-4])
            # if iteration == 0:
                # continue
            # with open(os.path.join(steps_dir, file), "rb") as file_stream:
                # rho_values[iteration] = pickle.load(file_stream)["rho"]
        # ax = axes.ravel()[-1]
        # iterations = sorted(list(rho_values.keys()))
        # ax.plot(iterations, [rho_values[it] for it in iterations], color=color)


    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='lower center',       # anchor point on the legend box
        bbox_to_anchor=(0.5, 0),  # (x=center, y=just below figure)
        ncol=len(labels),      # one line display
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.25)
    label_subplots_letter(axes, xshift=-0.15, yshift=0.15)

    fig.savefig(os.path.join(figure_dir(), f"mdl_{args.graph}.pdf"))
    plt.show()
