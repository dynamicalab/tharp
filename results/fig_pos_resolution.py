import json
import os
import pickle
from matplotlib.lines import Line2D
from tqdm import tqdm

import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt

from mdl_hyperbolic.models import hyperbolic_dl_terms
from plot_tools import color_gradient, label_subplots_letter, text_width, figure_dir, cache_dir
from mdl_hyperbolic.optimize import optimize_quantization


jax.config.update('jax_enable_x64', True) # enable double precision floating-point


if __name__ == "__main__":
    OVERWRITE_CACHE = False

    cache_file = str(cache_dir().joinpath("pos_resolution.bin"))
    if not OVERWRITE_CACHE and os.path.isfile(cache_file):
        with open(cache_file, 'rb') as file_stream:
            data = pickle.load(file_stream)
    else:
        dims = [2, 3, 4]
        dxs = np.logspace(-5, -1, 100)

        base_name = ".output/community_case_study/synth_communities"
        graph_file = base_name + ".txt"
        fit_file = base_name + "-radam-fits.json"

        edges = jnp.asarray(np.array(np.loadtxt(graph_file, dtype=np.int32)))
        with open(fit_file, "r") as file_stream:
            fits = json.load(file_stream)["fits"]

        data = {"dxs": dxs, "log-likelihoods": {d: [] for d in dims}, "dls": {d: [] for d in dims}, "dx hat": {}}

        delta = 1e-5
        quantization = {p: delta for p in ["beta", "r", "rho", "nu"]}
        for dim in tqdm(dims):
            fit = fits[str(dim)]
            fit["pos"] = jnp.asarray(fit["pos"])
            data["dx hat"][dim] = optimize_quantization(fit, edges, delta=delta)
            for dx in dxs:
                quantization["pos"] = dx
                neg_likelihood, neg_prior = hyperbolic_dl_terms(**fit, edges=edges, quantization=quantization)

                data["log-likelihoods"][dim].append(-neg_likelihood)
                data["dls"][dim].append(neg_likelihood + neg_prior)

        with open(cache_file, 'wb') as file_stream:
            pickle.dump(data, file_stream)

    fig, axes = plt.subplots(1, 2, figsize=(text_width, 0.3*text_width), sharex=True)

    colors = color_gradient(len(data["dls"]))

    last = np.argwhere(data["dxs"] > 0.01).ravel()[0]
    # Description length
    ax = axes[0]
    for color, (dim, values) in zip(colors, data["dls"].items()):
        ax.plot(data["dxs"][:last], values[:last], color=color, label=f"$D={dim}$")
        ax.axvline(data["dx hat"][dim]["pos"], color=color, ls="--")
    ax.set_xlabel("$\\Delta_x$")
    ax.set_ylabel("Description length")
    ax.set_xscale("log")

    # Log-likelihood
    ax = axes[1]
    for color, (dim, values) in zip(colors, data["log-likelihoods"].items()):
        ax.plot(data["dxs"][:last], values[:last], color=color)
        ax.axvline(data["dx hat"][dim]["pos"], color=color, ls="--")
    ax.set_xlabel("$\\Delta_x$")
    ax.set_ylabel("Log-likelihood")
    ax.set_xscale("log")

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.25)

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    handles.append(Line2D([], [], ls="--", color="gray"))
    labels.append(r"$\hat{\Delta}_x$")
    fig.legend(
        handles, labels,
        loc='lower center',       # anchor point on the legend box
        bbox_to_anchor=(0.5, 0),  # (x=center, y=just below figure)
        ncol=len(labels),      # one line display
    )
    label_subplots_letter(axes, xshift=-0.25, yshift=0.05)

    fig.savefig(str(figure_dir().joinpath("pos_resolution.pdf")))
    plt.show()
