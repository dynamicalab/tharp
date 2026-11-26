import json
import os
from pathlib import Path
import pickle

from tqdm import tqdm
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import rustworkx as rx

from mdl_hyperbolic.geometry import hyperbolic_distance_matrix, hypersphere_distance_matrix
from embedding_properties import average_community_concentration
from mdl_hyperbolic.models import hyperbolic_loglikelihood, map_hyperbolic_model_to_sd, sd_loglikelihood
from plot_tools import alg_colors, alg_to_label, label_subplots_letter, text_width, figure_dir, cache_dir


jax.config.update('jax_enable_x64', True) # enable double precision floating-point


if __name__ == "__main__":
    OVERWRITE_CACHE = False

    output_dir = Path(".output/dmercator_benchmark/sd_caps")
    algs = ["dmercator", "gd", "radam"]

    og_dims = np.arange(2, 4)
    fig, axes = plt.subplots(2, len(og_dims), figsize=(0.8*text_width, 0.5*text_width), sharex=True)


    def eval_likelihood(edges, fits):
        sphere_pos, kappas, sphere_radius, mu, _ = map_hyperbolic_model_to_sd(jnp.asarray(fits["pos"]), fits["r"])
        return sd_loglikelihood(edges, sphere_pos, kappas, fits["beta"], sphere_radius, mu)
        # return hyperbolic_loglikelihood(edges=edges, pos=jnp.asarray(fits["pos"]), beta=fits["beta"], r=fits["r"])

    def eval_sd_cc(pos, communities):
        pos = jnp.asarray(pos)
        pos = pos/jnp.linalg.norm(pos, axis=1)[:, None]
        return average_community_concentration(hypersphere_distance_matrix(pos), communities)

    def eval_hd_cc(pos, communities):
        return average_community_concentration(hyperbolic_distance_matrix(jnp.asarray(pos)), communities)


    cache = cache_dir()
    for ax_col, og_dim in zip(axes.T, og_dims):
        for i, alg in enumerate(algs):

            cache_file = str(cache_dir().joinpath(f"sd_uniform_comm_concentration_{og_dim}_{alg}.pkl"))
            if not OVERWRITE_CACHE and os.path.isfile(cache_file):
                with open(cache_file, "rb") as file_stream:
                    res, og_res = pickle.load(file_stream)
                print(f"Read {alg} - {og_dim}D results from cache.")
            else:
                res = { key: [] for key in ["hd_ccs", "sd_ccs", "likelihoods"] }
                og_res = { key: [] for key in ["hd_ccs", "sd_ccs", "likelihoods"] }

                for graph_id in tqdm(range(1, 51)):
                    basepath = str(output_dir.joinpath(f"{og_dim}d", f"graph{graph_id}"))

                    if not os.path.isfile(basepath+f"-{alg}-fits.json"):
                        print("skipping", basepath+f"-{alg}-fits.json")
                        continue

                    edges = jnp.asarray(np.loadtxt(basepath+".txt", dtype=np.int32))
                    graph = rx.PyGraph.read_edge_list(basepath+".txt", deliminator=" ", comment="#", multigraph=False, allow_self_loops=False)

                    with open(basepath+"-parameters.json", "r") as file_stream:
                        data = json.load(file_stream)
                        og_params = {"pos": data["pos"], "beta": data["beta"], "r": data["r"]}
                        communities = np.asarray(data.get("labels"), dtype=np.int64)
                    og_res["likelihoods"].append(eval_likelihood(edges, og_params))
                    og_res["hd_ccs"].append(eval_hd_cc(og_params["pos"], communities))
                    og_res["sd_ccs"].append(eval_sd_cc(og_params["pos"], communities))

                    with open(basepath+f"-{alg}-fits.json", "r") as file_stream:
                        fits = json.load(file_stream)["fits"]

                    dims = list(map(int, fits.keys()))
                    res["dims"] = dims
                    res["likelihoods"].append([eval_likelihood(edges, fits[str(d)]) for d in dims])
                    res["sd_ccs"].append([eval_sd_cc(fits[str(d)]["pos"], communities) for d in dims])
                    res["hd_ccs"].append([eval_hd_cc(fits[str(d)]["pos"], communities) for d in dims])

                with open(cache_file, "wb") as file_stream:
                    pickle.dump((res, og_res), file_stream)
            dims = np.array(res["dims"])

            label = alg_to_label(alg)
            inter_spacing = 0.4
            intra_spacing = 0.05
            n_bars = len(algs)
            width = (1 - inter_spacing - (n_bars-1)*intra_spacing)/n_bars
            assert abs(n_bars * width + (n_bars-1) * intra_spacing + inter_spacing - 1.) < 1e-5
            shift = 0.5*inter_spacing + i*width + (i-1)*intra_spacing # relative to D-1
            color = alg_colors[label]
            for ax, (values, og_value) in zip(ax_col, list(map(lambda k: (res[k], og_res[k]), ["sd_ccs", "likelihoods"]))):
                bplot = ax.boxplot(
                    np.array(values), positions=dims-0.5+shift, widths=width, label=label, patch_artist=True,
                    medianprops={"color": "k"},
                    boxprops={"facecolor": color, "edgecolor": "none"},
                    whiskerprops={"color": color},
                    flierprops={"marker": ".", "markersize": 3, "markerfacecolor": color, "markeredgecolor": "none"}
                )

                if og_value != []:
                    lo, med, hi = np.percentile(og_value, [25, 50, 75])
                    xlims = ax.get_xlim()
                    ax.fill_between(xlims, [lo]*2, [hi]*2, color="gray", alpha=0.5)
                    ax.axhline(med, color="k")

        ax_col[0].set_title(f"Sampled from $\\mathbb{{S}}^{og_dim-1}$")

    # axes[0, 0].set_ylabel(r"$c_C$ using $\mathbb{H}^D$")
    # for ax in axes[0]:
        # ax.set_xlim(1.5, 6.5)
        # ax.set_xticks(dims, []*len(dims))

    axes[0, 0].set_ylabel(r"$c_C$")
    for ax in axes[1]:
        ax.set_xlim(1.5, 6.5)
        ax.set_xticks(dims, list(map(str, dims)))

    axes[1, 0].set_ylabel("$\\mathbb{S}^{D-1}$ log-likelihood")
    for ax in axes[1]:
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.set_xlim(1.5, 6.5)
        ax.set_xlabel("Dimension")
        ax.set_xticks(dims, list(map(str, dims)))


    labels = list(map(alg_to_label, algs))
    handles = [Patch(color=alg_colors[label]) for _, label in enumerate(labels)]
    fig.legend(
        handles, labels,
        loc='lower center',
        bbox_to_anchor=(0.5, 0),
        ncol=len(labels),
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)
    label_subplots_letter(axes, yshift=.1)


    fig.savefig(os.path.join(figure_dir(), "comm_concentration_sd_model.pdf"))
    plt.show()
