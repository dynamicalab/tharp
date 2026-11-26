import json
import os
from concurrent.futures import ProcessPoolExecutor
import pickle
from argparse import ArgumentParser

from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Bbox
import numpy as np
import matplotlib.pyplot as plt
import rustworkx as rx
import networkx as nx
import jax
import jax.numpy as jnp
from tqdm import tqdm

from mdl_hyperbolic.geometry import expand_positions, hyperbolic_distance, hyperbolic_distance_matrix, hypersphere_distance_matrix
from mdl_hyperbolic.models import hyperbolic_loglikelihood, sd_connection_prob_pairs, sd_loglikelihood

from embedding_properties import average_community_concentration, count_chordless_cycles, connection_prob_matrix
from plot_tools import label_subplots_letter, text_width, figure_dir, alg_colors, cache_dir, alg_to_label


def _init_worker(probs):
    global _probs
    _probs = probs

def _sample_and_count(_):
    a = np.triu(np.random.rand(*_probs.shape) < _probs, k=1).astype(np.double)
    a = a + a.T
    pred_graph = rx.PyGraph.from_adjacency_matrix(a)
    pred_edges = np.asarray(pred_graph.edge_list())
    pred_props = {}
    pred_props["cycles"] = count_chordless_cycles(pred_edges)
    nx_pred_graph = nx.from_numpy_array(a)
    pred_props["assortativity"] = nx.degree_assortativity_coefficient(nx_pred_graph)
    return pred_props

def compute_properties(probs, original_adj, n_samples=50, n_workers=None):
    props = {}
    props["pred_error"] = np.average(np.abs(original_adj - probs))
    props["expected_degree"] = np.average(jnp.sum(probs, axis=0))

    with ProcessPoolExecutor(
        max_workers=n_workers, initializer=_init_worker, initargs=(probs,)
    ) as executor:
        tmp = list(executor.map(_sample_and_count, range(n_samples)))

    props["cycles"] = [v["cycles"] for v in tmp]
    props["assortativity"] = [v["assortativity"] for v in tmp]
    return props


def map_to_hypersphere(pos):
    pos = jnp.asarray(pos)
    return pos/jnp.linalg.norm(pos, axis=1)[:, None]


def eval_sd_cc(pos, communities):
    pos = jnp.asarray(pos)
    pos = pos/jnp.linalg.norm(pos, axis=1)[:, None]
    return average_community_concentration(hypersphere_distance_matrix(pos), communities)

def eval_hd_cc(pos, communities):
    return average_community_concentration(hyperbolic_distance_matrix(jnp.asarray(pos)), communities)


if __name__ == "__main__":
    jax.config.update('jax_enable_x64', True) # enable double precision floating-point

    parser = ArgumentParser()
    parser.add_argument("objective", choices=["mle", "mdl"])
    parser.add_argument("graph", choices=["comm", "unif"])
    parser.add_argument("--ignore-cache", action="store_true")
    args = parser.parse_args()

    if args.graph == "comm":
        path = ".output/community_case_study/"
    else:
        path = ".output/unif_case_study/"

    if args.objective == "mle":
        path += "mle/"

    if args.graph == "comm":
        base_name = path + "synth_communities"
    else:
        base_name = path + "graph1000_4D"

    graph_file = base_name + ".txt"
    og_fit = base_name + "-parameters.json"
    fit_file_format = base_name + "-{}-fits.json"
    dmercator_file = base_name + "-dmercator-fits.json"

    def parse_fit_file(file_name):
        with open(file_name, "r") as file_stream:
            fits = json.load(file_stream)["fits"]
            for d in fits:
                fits[d]["pos"] = jnp.asarray(fits[d]["pos"])
                if "kappa" in fits[d]:
                    fits[d]["kappa"] = jnp.asarray(fits[d]["kappa"])
        return fits

    # Files are parsed one at the time to limit GPU usage
    fits = {
        "GD": lambda: parse_fit_file(fit_file_format.format("gd")),
        "RGD": lambda: parse_fit_file(fit_file_format.format("rgd")),
        "RAdam": lambda: parse_fit_file(fit_file_format.format("radam")),
        "D-Mercator": lambda: parse_fit_file(dmercator_file),
        "D-Mercator*": lambda: parse_fit_file(dmercator_file),
    }

    cache_file = str(cache_dir().joinpath(f"fig_{args.objective}_{args.graph}_properties.pkl"))
    use_cache = not args.ignore_cache and os.path.isfile(cache_file)

    # Compute properties
    if use_cache:
        with open(cache_file, "rb") as file_stream:
            og_props, all_props = pickle.load(file_stream)
    else:
        import multiprocessing as mp
        mp.set_start_method("spawn") # For safe multi processing inside jax

        with open(og_fit, "r") as file_stream:
            og_params = json.load(file_stream)
            og_params["pos"] = jnp.asarray(og_params["pos"])
            communities = og_params.get("labels")
            if communities is not None:
                communities = np.asarray(communities, dtype=np.int64)

        edges = jnp.asarray(np.loadtxt(graph_file, dtype=np.int32))
        graph = rx.PyGraph.read_edge_list(graph_file, deliminator=" ", comment="#", multigraph=False, allow_self_loops=False)
        n = graph.num_nodes()
        degrees = np.array([graph.degree(u) for u in range(n)])

        adj = np.asarray(rx.adjacency_matrix(graph))
        nx_graph = nx.from_numpy_array(adj)
        og_probs = connection_prob_matrix(expand_positions(jnp.array(og_params["pos"])), og_params["beta"], og_params["r"], dist=hyperbolic_distance)

        og_props = {
            "expected_degree": np.average(degrees),
            "cycles": count_chordless_cycles(edges),
            "success_rate": rx.hyperbolic_greedy_success_rate(graph, og_params["pos"]),
            "pred_error": np.average(np.abs(adj-og_probs)),
            "assortativity": nx.degree_assortativity_coefficient(nx_graph),
            "likelihood": hyperbolic_loglikelihood(edges=edges, pos=og_params["pos"], beta=og_params["beta"], r=og_params["r"]),
            "gen_dim": og_params["pos"].shape[1],
        }
        if communities is not None:
            og_props["sd_cc"] = eval_sd_cc(og_params["pos"], communities)
            og_props["hd_cc"] = eval_hd_cc(og_params["pos"], communities)

        all_props = dict()
        for alg in fits:
            print(alg)
            alg_fits = fits[alg]()
            dims = list(map(int, alg_fits.keys()))
            dims.sort()

            props = dict()
            for dim in tqdm(dims):
                params = alg_fits[str(dim)]
                if "*" in alg:
                    kappas = np.array(params["kappa"])
                    sphere_pos = params["pos"]/np.linalg.norm(params["pos"], axis=1)[:, None]
                    d = int(dim) - 1
                    hd_radius = 2*np.log(2*params["r_s"]) - (2/d)*np.log(params["mu"]*params["kappa0"]**2)
                    radii = hd_radius - 2./d*np.log(kappas/params["kappa0"])

                    iu, iv = np.indices((n, n))
                    iu, iv = iu.flatten(), iv.flatten()
                    probs = sd_connection_prob_pairs(
                        pos1=sphere_pos[iu], pos2=sphere_pos[iv],
                        kappa1=params["kappa"][iu], kappa2=params["kappa"][iv],
                        beta=params["beta"], sphere_radius=params["r_s"], mu=params["mu"]).reshape((n, n))

                    loglikelihood = sd_loglikelihood(
                        edges=edges, sphere_pos=sphere_pos, kappas=kappas, beta=params["beta"], sphere_radius=params["r_s"], mu=params["mu"]
                    )
                    success_rate = rx.approx_hyperbolic_greedy_success_rate(graph, radii, sphere_pos)
                else:
                    probs = connection_prob_matrix(expand_positions(jnp.array(params["pos"])), params["beta"], params["r"], dist=hyperbolic_distance)
                    loglikelihood = hyperbolic_loglikelihood(edges=edges, pos=params["pos"], beta=params["beta"], r=params["r"])
                    success_rate = rx.hyperbolic_greedy_success_rate(graph, params["pos"])
                probs = jnp.fill_diagonal(probs, 0, inplace=False)

                props[dim] = compute_properties(np.array(probs), original_adj=adj)
                props[dim]["success_rate"] = success_rate
                props[dim]["loglikelihood"] = loglikelihood
                if communities is not None:
                    props[dim]["sd_cc"] = eval_sd_cc(params["pos"], communities)
                    props[dim]["hd_cc"] = eval_hd_cc(params["pos"], communities)

            del alg_fits # Clear unused GPU memory
            all_props[alg] = props

        with open(cache_file, "wb") as file_stream:
            pickle.dump((og_props, all_props), file_stream)

    # Plot results
    plot_cc = "sd_cc" in og_props

    if plot_cc:
        fig, axes = plt.subplots(3, 3, figsize=(text_width, .6*text_width))
    else:
        fig, axes = plt.subplots(3, 3, figsize=(text_width, .56*text_width))

    first_algo = next(iter(fits))
    for alg in fits:
        ax_iter = iter(axes.ravel())
        color = alg_colors[alg]
        props = all_props[alg]

        dims = list(map(int, all_props[alg].keys()))
        dims.sort()

        og_style = {"label": "Original", "marker": "*", "color": "k", "zorder": 4, "s": 30, "edgecolor": "none"}
        style = {"color": color, "marker":".", "ls":"--" if "*" in alg else "-", "markersize": 4, "lw": 1.}

        # Likelihood
        ax = next(ax_iter)
        if alg == first_algo:
            ax.scatter(og_props["gen_dim"], og_props["likelihood"], **og_style)
        ax.plot(dims, [props[dim]["loglikelihood"] for dim in dims], label=alg, **style)
        ax.set_ylabel("Log-likelihood")
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

        # # L1 prob. error
        # ax = next(ax_iter)
        # ax.plot(dims, [props[dim]["pred_error"] for dim in dims], **style)
        # if alg == first_algo:
            # ax.scatter(og_props["gen_dim"], og_props["pred_error"], **og_style)
        # ax.set_ylabel("Av. $L^1$ error")

        # Expected degrees
        ax = next(ax_iter)
        ax.plot(dims, [props[dim]["expected_degree"] for dim in dims], **style)
        if alg == first_algo:
            ax.scatter(og_props["gen_dim"], og_props["expected_degree"], **og_style)
        ax.set_ylabel(r"$\mathbb{E}[\bar{K}]$")

        # Degree assortativity
        ax = next(ax_iter)
        lo, med, hi = np.percentile([props[dim]["assortativity"] for dim in dims], [25, 50, 75], axis=1)
        ax.plot(dims, med, **style)
        ax.fill_between(dims, lo, hi, color=color, alpha=0.3)
        if alg == first_algo:
            ax.scatter(og_props["gen_dim"], og_props["assortativity"], **og_style)
        ax.set_ylabel(r"Deg. assortativity")

        # Chordless cycles (3,4,5-cycles)
        cycles = np.array([props[dim]["cycles"] for dim in dims])
        all_perc = np.percentile(cycles, [25, 50, 75], axis=1).T
        for i, (perc, og_val) in enumerate(zip(all_perc, og_props["cycles"])):
            lo, med, hi = perc.T
            ax = next(ax_iter)
            ax.plot(dims, med, **style)
            ax.fill_between(dims, lo, hi, color=color, alpha=0.3)
            if alg == first_algo:
                ax.scatter(og_props["gen_dim"], og_val, **og_style)
            ax.set_ylabel(f"{i+3}-cycles")
            ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 0))

        # Greedy routing
        ax = next(ax_iter)
        ax.plot(dims, [props[dim]["success_rate"] for dim in dims], **style)
        if alg == first_algo:
            ax.scatter(og_props["gen_dim"], og_props["success_rate"], **og_style)
        ax.set_ylabel("GR success")

        # Community concentration
        if plot_cc:
            ax = next(ax_iter)
            ax.plot(dims, [props[dim]["sd_cc"] for dim in dims], **style)
            if alg == first_algo:
                ax.scatter(og_props["gen_dim"], og_props["sd_cc"], **og_style)
            ax.set_ylabel("$c_C$")


        # D-Mercator settings subplots. Assumes cache was built in their respective scripts.
        # Community concentration
        algs = ["dmercator", "gd", "radam"]
        ax_gr = next(ax_iter)
        ax_cc = next(ax_iter)
        dmercator_dim = 3 # Fixed
        for i, alg in enumerate(algs):
            gr_cache = str(cache_dir().joinpath(f"sd_uniform_gr_{dmercator_dim}_{alg}.pkl"))
            with open(gr_cache, "rb") as file_stream:
                res_gr = pickle.load(file_stream)
            cc_cache = str(cache_dir().joinpath(f"sd_uniform_comm_concentration_{dmercator_dim}_{alg}.pkl"))
            with open(cc_cache, "rb") as file_stream:
                res_cc, og_res_cc = pickle.load(file_stream)

            dims = np.array(res_cc["dims"])

            label = alg_to_label(alg)
            inter_spacing = 0.4
            intra_spacing = 0.05
            n_bars = len(algs)
            width = (1 - inter_spacing - (n_bars-1)*intra_spacing)/n_bars
            assert abs(n_bars * width + (n_bars-1) * intra_spacing + inter_spacing - 1.) < 1e-5
            shift = 0.5*inter_spacing + i*width + (i-1)*intra_spacing # relative to D-1
            color = alg_colors[label]
            for ax, values, og_value in zip([ax_gr, ax_cc], [res_gr["grs"], res_cc["sd_ccs"]], [res_gr["og_grs"], og_res_cc["sd_ccs"]]):
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
                    ax.fill_between(xlims, [lo]*2, [hi]*2, color="#CFCFCF", alpha=0.5, zorder=0)
                    ax.axhline(med, color="#3B3B3B", lw=0.75, zorder=0)
                ax.set_xticks(dims, dims)
        ax_cc.set_ylabel(r"$c_C$")
        ax_gr.set_ylabel(r"GR success")

    for ax in axes.ravel():
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    for ax in axes[-1]:
        ax.set_xlabel("Dimension")

    fig.tight_layout()
    if plot_cc:
        fig.subplots_adjust(hspace=0.35)
        label_subplots_letter(axes.ravel()[:-1], xshift=-0.3, yshift=0.22)
        ax = axes.ravel()[-1]
        pos = ax.get_position()
        shifted_pos = Bbox.from_bounds(pos.x0 - 0.02, pos.y0 - 0.03, pos.width, pos.height)
        fig.delaxes(ax)

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles, labels,
            bbox_to_anchor=shifted_pos, # (x=center, y=just below figure)
        )
    else:
        fig.subplots_adjust(bottom=0.2, hspace=0.45)
        label_subplots_letter(axes, xshift=-0.25, yshift=0.22)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles, labels,
            loc='lower center',      # anchor point on the legend box
            bbox_to_anchor=(0.5, 0), # (x=center, y=just below figure)
            ncol=len(labels),        # one line display
        )

    fig.savefig(os.path.join(figure_dir(), f"{args.objective}_{args.graph}_properties.pdf"), bbox_inches="tight")
    plt.show()
