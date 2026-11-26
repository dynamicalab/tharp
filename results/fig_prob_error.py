import json
import os

import rustworkx as rx
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from mdl_hyperbolic.geometry import expand_positions, hyperbolic_distance
from mdl_hyperbolic.sgd_sampling import get_uniform_non_edge_sampler
from mdl_hyperbolic.models import connection_prob_pairs
from plot_tools import figure_dir, text_width


path = ".output/unif_case_study/mle/"
base_name = path + "graph1000_4D"

graph_file = base_name + ".txt"
fit_file_format = base_name + "-{}-fits.json"

edges = np.array(np.loadtxt(graph_file, dtype=np.int32))

with open(fit_file_format.format("gd"), "r") as file_stream:
    fits = json.load(file_stream)["fits"]
    dims = [key for key in fits if "pos" in fits[key]]
    dims.sort(key=int)


edges = np.loadtxt(graph_file, dtype=np.int32)
edges = jnp.asarray(np.unique(edges, axis=0))
n = jnp.max(edges)+1

batch_size = 1024
sample_size = 1e4
n_loops = int(sample_size/batch_size)
sampler = get_uniform_non_edge_sampler(n, edges, batch_size=batch_size)

iter_key = jax.random.PRNGKey(0)

dims = list(fits.keys())
dims.sort(key=int)

fig, axes = plt.subplots(3, np.ceil(len(dims)/3), sharey=True, sharex=True, figsize=(.6*text_width, .4*text_width))

def get_non_edges(edges, n):
    # Ensure edges are formatted with u < v
    edges = np.sort(edges, axis=1)

    # Generate all upper-triangle pairs (u < v)
    iu, iv = np.triu_indices(n, k=1)

    # Convert 2D indices to unique 1D IDs
    all_ids = iu * n + iv
    edge_ids = edges[:, 0] * n + edges[:, 1]

    # Find IDs in all_ids that are NOT in edge_ids
    non_edge_ids = np.setdiff1d(all_ids, edge_ids)

    # Unflatten 1D IDs back to 2D coordinates
    non_edges = np.column_stack((non_edge_ids // n, non_edge_ids % n))
    return non_edges.T

purple = "#967bc0"
orange = "#ec9a29"


graph = rx.PyGraph.read_edge_list(graph_file, deliminator=" ", comment="#", multigraph=False, allow_self_loops=False)

iu, iv = get_non_edges(edges, n)
for u, v in zip(iu, iv):
    assert(not graph.has_edge(u, v))


for ax, dim in zip(axes.ravel(), dims):
    fit = fits[dim]
    pos = expand_positions(jnp.asarray(fit["pos"]))

    edge_probs = connection_prob_pairs(pos[edges[:, 0]], pos[edges[:, 1]], beta=fit["beta"], r=fit["r"], dist=hyperbolic_distance)
    non_edges_probs = connection_prob_pairs(pos[iu], pos[iv], beta=fit["beta"], r=fit["r"], dist=hyperbolic_distance)

    ax.hist([edge_probs, non_edges_probs], label=["Edges", "Non-edges"], color=[purple, orange], density=True)
    ax.text(0.5, 10, f"$D={dim}$")
    ax.yaxis.set_visible(False)
    ax.spines[["left"]].set_visible(False)

for ax in axes[-1]:
    ax.set_xlabel("Edge probability $p_{uv}$")
axes.ravel()[-1].legend()
plt.tight_layout()
fig.savefig(os.path.join(figure_dir(), f"unif_mle_prob_error.pdf"), bbox_inches="tight")
plt.show()
