import json
import os
import pickle
import re
from argparse import ArgumentParser

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from tqdm import tqdm

from mdl_hyperbolic.models import hyperbolic_loglikelihood, prior_logdensity
from plot_tools import text_width, alg_colors, alg_to_label


jax.config.update('jax_enable_x64', True) # enable double precision floating-point


def extract_alg_name(dir_name):
    ms = re.match(r".*steps_(.*)", dir_name)
    if ms is None:
        return "?"
    return ms[1]


parser = ArgumentParser()
parser.add_argument("graph_file")
parser.add_argument("dir_prefix", nargs="+")
parser.add_argument("--dims", type=int, nargs="+")
parser.add_argument("--mle", action="store_true")
parser.add_argument("--save", type=str)
parser.add_argument("--gen-file", type=str)
args = parser.parse_args()

n_cols = 3
n_rows = np.ceil(len(args.dims)/n_cols).astype(np.int32)
fig, axes = plt.subplots(n_rows, n_cols, sharex=True, figsize=(text_width, 0.5*text_width))
empty_plots = n_rows*n_cols - len(args.dims)
if n_rows == 1:
    axes = np.array([axes])
if empty_plots>0:
    for ax in axes.ravel()[-empty_plots:]:
        fig.delaxes(ax)


if args.gen_file:
    with open(args.gen_file, "r") as fstream:
        data = json.load(fstream)
        gen_params = {
            "pos": jnp.asarray(data["pos"]),
            "beta": data["beta"],
            "r": data["r"],
            "rho": data["r"],
            "nu": 5.,
        }
        gen_dim = gen_params["pos"].shape[1]
else:
    gen_params = None

edges = jnp.asarray(np.loadtxt(args.graph_file, dtype=np.int32))

gradient_algs = ["RAdam", "RGD", "GD"]
handles = [Patch(color=color) for name, color in alg_colors.items() if name in gradient_algs]
handles += [Line2D([], [], color="#D1D1D1", ls=ls) for ls in ["-", "--"]]
labels = [name for name in alg_colors if name in gradient_algs] + ["Exact", "Stochastic"]
if gen_params:
    handles += [Line2D([], [], color="k", ls="-")]
    labels += ["Original parameters"]

if args.mle:
    @jax.jit
    def loss(params):
        return hyperbolic_loglikelihood(pos=params["pos"], beta=params["beta"], r=params["r"], edges=edges)
else:
    @jax.jit
    def loss(params):
        pos, beta, r, rho, nu = list(params[p] for p in ["pos", "beta", "r", "rho", "nu"])
        return hyperbolic_loglikelihood(pos=pos, beta=beta, r=r, edges=edges)\
                + prior_logdensity(pos=pos, beta=beta, r=r, rho=rho, nu=nu)

gen_likelihood = loss(gen_params) if gen_params else None

for ax, dim in zip(axes.ravel(), tqdm(args.dims)):
    ax.set_title(f"$D={dim}$")
    for prefix in args.dir_prefix:
        alg_name = extract_alg_name(prefix)
        ls = "--" if alg_name[0] == "s" else "-"
        color = alg_colors[alg_to_label(alg_name)]

        output_directory = f"{prefix}/{dim}"
        loss_values = {}
        for file in os.listdir(output_directory):
            iteration = int(file[len("embedding_"):-4])
            if iteration == 0:
                continue

            with open(os.path.join(output_directory, file), "rb") as file_stream:
                params = pickle.load(file_stream)
            loss_values[iteration] = loss(params)
        xvals = list(loss_values)
        xvals.sort()
        ax.plot(xvals, [loss_values[i] for i in xvals], color=color, ls=ls)
    if gen_likelihood and int(dim) >= gen_dim:
        ax.axhline(gen_likelihood, color="k", label="Original parameters", zorder=0)

for ax in axes[:, 0]:
    if args.mle:
        ax.set_ylabel("Log-likelihood")
    else:
        ax.set_ylabel("Posterior\ndensity")
for ax in axes[-1, :]:
    ax.set_xlabel("Iteration")

fig.legend(
    handles, labels,
    loc='lower center',      # anchor point on the legend box
    bbox_to_anchor=(0.5, 0), # (x=center, y=just below figure)
    ncol=len(labels),        # display on one line
)
fig.tight_layout()
fig.subplots_adjust(bottom=0.2)

if args.save:
    fig.savefig(args.save)
plt.show()
