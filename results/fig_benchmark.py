import os
import json
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from plot_tools import alg_to_label, label_subplots_letter, text_width, figure_dir, alg_colors


with open(".output/benchmark/results.json", "r") as file_stream:
    res = json.load(file_stream)

fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(text_width, 0.5*text_width))


dims = list(res["dims"])
dims.sort(key=int)

algs = ["dmercator", "gd", "sgd", "rgd", "srgd", "radam", "sradam"]
handles = []
labels = []
for ax, dim in zip(axes.ravel(), dims):
    for alg in algs:
        label = alg_to_label(alg)
        color = alg_colors[label]
        ls = "--" if alg[0] == "s" else "-"
        ax.plot(res["ns"], res[alg][str(dim)], color=color, label=label, ls=ls)

        if len(labels) <= 0.5*(len(algs)-1) and alg[0] != "s":
            handles.append(Line2D([], [], color=color))
            labels.append(label)

    ax.set_title(f"$D={dim}$")
    ax.set_yscale("log")

for ax in axes:
    ax[0].set_ylabel("Time [s]")
for ax in axes[-1]:
    ax.set_xlabel("Number of vertices $n$")

handles += [Line2D([], [], color="gray", ls=ls) for ls in ["-", "--"]]
labels += ["Exact", "Stochastic"]
fig.legend(
    handles, labels,
    loc='lower center',
    bbox_to_anchor=(0.5, 0),
    ncol=len(labels),
)

fig.tight_layout()
fig.subplots_adjust(bottom=0.2)
label_subplots_letter(axes, xshift=-0.08, yshift=0.15)

fig.savefig(os.path.join(figure_dir(), "benchmark.pdf"))
plt.show()
