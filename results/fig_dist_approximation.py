import os
from matplotlib import pyplot as plt
from matplotlib import lines, patches
import numpy as np
import jax

from plot_tools import color_gradient, format_ticks, label_subplots_letter, text_width, figure_dir


def approx_dist(r1, r2, da):
    return r1 + r2 + 2*np.log(da/2)

def exact_dist(r1, r2, da):
    return np.acosh(np.cosh(r1)*np.cosh(r2) - np.sinh(r1)*np.sinh(r2)*np.cos(da))

def connection_prob(dist, beta, r):
    return jax.nn.sigmoid(0.5 * beta * (r - dist))


fig, axes = plt.subplots(1, 2, figsize=(0.8*text_width, 0.3*text_width))

beta = 2.
r = 6.

dist_values = np.linspace(0.01, np.pi, 100)
radii = np.arange(0., 8, 2)
colors = color_gradient(len(radii))
for r1, color in zip(radii, colors):
    r2 = r1+5
    axes[0].plot(dist_values, exact_dist(r1, r2, dist_values), color=color, label=f"$r={r1:.0f}$")
    axes[0].plot(dist_values, approx_dist(r1, r2, dist_values), color=color, ls="--")

    hd_prob = connection_prob(exact_dist(r1, r2, dist_values), beta, r)
    sd_prob = connection_prob(approx_dist(r1, r2, dist_values), beta, r)
    axes[1].plot(dist_values, hd_prob, color=color, label=f"$r_u={r1:.0f}$")
    axes[1].plot(dist_values, sd_prob, color=color, ls="--")

axes[0].axhline(0, color="k", lw=0.9, zorder=0)
axes[0].set_ylabel(f"$d(x_u, x_v)$")
axes[0].axhline(r, color="k", ls="dotted", alpha=0.5, zorder=0)
# axes[0].text(np.pi/5, r, "$d(x, y)=\\phi$", va="bottom", size="small", color="k" , alpha=0.5)


handles = [patches.Patch(color=c) for c in colors] + [lines.Line2D([], [], color="gray", ls=ls) for ls in ["-", "--"]]
labels = [f"$r_u={radius:.0f}$" for radius in radii] + [r"Exact ($\mathbb{H}^D$)", r"Approx. ($\mathbb{S}^{D-1}$)"]

fig.legend(
    handles, labels,
    loc='lower center',
    bbox_to_anchor=(0.5, 0),
    ncol=len(labels),
)


axes[1].set_ylabel(r"$p_{uv}(x, \beta, \phi)$")
for ax in axes:
    ax.set_xlim((0, None))
    ax.set_xticks(*format_ticks(4))
    ax.set_xlabel(r"$\Delta\theta(\theta_u, \theta_v)$")



fig.tight_layout()

ax = axes[0]
# Add phi_value to the existing yticks
existing_ticks = ax.get_yticks().tolist()
all_ticks = sorted(set(existing_ticks + [r]))
ax.set_yticks(all_ticks)

# Build custom labels: use φ symbol for phi_value, default for the rest
labels = []
for t in all_ticks:
    if np.isclose(t, r):
        labels.append(r'$\phi$')
    else:
        labels.append(f'{t:.0f}')
ax.set_yticklabels(labels)

# Color only the phi tick label gray
for tick, label in zip(ax.yaxis.get_major_ticks(), labels):
    if label == r'$\phi$':
        tick.label1.set_color('gray')




fig.subplots_adjust(bottom=0.4)
label_subplots_letter(axes, xshift=-0.2, yshift=0.1)

fig.savefig(os.path.join(figure_dir(), "dist_approximation.pdf"))
