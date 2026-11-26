import os

import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt
import jax

from mdl_hyperbolic.geometry import time_coordinates, log_sinh

from mdl_hyperbolic.priors import log_I_r
from plot_tools import label_subplots_letter, text_width, figure_dir, color_gradient


jax.config.update('jax_enable_x64', True) # enable double precision floating-point

def base_prior_lpdf(radii, nu, rho, dim):
    return -jax.nn.softplus(nu*(dim-1)*(radii-rho))


fig, axes = plt.subplots(2, 2, figsize=(text_width, .45*text_width), sharey="row")


dims = [2, 3, 4, 5]
rho = 5.
x_rho = np.sinh(rho)
rvalues = np.linspace(0.05, 6.5, 200)
xvalues = np.linspace(0, np.sinh(np.max(rvalues)), 200)
nus = [1.001, 1.01, 1.5, 100.]
base_nu = 1.5
for dim in dims:
    padding = np.zeros((len(xvalues), dim-1))
    padded_x = np.hstack([xvalues[:, None], padding])

    x_time = time_coordinates(padded_x)
    x_radii = np.acosh(x_time)
    color = color_gradient(len(dims))[dims.index(dim)]

    # Extended uniform density...
    lognorm = log_I_r(dim, rho, base_nu)

    # ... in geodesic coords
    axes[0, 0].plot(rvalues, base_prior_lpdf(rvalues, base_nu, rho, dim) - lognorm + log_sinh(rvalues),
                 ls="-", label=f"$D={dim}$", color=color)

    # ... in hyperboloid coords
    axes[1, 0].plot(xvalues, base_prior_lpdf(x_radii, base_nu, rho, dim) - lognorm - np.log(x_time),
                 ls="-", label=f"$D={dim}$", color=color)

    # Uniform density...
    norm = integrate.quad(lambda x: np.power(np.sinh(x), dim-1), 0., rho)[0]
    base_logdensity = -np.log(norm)
    # ... in geodesic coords
    axes[0, 0].plot(rvalues[rvalues<rho], base_logdensity + log_sinh(rvalues[rvalues<rho]),
                 ls="dotted", color=color)

    # Extended uniform density in hyperboloid coords
    axes[1, 0].plot(xvalues[x_radii < rho], base_logdensity - np.log(x_time[x_radii < rho]),
                 ls="dotted", color=color)

    # In col 2, we vary nu instead of dim
    if dim == 4:
        for nu in nus:
            # Extended uniform density...
            lognorm = log_I_r(dim, rho, nu)
            base_logdensity = base_prior_lpdf(x_radii, nu, rho, dim) - lognorm

            color = color_gradient(len(nus))[nus.index(nu)]
            # ... in geodesic coords
            axes[0, 1].plot(rvalues, base_prior_lpdf(rvalues, base_nu, rho, dim) - lognorm + log_sinh(rvalues),
                         ls="-", label=f"$\\xi={nu}$", color=color)

            # ... in hyperboloid coords
            axes[1, 1].plot(xvalues, base_prior_lpdf(x_radii, base_nu, rho, dim) - lognorm - np.log(x_time),
                         ls="-", color=color)

        # Uniform density...
        norm = integrate.quad(lambda x: np.power(np.sinh(x), dim-1), 0., rho)[0]
        base_logdensity = -np.log(norm)
        # ... in geodesic coords
        axes[0, 1].plot(rvalues[rvalues<rho], base_logdensity + log_sinh(rvalues[rvalues<rho]),
                     ls="dotted", label="Uniform", color="k", lw=2)

        # Extended uniform density in hyperboloid coords
        axes[1, 1].plot(xvalues[x_radii < rho], base_logdensity - np.log(x_time[x_radii < rho]),
                     ls="dotted", color="k", lw=2)


for ax in axes[0]:
    ax.axvline(rho, color="#919191", zorder=0)
    ax.text(rho+0.1, -22, r"$r=\rho$", color="#919191", zorder=0, fontsize="small")
    ax.set_xlabel("$r$")

for ax in axes[1]:
    ax.axvline(x_rho, color="#919191", zorder=0)
    ax.text(x_rho+5, -32, r"$x^1=\sinh \rho$", color="#919191", zorder=0, fontsize="small")
    ax.set_xlabel("$x^1$")

for ax in axes[:, 0]:
    ax.set_ylabel("Log-density")

for ax in axes.ravel():
    ax.set_xlim((0, None))

axes[0, 0].plot([], [], color="gray", label="Uniform", ls="dotted")
axes[0, 0].plot([], [], color="gray", label="Extended uniform", ls="-")
axes[0, 1].plot([], [], color="gray", label="Extended uniform", ls="-")


legends = []
for col in range(2):
    handles, labels = axes[0, col].get_legend_handles_labels()

    x_anchor = 0.3 if col == 0 else 0.77
    leg = fig.legend(
        handles, labels,
        loc='lower center',
        bbox_to_anchor=(x_anchor, 0),
        ncols=3,
    )
    legends.append(leg)

fig.tight_layout()
fig.subplots_adjust(bottom=0.27)
label_subplots_letter(axes, xshift=-0.1, yshift=0.15)

fig.savefig(os.path.join(figure_dir(), "pos_prior.pdf"))
plt.show()
