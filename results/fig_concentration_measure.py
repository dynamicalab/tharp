import os

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import jax
import numpy as np
from scipy.special import gamma as gamma_fn

from mdl_hyperbolic.models import estimate_parameters_for_uniform
from mdl_hyperbolic.generators import sample_quasiuniform_hyperbolic_ball, sample_quasiuniform_radius, sample_unit_vector_uniformly
from mdl_hyperbolic.geometry import geodesic_to_hyperboloid, hyperbolic_distance_matrix, expand_positions, hyperboloid_to_geodesic
from plot_tools import color_gradient, figure_dir, label_subplots_letter, text_width


jax.config.update('jax_enable_x64', True) # enable double precision floating-point


def key_gen():
    k = jax.random.PRNGKey(0)
    while True:
        k, new_key = jax.random.split(k, 2)
        yield new_key

key = key_gen()


dims = np.arange(4, 11)
expected_degree = 10
gamma = 2.7
n = 500
base_beta = 2.5

shift = 1

fig, axes = plt.subplots(1, 2, figsize=(text_width, 0.3*text_width), sharey=True)


for color, og_dim in zip(color_gradient(len(dims)), dims):
    beta = base_beta * (og_dim-1)
    r, ball_r, alpha = estimate_parameters_for_uniform(n, og_dim, beta, gamma, expected_degree)
    pos = sample_quasiuniform_hyperbolic_ball(og_dim, ball_r, key=next(key), num_samples=n, alpha=alpha)
    radii, angles = hyperboloid_to_geodesic(pos)

    full_dist = hyperbolic_distance_matrix( expand_positions(pos) )
    average_errors = []
    ms = np.arange(1, og_dim-1)
    for m in ms:
        # compute Z_{U_{D-m}}
        sphere_pos = geodesic_to_hyperboloid(np.full_like(radii, np.asinh(1)), angles[:, m:])
        sub_pos = sphere_pos / np.linalg.norm(sphere_pos, axis=1)[:, None] * np.sinh(radii)[:, None]
        sub_dist = hyperbolic_distance_matrix( expand_positions(sub_pos) )
        average_errors.append( np.average(np.abs(full_dist-sub_dist)) )
    axes[0].plot(ms, average_errors, color=color)
    axes[1].plot(og_dim - ms, average_errors, color=color)

    d = og_dim
    # First-order approximation
    var = 1/d + 1/(d-ms) - 2/(d-ms)*(gamma_fn(d/2)*gamma_fn(0.5*(d-ms+1)) / (gamma_fn(0.5*(d+1)) * gamma_fn(0.5*(d-ms)) ))**2
    expected_error = np.sqrt(2/np.pi) * np.sqrt(var) * np.sinh(ball_r)**2 / np.sqrt(np.cosh(ball_r)**4 - 1)

    # axes[0].plot(ms, expected_error, color=color, ls="--")
    # axes[1].plot(og_dim - ms, expected_error, color=color, ls="--")

    yvals = average_errors
    axes[0].text(ms[-1], yvals[-1] + 0.05, f"$D={og_dim}$", color=color, fontsize="xx-small", va="top", ha="center")
    axes[1].text(og_dim - ms[0], yvals[0] - 0.02, f"$D={og_dim}$", color=color, fontsize="xx-small", va="top", ha="center")

# handles = [Line2D([], [], color="gray", ls=ls) for ls in ["-", "--"]]
# labels = ["Numerical approx.", "Analytical approx."]
# axes[0].legend(handles, labels)

axes[0].set_xlabel("Dimensions removed $m$")
axes[1].set_xlabel("Reduced dimension $D-m$")
axes[0].set_ylabel("$\\mathbb{E}[|d(U, V) - d(U_{D-m}, V_{D-m})|]$")
axes[0].set_ylim((0.0, None))

plt.tight_layout()
label_subplots_letter(axes, xshift=-0.08, yshift=0.03)

plt.savefig(os.path.join(figure_dir(), "concentration_measure.pdf"))
plt.show()
