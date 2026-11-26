import json
import os

from scipy import stats, special
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from mdl_hyperbolic.models import connection_prob_pairs, map_sd_to_hyperbolic_model, optimize_kappa0, sd_connection_prob_pairs
from mdl_hyperbolic.geometry import expand_positions, hyperbolic_distance, time_coordinates
from mdl_hyperbolic.generators import sample_unit_vector_uniformly
from plot_tools import label_subplots_letter, text_width, figure_dir


jax.config.update('jax_enable_x64', True) # enable double precision floating-point


dmercator_file = ".output/community_case_study/synth_communities-dmercator-fits.json"

with open(dmercator_file, "r") as file_stream:
    dmercator_fits = json.load(file_stream)["fits"]
dims = [2, 3, 6, 10]

fig = plt.figure(figsize=(text_width, 0.5*text_width))

gs = fig.add_gridspec(
    4, len(dims),
    height_ratios=[1, 1, 0.3, 1],
    hspace=0.3,
    wspace=0.3,
)

axes = []
for j, dim in enumerate(dims):
    axes.append([])
    # d = dim - 1
    # n = 500
    # average_kappa = 15
    # beta = 2.5*d
    # kappas = stats.expon.rvs(size=n, loc=1., scale=average_kappa)
    # key = jax.random.PRNGKey(0)
    # sphere_pos = sample_unit_vector_uniformly(dim, key=key, num_samples=n)

    # sphere_radius = (n/(2*np.pi**((d+1)/2)) * special.gamma((d+1)/2))**(1/d)
    # mu = beta*special.gamma(d/2)*np.sin(d*np.pi/beta) / (2*np.pi**(1+d/2)*average_kappa)
    # positions = sample_unit_vector_uniformly(dim, key, num_samples=n)

    fit = dmercator_fits[str(dim)]
    kappas = np.array(fit["kappa"])
    pos = np.array(fit["pos"])
    sphere_radius = fit["r_s"]
    mu = fit["mu"]
    beta = fit["beta"]
    t = time_coordinates(pos)
    sphere_pos = pos/t[:, None]
    n = len(kappas)

    iu, iv = np.triu_indices(n, k=1)
    def probs(kappa0):
        sd_probs = sd_connection_prob_pairs(
                pos1=sphere_pos[iu], pos2=sphere_pos[iv],
                kappa1=kappas[iu], kappa2=kappas[iv],
                beta=beta, mu=mu, sphere_radius=sphere_radius)
        hd_pos, hd_r = map_sd_to_hyperbolic_model(sphere_pos, kappas, mu, kappa0, sphere_radius)
        x_full = expand_positions(hd_pos)
        hd_probs = connection_prob_pairs(x_full[iu], x_full[iv], beta, hd_r, dist=hyperbolic_distance)
        return sd_probs, hd_probs

    # Row 1 -- Prob correlation with kappa0=kappa_min
    ax1 = fig.add_subplot(gs[0, j])
    axes[-1].append(ax1)
    # ax = axes[0, j]
    kappa_min = np.min(kappas)
    ax1.scatter(*probs(kappa_min), alpha=0.01, s=2, marker=".")
    ax1.set_title(f"$D={dim}$")
    ax1.tick_params(labelbottom=False)
    if j > 0:
        ax1.tick_params(labelleft=False)

    # Row 2 -- Prob correlation with kappa0=kappa_min
    # ax = axes[1, j]
    ax2 = fig.add_subplot(gs[1, j], sharex=ax1)
    axes[-1].append(ax2)
    kappa_opt = optimize_kappa0(sphere_pos, kappas, beta, mu, sphere_radius)
    ax2.scatter(*probs(kappa_opt), alpha=0.01, s=2, marker=".")
    ax2.set_xlabel(r"$p_{uv}^\mathbb{H}$")
    if j > 0:
        ax2.tick_params(labelleft=False)

    # Row 3 -- Plot L^2 error
    ax3 = fig.add_subplot(gs[3, j])
    axes[-1].append(ax3)
    # ax = axes[2, j]

    sd_probs = probs(1.)[0]
    @jax.jit
    def prob_l2_error(kappa0):
        pos, r = map_sd_to_hyperbolic_model(sphere_pos, kappas, mu, kappa0, sphere_radius)
        x_full = expand_positions(pos)
        hd_prob = connection_prob_pairs(x_full[iu], x_full[iv], beta, r, dist=hyperbolic_distance)
        return jnp.average(jnp.power(sd_probs-hd_prob, 2))

    kappa0s = np.linspace(1e-3, 20, 100)
    ax3.plot(kappa0s, [prob_l2_error(k0) for k0 in kappa0s])
    ax3.axvline(kappa_min, ls="--", color="k")
    ax3.axvline(kappa_opt, ls="--")
    ax3.set_xlabel(r"$\kappa_0$")
axes = np.asarray(axes).T

axes[0, 0].set_ylabel(r"$p_{uv}^\mathbb{S}$")
axes[1, 0].set_ylabel(r"$p_{uv}^\mathbb{S}$")
axes[2, 0].set_ylabel(r"Average $L^2$ error")
for ax in axes[2]:
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))


fig_dir = figure_dir()
label_subplots_letter(axes, xshift=-0.22, yshift=0.3)

fname = os.path.splitext(os.path.basename(dmercator_file))[0]
suffix = "-dmercator-fits"
if fname.endswith(suffix):
    fname = fname[:-len(suffix)]
fig.savefig(os.path.join(fig_dir, f"{fname}_prob_correlation.png"), bbox_inches="tight", dpi=400)
