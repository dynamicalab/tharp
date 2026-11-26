import os

import jax
from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np
import rustworkx as rx

from mdl_hyperbolic.models import hyperbolic_loglikelihood
from mdl_hyperbolic.generators import sample_quasiuniform_hyperbolic_ball

from plot_tools import figure_dir


def rot_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


jax.config.update('jax_enable_x64', True) # enable double precision floating-point

n = 50
r = 5.0
original_beta = 3.0
pos = sample_quasiuniform_hyperbolic_ball(num_dim=2, num_samples=n, radius=r, key=jax.random.PRNGKey(0))
g = rx.hyperbolic_random_graph(pos, r, original_beta, seed=42)
edges = jax.numpy.asarray(list(g.edge_list()))

vertex = 0
angles = np.linspace(0, 2*np.pi, 100)


fig, ax = plt.subplots(figsize=(6, 4))
line, = ax.plot(angles, np.zeros_like(angles))

num_frames = 100
betas = np.linspace(0, 1, num_frames)
ax.set_ylim((-20, 20))
ax.set_xlabel("Angle")
ax.set_ylabel("log-likelihood - theta-averaged log-likelihood")

def animate(i):
    beta = betas[i]
    likelihoods = []
    for i, theta in enumerate(angles):
        rot_pos = np.copy(pos)
        rot_pos[vertex] = rot_matrix(theta) @ pos[vertex]
        likelihoods.append(hyperbolic_loglikelihood(edges=edges, pos=rot_pos, beta=beta, r=r))
    average = np.average(likelihoods)
    line.set_ydata(np.array(likelihoods)-average)
    ax.set_title(f"$\\beta={beta:.2f}$")
    return line,


anim = animation.FuncAnimation(fig, animate,
                               frames=num_frames, interval=10, blit=False)

anim.save(os.path.join(figure_dir(), "likelihood_beta.gif"))
