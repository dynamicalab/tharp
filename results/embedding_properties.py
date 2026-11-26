import subprocess
from pathlib import Path
import os
import tempfile
from functools import partial

import jax
import numpy as np
import jax.numpy as jnp
import rustworkx as rx

from mdl_hyperbolic.models import connection_prob_pairs, sd_connection_prob_pairs
from mdl_hyperbolic.geometry import expand_positions, hyperbolic_distance


exec_path = Path(__file__).parent.joinpath("chordless_cycles/cyclesmap")
def count_chordless_cycles(edges: np.ndarray) -> tuple[float, float, float]:
    """
    Runs cyclesmap on an edge list and returns the three cycle coefficients.

    Parameters
    ----------
    edges : np.ndarray of shape (M, 2)
        Edge list with vertices indexed from 0 to n-1.

    Returns
    -------
    (clustering, square_coeff, pentagon_coeff) : tuple of float
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f1, \
         tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f2:
        tmp1 = f1.name
        tmp2 = f2.name
        # Write edges re-indexed from 1
        np.savetxt(tmp1, edges+1, fmt="%d")
    try:
        subprocess.run([exec_path, tmp1, tmp2], check=True)
        with open(tmp2, 'r') as f:
            line = f.read().strip()
        values = [float(x.strip()) for x in line.split(',')]
        clustering, square_coeff, pentagon_coeff = values[0], values[1], values[2]
        return clustering, square_coeff, pentagon_coeff
    finally:
        os.unlink(tmp1)
        os.unlink(tmp2)


def degrees_ccdf(graph: rx.PyGraph):
    """
    Empirical complementary cumulative degree distribution of the given graph.
    """
    degrees = [graph.degree(u) for u in graph.node_indices()]
    degrees = np.sort(degrees)
    x_values = np.unique(degrees)
    cdf_values = np.searchsorted(degrees, x_values, side='right') / graph.num_nodes()
    return x_values, 1.-cdf_values


def sd_expected_degree(params):
    """
    Expected degree of vertices computed with the S^{D-1} random graph model.
    """
    n = params["pos"].shape[0]
    sphere_pos = params["pos"]/np.linalg.norm(params["pos"], axis=1)[:, None]
    iu, iv = np.indices((n, n))
    iu, iv = iu.flatten(), iv.flatten()
    probs = sd_connection_prob_pairs(
        pos1=sphere_pos[iu], pos2=sphere_pos[iv],
        kappa1=params["kappa"][iu], kappa2=params["kappa"][iv],
        beta=params["beta"], sphere_radius=params["r_s"], mu=params["mu"])
    probs = jnp.fill_diagonal(probs.reshape((n, n)), 0, inplace=False)
    return jnp.sum(probs, axis=0)


def connection_prob_matrix(pos, beta, r, dist):
    n = pos.shape[0]
    iu, iv = np.indices((n, n))
    return connection_prob_pairs(pos[iu], pos[iv], beta, r, dist=dist).reshape(n, n)


def expected_degrees(params):
    """
    Expected degree of vertices computed with the H^D random graph model.
    """
    probs = connection_prob_matrix(expand_positions(jnp.array(params["pos"])), params["beta"], params["r"], dist=hyperbolic_distance)
    probs = jnp.fill_diagonal(probs, 0, inplace=False)
    return jnp.sum(probs, axis=0)


def expected_degrees_ccdf(params):
    """
    Empirical complementary cumulative of degree distribution found for the expected degrees.
    """
    degs = expected_degrees(params)
    data_sorted = np.sort(degs)
    p = 1. - 1. * np.arange(len(degs)) / (len(degs) - 1)
    return data_sorted, p


@partial(jax.jit, static_argnames=["proportion"])
def community_concentration(vertex, distances, communities, proportion=0.1):
    n = distances.shape[0]
    k = int(np.ceil(n * proportion))
    d = distances[vertex].at[vertex].set(jnp.inf)
    top_closests = jnp.argsort(d)[:k]
    same_community = jnp.sum(communities[top_closests] == communities[vertex])
    community_size = jnp.sum(communities == communities[vertex])
    return same_community / community_size / proportion

@partial(jax.jit, static_argnames=["proportion"])
def average_community_concentration(distances, communities, proportion=0.1):
    n = distances.shape[0]
    return jnp.mean(jax.vmap(
        lambda v: community_concentration(v, distances, communities, proportion)
    )(jnp.arange(n)))
