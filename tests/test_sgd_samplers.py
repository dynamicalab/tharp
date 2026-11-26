import numpy as np
import rustworkx as rx
import jax.numpy as jnp
import jax
import pytest

from mdl_hyperbolic.geometry import hyperbolic_distance
from mdl_hyperbolic.sgd_sampling import *
from mdl_hyperbolic.models import connection_prob_pairs, hyperbolic_loglikelihood


jax.config.update('jax_enable_x64', True) # enable double precision floating-point

graph = rx.PyGraph()
graph.add_nodes_from(range(5))
graph.add_edges_from_no_data([(0, 1), (0, 2), (0, 3), (1, 2), (2, 3), (3, 4)])
edges = jnp.array(graph.edge_list())
edges = jnp.sort(edges)  # Required for edge hashing in samplers
non_edges = jnp.array(rx.graph_complement(graph).edge_list())
n = graph.num_nodes()
num_edges = edges.shape[0]
num_non_edges = int(n*(n-1)/2) - num_edges

seed_key = jax.random.PRNGKey(0)
keys = iter(jax.random.split(seed_key, 100))


def exact_edge_contribution(pos, beta, r):
    x = expand_positions(pos)
    xi = x[edges[:, 0]]
    xj = x[edges[:, 1]]
    p_edges = connection_prob_pairs(xi, xj, beta, r, dist=hyperbolic_distance)
    p_edges = jnp.clip(p_edges, 1e-4, 1 - 1e-4)
    return -jnp.sum(jnp.log(p_edges))


def exact_non_edge_contribution(pos, beta, r):
    x = expand_positions(pos)
    xi = x[non_edges[:, 0]]
    xj = x[non_edges[:, 1]]
    p_non_edges = connection_prob_pairs(xi, xj, beta, r, dist=hyperbolic_distance)
    p_non_edges = jnp.clip(p_non_edges, 1e-4, 1 - 1e-4)
    return -jnp.sum(jnp.log1p(-p_non_edges))


def exact_objective(pos, beta, r):
    return exact_edge_contribution(pos, beta, r) + exact_non_edge_contribution(pos, beta, r)


def approx_objective(pos, beta, r, pairs, weights, is_edge):
    return -estimate_pair_loglikelihood(pos, beta, r, pairs, weights, is_edge, dist=hyperbolic_distance)


dim = 4
params = {
    "pos": jnp.sinh(3.)*jax.random.uniform(next(keys), shape=(n, dim)),
    "beta": (dim-1)*2.5,
    "r": 4.,
}


def test_objective_function_valid():
    assert pytest.approx(-hyperbolic_loglikelihood(edges=edges,
                         **params)) == exact_objective(**params)


def test_uniform_edge_sampler():
    sampler = get_uniform_edge_sampler(edges, batch_size=10000)
    pairs, weights, is_edge = sampler(next(keys))

    assert np.all([graph.has_edge(*pair) for pair in pairs])
    assert jnp.all(is_edge)

    exact = exact_edge_contribution(**params)
    approx = approx_objective(**params, pairs=pairs,
                              weights=weights, is_edge=is_edge)
    assert pytest.approx(exact, rel=0.02) == approx


def test_uniform_non_edge_sampler():
    sampler = get_uniform_non_edge_sampler(n, edges, batch_size=10000)
    pairs, weights, is_edge = sampler(next(keys))

    assert not np.any([graph.has_edge(*pair) for pair in pairs])
    assert not jnp.any(is_edge)

    exact = exact_non_edge_contribution(**params)
    approx = approx_objective(**params, pairs=pairs,
                              weights=weights, is_edge=is_edge)
    assert pytest.approx(exact, rel=0.02) == approx


def test_uniform_pair_sampler():
    sampler = get_uniform_pair_sampler(n, edges, batch_size=10000)
    pairs, weights, is_edge = sampler(next(keys))

    assert np.all(np.array([graph.has_edge(*pair)
                  for pair in pairs]) == is_edge)

    exact = exact_objective(**params)
    approx = approx_objective(**params, pairs=pairs,
                              weights=weights, is_edge=is_edge)
    assert pytest.approx(exact, rel=0.05) == approx


def test_neighbor_sampler():
    sampler = get_neighborhood_sampler(
        graph, n_sampled_vertices=200, n_sampled_neighbors=200)
    pairs, weights, is_edge = sampler(next(keys))

    assert np.all(np.array([graph.has_edge(*pair) for pair in pairs]))
    assert np.all(is_edge)

    exact = exact_edge_contribution(**params)
    approx = approx_objective(**params, pairs=pairs,
                              weights=weights, is_edge=is_edge)
    assert pytest.approx(exact, rel=0.05) == approx


def test_bernoulli_sampler():
    sampler = get_neighborhood_bernoulli_sampler(
        graph, n_sampled_vertices=200, n_sampled_neighbors=200, edge_prob=0.1)
    pairs, weights, is_edge = sampler(next(keys))

    assert np.all(np.array([graph.has_edge(*pair)
                  for pair in pairs]) == is_edge)

    exact = exact_objective(**params)
    approx = approx_objective(**params, pairs=pairs,
                              weights=weights, is_edge=is_edge)
    assert pytest.approx(exact, rel=0.05) == approx
