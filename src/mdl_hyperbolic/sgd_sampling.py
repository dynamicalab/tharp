from typing import Callable

import jax
import jax.numpy as jnp

import rustworkx as rx

from mdl_hyperbolic.geometry import expand_positions
from mdl_hyperbolic.models import connection_prob_pairs


def estimate_gradient(key, pair_samplers: list[Callable], grad_fn: Callable, params):
    """
    Helper function that merges samples of `pair_samplers` and pass them to
    `grad_fn` along with `params`. The gradient of indices (vertices) that do
    not appear in the merged sampled are set to 0.

    Params:
        key: JAX PRNG key.
        pair_samplers:
            functions that given a JAX PRNG key, return (sample, weights, is_edge)
            where `sample` is sample of pairs, `weights` are their SGD weights
            and `is_edge` indicates if the sampled pairs are edges.
        grad_fn:
            function that given `params` and the output of a pair sampler,
            returns an unbiaised gradient estimation.
        params:
            model parameters.

    Returns:
        (loss, grads) estimated objective function value and gradient, respectively.
    """
    keys = jax.random.split(key, len(pair_samplers))

    outputs = [sampler(k) for sampler, k in zip(pair_samplers, keys)]
    all_samples = jnp.concatenate([r[0] for r in outputs])
    all_weights = jnp.concatenate([r[1] for r in outputs])
    is_edge = jnp.concatenate([r[2] for r in outputs])

    # Mask on vertices that appear in the log-likelihood estimation
    mask = jnp.zeros(params["pos"].shape[0], dtype=jnp.float32)
    mask = mask.at[all_samples.flatten()].set(1)[:, None]

    loss, grads = grad_fn(params, pairs=all_samples, weights=all_weights, is_edge=is_edge.astype(jnp.bool))
    grads["pos"] = grads["pos"] * mask

    return loss, grads


def estimate_pair_loglikelihood(
        pos: jnp.ndarray,
        beta,
        r,
        pairs: jnp.ndarray,
        weights: jnp.ndarray,
        is_edge: jnp.ndarray,
        dist,
        clip_threshold: float = 1e-4,
    ):
    """
    Estimates log-likelihood given a sample of k pairs, their weight and
    whether they are connected by an edge.

    Args:
        pos, beta, r, dist: model's parameters. See documentation of `connection_prob_pairs`.
        pairs: (k, 2) sample of pairs.
        weights:
            (k, 2) weight of each pair to obtain an unbiaised estimator.
            The weight is 1/p, where p is the marginalized probability of sampling
            this pair.
        is_edges: (k, 2) boolean array indicating whether each pair is an edge or not.
        clip_threshold: maximum distance to 0 and 1 of the edge probability.

    Returns:
        Log-likelihood estimation.
    """
    x_full = expand_positions(pos)

    probs = connection_prob_pairs(x_full[pairs[:, 0]], x_full[pairs[:, 1]], beta, r, dist=dist)
    clipped_probs = jnp.clip(probs, clip_threshold, 1. - clip_threshold)

    # Use jnp.where to avoid boolean indexing
    return jnp.average(jnp.where(
        is_edge,
        weights * jnp.log(clipped_probs),
        weights * jnp.log(1-clipped_probs)
    ))


def get_uniform_edge_sampler(edges, batch_size):
    """
    Creates a function that samples edges uniformly.

    Args:
        edges: (m, 2) edges of the graph.
        batch_size: number of edges to sample.

    Returns:
        edge sampling function.
    """
    n_edges = edges.shape[0]
    if n_edges == 0:
        raise ValueError("Cannot sample edges from empty graph.")

    @jax.jit
    def sample_edges(key):
        idx = jax.random.choice(key, n_edges, (batch_size,))
        return edges[idx], jnp.full(batch_size, n_edges), jnp.ones(batch_size)
    return sample_edges


def get_uniform_non_edge_sampler(n, sorted_edges, batch_size):
    """
    Creates a function that samples non-edges uniformly.

    Args:
        n: graph size.
        sorted_edges: (m, 2) edges of the graph, sorted by column.
        batch_size: number of non-edges to sample.

    Returns:
        non-edge sampling function.
    """
    edge_hash = sorted_edges[:, 0] * n + sorted_edges[:, 1]
    n_non_edges = int(.5*n*(n-1)) - sorted_edges.shape[0]
    if n_non_edges == 0:
        raise ValueError("Cannot sample non-edges from complete graph.")

    def sample_non_edges(key):
        collected = []
        while len(collected) < batch_size:
            key, subkey1, subkey2 = jax.random.split(key, 3)
            i = jax.random.randint(subkey1, shape=(batch_size,), minval=0, maxval=n)
            j = jax.random.randint(subkey2, shape=(batch_size,), minval=0, maxval=n)
            pair_sample = jnp.sort(jnp.stack([i, j], axis=1), axis=1)
            pairs_hash = pair_sample[:, 0] * n + pair_sample[:, 1]
            is_non_edge = (pair_sample[:, 0] != pair_sample[:, 1]) & \
                          ~(pairs_hash[:, None] == edge_hash[None, :]).any(axis=1)
            collected.extend(pair_sample[is_non_edge].tolist())
        nonedge_sample = jnp.array(collected[:batch_size])
        return nonedge_sample, jnp.full(batch_size, n_non_edges), jnp.zeros(batch_size)
    return sample_non_edges


def get_uniform_pair_sampler(n, sorted_edges, batch_size):
    """
    Creates a function that samples pairs uniformly.

    Args:
        sorted_edges: (m, 2) edges of the graph, sorted by column.
        batch_size: number of pairs to sample.

    Returns:
        pair sampling function.
    """
    edge_hash = sorted_edges[:, 0] * n + sorted_edges[:, 1]
    n_total = int(.5 * n * (n - 1))

    @jax.jit
    def sample_pairs(key: jax.Array):
        # Sample a flat index in [0, n_total)
        flat_idx = jax.random.randint(key, shape=(batch_size,), minval=0, maxval=n_total)

        # Convert flat index to (i, j) with i < j
        # Row i contains (n-1) + (n-2) + ... + (n-i) pairs; invert this via quadratic formula
        i = jnp.floor(
            (2 * n - 1 - jnp.sqrt((2 * n - 1) ** 2 - 8 * flat_idx)) / 2
        ).astype(jnp.int32)
        j = flat_idx - i * (2 * n - i - 1) // 2 + i + 1

        pairs = jnp.stack([i, j], axis=1)
        pairs_hash = pairs[:, 0] * n + pairs[:, 1]
        is_edge = (pairs_hash[:, None] == edge_hash[None, :]).any(axis=1)
        weights = jnp.full(batch_size, n_total)
        return pairs, weights, is_edge
    return sample_pairs


def get_neighborhood_bernoulli_sampler(graph: rx.PyGraph, n_sampled_vertices: int, n_sampled_neighbors: int, edge_prob: float):
    """
    Creates a function that sample pairs in two steps, based on the neighborhood.

    Sampling is performed in two three steps:
        1. Sample a vertex `u` uniformly.
        2. Choose to sample edges with probability `edge_prob` and non-edges otherwise.
        3a. If sampling edges, sample `n_sampled_neighbors` neighbors of `u`.
        3b. If sampling non-edges, sample uniformly `n_sampled_neighbors`
            vertices different from `u`. Note that this may yield neighbors of
            `u`, but only rarely for sparse graphs. The return value indicates
            whether each sampled pair is an edge.
    This sampling is performed `n_sampled_vertices` times, resulting in a pair sample of size
    `n_sampled_vertices`*`n_sampled_neighbors`.

    Args:
        graph: graph to sample from.
        n_sampled_vertices: number of vertices to sample uniformly.
        n_sampled_neighbors: number of vertices to sample conditioned on the first one.
        edge_prob: probability of sampling edges explicitely.

    Returns:
        function that samples both edges and non-edges.
    """
    n = graph.num_nodes()
    degrees = jnp.array([graph.degree(u) for u in range(n)], dtype=jnp.int32)

    offsets = jnp.array([0] + list(jnp.cumsum(degrees)), dtype=jnp.int64)  # (n+1,)
    flat_neighbors = jnp.concatenate([
        jnp.asarray(graph.neighbors(u), dtype=jnp.int32) for u in range(n)
    ])

    # Build edge hash set: store sorted_u * n + sorted_v for every edge
    edge_hashes = set()
    for u in range(n):
        for v in graph.neighbors(u):
            if u < v:
                edge_hashes.add(u * n + v)
    edge_hashes = jnp.array(sorted(edge_hashes), dtype=jnp.int64)  # (E,)

    @jax.jit
    def sample(key: jax.Array):
        vertex_key, neighbor_key = jax.random.split(key)

        vertices = jax.random.randint(
            vertex_key, shape=(n_sampled_vertices,), minval=0, maxval=n
        )
        per_vertex_keys = jax.random.split(neighbor_key, n_sampled_vertices)

        def sample_one(k, u):
            k1, k2, k3 = jax.random.split(k, 3)
            start = offsets[u]
            neighbor_indices = jax.random.randint(
                k1, shape=(n_sampled_neighbors,), minval=0, maxval=degrees[u]
            )
            neighbor_vs = flat_neighbors[start + neighbor_indices]

            random_vs_raw = jax.random.randint(
                k2, shape=(n_sampled_neighbors,), minval=0, maxval=n - 1
            )
            random_vs = jnp.where(random_vs_raw >= u, random_vs_raw + 1, random_vs_raw)

            a = jax.random.bernoulli(k3, p=edge_prob, shape=(n_sampled_neighbors,))
            vs = jnp.where(a, neighbor_vs, random_vs)

            # Sort each (u, v) pair and compute hash
            u_broadcast = jnp.full((n_sampled_neighbors,), u, dtype=jnp.int32)
            sorted_lo = jnp.minimum(u_broadcast, vs)
            sorted_hi = jnp.maximum(u_broadcast, vs)
            hashes = (
                sorted_lo.astype(jnp.int64) * jnp.int64(n)
                + sorted_hi.astype(jnp.int64)
            )

            # Edge indicator: neighbor branch is always an edge;
            # random branch checks hash membership via sorted binary search
            random_are_edges = jnp.isin(hashes, edge_hashes, method="binary_search")
            is_edge = jnp.where(a, jnp.ones(n_sampled_neighbors, dtype=jnp.bool_), random_are_edges)

            pairs = jnp.stack([u_broadcast, vs], axis=1)
            return pairs, is_edge

        pairs, is_edges = jax.vmap(sample_one)(per_vertex_keys, vertices)
        pairs = pairs.reshape(-1, 2)
        is_edges = is_edges.reshape(-1)

        k1 = degrees[pairs[:, 0]].astype(jnp.float32)
        k2 = degrees[pairs[:, 1]].astype(jnp.float32)

        edge_weights = n / (edge_prob * (1.0 / k1 + 1.0 / k2) + 2.0 * (1.0 - edge_prob) / (n - 1))
        non_edge_weights = jnp.full(pairs.shape[0], (1.0 / (1.0 - edge_prob)) * n * (n - 1) / 2.0)
        weights = jnp.where(is_edges, edge_weights, non_edge_weights)

        return pairs, weights, is_edges

    return sample
