from functools import partial
import numpy as np
from mdl_hyperbolic.models import *
from mdl_hyperbolic.geometry import expand_positions, sd_hyperbolic_distance
import pytest

import jax


jax.config.update('jax_enable_x64', True) # enable double precision floating-point

pos = expand_positions(jnp.array([[0., 0.], [5., 0.], [-3., -2.]]))
dist = jnp.array([
    [0., np.asinh(5), np.asinh(np.sqrt(13))],
    [np.asinh(5), 0., np.acosh(np.sqrt(26*14)+15)],
    [np.asinh(np.sqrt(13)), np.acosh(np.sqrt(26*14)+15), 0.]
])
beta = 2.
r = 1.
probs = 1/(1+np.exp(0.5*beta*(dist - r)))


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def test_connection_prob():
    res = connection_prob_pairs(jnp.array([pos[0], pos[0], pos[0]]), pos, beta, r, dist=hyperbolic_distance)
    assert pytest.approx(probs[0, 0]) == res[0]
    assert pytest.approx(probs[0, 1]) == res[1]


def test_hyperbolic_loglikelihood():
    edges = jnp.array([[0, 1], [1, 2]])
    expected = np.log(probs[0, 1]) + np.log(probs[1, 2]) + np.log(1-probs[0, 2])
    res = hyperbolic_loglikelihood(pos=pos, beta=beta, r=r, edges=edges)
    assert pytest.approx(expected) == res


def test_discretization():
    quantized = discretize({"x": 1.12345, "y": 1235.4}, {"x": 0.001, "y": 10.})
    assert pytest.approx(quantized["x"], abs=1e-14) == 1.124
    assert pytest.approx(quantized["y"], abs=1e-14) == 1240


def test_sd_distance():
    p1 = np.array([1., 0., 0.,])
    p2 = np.array([np.sqrt(2)/2, np.sqrt(2)/2., 0.,])
    rotation = rotation_matrix([1., 0.4, -0.7], 1.4)

    assert pytest.approx(sd_dist(p1[None, :], p2[None, :])) == np.pi/4
    assert pytest.approx(sd_dist((rotation@p1)[None, :], (rotation@p2)[None, :])) == np.pi/4


def test_sd_connection_prob():
    p1 = np.array([1., 0., 0.,])
    p2 = np.array([np.sqrt(2)/2, np.sqrt(2)/2., 0.,])
    p3 = np.array([.5, 0., np.sqrt(3)/2])
    d = len(p1)-1
    kappas = np.array([10., 2.2, 0.4])
    mu = 1.5
    sphere_radius = 5.8

    res = sd_connection_prob_pairs(
            pos1=np.array([p1, p1]), pos2=np.array([p2, p3]),
            kappa1=np.full(2, kappas[0]), kappa2=kappas[1:],
            beta=beta, mu=mu, sphere_radius=sphere_radius
        )
    assert pytest.approx(1/(1+(sphere_radius*(np.pi/4)/(mu*kappas[0]*kappas[1])**(1/d))**beta)) == res[0]
    assert pytest.approx(1/(1+(sphere_radius*(np.pi/3)/(mu*kappas[0]*kappas[2])**(1/d))**beta)) == res[1]


def test_sd_loglikelihood():
    ps = np.array([
        [1., 0., 0.,],
        [.5, 0., np.sqrt(3)/2],
        [np.sqrt(2)/2, np.sqrt(2)/2., 0.]
    ])
    d = ps.shape[1]-1
    kappas = np.array([10., 2.2, 0.4])
    mu = 1.5
    sphere_radius = 5.8
    edges = jnp.array([[0, 1], [1, 2]])

    u = np.array([0, 0, 1])
    v = np.array([1, 2, 2])
    probs = sd_connection_prob_pairs(
            pos1=ps[u], pos2=ps[v],
            kappa1=kappas[u], kappa2=kappas[v],
            beta=beta, mu=mu, sphere_radius=sphere_radius
        )
    res = sd_loglikelihood(
            edges=edges,
            sphere_pos=ps, kappas=kappas,
            beta=beta, mu=mu, sphere_radius=sphere_radius
        )
    expected = np.log(probs[0]) + np.log(probs[2]) + np.log(1-probs[1])
    assert pytest.approx(expected) == res


def test_hd_sd_mapping_preserves_likelihood():
    approx_hd_loglikelihood = partial(hyperbolic_loglikelihood, dist=sd_hyperbolic_distance)

    edges = jnp.array([[0, 1], [1, 2]])
    expected = approx_hd_loglikelihood(edges, pos, beta, r)

    sphere_pos, kappas, sphere_radius, mu, kappa0 = map_hyperbolic_model_to_sd(pos, r)
    sd_likelihood = sd_loglikelihood(edges, sphere_pos, kappas, beta, sphere_radius, mu)
    assert pytest.approx(expected) == sd_likelihood

    new_pos, new_r = map_sd_to_hyperbolic_model(sphere_pos, kappas, mu, kappa0, sphere_radius)
    new_hd_likelihood = approx_hd_loglikelihood(edges, new_pos, beta, new_r)
    assert pytest.approx(expected) == new_hd_likelihood
