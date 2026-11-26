import pytest

import jax
import jax.numpy as jnp
from mdl_hyperbolic.geometry import *


jax.config.update('jax_enable_x64', True) # enable double precision floating-point


# Tangent vectors must have null dot product with the corresponding point.
def assert_tangent(x, v):
    for value in lorentz_dot(x, v):
        assert value == pytest.approx(0., abs=1e-6)

def assert_on_hyperboloid(x):
    ts = time_coordinates(x[:, :-1])
    for t, x_t in zip(x[:, -1], ts):
        assert t == pytest.approx(x_t, abs=1e-6)


def test_time_coordinates():
    pos = jnp.array([[3., 5.], [10., 4.]])
    time = time_coordinates(pos)
    assert jnp.all(time == jnp.array([jnp.sqrt(35), jnp.sqrt(117)]))


def test_lorentz_dot():
    x = [3., 4., 5.]
    y = [0., 5., 10.]
    p1 = jnp.array([x, x])
    p2 = jnp.array([x, y])
    res = lorentz_dot(p1, p2)
    assert res[0] == 0.
    assert res[1] == -30


def test_hyperbolic_distance():
    x = [0., 0.]
    y = [5., 0.]
    p1 = expand_positions(jnp.array([x, y]))
    p2 = expand_positions(jnp.array([y, y]))
    res = hyperbolic_distance(p1, p2)
    assert pytest.approx(res[0]) == jnp.asinh(5)
    assert pytest.approx(res[1]) == 0.

    res = hyperbolic_distance_stable(p1, p2)
    assert pytest.approx(res[0], abs=1e-3) == jnp.asinh(5)
    assert pytest.approx(res[1], abs=1e-3) == 0.
    # The stable distance introduces a correction to prevent undefined gradients.
    # See documentation of the function.

    assert 0. <= res[1] < 0.001


def test_hyperbolic_distance_matrix():
    ps = expand_positions(jnp.array([[0., 0.], [5., 0.]]))
    res = hyperbolic_distance_matrix(ps)
    expected = jnp.array([[0., jnp.asinh(5)], [jnp.asinh(5), 0.]])
    assert jnp.all(pytest.approx(res) == expected)


def test_project_tangent():
    xs = expand_positions(jnp.array([[-10, 10.], [5., 5.]]))
    vs = jnp.array([[3., 4., 105.], [4., -5., -100.]])
    vs = project_tangent(xs, vs)
    assert_tangent(xs, vs)


def test_exp_map():
    ps = expand_positions(jnp.array([[0., 0.], [5., 10.]]))
    vs = project_tangent(ps, jnp.array([[1., 0., 0.], [-1., 1., 0.]]))
    res = exp_map(ps, vs)
    assert_on_hyperboloid(res)
    assert res[0, 1] == 0.
    assert res[0, 0] > 0.


def test_log_map():
    xs = expand_positions(jnp.array([[0., 0.], [5., 10.]]))
    ys = expand_positions(jnp.array([[1., 0.], [-3., 12.]]))
    vs = log_map(xs, ys)
    assert_tangent(xs, vs)
    assert vs[0, 0] > 0.
    assert vs[0, 1] == 0.


def test_log_exp_identity():
    xs = expand_positions(jnp.array([[0., 0.], [5., 10.]]))
    ys = expand_positions(jnp.array([[1., 0.], [-3., 12.]]))
    vs = project_tangent(xs, jnp.array([[1., 0., 0.], [-1., 1., 0.]]))

    res = exp_map(xs, log_map(xs, ys))
    assert jnp.allclose(res[:, :-1], ys[:, :-1], atol=0.005)

    res = log_map(xs, exp_map(xs, vs))
    assert jnp.allclose(res[:, :-1], vs[:, :-1], atol=0.005)


def test_parallel_transport():
    xs = expand_positions(jnp.array([[0., 0.], [5., 10.]]))
    ys = expand_positions(jnp.array([[1., 0.], [-3., 12.]]))
    vs = project_tangent(xs, jnp.array([[1., 0., 0.], [-1., 1., 0.]]))

    forward = parallel_transport(xs, ys, vs)
    assert_tangent(ys, forward)
    backwards = parallel_transport(ys, xs, forward)
    assert_tangent(xs, backwards)
    assert jnp.allclose(vs, backwards, atol=1e-5)


def test_translation():
    xs = jnp.array([[0., 0.], [-1., 0.5], [2., 10.]])
    v = jnp.array([1./jnp.sqrt(2), 1./jnp.sqrt(2)])
    res = hyperbolic_translate(xs, v, 2.5)

    distance_xs = hyperbolic_distance_matrix(expand_positions(xs))
    distance_res = hyperbolic_distance_matrix(expand_positions(res))

    assert jnp.allclose(distance_xs, distance_res, atol=1e-5)
    assert res[0, 0] == res[0, 1]
    assert res[0, 0] > 0.
    assert res[1, 0] > 0.
    assert res[1, 1] > 0.


def test_change_of_coordinates_identity():
    key = jax.random.PRNGKey(0)

    for dim in [2, 5, 10, 100]:
        x = jax.random.uniform(key, shape=(10, dim))
        diff = x - geodesic_to_hyperboloid(*hyperboloid_to_geodesic(x))
        assert jnp.max(jnp.abs(diff)) < 1e-10
