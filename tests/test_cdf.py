"""
Verifies that binned cdf over the domain sums to 1.
"""
import pytest
import numpy as np

import jax
from mdl_hyperbolic.priors import *


jax.config.update('jax_enable_x64', True) # enable double precision floating-point


def hypercube_grid(bounds, num_points, dim):
    """
    Construct a regular grid over a hypercube (or hyper-rectangle).
    """
    axes = [np.linspace(*bounds, num_points)] * dim
    bin_size = axes[0][1] - axes[0][0]
    grid = map(lambda x: x.ravel(), np.meshgrid(*axes, indexing="ij"))
    return np.stack(tuple(grid)).T, bin_size


def test_cauchy_prior():
    xmax = 700
    q = 5e-1

    sum_on_range_approx = sum(np.exp(cauchy_logprior(np.arange(-xmax, xmax+q, q), dispersion=0.1, quantization=q)))
    assert pytest.approx(sum_on_range_approx, 1e-4) == 1.


def test_normal_prior():
    xmax = 100
    q = 5e-1
    mean = 2

    sum_on_range_approx = sum(np.exp(normal_logprior(mean+np.arange(-xmax, xmax+q, q), mean=mean, std=0.3, quantization=q)))
    assert pytest.approx(sum_on_range_approx, 1e-4) == 1.


def test_trunc_normal_prior():
    x0 = 3
    xmax = 100
    q = 5e-1
    mean = 10

    sum_on_range_approx = sum(np.exp(trunc_normal_logprior(np.arange(x0, xmax+q, q), center=mean, sigma=0.3, lower_bound=x0, quantization=q)))
    assert pytest.approx(sum_on_range_approx, 1e-4) == 1.


def test_extended_uniform_hyperboloid_mc():
    rho = 0.5
    r_max = 5*rho
    xmax = np.sinh(r_max)
    n_points = 20
    nu = 4.
    for dim in [2, 3]:
        points, q = hypercube_grid((-xmax, xmax), n_points, dim)
        sum_on_range_approx = np.exp(extended_uniform_logprior_mc(points, rho=rho, nu=nu, quantization=q, n_samples=500, key=jax.random.PRNGKey(10)))
        assert pytest.approx(jnp.sum(sum_on_range_approx), 1e-2) == 1.
