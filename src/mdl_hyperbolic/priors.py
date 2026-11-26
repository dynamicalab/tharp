from jax.scipy.stats import norm
from jax.scipy import special
import jax.numpy as jnp
import jax
import numpy as np

from .geometry import time_coordinates, eps


def logdiffexp(a, b):
    """log(exp(a) - exp(b)), assuming a >= b."""
    return a + jnp.log1p(-jnp.exp(b - a))


def trunc_normal_logprior(x, center, sigma, lower_bound, quantization):
    """
    Compute the logarithm of the probability $\\ln P[X \\in [x-quantization, x)]$,
    assuming a normal distribution of mean `center` and standard deviation
    `sigma`, truncated below at `lower_bound`. `center` and `sigma` refer to
    the underlying normal's statistics and not the truncated normal's.

    Args:
        x: interval upper bound.
        center: mean of the underyling normal.
        sigma: standard deviation of the underlying normal.
        lower_bound: lower bound of the truncated normal.
        quantization: interval length.

    Returns:
        Log-probability.
    """
    q = quantization

    alpha = (lower_bound - center) / sigma
    z = (x - center) / sigma
    z_prev = z - q / sigma

    log_Z = special.log_ndtr(-alpha)

    # Small-bin approximation
    log_p_approx = (
        norm.logpdf(z)
        + jnp.log(q)
        - jnp.log(sigma)
        - log_Z
    )

    # Exact bin probability
    log_p_exact = (
        logdiffexp(
            special.log_ndtr(z),
            special.log_ndtr(z_prev),
        )
        - log_Z
    )
    use_approx = q / sigma < 1e-3
    return jnp.where(use_approx, log_p_approx, log_p_exact)


def normal_logprior(x, mean, std, quantization):
    """
    Computes the logarithm of the probability that x is in [x-quantization, x)
    if the random variable is normally distributed with mean `mean` and
    standard deviation `std`.

    Args:
        x: interval upper bound.
        mean: mean of the normal.
        std: standard deviation of the normal.
        quantization: interval length.

    Returns:
        Log-probability.
    """
    return jnp.log(
        norm.cdf(x, loc=mean, scale=std)
        - norm.cdf(x - quantization, loc=mean, scale=std)
    )


def cauchy_logprior(x, dispersion, quantization):
    """
    Computes the logarithm of the probability that x is in [x-quantization, x)
    if the random variable has a Cauchy distribution of dispersion parameter
    `dispersion`.

    Args:
        x: interval upper bound.
        dispersion: dispersion of the Cauchy.
        quantization: interval length.

    Returns:
        Log-probability.
    """
    d = dispersion
    return -jnp.log(jnp.pi) + jnp.log(jnp.arctan(d*quantization / (d*d + x*(x-quantization))))


def extended_uniform_unnormalized_logdensity(x, rho, nu):
    """
    Computes the unnormalized log-density of the extended uniform distribution
    over a hyperbolic ball with decaying density outside.

    Args:
        x: hyperboloid coordinates without time.
        rho: radius of the hyperbolic ball.
        nu: sharpness of the density decay outside the hyperbolic ball.

    Returns:
        Log-density.
    """
    time = time_coordinates(x)
    radii = jnp.acosh(jnp.clip(time, min=1+eps))
    dim = x.shape[1]
    return -jax.nn.softplus(nu*(dim-1)*(radii-rho)) - jnp.log(time)


def extended_uniform_logdensity(x, rho, nu):
    """
    Computes the density of the extended uniform distribution over a hyperbolic
    ball with decaying density outside.

    Args:
        x: hyperboloid coordinates without time.
        rho: radius of the hyperbolic ball.
        nu: sharpness of the density decay outside the hyperbolic ball.

    Returns:
        Log-density.
    """
    return extended_uniform_unnormalized_logdensity(x=x, rho=rho, nu=nu)\
            - extended_uniform_lognormalization(dim=x.shape[1], rho=rho, nu=nu)


def extended_uniform_logprior_mc(x, rho, nu, quantization, key, n_samples=500):
    """
    Vectorized version of `extended_uniform_logprior_mc_single`. Computes the
    probability that each coordinate is in a hypercube of length q in the
    extended uniform distribution. The integral is estimated with Monte Carlo.

    Args:
        x: (n, D) hyperboloid coordinates without time.
        nu: sharpness of the density decay outside the hyperbolic ball.
        rho: radius of the hyperbolic ball.
        quantization: width of the quantization bin.
        key: jax random key to generate the Monte Carlo sample.

    Returns:
        Log-prior estimation.
    """
    keys = jax.random.split(key, x.shape[0])
    batched = jax.vmap(extended_uniform_logprior_mc_single, in_axes=(0, None, None, None, 0, None))
    return batched(x, rho, nu, quantization, keys, n_samples)


def extended_uniform_logprior_mc_single(x, rho, nu, quantization, key, n_samples):
    """
    Computes the probability that one coordinate is in a hypercube of length q
    in the extended uniform distribution. Estimation is done with Monte Carlo.

    Args:
        x: hyperboloid coordinates without time.
        rho: radius of the hyperbolic ball.
        nu: sharpness of the density decay outside the hyperbolic ball.
        quantization: width of the quantization bin.
        key: jax random key to generate the Monte Carlo sample.

    Returns:
        Log-prior estimation.
    """
    dim = x.shape[0]
    lower = x - quantization
    u = jax.random.uniform(key, (n_samples, dim))
    samples = lower + u * quantization
    log_vals = extended_uniform_unnormalized_logdensity(samples, nu=nu, rho=rho)
    log_volume = dim*jnp.log(quantization)
    log_norm = extended_uniform_lognormalization(dim=dim, rho=rho, nu=nu)

    return log_volume + jax.scipy.special.logsumexp(log_vals) - jnp.log(n_samples) - log_norm


N_QUAD = 128
# Gauss-Legendre nodes on [-1, 1]
x, w = np.polynomial.legendre.leggauss(N_QUAD)


def log_I_r(dim, rho, nu):
    """
    Log-normalization of the extended uniform's radial coordinate marginal
    sinh^(D-1)(r)/(1+exp(nu(D-1)(r-rho))).

    Args:
        dim: dimension of the hyperbolic space.
        rho: hyperbolic ball radius.
        nu: decay of the density outside the ball.

    Returns:
        Log-normalization.
    """
    m = dim - 1

    # Left: [0, rho]
    r = 0.5 * rho * (x + 1.0)
    wr = 0.5 * rho * w

    log_sinh = r - jnp.log(2.0) + jnp.log1p(-jnp.exp(-2.0 * r))
    log_f = (m * log_sinh - jax.nn.softplus(nu * m * (r - rho)))

    I_left = jnp.exp(jax.scipy.special.logsumexp(jnp.log(wr) + log_f))

    # Right: [rho, infinity]
    z = 0.5 * (x + 1.0)
    wz = 0.5 * w

    lam = (nu - 1.0) * m
    # (1 - exp(-2 rho) z^(2/lambda))^m
    # Compute using log1p for stability.
    log_numerator = m * jnp.log1p(-jnp.exp(-2.0 * rho) * z**(2.0/lam))
    log_integrand = log_numerator - jnp.log1p(z**(nu / (nu - 1.0)))
    I_right = jnp.exp(m * rho) / (lam * 2.0**m) * jnp.sum(wz * jnp.exp(log_integrand))

    return jnp.log(I_left + I_right)


def extended_uniform_lognormalization(dim: int, rho, nu):
    """
    Log-normalization of the extended uniform distribution in geodesic coordinates.

    Args:
        dim: dimension of the hyperbolic space.
        rho: hyperbolic ball radius.
        nu: decay of the density outside the ball.

    Returns:
        Log-normalization.
    """
    log_angle_norm = sum(log_sin_power_normalization(dim-k-1) for k in range(1, dim))
    return log_I_r(dim, rho, nu) + log_angle_norm


def log_sin_power_normalization(power):
    """
    Log-integral of sin^power over [0, pi) if power>0 and over [0, 2pi) otherwise.

    Args:
        power: power of sin in integral.

    Returns:
        Log-integral.
    """
    if power == 0:
        return jnp.log(2*jnp.pi)
    elif power % 2 == 0:
        return jnp.log(jnp.pi)+jnp.sum(jnp.array([jnp.log((2*j-1)/(2*j)) for j in 1+jnp.arange(int(power/2))]))
    else:
        return jnp.log(2)+jnp.sum(jnp.array([jnp.log((2*j)/(2*j+1)) for j in 1+jnp.arange(int((power-1)/2))]))
