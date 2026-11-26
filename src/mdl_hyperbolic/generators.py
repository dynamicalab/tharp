"""
Graph generators for hyperbolic models.

Authors:
- JGY <jyoung22@uvm.edu>
- Ariane Lizotte <ariane.lizotte.2@ulaval.ca>
"""
import jax
import jax.numpy as jnp

from mdl_hyperbolic.geometry import hypershere_exp_map


def sample_quasiuniform_radii_2d(disk_radius, key, alpha=1., num_samples=1, dtype=jnp.double):
    """
    Samples radial coordinates almost uniformly in a hyperbolic ball.

    The radial coordinates are for the hyperbolic plane polar coordinates,
    which correspond to the distance to the origin. This implementation uses
    inverse transform sampling, which is only possible in the hyperbolic plane
    (2 dimensions). Sampling is uniform when `alpha` is 1. Larger (smaller)
    values of `alpha` skew towards larger (smaller) radii.

    Args:
        disk_radius: radius of the hyperbolic ball.
        key: jax rng key.
        alpha: skews vertices closer (<1) or farther from origin (>1) compared to unifomity.
        num_samples: number of samples to draw.
        dtype: floating point type to use

    Returns:
        Radii sample.
    """
    u = jax.random.uniform(key, shape=(num_samples,), dtype=dtype)
    return jnp.arccosh(u*(jnp.cosh(alpha*disk_radius) - 1) + 1)/alpha


def sample_trunc_exponential(power, sup, key, num_samples=1, dtype=jnp.double):
    """
    Samples the distribution $f(x)\\propto e^{\\text{power} x}$ with $x\\in [0, \\text{sup}]$.

    Args:
        power: power of the exponential density
        sup: truncation upper bound
        key: JAX PRNG key
        num_samples: number of samples to draw
        dtype: floating point type to use

    Returns:
        Sample.
    """
    u = jax.random.uniform(key, shape=(num_samples,), dtype=dtype)
    return 1/power*(jnp.log(u*(jnp.exp(power*sup)-1)+1))


def sample_quasiuniform_radius(num_dim, radius, key, num_samples=1, alpha=1., dtype=jnp.double):
    """
    Samples radial coordinates uniformly (if `alpha=1`) in a hyperbolic ball.

    The radial coordinates are for the hyperbolic plane polar coordinates,
    which correspond to the distance to the origin. Sampling is uniform when
    `alpha` is 1. Larger (smaller) values of `alpha` skew towards larger
    (smaller) radii. The implementation uses rejection sampling.

    Args:
        num_dim: number of dimensions of the hyperbolic ball.
        radius: radius of the hyperbolic ball.
        key: jax rng key.
        num_samples: number of samples to draw.
        alpha: controls the skewness in values of radii compared to uniformity.
        dtype: floating point type to use

    Returns:
        Radii sample.
    """
    samples = jnp.zeros((0,), dtype=dtype)
    current_key = key
    if radius == 0:
        return jnp.zeros(num_samples)
    while samples.shape[0] < num_samples:
        current_key, proposal_key, u_key = jax.random.split(current_key, 3)
        proposals = sample_trunc_exponential(alpha*(num_dim-1), radius, proposal_key, num_samples=num_samples, dtype=dtype)
        u = jax.random.uniform(u_key, shape=(num_samples,), dtype=dtype) + 1e-16 # Making sure that u!=0
        accepted = proposals[jnp.log(u) < (num_dim-1)*jnp.log(1.-jnp.exp(-2.*alpha*proposals))]
        samples = jnp.concatenate((samples, accepted), axis=0)
    return samples[:num_samples]


def sample_unit_vector_uniformly(num_dim:int, key, num_samples:int=1, dtype=jnp.double):
    """
    Samples points on the (`num_dim`-1)-sphere uniformly.

    Args:
        num_dim: number of dimensions of the hyperbolic ball.
        key: JAX PRNG key.
        num_samples: number of samples to draw.
        dtype: floating point type to use

    Returns:
        (num_samples, num_dim) Sample in cartesian coordinates.
    """
    y = jax.random.normal(key, shape=(num_samples, num_dim), dtype=dtype)
    return y / jnp.linalg.norm(y, axis=1).reshape(-1, 1)


def sample_quasiuniform_hyperbolic_ball(num_dim:int, radius:float, key, num_samples:int=1, alpha:float=1., dtype=jnp.double):
    """
    Samples radial coordinates uniformly (if `alpha=1`) in a hyperbolic ball.

    Args:
        num_dim: number of dimensions of the hyperbolic ball.
        radius: radius of the hyperbolic ball.
        key: jax rng key.
        num_samples: number of samples to draw.
        alpha: skews vertices closer (<1) or farther from origin (>1) compared to unifomity.
        dtype: floating point type to use.

    Returns:
        (num_samples, num_dim) Sample in hyperboloid coordinates without time.
    """
    if num_dim < 2:
        raise ValueError("Dimension must be greater than 1.")
    angle_key, radii_key = jax.random.split(key, 2)

    radii = sample_quasiuniform_radius(num_dim, radius, radii_key, num_samples=num_samples, alpha=alpha, dtype=dtype)
    v = sample_unit_vector_uniformly(num_dim, angle_key, num_samples=num_samples, dtype=dtype)
    return v*jnp.sinh(radii).reshape(-1, 1)


def sample_hyperbolic_normal(num_dim, gamma, key, num_samples=1, dtype=jnp.double):
    """
    Samples the hyperbolic normal distribution.

    Args:
        num_dim: number of dimensions of the hyperbolic space.
        gamma: dispersion parameter of the distribution.
        key: JAX PRNG key.
        dtype: floating point type to use.

    Returns:
        (num_samples, num_dim) Sample in hyperboloid coordinates without time.
    """
    angle_key, radii_key = jax.random.split(key, 2)
    r = jnp.abs(gamma*jax.random.normal(radii_key, shape=(num_samples,), dtype=dtype))
    v = sample_unit_vector_uniformly(num_dim, angle_key, num_samples=num_samples)
    return v*jnp.sinh(r).reshape(-1, 1)


def _sample_hypersphere_normal(key, center, sigma, dtype):
    z = jax.random.normal(key, shape=center.shape, dtype=dtype)

    # project onto tangent space
    v = z - jnp.dot(z, center) * center

    # isotropic Gaussian in T_center S^D
    v = sigma * v

    return hypershere_exp_map(center, v)


def sample_hypersphere_normal(key, center, sigma, num_samples:int=1, dtype=jnp.double):
    """
    Samples points from an isotropic intrinsic Gaussian centered at 'center'.

    Args:
        key: JAX PRNG key.
        center: (D+1,) unit vector
        sigma: tangent-space standard deviation
        num_samples: number of samples to draw.
        dtype: floating point type to use

    Returns:
        (num_samples, D+1) sample in Euclidean coordinates.
    """
    keys = jax.random.split(key, num_samples)
    return jax.vmap(
        lambda k: _sample_hypersphere_normal(k, center, sigma, dtype)
    )(keys)


def sample_uniform_cap(key, center, theta, num_samples:int=1, dtype=jnp.double):
    """
    Uniformly sample n points from a spherical cap on S^D.

    Args:
        key : JAX PRNG key
        center : (D+1,) Unit vector defining the cap center.
        theta : float Geodesic radius (radians).
        num_samples : int Number of samples.
        dtype: floating point type to use

    Returns:
        samples : (num_samples, D+1) Uniform samples in the cap.
    """

    ambient_dim = center.shape[0]
    D = ambient_dim - 1

    if D == 1:
        center_angle = jnp.atan2(center[1], center[0])

        angles = center_angle + theta * jax.random.uniform(
            key,
            shape=(num_samples,),
            minval=-1.0,
            maxval=1.0,
            dtype=dtype,
        )

        return jnp.stack(
            [jnp.cos(angles), jnp.sin(angles)],
            axis=-1,
        )

    zmin = jnp.cos(theta)

    key_z, key_u = jax.random.split(key)

    # Sample z = center·y
    def sample_z(key):
        def cond_fun(state):
            _, accepted, _ = state
            return ~accepted

        def body_fun(state):
            key, _, _ = state
            key, k1, k2 = jax.random.split(key, 3)
            z = jax.random.uniform(
                k1,
                minval=zmin,
                maxval=1.0,
                dtype=dtype
            )
            weight = (1.0 - z**2) ** ((D - 2) / 2)
            accepted = (
                jax.random.uniform(k2)
                < weight
            )
            return key, accepted, z

        _, _, z = jax.lax.while_loop(
            cond_fun,
            body_fun,
            (key, False, 0.0),
        )
        return z

    keys = jax.random.split(key_z, num_samples)
    z = jax.vmap(sample_z)(keys)

    # Sample uniform tangent directions
    v = jax.random.normal(key_u, shape=(num_samples, ambient_dim))

    # project to tangent space at center
    v = v - jnp.sum(v * center, axis=-1, keepdims=True) * center

    v = v / jnp.linalg.norm(v, axis=-1, keepdims=True)

    samples = (
        z[:, None] * center
        + jnp.sqrt(1.0 - z**2)[:, None] * v
    )

    return samples
