import jax
from jax import numpy as jnp


eps = 1e-7


def time_coordinates(x: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the time coordinate in the hyperboloid model based on the spatial coordinates.

    Args:
        x: positions (n, D) in the hyperboloid excluding the time coordinate.

    Returns:
        time coordinates (D+1,) in the hyperboloid.
    """
    return jnp.sqrt(jnp.sum(x * x, axis=-1) + 1.0)


def expand_positions(x: jnp.ndarray) -> jnp.ndarray:
    """
    Expands the positions by deducing the time coordinate. The time coordinate is
    appended as the last column.

    Args:
        x: positions (n, D) in the hyperboloid excluding the time coordinate.

    Returns:
        positions (n, D+1) in the hyperboloid including the time coordinate.
    """
    return jnp.concatenate((x, time_coordinates(x)[:, None]), axis=-1)


def lorentz_dot(x, y):
    """
    Vectorized Lorentz dot product.

    Args:
        x, y: (n, D+1) positions on the hyperboloid with time coordinates.

    Returns:
        Lorentz dot product
    """
    return jnp.sum(x[:, :-1]*y[:, :-1], axis=1) - x[:, -1]*y[:, -1]


def hyperbolic_distance(x, y):
    """
    Vectorized hyperbolic distance between two sets of points in thj
    hyperboloid model.

    Args:
        x, y: (n, D+1) positions on the hyperboloid with time coordinates.

    Returns:
        Hyperbolic distance
    """
    ds = x[:, :-1] - y[:, :-1]
    dt = x[:, -1] - y[:, -1]
    q = jnp.sum(ds**2, axis=1) - dt**2
    q = jnp.maximum(q, 0.0)
    return 2.0 * jnp.arcsinh(0.5 * jnp.sqrt(q))


def hyperbolic_distance_stable(x, y):
    """
    Vectorized hyperbolic distance on the hyperboloid with small correction to avoid
    undefined gradient.

    The minimum distance returned is 2asinh(sqrt(q)/2).

    Args:
        x, y: (n, D+1) positions on the hyperboloid with time coordinates.

    Returns:
        Hyperbolic distance
    """
    ds = x[:, :-1] - y[:, :-1]
    dt = x[:, -1] - y[:, -1]
    q = jnp.sum(ds**2, axis=1) - dt**2    # = ||x - y||_L^2 >= 0
    q = jnp.maximum(q, 0.0)               # only guards round-off noise
    return 2.0 * jnp.arcsinh(0.5 * jnp.sqrt(q + eps))


def sd_hyperbolic_distance(x, y):
    """
    Vectorized approximate hyperbolic distance between two points in the analogous S^D-1 model.

    Args:
        x, y: (n, D+1) positions on the hyperboloid with time coordinates.

    Returns:
        Approximated hyperbolic distance.
    """
    rx = jnp.acosh( jnp.clip(x[:, -1], min=1.+eps))
    ry = jnp.acosh( jnp.clip(y[:, -1], min=1.+eps))

    cos_theta = jnp.sum(x[:, :-1]/jnp.sinh(rx[:, None]) * y[:, :-1]/jnp.sinh(ry[:, None]), axis=1)
    return rx + ry + 2.*jnp.log(0.5*jnp.acos( jnp.clip(cos_theta, min=-1, max=1)))


def sd_hyperbolic_distance_stable(x, y):
    """
    Vectorized, smooth, numerically stable approximation of the S^D-1 model approximate hyperbolic distance.
    d ≈ r_x + r_y + 2 log(θ/2).

    The distance approximation itself is approximated to avoid its divergence at θ=0, x=0 or y=0.

    Args:
        x, y: (n, D+1) points on the hyperboloid (time coordinate last).
    Returns:
        Approximated hyperbolic distance.
    """
    xs, ys = x[:, :-1], y[:, :-1]

    # sinh(r) = ||x_spatial||, smoothed at the origin
    sx = jnp.sqrt(jnp.sum(xs**2, axis=1) + eps)
    sy = jnp.sqrt(jnp.sum(ys**2, axis=1) + eps)
    rx, ry = jnp.arcsinh(sx), jnp.arcsinh(sy)

    # unit directions
    ux, uy = xs / sx[:, None], ys / sy[:, None]

    # stable half-angle: tan(θ/2) = ||u - v|| / ||u + v||
    a2 = 0.25 * jnp.sum((ux - uy) ** 2, axis=1)   # = sin^2(θ/2)
    b2 = 0.25 * jnp.sum((ux + uy) ** 2, axis=1)   # = cos^2(θ/2)
    half_theta = jnp.arctan2(jnp.sqrt(a2 + eps), jnp.sqrt(b2 + eps))

    return rx + ry + 2.0 * jnp.log(half_theta)


def hyperbolic_distance_matrix(x: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the hyperbolic distance between each pair of points.

    Args:
        x: positions (n, D+1) in the hyperboloid including the time coordinate.

    Returns:
        (n, n) Distance matrix.
    """
    v, t = x[:, :-1], x[:, -1]
    mip = jnp.outer(t, t) - v @ v.T
    arg = jnp.clip(mip, min=1.)
    return jnp.arccosh(arg)


def sd_dist(pos1, pos2):
    """
    Computes the angular distance between each pair of points.

    Args:
        pos1, pos2: Euclidean positions (n, D+1) on the unit D-sphere.

    Returns:
        (n, n) Distance matrix.
    """
    dot_product = jnp.sum(pos1 * pos2, axis=1)
    return jnp.arccos(jnp.clip(dot_product, -1.0, 1.0))



def hypersphere_distance_matrix(x: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the angular distance between each pair of points.

    Args:
        x: Euclidean positions (n, D+1) on the D-sphere.

    Returns:
        (n, n) Distance matrix.
    """
    dot_product = x @ x.T # (n, D+1) @ (D+1, n) -> (n, n)
    return jnp.arccos(jnp.clip(dot_product, -1.0, 1.0))


def project_tangent(x, v):
    """
    Vectorized projection of vectors onto the hyperboloid model's tangent space
    at x. Applied row-wise.

    Args:
        v: (n, D+1) vectors.
        x: (n, D+1) points on hyperboloid with time coordinate.

    Returns:
        (n, D+1) Tangent vectors.
    """
    inner = lorentz_dot(x, v)[..., None]
    return v + inner * x


def sinhc(x):
    """Numerically stable sinh(x)/x with 4th order Taylor approximation."""
    x2 = x * x
    # 4th-order Taylor expansion around 0
    taylor = 1.0 + x2 / 6.0 + x2 * x2 / 120.0
    return jnp.where(jnp.abs(x) < 1e-4, taylor, jnp.sinh(x) / x)


def log_sinh(x):
    """Numerically stable ln(sinh(x))."""
    x = jnp.asarray(x)
    abs_x = jnp.abs(x)

    # Use np.where for clean conditional computation
    # Small values: compute directly
    # Large values: use asymptotic expansion
    return jnp.where(
        abs_x < 1.0,
        jnp.log(jnp.maximum(jnp.sinh(abs_x), 1e-300)),  # Direct computation
        abs_x + jnp.where(
            jnp.exp(-2 * abs_x) < 1e-15,
            -jnp.log(2),  # Very large: x - log(2)
            jnp.log1p(-jnp.exp(-2 * abs_x)) - jnp.log(2)  # Moderate: use log1p
        )
    )


def exp_map(x, v):
    """
    Vectorized exponential map on the hyperboloid.

    Args:
        x: (n, D+1) points.
        v: (n, D+1) tangent vectors.

    Returns:
        (n, D+1) Exponential map.
    """
    v_norm_sq = lorentz_dot(v, v)
    v_norm_sq = jnp.maximum(v_norm_sq, 0.0)

    v_norm = jnp.sqrt(v_norm_sq)[..., None]

    return (
        jnp.cosh(v_norm) * x
        + sinhc(v_norm) * v
    )


# Equation from Manopt https://manoptjl.org/v0.1/manifolds/hyperbolic/
def log_map(x, y, eps=1e-8):
    """
    Vectorized logarithmic map of the hyperboloid model.
    Map points y back to the tangent space at x.

    Args:
        x: (n, D+1) base points on the hyperboloid
        y: (n, D+1) target points on the hyperboloid

    Returns:
        (n, D+1) tangent vectors at x
    """
    xy = lorentz_dot(x, y)  # (n,), should be <= -1
    xy_clamped = jnp.clip(-xy, min=1.0)
    dist = jnp.arccosh(xy_clamped)  # (n,)

    # Numerator: y + <x,y>_L * x
    # This is the component of y orthogonal to x in the Lorentzian sense
    u = y + xy[:, None] * x  # (n, D+1)

    # The Lorentzian norm of u equals sinh(dist) exactly in exact arithmetic.
    # Using sinh(dist) directly is more stable than computing the norm of u,
    # especially when x and y are close (dist -> 0, u -> 0, norm(u) -> 0)
    # and when x and y are far (u has large components that cancel)
    sinh_dist = jnp.sinh(dist)  # (n,)

    # When x and y are very close, both dist and sinh_dist approach 0.
    # The ratio dist/sinh_dist -> 1 in that limit, and the whole log_map -> 0.
    # We use where() to avoid 0/0 NaN while keeping correct gradients.
    safe_sinh = jnp.where(sinh_dist < eps, jnp.ones_like(sinh_dist), sinh_dist)
    scale = jnp.where(sinh_dist < eps, jnp.zeros_like(dist), dist / safe_sinh)

    return scale[:, None] * u  # (n, D+1)


def parallel_transport(x, y, v):
    """
    Parallel transport of tangent vector v from T_x M to T_y M
    along the geodesic from x to y on the hyperboloid.

    Args:
        x: (n, D+1) source points on the hyperboloid
        y: (n, D+1) target points on the hyperboloid
        v: (n, D+1) tangent vectors at x to be transported

    returns: (n, D+1) tangent vectors at y
    """
    xy = lorentz_dot(x, y)
    yv = lorentz_dot(y, v)

    coeff = yv / (1.0 - xy)
    transported = v + coeff[:, None] * (x + y)
    return project_tangent(y, transported)


def hyperbolic_translate(points, u, t):
    """
    Vectorized translation of points on hyperboloid in direction u at a
    distance t.

    Args:
        points : (n, D) hyperboloid coordinates.
        u : (D,) unit spatial direction in hyperboloid.
        t : translation distance.

    Returns:
        (n, D) Translated points.
    """
    spatial = points
    time = time_coordinates(points)

    cosh = jnp.cosh(t)
    sinh = jnp.sinh(t)

    # projection onto direction u
    xu = spatial @ u # (N,)

    # orthogonal component
    spatial_perp = spatial - jnp.outer(xu, u)

    # boosted components
    xu_new = cosh * xu + sinh * time

    return spatial_perp + jnp.outer(xu_new, u)


def to_riemannian_gradient(x_full, grads):
    """
    Apply the inverse Riemannian metric correction to Euclidean gradients.

    Args:
        x_full: (n, D+1) hyperboloid coordinates including the time coordinate
        grads: (n, D) Euclidean gradient w.r.t. spatial coordinates

    Returns:
        (n, D+1) Riemannian gradient with time component
    """
    x_spatial = x_full[:, :-1]

    # Spatial metric: eye(D) + outer(x, x), applied to grad
    # w_i = (I + x_i x_i^T) grad_i = grad_i + x_i * (x_i . grad_i)
    dots = jnp.sum(x_spatial * grads, axis=-1, keepdims=True)  # (n, 1)
    w = grads + x_spatial * dots                               # (n, D)

    # Time coordinate of Riemannian gradient
    v_time = jnp.sum(w * x_spatial, axis=-1) / x_full[:, -1]  # (n,)
    riemannian_grad_full = jnp.concatenate([w, v_time[:, None]], axis=-1)  # (n, D+1)
    return riemannian_grad_full


def geodesic_to_hyperboloid(radii, angles):
    """
    Convert geodesic coordinates to hyperboloid coordinates.

    The first D-2 angles are in range [0, π] and the last angle is in range [-π, π].

    Args:
        radii: (n,) contains radial coordinates.
        angles: (n, D-1) contains angular coordinates.

    Returns:
        (n, D) Hyperboloid coordinates.
    """
    n = angles.shape[0]
    d = angles.shape[1] + 1  # dimension of hyperbolic space

    angles_t = angles.T
    # Compute cumulative sine products
    sin_products = jnp.ones(n)

    # Compute hyperboloid coordinates
    hyperboloid = []
    for i in range(d):
        if i < d-1:
            # xᵢ = sinh(r) * sin(θ₁) * ... * sin(θᵢ₋₁) * cos(θᵢ)
            hyperboloid.append(jnp.sinh(radii) * sin_products * jnp.cos(angles_t[i]))
            sin_products = sin_products*jnp.sin(angles_t[i])
        else:
            # Last coordinate: xₐ = sinh(r) * sin(θ₁) * ... * sin(θₐ₋₁)
            hyperboloid.append(jnp.sinh(radii) * sin_products)
    return jnp.asarray(hyperboloid).T


def hyperboloid_to_geodesic(pos):
    """
    Convert hyperboloid coordinates to geodesic coordinates.

    Args:
        pos : (n, D) positions on the hyperboloid without the time coordinate.

    Returns:
        (radii, angles), where shapes are radii (n,) and angles (n, D-1).
    """
    n, d = pos.shape

    hyperboloid_coords_t = pos.T
    # Compute angles in reverse
    angles = [jnp.arctan2(hyperboloid_coords_t[-1], hyperboloid_coords_t[-2])]
    partial_sums = hyperboloid_coords_t[-1]**2 + hyperboloid_coords_t[-2]**2

    clip_value = 1e-8 if pos.dtype == jnp.float32 else 1e-15
    for i in range(d-3, -1, -1):
        partial_sums = partial_sums + hyperboloid_coords_t[i]**2
        r_partial = jnp.sqrt(partial_sums)

        # Avoid division by zero
        a = jnp.arccos(jnp.clip(hyperboloid_coords_t[i] / r_partial, -1, 1))
        a = jnp.where(r_partial < clip_value, 0., a)
        angles.append(a)

    return jnp.arcsinh(jnp.sqrt(partial_sums)), jnp.array(angles[::-1]).T


def sinc(x):
    """
    sin(x)/x with the correct limit at x=0.
    """
    x2 = x * x
    return jnp.where(
        jnp.abs(x) < 1e-4,
        1.0 - x2 / 6.0 + x2 * x2 / 120.0,
        jnp.sin(x) / x,
    )


def hypershere_exp_map(x, v):
    """
    Vectorized exponential map on the unit sphere at x in direction v.

    Args:
        x: (n, D+1) Euclidean coordinates on D-sphere
        v: (n, D+1) Euclidean tangent vectors at x

    Returns:
        (n, D+1) Exponential map.
    """
    v_norm = jnp.linalg.norm(v, axis=-1, keepdims=True)
    return (
        jnp.cos(v_norm) * x
        + sinc(v_norm) * v
    )
