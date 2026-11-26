"""
Hyperbolic embedding objective functions.

Authors:
- JGY <jyoung22@uvm.edu>
- Ariane Lizotte <ariane.lizotte.2@ulaval.ca>
"""

import jax
import jax.numpy as jnp
from jax.scipy import special
from scipy import optimize

from .geometry import expand_positions, hyperbolic_distance, sd_dist
from .priors import cauchy_logprior, extended_uniform_logdensity, trunc_normal_logprior, extended_uniform_logprior_mc, log_sin_power_normalization


# Hyperparameter values used in the paper
default_sigmas = {"beta": 10., "r": 3., "rho": 3., "nu": 3.}
default_beta_m = 2.5


def connection_prob_pairs(xs: jnp.ndarray, ys: jnp.ndarray, beta: float, r: float, dist):
    """
    Connection probability between each pair of `xs` and `ys`.

    Args:
        xs, ys: (n, D+1) positions in the hyperboloid model including the time coordinate.
        beta: inverse temperature.
        r: midpoint of the connection function.

    Returns:
        Edge probabilities.
    """
    return jax.nn.sigmoid(0.5 * beta * (r - dist(xs, ys)))


def hyperbolic_loglikelihood(
    edges: jnp.ndarray,   # (m,2)
    pos: jnp.ndarray,
    beta: float,
    r: float,
    block_size: int = 1024,
    clip_threshold: float = 1e-4,
    dist = hyperbolic_distance,
) -> float:
    """
    Log-likelihood of the H^D hyperbolic graph model.

    Args:
        edges: (m, 2) edges of the graph.
        pos: (n, D) vertex positions in the hyperboloid model without time coordinate.
        beta: inverse temperature.
        r: midpoint of the connection function.
        clip_threshold: maximum distance to 0 and 1 of the edge probability.
        block_size: Controls how many terms of the non-edges are computed simultaneously.
            The sum over non-edges is computed by summing submatrices of size (block_size, block_size).
    Returns:
        Log-likelihood.
    """
    x = expand_positions(pos)
    xi = x[edges[:, 0]]
    xj = x[edges[:, 1]]

    p_edges = connection_prob_pairs(xi, xj, beta, r, dist)
    p_edges = jnp.clip(
        p_edges,
        clip_threshold,
        1 - clip_threshold
    )

    log_p = jnp.log(p_edges)
    log_1mp = jnp.log1p(-p_edges)

    # Edges with correction term
    edge_term = jnp.sum(log_p - log_1mp)

    # Non edges
    # sum_{i<j} log(1 - p_ij)
    background = _background_log1m_sum(
        x, beta, r, block_size=block_size, clip_threshold=clip_threshold, dist=dist
    )

    return background + edge_term


def _background_log1m_sum(
    x,
    beta,
    r,
    dist,
    block_size=1024,
    clip_threshold=1e-4,
):
    """Computes sum_{i<j} log(1 - p_ij) without constructing an (n,n) matrix."""
    n = x.shape[0]
    total = 0.0

    for i0 in range(0, n, block_size):
        xi = x[i0:i0 + block_size]

        for j0 in range(i0, n, block_size):
            xj = x[j0:j0 + block_size]

            B = xi.shape[0]
            M = xj.shape[0]

            xi_rep = jnp.repeat(xi, M, axis=0)   # (B*M, D+1)
            xj_rep = jnp.tile(xj, (B, 1))        # (B*M, D+1)

            P = connection_prob_pairs(
                xi_rep,
                xj_rep,
                beta,
                r,
                dist=dist
            )
            P = P.reshape(B, M)

            P = jnp.clip(
                P,
                clip_threshold,
                1 - clip_threshold,
            )

            block = jnp.log1p(-P)

            # remove diagonal when same block
            if i0 == j0:
                block = jnp.triu(block, k=1)

            total += jnp.sum(block)
    return total


def discretize(params: dict[str, float|jnp.ndarray], quantization: dict[str, float]):
    """
    Apply x -> quantization*ceil(x/quantization) on each value of `params`.
    This expresses each parameter up to `quantization` precision.

    Args:
        params: `dict` of continuous parameters to discretize.
        quantization: width of the discrete bins for each parameter.
    """
    discretize_value = lambda x, q: q*jnp.ceil(x/q)
    return {
        param: discretize_value(value, quantization[param])
            for param, value in params.items()
    }


def sd_connection_prob_pairs(pos1, pos2, kappa1, kappa2, beta, mu, sphere_radius, clip_threshold: float = 1e-4):
    """
    Vectorized edge probability of the S^D-1 model.

    Args:
        pos1, pos2: (n, D) Euclidean coordinates on the unit (D-1)-sphere.
        kappa1, kappa2: Hidden degrees.
        beta: inverse temperature.
        sphere_radius: radius of the sphere.
        clip_threshold: maximum distance to 0 and 1 of the edge probability.

    Returns:
        Edge probabilities.
    """
    d = pos1.shape[1]-1
    dist = sd_dist(pos1, pos2)
    chi = dist * sphere_radius / (mu * (kappa1 * kappa2))**(1/d)
    return jnp.clip(1 / (1 + chi**beta), clip_threshold, 1 - clip_threshold)


def sd_loglikelihood(edges, sphere_pos, kappas, beta, sphere_radius, mu, clip_threshold: float = 1e-4):
    """
    Log-likelihood of the S^D-1 graph model.

    Args:
        edges: (m, 2) edges of the graph.
        sphere_pos: (n, D) Euclidean vertex positions on the unit (D-1)-sphere.
        kappas: (n,) vertex hidden degrees.
        beta: inverse temperature.
        r: midpoint of the connection function.
        clip_threshold: maximum distance to 0 and 1 of the edge probability.
    Returns:
        Log-likelihood.
    """
    n = sphere_pos.shape[0]

    # Contribution of non-edges
    iu, iv = jnp.triu_indices(n, k=1)
    p = sd_connection_prob_pairs(sphere_pos[iu], sphere_pos[iv], kappas[iu], kappas[iv], beta, mu, sphere_radius, clip_threshold)
    loglikelihood_nonedges = jnp.sum(jnp.log1p(-p))

    # Contribution of edges (corresponding non-edges terms are added cancel out the previous calculation)
    iu, iv = edges.T
    p = sd_connection_prob_pairs(sphere_pos[iu], sphere_pos[iv], kappas[iu], kappas[iv], beta, mu, sphere_radius, clip_threshold)
    loglikelihood_edges = -jnp.sum(jnp.log1p(-p)) + jnp.sum(jnp.log(p))
    return loglikelihood_nonedges + loglikelihood_edges


def hyperbolic_dl_terms(
    pos: jnp.ndarray,
    beta: float,
    r: float,
    edges: jnp.ndarray,
    rho: float,
    nu: float,
    quantization: dict[str, float],
    sigmas: dict[str, float] = default_sigmas,
    beta_m: float = default_beta_m,
    block_size: int = 1024,
    key = None
    ):
    """
    Computes the log-likelihood and the log-prior terms of the H^D hyperbolic graph model.
    The description length is the sum over the returned values.

    Args:
        pos: (n, D) vertex positions in the hyperboloid without the time coordinate.
        beta: inverse temperature.
        r: midpoint of the connection function.
        edges: (m, 2) edges of the graph.
        rho: radius of the hyperbolic ball for the extended uniform prior.
        nu: sharpness of the density drop outside the hyperbolic ball (extended uniform prior).
        quantization: width of the discrete bins for each parameter.
        sigmas: standard deviation/dispersion of each hyperparameter.
        beta_m: median of the truncated normal prior for beta/(D-1).
        block_size: Controls how many terms of the non-edges are computed simultaneously in the likelihood.
            The sum over non-edges is computed by summing submatrices of size (block_size, block_size).
        key: Jax RNG key used for the Monte Carlo estimation of the prior on positions.

    Returns:
        Negative log-likelihood, Negative log-prior
    """
    dim = pos.shape[1]
    binned = discretize({"pos": pos, "beta": beta/(dim-1), "r": r, "rho": rho, "nu": nu}, quantization)

    log_likelihood = hyperbolic_loglikelihood(edges=edges, pos=binned["pos"], beta=(dim-1)*binned["beta"], r=binned["r"], block_size=block_size)

    if key is None:
        key = jax.random.PRNGKey(10)
    log_prior_pos = jnp.sum(extended_uniform_logprior_mc(binned["pos"], rho=binned["rho"], nu=binned["nu"], quantization=quantization["pos"], key=key))

    log_prior_beta = trunc_normal_logprior(binned["beta"], center=beta_m, sigma=sigmas["beta"], lower_bound=0, quantization=quantization["beta"])
    # log(2) is the correction for the truncation at zero
    log_prior_r = cauchy_logprior(binned["r"], dispersion=sigmas["r"], quantization=quantization["r"])+jnp.log(2)
    log_prior_rho = cauchy_logprior(binned["rho"], dispersion=sigmas["rho"], quantization=quantization["rho"])+jnp.log(2)
    log_prior_nu = cauchy_logprior(binned["nu"]-1, dispersion=sigmas["nu"], quantization=quantization["nu"])+jnp.log(2)
    return -log_likelihood, -log_prior_pos - log_prior_beta - log_prior_r - log_prior_rho - log_prior_nu



def hyperbolic_description_length(*args, **kwargs):
    """Description length of the H^D hyperbolic random graph model.
    See documentation of hyperbolic_dl_terms."""
    return sum(hyperbolic_dl_terms(*args, **kwargs))


def prior_logdensity(
    pos: jnp.ndarray,
    beta: float,
    r: float,
    rho: float,
    nu: float,
    sigmas: dict[str, float]=default_sigmas,
    beta_m: float=default_beta_m,
) -> float:
    """
    Unnormalized prior distribution of the H^D hyperbolic graph model.

    Args:
        pos: positions (n, D) in the hyperboloid without time coordinate.
        beta: inverse temperature.
        r: midpoint of the connection function.
        rho: radius of the hyperbolic ball for the extended uniform prior.
        nu: sharpness of the density decay outside the hyperbolic ball (extended uniform prior).
        sigmas: standard deviation/dispersion of each hyperparameter.
        beta_m: median of the truncated normal prior for beta/(D-1).

    Returns:
        Prior log density of the hyperbolic graph model.
    """
    dim = pos.shape[1]
    log_prior_pos = jnp.sum(extended_uniform_logdensity(pos, rho, nu))
    # Full range (not truncated) densities are used because the normalization constants don't matter for optimization
    log_prior_beta = jax.scipy.stats.norm.logpdf(beta/(dim-1), scale=sigmas["beta"], loc=beta_m)
    log_prior_r = jax.scipy.stats.cauchy.logpdf(r, scale=sigmas["r"])
    log_prior_rho = jax.scipy.stats.cauchy.logpdf(rho, scale=sigmas["rho"])
    log_prior_nu = jax.scipy.stats.cauchy.logpdf(nu-1, scale=sigmas["nu"])

    return log_prior_pos + log_prior_beta + log_prior_r + log_prior_rho + log_prior_nu


def map_sd_to_hyperbolic_model(sphere_pos, kappas, mu, kappa0, sphere_radius):
    """
    Finds the H^D model parametrization of a S^D-1 model embedding.

    Args:
        sphere_pos: (n, D) Euclidean vertex positions in the unit (D-1)-sphere.
        kappas: (n, ) hidden degrees.
        mu: hidden degree global scaling parameter.
        kappa0: dummy parameter related to the minimum kappa.
        sphere_radius: sphere radius.

    Returns:
        (pos, r), where pos (n, D) are the spatial coordinates of the
        hyperboloid model and r is the connection length parameter.
    """
    d = sphere_pos.shape[1]-1
    hd_radius = 2*jnp.log(2*sphere_radius) - (2/d)*jnp.log(mu*kappa0**2)
    radii = hd_radius - 2./d*jnp.log(kappas/kappa0)
    radii = jnp.where(radii < 0., 0., radii)
    return jnp.sinh(radii)[:, None] * sphere_pos, hd_radius


def map_hyperbolic_model_to_sd(hyperbolic_pos, r):
    """
    Finds the S^D-1 model parametrization of a H^D model embedding.

    Args:
        hyperbolic_pos : (n, D) spatial vertex coordinates in the hyperboloid model.
        r : Midpoint of the connection function.

    Returns:
        (sphere_pos, kappas, sphere_radius, mu, kappa0)
            where
                - sphere_pos (n, D) are the Euclidean vertex coordinates in the unit S^D-1 sphere;
                - kappas (n,) are the hidden degrees;
                - mu: hidden degree global scaling parameter;
                - sphere_radius is the sphere radius;
    """
    d = hyperbolic_pos.shape[1] - 1

    spatial_norm = jnp.linalg.norm(hyperbolic_pos, axis=1)
    radii = jnp.asinh(spatial_norm)

    sphere_pos = hyperbolic_pos / jnp.maximum(spatial_norm[:, None], 1e-12)

    # arbitrary direction for the origin
    sphere_pos = sphere_pos.at[spatial_norm < 1e-12].set(
        jnp.eye(hyperbolic_pos.shape[1])[0]
    )

    # These parameters are tied together and don't affect the connection prob.
    # With these values however, kappas are not the expected degrees.
    kappa0 = 1
    mu = 1
    sphere_radius = 0.5 * jnp.exp(
        0.5 * (r + (2 / d) * jnp.log(mu * kappa0**2))
    )

    kappas = kappa0 * jnp.exp(0.5 * d * (r - radii))
    return sphere_pos, kappas, sphere_radius, mu, kappa0


def optimize_kappa0(sphere_pos, kappas, beta, mu, sphere_radius):
    """
    Finds the kappa0 minimizing the L^2 error between H^D and S^D-1 edge probabilities.

    Args:
        sphere_pos: (n, D) Euclidean vertex positions in the unit (D-1)-sphere.
        kappas: (n, ) hidden degrees.
        beta: inverse temperature.
        mu: hidden degree global scaling parameter.
        sphere_radius: sphere radius.

    Returns:
        Optimized kappa0 value.
    """
    n = sphere_pos.shape[0]

    iu, iv = jnp.triu_indices(n, k=1)
    sd_1_probs = sd_connection_prob_pairs(
            pos1=sphere_pos[iu], pos2=sphere_pos[iv],
            kappa1=kappas[iu], kappa2=kappas[iv],
            beta=beta, mu=mu, sphere_radius=sphere_radius)

    kappa0_max = 200
    to_unrestricted = lambda x: jax.scipy.special.logit(x/kappa0_max)
    from_unrestricted = lambda x: jax.scipy.special.expit(x)*kappa0_max

    @jax.jit
    def loss(_kappa0):
        pos, r = map_sd_to_hyperbolic_model(sphere_pos, kappas, mu, from_unrestricted(_kappa0), sphere_radius)
        x_full = expand_positions(pos)
        hd_prob = connection_prob_pairs(x_full[iu], x_full[iv], beta, r, dist=hyperbolic_distance)
        return jnp.sum(jnp.power(sd_1_probs-hd_prob, 2))

    return from_unrestricted(optimize.minimize(loss, to_unrestricted(jnp.min(kappas))).x[0])


def estimate_parameters_for_uniform(n, dim, beta, gamma, expected_degree):
    alpha = (gamma-1)/2
    tau = (dim-1)/beta
    d = dim - 1

    if tau<1:
        if dim==2:
            # Eq.(22) https://journals.aps.org/pre/abstract/10.1103/PhysRevE.82.036106
            xi = (alpha/(alpha-0.5))
            # Factor 0.5 to compensate the 2 factor multiplied later
            ball_r = 0.5*optimize.fsolve(
                lambda x: expected_degree - 2/jnp.pi*xi*xi*n*(
                    jnp.exp(-x/2)+jnp.exp(-alpha*x)*(alpha*x/2*(jnp.pi/(4*alpha*alpha) - (jnp.pi-1)/alpha + jnp.pi-2)-1)),
                x0=8.
            )[0]
            r = ball_r
        else:
            # Eqs.(43-46) https://journals.aps.org/pre/abstract/10.1103/PhysRevE.109.054131
            if gamma == 2.:
                constant = 2**d / (d * jnp.exp(log_sin_power_normalization(d-1))) / jnp.sinc(tau)
                nu = optimize.fsolve(lambda x: expected_degree - x*constant*jnp.log(n/x), x0=1.)[0]
                ball_r = jnp.log(n/nu)
                r = ball_r - jnp.log(ball_r)
            else:
                constant = 2**d / (d * jnp.exp(log_sin_power_normalization(d-1))) * ((gamma-1)/(gamma-2))**2 / jnp.sinc(tau)
                nu = optimize.fsolve(lambda x: expected_degree - x*constant*(1-2*(n/x)**(2-gamma)+(n/x)**(2*(2-gamma))), x0=1.)[0]
                ball_r = jnp.log(n/nu)
                r = ball_r
    else:
        # Possible for tau=1, but computing nu requires solving transcendental with special functions.
        # Also possible for tau>1, but requires computing an integral numerically.
        # See Sections IVB and IVC of https://journals.aps.org/pre/abstract/10.1103/PhysRevE.109.054131
        raise NotImplementedError
    ball_r = ball_r*2/d
    r = r*2/d
    return r, ball_r, alpha


def uniform_degree_lpdf(k, kappa0, a):
    """Log-density of the expected degree distribution for the hyperbolic graph
    model when points are near-uniformly distributed in a hyperbolic ball."""
    return (
        jnp.log(a)
        + a * jnp.log(kappa0)
        + jnp.log(special.gammaincc(k - a, kappa0))
        + special.gammaln(k - a)
        - special.gammaln(k + 1)
    )


def fit_pareto_poisson(sample):
    """
    Estimate kappa0 and a for the Pareto-mixed Poisson distribution via MLE.

    Parameters:
        sample: array of positive integer observations

    Returns:
        estimation of kappa0, a
    """
    sample = jnp.asarray(sample, dtype=float)

    # Initial guess: method of moments or simple heuristics
    a0 = 1.5
    kappa0_0 = a0 * jnp.mean(sample) / (jnp.mean(sample) - 1)  # rough moment estimate

    x0 = [kappa0_0, a0]
    bounds = [(1e-6, None), (1e-6, None)]  # both params must be positive

    def neg_log_likelihood(params):
        kappa0, a = params

        # Enforce valid parameter domain
        if kappa0 <= 0 or a <= 0:
            return jnp.inf

        # k must satisfy k > a for gammaincc to be defined
        if jnp.any(sample <= a):
            return jnp.inf

        ll = jnp.sum(uniform_degree_lpdf(sample, kappa0, a))
        if not jnp.isfinite(ll):
            return jnp.inf
        return -ll

    result = optimize.minimize(
        neg_log_likelihood,
        x0=x0,
        method="Nelder-Mead", # gradient-free, robust to the special functions
        bounds=bounds,
        options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 10_000},
    )

    return result.x
