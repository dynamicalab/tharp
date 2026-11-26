"""
Optimization loop and utility.

Authors:
- JGY <jyoung22@uvm.edu>
- Ariane Lizotte <ariane.lizotte.2@ulaval.ca>
"""
from functools import partial
from typing import NamedTuple, Callable, Optional

from optax import GradientTransformation
from optax.transforms import freeze
from scipy.optimize import minimize_scalar
from tqdm import tqdm
import jax
import jax.numpy as jnp
import optax
import rustworkx as rx

from mdl_hyperbolic.geometry import exp_map, expand_positions, lorentz_dot, parallel_transport, project_tangent, to_riemannian_gradient, sd_hyperbolic_distance_stable, hyperbolic_distance_stable
from mdl_hyperbolic.models import hyperbolic_loglikelihood, prior_logdensity, hyperbolic_description_length

from .sgd_sampling import estimate_gradient, estimate_pair_loglikelihood, get_neighborhood_bernoulli_sampler


params_max = 2000 # Maximum value allowed for the model parameters (beta/(D-1), r, rho, nu-1)
rho_min = 0.01 # Minimum rho value allowed. May cause optimization issues if too small.

# Approach of Stan https://mc-stan.org/docs/2_18/reference-manual/logit-transform-jacobian-section.html
# logit: (0, 1) -> R
# expit: R -> (0, 1)
def to_unrestricted_params(dim: int, **params):
    """
    Maps each restricted parameter to real numbers using the logit function.

    Unrestricted parameters are distinguished by prepending '_' to their name.
    The allowed parameter intervals are:
        beta: ((D-1), 2000(D-1))
        r: (0, 2000)
        rho: (0.01, 2000.01)
        nu: (1, 2001)
    Any parameter name not included in the previous list, including vertex
    positions, are untouched.

    Params:
        dim: embedding dimension used for beta scaling.
        **params: parameters to remap.

    Returns:
        Unrestricted parameters.
    """
    forward_map = lambda x: jax.scipy.special.logit(x/params_max)

    unrestricted = {}
    if "pos" in params:
        unrestricted["pos"] = params["pos"]
    if "beta" in params:
        unrestricted["_beta"] = forward_map(params["beta"]/(dim-1))
    if "r" in params:
        unrestricted["_r"] = forward_map(params["r"])
    if "rho" in params:
        unrestricted["_rho"] = forward_map(params["rho"] - rho_min)
    if "nu" in params:
        # nu > 1
        unrestricted["_nu"] = forward_map(params["nu"]-1)
    return unrestricted


def from_unrestricted_params(dim: int, **params):
    """
    Maps each parameter from real numbers to the original scale using the expit function.

    This function works as the inverse of `to_unrestricted_params`. See the
    latter's documentation for additional details.

    Params:
        dim: embedding dimension used for beta scaling.
        **params: unrestricted parameters to remap.

    Returns:
        Parameters in their original scaling.
    """
    inv_map = lambda x: jax.scipy.special.expit(x)*params_max

    original_scale = {}
    if "pos" in params:
        original_scale["pos"] = params["pos"]
    if "_beta" in params:
        original_scale["beta"] = (dim-1)*inv_map(params["_beta"])
    if "_r" in params:
        original_scale["r"] = inv_map(params["_r"])
    if "_rho" in params:
        original_scale["rho"] = inv_map(params["_rho"]) + rho_min
    if "_nu" in params:
        original_scale["nu"] = inv_map(params["_nu"])+1
    return original_scale


def train(
    key,
    init_params: dict,
    grad_fn: Callable,
    optimizer: GradientTransformation,
    num_iter: int,
    callback: Optional[Callable] = None,
    hide_progress: bool = False,
):
    """
    Standard gradient descent optimization loop.

    Parameter values should be handled in a unrestricted space (R^n) for
    adequate optimization. The callback function can be used for custom
    progress tracking and saving parameters during optimization.

    Args:
        key: JAX PRNG key.
        init_params: initial parameters values in the unrestricted space.
        grad_fn(k, p):
            function where `k` is a JAX PRNG key and `p` is a dictionary of
            parameter values. Shall return objective function's gradients in a
            dictionary with keys identical to `p`.
        optimizer: Optax optimizer.
        num_iter: number of iterations to perform.
        callback(pbar, i, loss, params):
            Function called after each iteration, where `pbar` is the tqdm
            progress bar, `i` is the iteration index, `loss` is the current
            objective function's value and `params` contain the current
            parameter values.
        hide_progress: display progress bar.

    Returns:
        Optimized parameters.
    """
    params = init_params
    opt_state = optimizer.init(params)
    iter_key = key

    with tqdm(range(num_iter), desc="Training Progress", disable=hide_progress) as pbar:
        for i in pbar:
            iter_key, tmp_key = jax.random.split(iter_key)
            loss, grads = grad_fn(tmp_key, params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

            if callback:
                callback(pbar, i, loss, params)
    return params


def get_gradient_fn(
    edges: jnp.ndarray,
    graph: rx.PyGraph,
    use_exact_likelihood: bool,
    maximize_only_likelihood: bool,
    use_sd_model: bool=False,
):
    """
    Creates gradient function for maximum likelihood or maximum a posteriori fits.

    Args:
        edges: (m, 2) edges of the graph.
        graph: undirected graph.
        use_exact_likelihood: Use exact likelihood in optimization if True and stochastic likelihood estimator if False.
        maximize_only_likelihood: Maximum likelihood optimization if True and maximum a posteriori optimization if False.
        use_sd_model: Use S^D-1 model likelihood if True and H^D model likelihood if False.

    Returns:
        Gradient of the function to minimize during optimization.
    """
    dist = sd_hyperbolic_distance_stable if use_sd_model else hyperbolic_distance_stable
    eval_loglikelihood = partial(hyperbolic_loglikelihood, dist=dist, edges=edges) if use_exact_likelihood else partial(estimate_pair_loglikelihood, dist=dist)
    if maximize_only_likelihood:
        maximized_func = eval_loglikelihood
    else: # Otherwise maximize posterior distribution
        def maximized_func(pos, beta, r, rho, nu, *args, **kwargs):
            return eval_loglikelihood(pos=pos, beta=beta, r=r, *args, **kwargs) + prior_logdensity(pos=pos, beta=beta, r=r, rho=rho, nu=nu)

    def objective(params, **kwargs):
        n, dim = params["pos"].shape
        # Parameters are mapped back to their original values in the loss function
        return -maximized_func(**from_unrestricted_params(dim, **params), **kwargs)

    # Gradient
    exact_grad = jax.jit(jax.value_and_grad(objective))
    if use_exact_likelihood:
        grad_fn = jax.jit(lambda _key, params: exact_grad(params))
    else:
        edges = jnp.sort(edges) # Required for edge hashing in samplers
        pair_samplers = [
            get_neighborhood_bernoulli_sampler(
                graph, n_sampled_vertices=40, n_sampled_neighbors=200, edge_prob=0.1)
        ]
        @jax.jit
        def grad_fn(key, params):
            return estimate_gradient(key, pair_samplers, exact_grad, params)
    return grad_fn


def get_optimizer(
    name: str,
    pos_scheduler: optax.Schedule,
    other_scheduler: optax.Schedule,
    exact_exp: bool=True,
    fit_r: bool=True,
 ):
    """
    Create Optax optimizer.

    The `pos` parameter is optimized using the algorithm specified by `name`
    and other parameters are optimized using Adamw.

    Args:
        name:
            optimization algorithm for the `pos` parameter, among:
                - gd: Euclidean Adamw
                - rgd: Riemannian gradient descent
                - radam: Riemannian Adam
            Each algorithm supports its stochastic version by prepending its
            name with "s". Note that the stochastic optimizers are identical to
            exact optimizers (only the gradient function differs).
        pos_scheduler: learning rate scheduler of the `pos` optimizer.
        other_scheduler: learning rate scheduler of every other optimizer.
        exact_exp: Use exact exponential map in Riemannian algorithms if True
                   and use first-order approximation if False.
        fit_r: do not optimize parameter `r` (edge length parameter) if False.
    """
    # Position optimizer
    if name in ["rgd", "srgd"]:
        pos_optimizer = scale_by_rgd(pos_scheduler, exact_exp=exact_exp)
    elif name in ["radam", "sradam"]:
        pos_optimizer = scale_by_radam(pos_scheduler, exact_exp=exact_exp)
    elif name in ["gd", "sgd"]:
        pos_optimizer = optax.adamw(learning_rate=pos_scheduler)
    else:
        raise NotImplementedError(f"The {name} optimizer is not supported.")

    # Other parameters (beta, phi, rho, nu) optimizer
    fixed_parameters = lambda params: {name: not (name != "_r" or fit_r) for name in params}
    labels = lambda params: {name: ("pos" if name=="pos" else "other") for name in params}
    optimizer = optax.chain(
        optax.multi_transform(
            {
                "pos": pos_optimizer,
                "other": optax.adamw(learning_rate=other_scheduler),
            },
            labels,
        ),
        freeze(fixed_parameters)
    )
    return optimizer


# Manual integration of RGD and RAdam to Optax's optimization framework

class RGDState(NamedTuple):
    iteration: jnp.ndarray


def scale_by_rgd(schedule, exact_exp, label="pos"):
    def init_fn(params):
        return RGDState(
            iteration=jnp.zeros([], jnp.int32)
        )

    @jax.jit
    def _update_rgd(pos, iteration, grad):
        x = expand_positions(pos)
        lr = schedule(iteration)
        rg = to_riemannian_gradient(x, grad)
        step = project_tangent(x, -lr*rg)
        x_new = exp_map(x, step)[:, :-1] if exact_exp\
                    else (x + step)[:, :-1]

        # Handle unstable behavior
        x_new = jnp.clip(x_new, min=-1e9, max=1e9)
        nan_mask = jnp.any(jnp.isnan(x_new), axis=1)
        x_new = jnp.where(nan_mask[:, None], pos, x_new)
        return x_new

    def update_fn(grads, state, params):
        updates = {name: None for name in grads}
        updates[label] = _update_rgd(params[label], state.iteration, grads[label]) - params[label]
        return updates, RGDState(state.iteration + 1)

    return optax.GradientTransformation(
        init_fn,
        update_fn,
    )


class RAdamState(NamedTuple):
    iteration: jnp.ndarray
    m: jnp.ndarray
    v: jnp.ndarray


def scale_by_radam(schedule, exact_exp, beta1=0.9, beta2=0.999, eps=1e-6, label="pos"):
    def init_fn(params):
        pos = params[label]
        n, d = pos.shape

        return RAdamState(
            iteration=jnp.zeros([], jnp.int32),
            m=jnp.zeros((n, d + 1)),
            v=jnp.zeros((n, 1)),
        )

    @jax.jit
    def _radam_update(
        pos,
        grad,
        iteration,
        m,
        v,
    ):
        x_full = expand_positions(pos)
        rgrad = to_riemannian_gradient(x_full, grad)

        # Momentum moving average
        m = beta1 * m + (1 - beta1) * rgrad
        # Gradient norm moving average
        grad_sq_norm = lorentz_dot(rgrad, rgrad)[:, None]
        grad_sq_norm = jnp.maximum(grad_sq_norm, 0.) # for numerical stability
        v = beta2 * v + (1 - beta2) * grad_sq_norm

        # step vector
        m_hat = m / (1 - beta1**(iteration+1))
        v_hat = v / (1 - beta2**(iteration+1))

        lr = schedule(iteration)
        step = m_hat / (jnp.sqrt(v_hat) + eps)
        step = project_tangent(x_full, -lr*step) # for numerical stability

        # Apply exponential map and keep only spatial part
        x_new = exp_map(x_full, step)[:, :-1] if exact_exp\
                    else (x_full + step)[:, :-1]

        # Prevent vertices from being too far from origin
        x_new = jnp.clip(x_new, min=-1e9, max=1e9)
        # Reset state when position diverges
        nan_mask = jnp.any(jnp.isnan(x_new), axis=1)
        x_new = jnp.where(nan_mask[:, None], pos, x_new)
        m = jnp.where(nan_mask[:, None], 0.0, m)
        v = jnp.where(nan_mask[:, None], 0.0, v)

        # Parallel transport momentum to new position
        m = parallel_transport(x_full, expand_positions(x_new), m)
        return x_new, iteration+1, m, v


    def update_fn(grads, state, params):
        x_new, iteration, m, v = _radam_update(
            params[label],
            grads[label],
            state.iteration,
            state.m,
            state.v,
        )

        updates = {name: None for name in grads}
        updates[label] = x_new - params[label]
        return updates, RAdamState(
            iteration=iteration,
            m=m,
            v=v,
        )

    return optax.GradientTransformation(
        init_fn,
        update_fn,
    )


def _optimize(f,
             low=-7,
             high=2,
             n_coarse=10,
             maxiter=20):
    """
    Minimize an expensive objective f(log10_x).

    Parameters
    ----------
    f : callable
        Objective taking log10(parameter).
    low, high : float
        Search interval in log10-space.
    n_coarse : int
        Number of coarse evaluations.

    Returns
    -------
    log10_x_best : float
    """
    xs = jnp.linspace(low, high, n_coarse)
    fs = jnp.array([f(x) for x in xs])

    i = jnp.argmin(fs)

    # minimum appears on boundary
    if i == 0:
        return xs[0]
    elif i == len(xs) - 1:
        return xs[-1]

    # ---- local refinement
    left = xs[i - 1]
    right = xs[i + 1]

    res = minimize_scalar(
        f,
        bounds=(left, right),
        method="bounded",
        options={"xatol": 0.05, "maxiter": maxiter},

    )
    return res.x


def optimize_quantization(fit, edges, delta=1e-5):
    def get_quantization(log10_dx):
        quantization = {key: delta for key in ["beta", "r", "nu", "rho"]}
        quantization["pos"] = 10**log10_dx
        return quantization

    def loss_func(log10_dx):
        return hyperbolic_description_length(
                pos=fit["pos"], beta=fit["beta"], r=fit["r"], rho=fit["rho"], nu=fit["nu"],
                edges=edges, quantization=get_quantization(log10_dx)
        )
    # Search in log10 space
    res = _optimize(f=loss_func, low=-7, high=2)
    return get_quantization(res)
