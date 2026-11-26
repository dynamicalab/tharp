"""
Dimension selection results.

Authors:
- JGY <jyoung22@uvm.edu>
- Ariane Lizotte <ariane.lizotte.2@ulaval.ca>
"""
from argparse import ArgumentParser
import json
import os
import pickle
import time
import warnings

import numpy as np
import jax
import rustworkx as rx
from jax import numpy as jnp
import optax

from mdl_hyperbolic.optimize import from_unrestricted_params, get_gradient_fn, get_optimizer, to_unrestricted_params, train
from mdl_hyperbolic.laplacian_eigenmap import laplacian_eigenmap


def can_create_file(filename):
    base_dir = os.path.dirname(filename)
    return (os.path.isdir(base_dir) or base_dir == "") and os.access(base_dir, os.W_OK)


def no_source_formatwarning(message, category, filename, lineno, line=None):
    return f"{filename}:{lineno}: {category.__name__}: {message}\n"

warnings.formatwarning = no_source_formatwarning


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, jnp.ndarray) or isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


jax.config.update('jax_enable_x64', True) # enable double precision floating-point

def main():
    parser = ArgumentParser()
    parser.add_argument("graph_path", type=str,
                        help="JSON file that contains the edge list of the graph to analyze.")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Path of the json output file. Default: graph_path file name appended with \"-<alg_name>-fits.json\".")
    parser.add_argument("-a", "--algorithm", choices=["gd", "rgd", "radam", "sgd", "srgd", "sradam"],
                        help="Optimization algorithm.")
    parser.add_argument("-s", "--seed", type=int, default=0,
                        help="Seed of the random generator used for the stochastic gradient descent.")
    parser.add_argument("--dims", type=int, default=list(range(2, 9)), nargs="+",
                        help="List of dimensions to fit.")
    parser.add_argument("--num-iter", type=int, default=10_000,
                        help="Number of iterations used in the maximization algorithm.")
    parser.add_argument("--init", type=str,
                        help="JSON file containing initial parameters and embedding to use in each dimension. "
                             "If no initialization is provided, positions are initialized using laplacian eigenmap.")
    parser.add_argument("--mle", action="store_true", default=False,
                        help="Maximize only likelihood instead of the full Bayesian model posterior.")
    parser.add_argument("--skip-le", action="store_true", default=False,
                        help="Use available laplacian eigenmap if available.")
    parser.add_argument("--save-every", type=int, default=None,
                        help="Write embedding every n-th iteration, where n is the value of this parameter.")
    parser.add_argument("--no-progress", action="store_true", default=False,
                        help="Hide progress bar.")
    parser.add_argument("--sd-model", action="store_true", default=False,
                        help="Use approximate hyperbolic distance corresponding to the S^D-1 model.")
    parser.add_argument("--pos-learning-rate", type=float, help="Learning rate of the position optimizer.")
    parser.add_argument("--other-learning-rate", type=float, help="Learning rate of other paramaters' optimizer.")
    parser.add_argument("--learning-rate-scale", type=float, default=1., help="Multiply all learning rates with this value.")
    args = parser.parse_args()

    file_name = os.path.splitext(args.graph_path)[0]
    output_file_name = file_name+f"-{args.algorithm}-fits{'-sd' if args.sd_model else ''}.json" if args.output is None else args.output
    if os.path.isfile(output_file_name):
        warnings.warn(f"The output file \"{output_file_name}\" exists. New fits will overwrite current ones.")
        with open(output_file_name, "r") as file_stream:
            fits = json.load(file_stream)
    elif not can_create_file(output_file_name):
        raise OSError(f"Cannot create the output file {output_file_name}. Verify that the directory exists and that writing is permitted.")
    else:
        fits = {"fits": {}}
    fits["metadata"] = {"start time": time.ctime(time.time())}

    edges = np.loadtxt(args.graph_path, dtype=np.int32)
    edges = jnp.asarray(np.unique(edges, axis=0))
    graph = rx.PyGraph.read_edge_list(args.graph_path, deliminator=" ", comment="#", multigraph=False, allow_self_loops=False)
    n = graph.num_nodes()

    # Setup objective function
    grad_fn = get_gradient_fn(
            edges=edges, graph=graph,
            use_exact_likelihood=args.algorithm in ["gd", "rgd", "radam"],
            maximize_only_likelihood=args.mle,
            use_sd_model=args.sd_model,
        )

    # Default learning rates
    if args.pos_learning_rate:
        pos_lr = args.pos_learning_rate
    else:
        match args.algorithm:
            case "rgd" | "srgd":
                pos_lr = 0.001
            case "radam" | "sradam":
                pos_lr = 0.01
            case "gd" | "sgd":
                pos_lr = 0.005
            case _:
                raise NotImplementedError(f"The {args.algorithm} optimizer is not supported.")
    other_lr = args.other_learning_rate if args.other_learning_rate else 0.01
    lr_scale = args.learning_rate_scale

    pos_scheduler = optax.cosine_decay_schedule(init_value=lr_scale*pos_lr, decay_steps=args.num_iter, alpha=0.01)
    other_scheduler = optax.cosine_decay_schedule(init_value=lr_scale*other_lr, decay_steps=args.num_iter, alpha=0.01)

    optimizer = get_optimizer(args.algorithm, pos_scheduler=pos_scheduler, other_scheduler=other_scheduler, exact_exp=True)

    if args.init:
        with open(args.init, "r") as file_stream:
            init_states = json.load(file_stream)["fits"]

        param_names = set(["pos", "beta", "r"]) if args.mle else set(["pos", "beta", "r", "rho", "nu"])
        def get_init(dim):
            init_params = {key: init_states[str(dim)].get(key) for key in param_names}
            if any(map(lambda value: value is None, init_params.values())):
                raise ValueError("Missing parameters in initialization.")
            return init_params
    else:
        num_components = rx.number_connected_components(graph)
        def get_init(dim):
            return laplacian_eigenmap(args.graph_path, graph, dim, num_components, skip=args.skip_le)

    # Fit
    for dim in args.dims:
        init_params = get_init(dim)
        init_params["pos"] = jnp.asarray(init_params["pos"])
        if args.mle:
            # rho cannot be inferred only from the likelihood
            if "rho" in init_params:
                del init_params["rho"]
            if "nu" in init_params:
                del init_params["nu"]
        else:
            # If rho is undefined, it's initialized to r
            if "rho" not in init_params:
                init_params["rho"] = init_params["r"]
            if "nu" not in init_params:
                init_params["nu"] = 1.5

        # Parameters must be real numbers in the optimizers.
        unrestricted_init_params = to_unrestricted_params(dim, **init_params)
        del init_params # Free memory

        if args.save_every:
            steps_directory = os.path.join(f"{file_name}_steps_{args.algorithm}", str(dim))
            if os.path.isfile(steps_directory):
                raise OSError("Steps directory is already a file")
            elif os.path.isdir(steps_directory):
                with os.scandir(steps_directory) as it:
                    if any(it):
                        warnings.warn(f"Output directory for steps \"{steps_directory}\" already exists. Its content will be overwritten.")
            else:
                os.makedirs(steps_directory)

        # Displays information about the embedding during optimization and saves to `steps_directory`
        # the parameters during training if specified in the program arguments.
        if args.no_progress:
            callback = lambda *args, **kwargs: None
        else:
            def callback(pbar, iteration, loss, _params):
                n, dim = _params["pos"].shape
                params = from_unrestricted_params(dim, **_params)
                beta, r = params["beta"], params["r"]
                if args.mle:
                    pbar.set_description(f"Loss: {loss:.4f}, Dim:{dim}, Params: {beta=:.2f} {r=:.2f}")
                else:
                    nu, rho = params["nu"], params["rho"]
                    pbar.set_description(f"Loss: {loss:.4f}, Dim:{dim}, Params: {beta=:.2f} {r=:.2f} {nu=:.2f} {rho=:.2f}")
                if args.save_every and iteration % args.save_every == 0 or iteration==args.num_iter:
                    with open(os.path.join(steps_directory, f"embedding_{iteration}.pkl"), "wb") as file_stream:
                        pickle.dump(params, file_stream)

        unrestricted_params = train(
                key=jax.random.PRNGKey(args.seed),
                init_params=unrestricted_init_params,
                grad_fn=grad_fn,
                optimizer=optimizer,
                num_iter=args.num_iter,
                callback=callback,
                hide_progress=args.no_progress,
            )

        # Output is saved in the original parameter scaling
        fits["fits"][str(dim)] = from_unrestricted_params(dim, **unrestricted_params)
        fits["fits"][str(dim)]["pos"] = fits["fits"][str(dim)]["pos"].tolist() # Frees memory from GPU

    fits["metadata"]["end time"] = time.ctime(time.time())
    fits["metadata"]["args"] = vars(args)

    with open(output_file_name, "w") as file_stream:
        json.dump(fits,
            file_stream,
            cls=NumpyEncoder
        )

if __name__ == "__main__":
    main()
