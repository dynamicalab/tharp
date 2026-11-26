import os
import json
from argparse import ArgumentParser

import jax
import numpy as np
import rustworkx as rx

from mdl_hyperbolic.models import estimate_parameters_for_uniform
from mdl_hyperbolic.generators import sample_quasiuniform_hyperbolic_ball


jax.config.update('jax_enable_x64', True) # enable double precision floating-point


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, jax.numpy.ndarray) or isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)



if __name__ == "__main__":
    overwritten_text = "Overwritten when options -k and -g are used."
    parser = ArgumentParser()
    parser.add_argument("output_path", type=str, help="Path to the json file that will contain the synthetic graph's information.")
    parser.add_argument("number_of_vertices", type=int, help="Number of vertices of the synthetic graph.")
    parser.add_argument("dimension", type=int, help="Dimension of the synthetic graph.")
    parser.add_argument("-b", "--beta", type=float, default=2.5, help="Inverse temperature in the H^D model.")
    parser.add_argument("-s", "--seed", type=int, default=42, help="Seed used to generate the coordinates and the graph.")
    parser.add_argument("-r", type=float, default=5., help="Length scale of edges in the H^D model." + overwritten_text)
    parser.add_argument("--ball-r", type=float, default=5., help="Radius of the hyperbolic ball from which coordinates are drawn." + overwritten_text)
    parser.add_argument("-a", "--alpha", type=float, default=5., help="Distortion parameter of the radii uniform distribution. Controls the power-law degree distribution's exponent (gamma)." + overwritten_text)
    parser.add_argument("--info", action="store_true", help="Print synthetic graph properties.")
    parser.add_argument("--ensure-connected", action="store_true", help="Resample until the graph is connected.")

    parser.add_argument("-g", "--gamma", type=float, help="Power-law degree distribution negative exponent. Must be used with --expected-degree.")
    parser.add_argument("-k", "--expected-degree", type=float, help="Expected degree. Must be used with --gamma.")
    args = parser.parse_args()

    if bool(args.gamma)^bool(args.expected_degree):
        raise parser.error("Only gamma or expected degree was provided. Either both or neither must be specified.")

    n = args.number_of_vertices
    dim = args.dimension
    beta = args.beta

    if args.gamma and args.expected_degree: # Ignore alpha, r and ball_r specified in the arguments
        r, ball_r, alpha = estimate_parameters_for_uniform(n, dim, beta, args.gamma, args.expected_degree)
    else:
        r = args.r
        ball_r = args.ball_r
        alpha = args.alpha

    key = jax.random.PRNGKey(args.seed)
    is_connected = None
    for _ in range(500):
        key, param_key, graph_key = jax.random.split(key, 3)
        positions = sample_quasiuniform_hyperbolic_ball(dim, ball_r, param_key, num_samples=n, alpha=alpha)

        seed_int = int(jax.random.randint(graph_key, (), 0, 2**31 - 1))
        g = rx.hyperbolic_random_graph(positions, r, beta, seed=seed_int)

        if not args.ensure_connected:
            break
        elif is_connected := rx.is_connected(g):
            break
    else:
        raise RuntimeError("Could not generate a connected graph for the given parameters. Adjust the parameters or change the seed.")

    if args.info:
        if is_connected is None:
            is_connected = rx.is_connected(g)
        print("Connected:\t{}".format(is_connected))
        print("Av. degree:\t{:.3f}".format(sum([g.degree(u) for u in g.node_indexes()])/n))
        print("Edge density:\t{:.3f}".format(g.num_edges()*2/(n*(n-1))))
        print("Transitivity:\t{:.3f}".format(rx.transitivity(g)))


    g.write_edge_list(args.output_path)
    prop_file = os.path.splitext(args.output_path)[0] + "-parameters.json"
    with open(prop_file, "w") as file_stream:
        json.dump(
            {
                "pos": positions,
                "beta": beta,
                "r": r,
                "args": vars(args)
            },
            file_stream,
            cls=NumpyEncoder
        )
