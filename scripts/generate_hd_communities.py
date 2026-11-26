import os
import json
from argparse import ArgumentParser

import jax
import numpy as np
import rustworkx as rx

from mdl_hyperbolic.generators import sample_quasiuniform_hyperbolic_ball
from mdl_hyperbolic.geometry import hyperbolic_translate


jax.config.update('jax_enable_x64', True) # enable double precision floating-point

if __name__ == "__main__":
    parser = ArgumentParser(description="Samples a graph from the H^D model with a community structure.")
    parser.add_argument("dim", type=int)
    parser.add_argument("--beta", type=float, default=2.5, help="Value of beta/(dim-1)")
    parser.add_argument("-r", type=float, default=2., help="Length scale of edges in the H^D model.")
    parser.add_argument("--n-per-comm", type=int, default=200, help="Number of vertices per community.")
    parser.add_argument("--community-dist", type=float, default=2.3, help="Community center's distance to origin.")
    parser.add_argument("-o", "--output", type=str, default="synth_sd_communities.txt", help="Edge list file name. The original parameters of the graph are saved to this file name with suffix '-parameters.json'.")
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("--show", action="store_true", help="Display embedding and match between expected (kappas) and actual degrees. Only works for dim<=3.")
    args = parser.parse_args()

    dim = args.dim
    key = jax.random.PRNGKey(args.seed)

    beta_div_d = args.beta # Normalized beta
    r = args.r

    community_r = args.community_dist # Distance of the communities' center to the origin
    # Contains unit vectors pointing in direction of the community center
    community_direction = []
    for d in range(dim):
        for x in [-1, 1]:
            p = np.zeros(dim)
            p[d] = x
            community_direction.append(p)
    community_sizes = [args.n_per_comm]*len(community_direction)

    beta = (dim-1)*beta_div_d

    graph_key, *param_keys = jax.random.split(key, 1+len(community_sizes))
    community_labels = []
    positions = []
    for k in range(len(community_sizes)):
        size = community_sizes[k]
        community_pos = sample_quasiuniform_hyperbolic_ball(dim, community_r, param_keys[k], num_samples=size)
        positions.append(hyperbolic_translate(community_pos, community_direction[k], .8*community_r))
        community_labels += [k]*size
    positions = np.concatenate(positions)
    n = positions.shape[0]
    assert n == sum(community_sizes)

    if args.show:
        if dim > 3:
            print(f"Cannot display embedding in dimension > 3.")
        else:
            from matplotlib import pyplot as plt
            from matplotlib.colors import ListedColormap
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            cmap = ListedColormap(['red', 'blue', 'green', 'orange', 'pink', 'blueviolet', 'darkturquoise', 'saddlebrown'])
            ax.scatter(*positions[:, :dim].T, c=community_labels, cmap=cmap)
            ax.set_aspect("equal")
            plt.show()


    seed_int = int(jax.random.randint(graph_key, (), 0, 2**31 - 1))
    g = rx.hyperbolic_random_graph(positions, r, beta, seed=seed_int)

    is_connected = rx.is_connected(g)
    print("Connected:\t{}".format(is_connected))
    print("Av. degree:\t{:.3f}".format(sum([g.degree(u) for u in g.node_indexes()])/n))
    print("Edge density:\t{:.3f}".format(g.num_edges()*2/(n*(n-1))))
    print("Transitivity:\t{:.3f}".format(rx.transitivity(g)))


    output_path = args.output
    print(f"\nWriting graph to {output_path}.")
    g.write_edge_list(output_path)
    prop_file = os.path.splitext(output_path)[0] + "-parameters.json"
    print(f"Writing parameters to {prop_file}.")
    with open(prop_file, "w") as file_stream:
        json.dump(
            {
                "pos": positions.tolist(),
                "beta": beta,
                "r": r,
                "labels": community_labels,
                "metadata": {"args": vars(args)},
            },
            file_stream,
        )
