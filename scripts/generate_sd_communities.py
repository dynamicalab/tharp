import json
import os
from argparse import ArgumentParser

import jax
import scipy
import numpy as np
import rustworkx as rx

from mdl_hyperbolic.generators import sample_hypersphere_normal, sample_uniform_cap
from mdl_hyperbolic.models import sd_connection_prob_pairs, map_sd_to_hyperbolic_model


jax.config.update('jax_enable_x64', True) # enable double precision floating-point


if __name__ == "__main__":
    parser = ArgumentParser(description="Samples a graph from the S^D-1 model with a community structure. One community is placed at each intersection with one axis (2^D-1 communities).")
    parser.add_argument("dim", type=int)
    parser.add_argument("--beta", type=float, default=2.5, help="Value of beta/(dim-1)")
    parser.add_argument("--kappa0", type=float, default=8)
    parser.add_argument("--average-kappa", type=float, default=20)
    parser.add_argument("--n-per-comm", type=int, default=200, help="Number of vertices per community.")
    # 0.25
    parser.add_argument('--normal-std', type=float, help="Standard deviation of normal. This parameter or --unif-cap must be set.")
    # 0.7
    parser.add_argument('--unif-cap', type=float, help="Spherical cap max angular separation. This parameter or --normal-std must be set.")
    parser.add_argument("-o", "--output", type=str, default="synth_sd_communities.txt", help="Edge list file name. The original parameters of the graph are saved to this file name with suffix '-parameters.json'.")
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("--ensure-connected", action="store_true", help="Sample until the graph is connected.")
    args = parser.parse_args()

    if not ((args.normal_std is None) ^ (args.unif_cap is None)):
        parser.error("Only one of the flags --normal-std and --unif-cap must be set.")

    num_dims = args.dim
    beta_div_d = args.beta # Normalized beta

    kappa0 = args.kappa0
    average_kappa = args.average_kappa
    d = num_dims-1
    beta = d*beta_div_d

    key = jax.random.PRNGKey(args.seed)
    np.random.seed(args.seed)

    community_centers = [] # On unit d-sphere
    for d in range(num_dims):
        for x in [-1, 1]:
            p = np.zeros(num_dims)
            p[d] = x
            community_centers.append(p)
    community_sizes = [args.n_per_comm]*len(community_centers)

    for _ in range(30):
        key, graph_key, *param_keys = jax.random.split(key, 2+len(community_sizes))
        community_labels = []
        positions = []
        for k in range(len(community_sizes)):
            size = community_sizes[k]
            if args.normal_std:
                community_pos = sample_hypersphere_normal(param_keys[k], center=community_centers[k], sigma=args.normal_std, num_samples=size)
            elif args.unif_cap:
                community_pos = sample_uniform_cap(param_keys[k], center=community_centers[k], theta=args.unif_cap, num_samples=size)
            else:
                raise ValueError("Distribution not set")
            positions.append(community_pos)
            community_labels += [k]*size
        positions = np.concatenate(positions)
        n = positions.shape[0]
        assert n == sum(community_sizes)

        sphere_radius = (n/(2*np.pi**((d+1)/2)) * scipy.special.gamma((d+1)/2))**(1/d)
        mu = beta*scipy.special.gamma(d/2)*np.sin(d*np.pi/beta) / (2*np.pi**(1+d/2)*average_kappa)

        gamma = average_kappa/(average_kappa-kappa0)
        kappas = kappa0*scipy.stats.pareto.rvs(gamma, size=n)

        # indices for upper triangle without diagonal
        iu, iv = np.triu_indices(n, k=1)

        p_upper = sd_connection_prob_pairs(positions[iu], positions[iv], kappas[iu], kappas[iv], beta, mu, sphere_radius)
        edges_upper = (np.random.rand(len(p_upper)) < p_upper).astype(int)

        A = np.zeros((n, n), dtype=int)
        A[iu, iv] = edges_upper
        A[iv, iu] = edges_upper  # make symmetric

        g = rx.PyGraph.from_adjacency_matrix(A.astype(np.double))
        if not args.ensure_connected:
            break
        elif is_connected := rx.is_connected(g):
            break
    else:
        raise RuntimeError("Could not generate a connected graph for the given parameters. Adjust the parameters or change the seed.")

    print("Connected:\t{}".format(rx.is_connected(g)))
    print("Av. degree:\t{:.3f}".format(sum([g.degree(u) for u in g.node_indexes()])/n))
    print("Edge density:\t{:.3f}".format(g.num_edges()*2/(n*(n-1))))
    print("Transitivity:\t{:.3f}".format(rx.transitivity(g)))

    hd_pos, hd_r = map_sd_to_hyperbolic_model(positions, kappas, mu, kappa0, sphere_radius)

    output_path = args.output
    print(f"\nWriting graph to {output_path}.")
    g.write_edge_list(output_path)
    prop_file = os.path.splitext(output_path)[0] + "-parameters.json"
    print(f"Writing parameters to {prop_file}.")
    with open(prop_file, "w") as file_stream:
        json.dump(
            {
                "pos": hd_pos.tolist(),
                "beta": beta,
                "r": float(hd_r),
                "labels": community_labels,
                "kappa": kappas.tolist(),
                "mu": mu,
                "kappa0": kappa0,
                "r_s": sphere_radius,
                "metadata": {"args": vars(args)},
            },
            file_stream,
        )
