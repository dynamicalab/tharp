from pathlib import Path
import os
import re
import tempfile

import numpy as np

from mdl_hyperbolic.models import map_sd_to_hyperbolic_model, optimize_kappa0


dmercator_path = Path(__file__).parent.joinpath("d-mercator/mercator")


# https://stackoverflow.com/questions/29418693/write-ranges-of-numbers-with-dashes
def _format_ranges(ints):
    seq = []
    final = []
    last = 0

    for index, val in enumerate(ints):
        if last + 1 == val or index == 0:
            seq.append(val)
            last = val
        else:
            if len(seq) > 1:
                final.append(str(seq[0]) + '-' + str(seq[len(seq)-1]))
            else:
                final.append(str(seq[0]))
            seq = []
            seq.append(val)
            last = val

        if index == len(ints) - 1:
            if len(seq) > 1:
                final.append(str(seq[0]) + '-' + str(seq[len(seq)-1]))
            else:
                final.append(str(seq[0]))

    final_str = ', '.join(map(str, final))
    return final_str


def check_for_missing_vertices(vertices, expected_n, algorithm_name):
    missing_vertices = set(np.arange(expected_n)) - set(vertices)
    if len(missing_vertices) > 0:
        missing_vertices = list(map(int, missing_vertices))
        sorted(missing_vertices)
        raise ValueError(f"Vertices {_format_ranges(missing_vertices)} "
                         f"are missing in {algorithm_name} embedding")


# From https://stackoverflow.com/questions/12643009/regular-expression-for-floating-point-numbers
float_regex = r"[+-]?(\d+([.]\d*)?([eE][+-]?\d+)?|[.]\d+([eE][+-]?\d+)?)"

def parse_dmercator_metadata(embedding_file):
    with open(embedding_file, "r") as file_stream:
        # First pass to determine dimension. If the line with "DIMENSION" doesn't exist
        # then the embedding is in 2 dimensions.
        inferred_dim = 2
        for line in file_stream.readlines():
            if (result := re.match(r"#\s+DIMENSION\s+(\d+)", line)):
                inferred_dim = int(result[1]) + 1
                break
        file_stream.seek(0)

        # The dimension changes the format for the inferred hyperbolic radius.
        r_name = "radius_H2" if inferred_dim == 2 else "radius_H\\^D\\+1"
        r_s_name = "radius_S1" if inferred_dim == 2 else "radius_S\\^D"

        beta = None
        r = None
        r_s = None
        mu = None
        for line in file_stream.readlines():
            if (result := re.match(f"#\\s+- beta:?\\s+({float_regex})", line)):
                beta = np.double(result[1])
            elif (result := re.match(f"#\\s+- {r_name}:?\\s+({float_regex})", line)):
                r = np.double(result[1])
            elif (result := re.match(f"#\\s+- {r_s_name}:?\\s+({float_regex})", line)):
                r_s = np.double(result[1])
            elif (result := re.match(f"#\\s+- mu:?\\s+({float_regex})", line)):
                mu = np.double(result[1])
    if beta is None or r is None or r_s is None or mu is None:
        raise ValueError("Couldn't read D-Mercator output file.")
    return inferred_dim, beta, r, r_s, mu


def embed_dmercator(edges, dim, fast=False):
    if dim < 2:
        raise ValueError(f"Cannot embed in D<{dim} dimensions.")

    if not os.path.isfile(dmercator_path):
        raise RuntimeError("D-Mercator executable not found.")

    if isinstance(edges, str):
        output_path = str(Path(edges).with_suffix('.inf_coord'))
        os.system(f"{dmercator_path} -q -d {dim-1} {'-f' if fast else ''} {edges}")
        embedding = np.loadtxt(output_path)
        check_for_missing_vertices(
            embedding[:, 0], np.max(embedding[:, 0])+1, "D-Mercator")
        _, beta, r, r_s, mu = parse_dmercator_metadata(output_path)
        if os.path.isfile(output_path):
            os.remove(output_path)
    else: # Create temp file with edge list
        fd, edgelist = tempfile.mkstemp()
        output_path = str(Path(edgelist).with_suffix('.inf_coord'))
        try:
            with os.fdopen(fd, 'w') as tmp:
                for u, v in edges:
                    tmp.write(f"{u} {v}\n")

            os.system(f"{dmercator_path} -q -d {dim-1} {'-f' if fast else ''} {edgelist}")
            embedding = np.loadtxt(output_path)
            check_for_missing_vertices(
                embedding[:, 0], np.max(embedding[:, 0])+1, "D-Mercator")
            _, beta, r, r_s, mu = parse_dmercator_metadata(output_path)
        finally:
            if os.path.isfile(output_path):
                os.remove(output_path)
            os.remove(edgelist)

    if r < 0:
        raise ValueError("Invalid D-Mercator result: negative hyperbolic radius.")

    if dim == 2:
        # When D=2, D-Mercator yields the angles.
        kappas = embedding[:, 1]
        angles_s_model = embedding[:, 2]
        sphere_pos = np.stack([np.cos(angles_s_model), np.sin(angles_s_model)]).T
    else:
        # When D>2, D-Mercator yields cartesian coordinates of the hypersphere.
        kappas = embedding[:, 1]
        # Ensure points are on unit sphere
        sphere_pos = embedding[:, 3:]/np.linalg.norm(embedding[:, 3:], axis=1)[:, None]

    kappa0 = optimize_kappa0(sphere_pos, kappas, beta, mu, r_s)
    pos, r = map_sd_to_hyperbolic_model(sphere_pos=sphere_pos, kappas=kappas, mu=mu, kappa0=kappa0, sphere_radius=r_s)
    return {"pos": pos, "beta": beta, "r": r, "r_s": r_s, "kappa": kappas, "mu": mu, "kappa0": kappa0}
