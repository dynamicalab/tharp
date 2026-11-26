from argparse import ArgumentParser
import json
import os
import time
import warnings

import numpy as np
import jax
import rustworkx as rx
from tqdm import tqdm

import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.resolve()))
from dmercator_api import embed_dmercator


jax.config.update('jax_enable_x64', True) # enable double precision floating-point


def can_create_file(filename):
    base_dir = os.path.dirname(filename)
    return (os.path.isdir(base_dir) or base_dir == "") and os.access(base_dir, os.W_OK)


def no_source_formatwarning(message, category, filename, lineno, line=None):
    return f"{filename}:{lineno}: {category.__name__}: {message}\n"

warnings.formatwarning = no_source_formatwarning


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray) or isinstance(o, jax.numpy.ndarray):
            return o.tolist()
        return super().default(o)


def main():
    parser = ArgumentParser()
    parser.add_argument("graph_path", type=str, help="Text file containing the edge list.")
    parser.add_argument("--dims", type=int, nargs="+", help="Dimensions to fit.", required=True)
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Path of the json output file. Default: graph_path file name appended with \"-<alg_name>-fits.json\".")
    args = parser.parse_args()

    graph = rx.PyGraph.read_edge_list(args.graph_path, deliminator=" ", comment="#", multigraph=False, allow_self_loops=False)
    if not rx.is_connected(graph):
        print("Graph is not connected. Cannot run D-Mercator.", file=sys.stderr)
        exit()

    file_name = os.path.splitext(args.graph_path)[0]
    output_file_name = file_name+f"-dmercator-fits.json" if args.output is None else args.output
    if os.path.isfile(output_file_name):
        warnings.warn(f"The output file \"{output_file_name}\" exists. New fits will overwrite current ones.")
        with open(output_file_name, "r") as file_stream:
            fits = json.load(file_stream)
    elif not can_create_file(output_file_name):
        raise OSError(f"Cannot create the output file {output_file_name}. Verify that the directory exists and that writing is permitted.")
    else:
        fits = {"fits": {}}
    fits["metadata"] = {"start time": time.ctime(time.time())}

    for d in tqdm(args.dims):
        fits["fits"][str(d)] = embed_dmercator(args.graph_path, d)
    fits["metadata"]["end time"] = time.ctime(time.time())
    fits["metadata"]["args"] = vars(args)

    with open(output_file_name, "w") as file_stream:
        json.dump(fits,
            file_stream,
            cls=NumpyEncoder
        )

if __name__ == "__main__":
    main()
