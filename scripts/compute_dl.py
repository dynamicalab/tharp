"""
Dimension selection results.

Authors:
- JGY <jyoung22@uvm.edu>
- Ariane Lizotte <ariane.lizotte.2@ulaval.ca>
"""
from argparse import ArgumentParser
from copy import deepcopy
import json
import warnings

import numpy as np
import jax
from jax import numpy as jnp

from mdl_hyperbolic.models import hyperbolic_description_length
from mdl_hyperbolic.optimize import optimize_quantization


def no_source_formatwarning(message, category, filename, lineno, line=None):
    return f"{filename}:{lineno}: {category.__name__}: {message}\n"

warnings.formatwarning = no_source_formatwarning


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, jax.numpy.ndarray) or isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


jax.config.update('jax_enable_x64', True) # enable double precision floating-point



def main():
    parser = ArgumentParser(description="Computes the description length of the hyperbolic model.")
    parser.add_argument("graph_path", type=str, help="Edge list text file.")
    parser.add_argument("fit_path", type=str, help="JSON fit file.")
    parser.add_argument("--delta", type=float, default=1e-5, help="Bin size for all parameters except positions.")
    args = parser.parse_args()

    edges = jnp.array(np.loadtxt(args.graph_path, dtype=np.int32))
    with open(args.fit_path, "r") as file_stream:
        data = json.load(file_stream)

    description_lengths = dict()
    delta = args.delta
    mendatory_parameters = ["pos", "beta", "r"]
    if "fits" not in data:
        print("The \"fits\" field was not found. Computing DL of root parameters if available.")
        if any([p not in data for p in mendatory_parameters]):
            raise ValueError("The provided file is missing embedding parameters. The file probably doesn't contain an embedding.")
        fits = {None: data}
    else:
        fits = data["fits"]

    for _fit in fits.values():
        fit = deepcopy(_fit) # prevent adding fields to original embedding
        fit["pos"] = jnp.array(fit["pos"])
        dim = len(fit["pos"][0])
        if "rho" not in fit:
            warnings.warn("The parameter rho was not found: r is used instead.")
            fit["rho"] = fit["r"]
        if "nu" not in fit:
            warnings.warn("The parameter nu was not found is set to 2.")
            fit["nu"] = 2.

        quantization = optimize_quantization(fit, edges, delta=delta)
        res = hyperbolic_description_length(pos=fit["pos"], beta=fit["beta"], r=fit["r"], rho=fit["rho"], nu=fit["nu"],
                                            edges=edges, quantization=quantization)
        description_lengths[dim] = (res, quantization)
        print(f"{dim}D: {res}")
    data["description length"] = description_lengths
    print("Dimension:", int(min(description_lengths, key=lambda k: description_lengths[k][0])))

    with open(args.fit_path, "w") as file_stream:
        json.dump(data,
            file_stream,
            cls=NumpyEncoder
        )

if __name__ == "__main__":
    main()
