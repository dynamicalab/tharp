import os
import json
from argparse import ArgumentParser
import jax.numpy as jnp


TARGET_DIMS = list(range(2, 16))

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


if __name__=="__main__":
    parser = ArgumentParser(description="This script identifies which fit failed (yield NaN parameters) or were not done. The script can check files directories.")
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()

    for path in args.paths:
        res = {}
        if os.path.isdir(path):
            files = os.listdir(path)
            dir = path
        elif os.path.isfile(path):
            files = [path]
            dir = ""
        else:
            print(f"{path} doesn't exist.")
            continue

        for file in files:
            if not file.endswith("-fits.json"):
                continue

            with open(os.path.join(dir, file), "r") as file_stream:
                fits = json.load(file_stream).get("fits")

            if fits is None:
                missing_dims = TARGET_DIMS
            else:
                missing_dims = []
                for dim in TARGET_DIMS:
                    fit = fits.get(str(dim))
                    if fit is None or any(jnp.any(jnp.isnan(jnp.asarray(val))) for val in fit.values()):
                        missing_dims.append(dim)

            if len(missing_dims) > 0:
                res[file] = missing_dims
                print(bcolors.FAIL + file, str(missing_dims) + bcolors.ENDC)
            else:
                print(file, str(missing_dims))
