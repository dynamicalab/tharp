import os
from subprocess import call
from time import perf_counter
from tqdm import tqdm
import json


run = lambda cmd: call(cmd, shell=True)
lrange = lambda *args: list(range(*args))
output_file = ".output/benchmark/results.json"


algs = ["dmercator", "gd", "sgd", "rgd", "srgd", "radam", "sradam"]
if os.path.isfile(output_file):
    with open(output_file, "r") as file_stream:
        results = json.load(file_stream)
    ns = results["ns"]
    dims = lrange(2, 6)  # Inferred
    results["dims"] = dims
else:
    ns = lrange(100, 1000, 100) + lrange(1000, 5001, 1000)
    dims = lrange(2, 6)  # Inferred
    results = {"ns": ns, "dims": dims}


print("Sampling graph benchmarks")
graph_format = ".output/benchmark/graph{n}.txt"

if not os.path.isdir(".output/benchmark"):
    os.makedirs(".output/benchmark")
gen_command = f"JAX_PLATFORMS='cpu' python scripts/generate_hd_graph.py {graph_format} {{n}} 2 -s {{n}} -b 2.5 -g 2.7 -k 20 --ensure-connected"
for n in tqdm(ns):
    if not os.path.isfile(graph_format.format(n=n)):
        run(gen_command.format(n=n))


generic_command = f"python scripts/fit_posterior.py {graph_format} --dims {{d}} -a {{alg}} --num-iter 10000"
dmercator_command = f"python scripts/fit_dmercator.py {graph_format} --dims {{d}}"

for alg in algs:
    if alg in results and all(len(ts) == len(ns) for ts in results[alg].values()):
        print(f"Skipping {alg} because benchmarking was already done.\n")
        continue

    print(f"Benchmarking {alg}")
    results[alg] = {d: [] for d in dims}
    for d in dims:
        print(f"Dim={d}")
        for n in ns:
            start = perf_counter()
            if alg == "dmercator":
                run(dmercator_command.format(n=n, d=d))
            else:
                run(generic_command.format(n=n, d=d, alg=alg))
            results[alg][d].append(perf_counter() - start)
        print()

    with open(output_file, "w") as file_stream:
        json.dump(results, file_stream)
