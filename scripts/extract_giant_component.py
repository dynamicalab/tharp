from argparse import ArgumentParser
import os

import numpy as np
import rustworkx as rx


def main():
    parser = ArgumentParser()
    parser.add_argument("graph_path", type=str, help="Edge list of the graph to analyze.")
    args = parser.parse_args()

    # graph = rx.PyGraph(multigraph=False)
    # with open(args.graph_path, "r") as file_stream:
        # nodes = {}
        # for line in file_stream.readlines():
            # if line[0] == "#":
                # continue
            # u, v = map(int, line.split())
            # if u not in nodes:
                # nodes[u] = len(nodes)
                # graph.add_node(u)
            # if v not in nodes:
                # nodes[v] = len(nodes)
                # graph.add_node(v)
            # graph.add_edge(nodes[u], nodes[v], None)
    graph = rx.PyGraph.read_edge_list(args.graph_path, deliminator=" ", comment="#", multigraph=False, allow_self_loops=False, labels=True)

    components = rx.connected_components(graph)
    if len(components) == 1:
        print("Graph is already connected.")
        # return

    giant_component_id = np.argmax(list(map(len, components)))
    connected_subgraph = graph.subgraph(list(components[giant_component_id]))

    output_file, ext = os.path.splitext(args.graph_path)
    with open(output_file+"_GC" + ext, "w") as file_stream:
        for u, v in connected_subgraph.edge_list():
            file_stream.write(f"{u} {v}\n")

if __name__ == "__main__":
    main()
