from argparse import ArgumentParser
import os
import sys
import xml.etree.ElementTree as ET
from xml.dom import minidom

import json
import numpy as np


def write_graphml(edges, params, output_path):
    embedding = params["pos"]
    graph_params = {name: value for name, value in params.items() if name != "pos"}
    n, dim = embedding.shape

    root = ET.Element('graphml', {
        'xmlns': 'http://graphml.graphdrawing.org/xmlns',
        'xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance',
        'xsi:schemaLocation': (
            'http://graphml.graphdrawing.org/graphml '
            'http://graphml.graphdrawing.org/graphml/graphml.xsd'
        )
    })

    # Separate scalar graph params from per-node array params
    scalar_params = {}
    node_array_params = {}
    for name, val in graph_params.items():
        if isinstance(val, (list, np.ndarray)) and len(val) == n:
            node_array_params[name] = val
        elif isinstance(val, (int, float, bool, str)):
            scalar_params[name] = val
        # else: unsupported shape — skip silently

    # Keys for scalar graph-level attributes
    for i, (name, val) in enumerate(scalar_params.items()):
        attr_type = (
            'boolean' if isinstance(val, bool) else
            'int'     if isinstance(val, int)   else
            'double'  if isinstance(val, float) else
            'string'
        )
        ET.SubElement(root, 'key', {'id': f'gp{i}', 'for': 'graph',
                                    'attr.name': name, 'attr.type': attr_type})

    # Keys for per-node array attributes (e.g. kappa)
    for name in node_array_params:
        ET.SubElement(root, 'key', {'id': f'na_{name}', 'for': 'node',
                                    'attr.name': name, 'attr.type': 'double'})

    # Keys for embedding coordinates
    for d in range(dim):
        ET.SubElement(root, 'key', {'id': f'd{d}', 'for': 'node',
                                    'attr.name': f'x{d}', 'attr.type': 'double'})

    graph = ET.SubElement(root, 'graph', {'id': 'G', 'edgedefault': 'undirected'})

    # Scalar graph-level data
    for i, (name, val) in enumerate(scalar_params.items()):
        data_el = ET.SubElement(graph, 'data', {'key': f'gp{i}'})
        data_el.text = str(val).lower() if isinstance(val, bool) else str(val)

    # Nodes: embedding + per-node array attributes
    for node_id in range(n):
        node_el = ET.SubElement(graph, 'node', {'id': str(node_id)})
        for name, arr in node_array_params.items():
            data_el = ET.SubElement(node_el, 'data', {'key': f'na_{name}'})
            data_el.text = str(arr[node_id])
        for d, val in enumerate(embedding[node_id]):
            data_el = ET.SubElement(node_el, 'data', {'key': f'd{d}'})
            data_el.text = str(val)

    # Edges
    for i, (u, v) in enumerate(edges):
        ET.SubElement(graph, 'edge', {'id': f'e{i}', 'source': str(u), 'target': str(v)})

    raw = ET.tostring(root, encoding='unicode')
    pretty = minidom.parseString(raw).toprettyxml(indent='  ')
    pretty = '\n'.join(pretty.split('\n')[1:])  # strip <?xml ...?> declaration
    with open(output_path, 'w') as f:
        f.write(pretty)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("graph_path", type=str, help="Text file that contains the edge list of the graph to analyze.")
    parser.add_argument("fit_path", type=str, help="JSON file that contains the edge list of the graph to analyze.")
    parser.add_argument("-o", "--output-file", type=str, default="", help="Output file path. Set to fit_path.graphml by default.")
    args = parser.parse_args()

    output_path = os.path.splitext(args.fit_path)[0] + ".graphml" if args.output_file == "" else args.output_file


    edges = np.loadtxt(args.graph_path, dtype=np.int64)
    with open(args.fit_path, "r") as file_stream:
        results = json.load(file_stream)
    if (dls := results.get("description length")) is None:
        print("The description length is not computed for this fit file. Run the \"compute_dl.py\" script first.", file=sys.stderr)
        exit(-1)

    inferred_dim = min(dls, key=lambda k: dls[k][1])
    params = results["fits"][inferred_dim]
    params["pos"] = np.array(params["pos"])

    write_graphml(
        edges=edges,
        params=params,
        output_path=output_path,
    )
