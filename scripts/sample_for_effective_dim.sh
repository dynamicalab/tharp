#!/bin/bash

# Run commands to file location
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"

beta="2.5"
n_graphs="20"

for dim in {2..7}; do
    output_directory="../.output/various_n/${dim}d"
    if [ -d "$output_directory" ]; then
        : # output_directory is a directory
    elif [ -f "$output_directory" ]; then
        echo "Error: The output directory for generated graphs is a file." 1>&2
        exit 1
    else
        mkdir -p $output_directory
    fi

    for n in {100..1000..100} {5000..20000..5000}; do
        echo "Generating graphs with $n vertices"
        for i in $(seq 1 ${n_graphs}); do
            edgelist_name="$output_directory/graph${n}_${i}.txt"
            if [ ! -f $edgelist_name ]; then
                echo "Generating graph ${i}/${n_graphs}"
                scaledbeta=$(echo "($dim-1)* ${beta}" | bc)
                JAX_PLATFORMS='cpu' python generate_hd_graph.py $edgelist_name $n $dim -s $i -b $scaledbeta -g 2.7 -k 20 --ensure-connected --info
            fi
        done
        echo ""
    done
done
