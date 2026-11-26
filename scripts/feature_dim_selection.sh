#!/bin/sh
edgelist="$1"

num_surrogates=50
num_pts_T_vs_beta=15
clustering_step=0.1
num_realizations_per_beta=1
max_dim=8
max_k=20

# In detecting-dimensionality, vertex indices start at 1.
cat ${edgelist} | awk '{print $1 + 1, $2 + 1}' > detecting-dimensionality/create_SD/RealNets/network.edge
cd detecting-dimensionality/create_SD

echo "1/4 Counting cycles of graph"
if ! ../cyclesmap/cyclesmap RealNets/network.edge network_features; then
    echo "Error: cyclesmap failed." >&2; exit 1
fi

echo "2/4 Generating synthetic graphs"
if ! sh create_SD.sh network ${num_surrogates} ${num_pts_T_vs_beta} ${clustering_step} ${num_realizations_per_beta} ${max_dim} > /dev/null 2>&1; then
    echo "Error: create_SD failed." >&2; exit 1
fi

cd ../create_feats

echo "3/4 Counting cycles of synthetic graphs"
if ! sh create_feats.sh ../create_SD/SDnets/network ../create_SD/SDfeats/network > /dev/null; then
    echo "Error: create_feats failed." >&2; exit 1
fi

cd ..

echo "4/4 Estimating dimension"
python -c "from dimension.dimension import dimension; dimension('create_SD/SDfeats/network','create_SD/network_features',['triangles', 'squares','pentagons'], ${max_k})" 2>/dev/null | grep "Dimension detected:"
