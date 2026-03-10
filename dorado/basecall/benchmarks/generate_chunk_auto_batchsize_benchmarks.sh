#!/bin/bash

set -ex

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <dorado executable> [device_string]"
    exit 1
fi

device_string=${2:-"auto"}
echo "Using device string -x $device_string"
data_dir=$(dirname $0)/../../../tests/data
dorado_bin=$(cd "$(dirname $1)"; pwd -P)/$(basename $1)
pod5_dir=${data_dir}/pod5/dna_r10.4.1_e8.2_400bps_5khz/

echo Running benchmarks for models of interest
for model_name in \
        dna_r10.4.1_e8.2_400bps_fast@v4.3.0 \
        dna_r10.4.1_e8.2_400bps_fast@v5.0.0 \
        dna_r10.4.1_e8.2_400bps_fast@v5.2.0 \
        dna_r10.4.1_e8.2_400bps_hac@v4.3.0 \
        dna_r10.4.1_e8.2_400bps_hac@v5.0.0 \
        dna_r10.4.1_e8.2_400bps_hac@v5.2.0 \
        dna_r10.4.1_e8.2_400bps_sup@v4.3.0 \
        dna_r10.4.1_e8.2_400bps_sup@v5.0.0 \
        dna_r10.4.1_e8.2_400bps_sup@v5.2.0 \
        rna004_130bps_fast@v5.1.0 \
        rna004_130bps_fast@v5.2.0 \
        rna004_130bps_hac@v5.1.0 \
        rna004_130bps_hac@v5.2.0 \
        rna004_130bps_sup@v5.1.0 \
        rna004_130bps_sup@v5.2.0 \
        ; do
    echo $model_name;
    $dorado_bin download --model $model_name
    $dorado_bin basecaller \
        $model_name $pod5_dir \
        -x $device_string \
        --skip-model-compatibility-check \
        --batchsize-benchmarks-file "${model_name}.csv" \
        --run-batchsize-benchmarks break \
        > /dev/null
done
