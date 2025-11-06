#!/bin/bash
population_size=$1
gpu=$2

layouts=(random3_large random3_large_n random3_m_large random3_m_large_n)
script_dir="$(cd "$(dirname "$0")" && pwd)"

for layout in "${layouts[@]}"; do
    echo "[wrapper] running train_mep_stage_1.sh for layout: ${layout}"
    bash "${script_dir}/train_mep_stage_1.sh" "${layout}" "${population_size}" "${gpu}"
done
