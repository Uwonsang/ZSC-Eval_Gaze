#!/bin/bash
gpu=$1

layouts=(random3_large random3_large_n random3_m_large random3_m_large_n)
script_dir="$(cd "$(dirname "$0")" && pwd)"

for layout in "${layouts[@]}"; do
    echo "[wrapper] running train_bias_agents.sh for layout: ${layout}"
    bash "${script_dir}/train_bias_agents.sh" "${layout}" "${gpu}"
done
