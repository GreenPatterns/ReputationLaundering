#!/usr/bin/env bash
# Multi-GPU launcher for CP-VPE + Credibility Gate + Inject-Then-Erase on kc-sse-ml-rn04 (4x NVIDIA L40S)
# Usage: bash keeper/cluster/run_cp_vpe_gpu.sh
# These are the per-snapshot gate runs (cp_vpe_*, the ablation of Table S7). Since ruling G1 (2026-10-27) the
# driver defaults to the count gate, so the snapshot gate is requested explicitly.

set -euo pipefail

PY="${PY:-$HOME/.conda/envs/ps/bin/python}"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

mkdir -p results/laundering

GPUS=(0 0 0 0)   # adapted (with the user's approval) from (0 1 2 3): this run has 1 GPU, 4 jobs share it
job_idx=0

for ds in bitcoinalpha bitcoinotc wiki-rfa; do
  for seed in 0 1 2 3 4; do
    out_json="results/laundering/cp_vpe_${ds}_s${seed}.json"
    if [[ -f "$out_json" ]]; then
      echo "[skip] $out_json already exists"
      continue
    fi
    gpu="${GPUS[$((job_idx % 4))]}"
    log_file="results/laundering/cp_vpe_${ds}_s${seed}.log"
    echo "[launch GPU $gpu] $ds seed $seed -> $log_file"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m keeper.cppr.cp_vpe_gpu \
      --ds "$ds" --seed "$seed" --epochs 150 --gate-norm snapshot > "$log_file" 2>&1 &
    job_idx=$((job_idx + 1))
    if (( job_idx % 4 == 0 )); then
      wait
    fi
  done
done

wait
# The aggregate is stored evidence: an existing one is replaced only if this script runs with
# KEEPER_OVERWRITE=1; otherwise the step is skipped with a message.
agg=results/laundering/cp_vpe_gpu_aggregate.json
if [[ -f "$agg" && "${KEEPER_OVERWRITE:-}" != 1 ]]; then
  echo "[skip] $agg exists; run with KEEPER_OVERWRITE=1 to rebuild it from the per-seed files"
else
  echo "[aggregate] Building $agg"
  CUDA_VISIBLE_DEVICES=0 "$PY" -m keeper.cppr.cp_vpe_gpu --aggregate --gate-norm snapshot
fi
echo "[done] All 15 GPU runs completed."
