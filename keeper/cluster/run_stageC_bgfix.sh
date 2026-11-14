#!/usr/bin/env bash
# Glue launcher (2026-10-20): reruns Stage C (CPPR) after the background-request
# fix in laundering/cp_vpe_gpu.py with both gate normalizations (ruling G1).
# 30 runs = 3 datasets x 5 seeds x {count gate (CPPR), snapshot gate (ablation)},
# spread over 4 GPUs with SLOTS concurrent runs per GPU. Skips finished outputs.
# Usage (repository root): bash keeper/cluster/run_stageC_bgfix.sh
set -uo pipefail

PY="${PY:-$HOME/.conda/envs/ps/bin/python}"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
GPUS=(0 1 2 3)
SLOTS=3
RES=results/laundering

jobs_list=()
for norm in snapshot count; do
  for ds in wiki-rfa bitcoinotc bitcoinalpha; do
    for seed in 0 1 2 3 4; do
      jobs_list+=("$norm $ds $seed")
    done
  done
done

idx=0
for spec in "${jobs_list[@]}"; do
  read -r norm ds seed <<< "$spec"
  if [[ "$norm" == snapshot ]]; then out="$RES/cp_vpe_${ds}_s${seed}"; else out="$RES/cpgate_${norm}_${ds}_s${seed}"; fi
  if [[ -f "$out.json" ]]; then echo "[skip] $out.json"; continue; fi
  while (( $(jobs -rp | wc -l) >= ${#GPUS[@]} * SLOTS )); do sleep 20; done
  gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
  echo "[$(date +%H:%M:%S) launch GPU $gpu] $norm $ds s$seed"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m keeper.cppr.cp_vpe_gpu --ds "$ds" --seed "$seed" \
    --epochs 150 --gate-norm "$norm" > "$out.log" 2>&1 &
  idx=$((idx + 1))
done
wait
# The aggregates are stored evidence: an existing one is replaced only if this script runs with
# KEEPER_OVERWRITE=1; otherwise its step is skipped with a message.
for norm in snapshot count; do
  if [[ "$norm" == snapshot ]]; then agg="$RES/cp_vpe_gpu_aggregate.json"; else agg="$RES/cpgate_${norm}_aggregate.json"; fi
  if [[ -f "$agg" && "${KEEPER_OVERWRITE:-}" != 1 ]]; then
    echo "[skip] $agg exists; run with KEEPER_OVERWRITE=1 to rebuild it from the per-seed files"
    continue
  fi
  CUDA_VISIBLE_DEVICES=0 "$PY" -m keeper.cppr.cp_vpe_gpu --aggregate --gate-norm "$norm"
done
echo "[$(date +%H:%M:%S)] STAGE C BGFIX DONE"
