#!/usr/bin/env bash
# Stage B full model run (approved 2026-10-17): the user's archived LocalGDTE+SCEM, built with the
# approved autograd-safe SCEM copy (scem_autograd_fix.py; merged into keeper/scem/scem.py on 2026-10-27),
# 3 datasets x 5 seeds, 150 epochs, 4 at a time on one GPU. Existing result files are skipped (Bitcoin-Alpha seed 0 is the smoke run, same code).
#SBATCH --partition=research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=orig-stageB
#SBATCH --output=/home/mucbp/PROJECTS/FedUnlearning-Argon/results/scem/stage_b/stageB-full-%j.out
set -u
cd /home/mucbp/PROJECTS/FedUnlearning-Argon
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
L=results/scem/stage_b
echo "host $(hostname) job ${SLURM_JOB_ID:-} CVD=${CUDA_VISIBLE_DEVICES:-} start $(date)"
for ds in wiki-rfa bitcoinotc bitcoinalpha; do          # longest first
  for seed in 0 1 2 3 4; do
    [ -f $L/model_${ds}_s${seed}.json ] && { echo "[skip] $ds s$seed exists" >&2; continue; }
    echo "$ds $seed"
  done
done | xargs -P 4 -n 2 sh -c '
  env -u PYTHONPATH -u VIRTUAL_ENV /home/mucbp/.conda/envs/ps/bin/python -u -m keeper.scem.stage_b model --ds "$0" --seed "$1" \
    > '"$L"'/model_"$0"_s"$1".log 2>&1
  echo "model $0 s$1 exit $? at $(date +%H:%M:%S)"'
# model_summary.json is stored evidence: it is replaced only if the job runs with KEEPER_OVERWRITE=1;
# otherwise the step is skipped with a message.
agg=$L/model_summary.json
if [[ -f "$agg" && "${KEEPER_OVERWRITE:-}" != 1 ]]; then
  echo "[skip] $agg exists; run with KEEPER_OVERWRITE=1 to rebuild it from the model records"
else
  env -u PYTHONPATH -u VIRTUAL_ENV /home/mucbp/.conda/envs/ps/bin/python -u -m keeper.scem.stage_b aggregate
fi
echo "exit $? at $(date)"
