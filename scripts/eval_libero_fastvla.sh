#!/usr/bin/env bash
# Evaluate OpenVLA-OFT + FastVLA on the four LIBERO suites with the paper's hyperparameters
# (per-suite rho; tau = 0.1 and window = 10 throughout), 50 trials per task (500 episodes per suite).
#
# Usage:  bash scripts/eval_libero_fastvla.sh [extra arguments for run_libero_eval.py]
#   e.g.  bash scripts/eval_libero_fastvla.sh --num_trials_per_task 5
#         bash scripts/eval_libero_fastvla.sh --fastvla_enable False   # uncompressed baseline
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

declare -A RHO=( [libero_spatial]=0.80 [libero_object]=0.70 [libero_goal]=0.75 [libero_10]=0.60 )

for suite in libero_spatial libero_object libero_goal libero_10; do
  python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint "moojink/openvla-7b-oft-finetuned-${suite//_/-}" \
    --task_suite_name "$suite" \
    --fastvla_rho "${RHO[$suite]}" --fastvla_tau 0.1 --fastvla_window 10 \
    --run_id_note "fastvla-rho${RHO[$suite]}" \
    "$@"
done
