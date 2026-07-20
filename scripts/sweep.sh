#!/usr/bin/env bash
#
# Launch a QCBM sweep. Extensions (and any other hyperparameters) are swept via Hydra multirun;
# within each combination the Python entrypoint fans out `runs_batch_size` auto-seeded runs in
# parallel (one wandb run per seed). No submitit — this is a plain wrapper you launch from the head
# node. (runs_batch_size is unrelated to mmd_batch_size, the MMD training mini-batch size.)
#
# Usage:
#   scripts/sweep.sh 'extension=none,all_to_all,metric_based,random'
#   scripts/sweep.sh 'extension=metric_based extension_threshhold=0.3,0.5,0.7'
#
#   # override sweep-wide settings via env vars:
#   DATASET=BAS N_QUBITS=9 RUNS_BATCH_SIZE=5 INIT_SEED=42 SIMULATOR=aer_statevec_cpu \
#       scripts/sweep.sh 'extension=none,metric_based,nearest_neighbor,all_to_all,random'
#
# A single invocation of this script (single node) is automatically registered as one real wandb
# Sweep (visible under the Sweeps tab, with parallel-coordinates etc.) — Python creates it on the
# first job and every job/seed-worker joins it via the WANDB_SWEEP_ID env var. No action needed.
#
# Multi-node: each node is a SEPARATE script invocation, so there's no shared parent process to
# create the sweep once for all of them. Instead:
#   1. Launch the FIRST node's slice normally (no WANDB_SWEEP_ID set) — Python creates the sweep and
#      prints it in the first job's log line: "Program started (..., sweep_id=<id>)".
#   2. Copy that <id>, then launch every OTHER node's slice with it pre-set, e.g.:
#        export WANDB_SWEEP_ID=<id> ; scripts/sweep.sh '<disjoint slice, e.g. different extensions>'
#      so all nodes' runs join the same sweep instead of creating their own.

set -euo pipefail

OVERRIDES="${1:?Provide a Hydra multirun override string, e.g. 'extension=none,metric_based'}"

# Sweep-wide settings (overridable via env).
DATASET="${DATASET:-BAS}"
N_QUBITS="${N_QUBITS:-9}"
RUNS_BATCH_SIZE="${RUNS_BATCH_SIZE:-5}"
INIT_SEED="${INIT_SEED:-42}"
SIMULATOR="${SIMULATOR:-aer_statevec_cpu}"
GPUS_PER_NODE="${GPUS_PER_NODE:-0}"          # >0 pins batch runs round-robin across GPUs
WANDB_MODE="${WANDB_MODE:-online}"

echo "WANDB_SWEEP_ID=${WANDB_SWEEP_ID:-<will be created>}"
echo "Sweeping: ${OVERRIDES}"
echo "dataset=${DATASET} N_qubits=${N_QUBITS} runs_batch_size=${RUNS_BATCH_SIZE} initial_random_seed=${INIT_SEED} simulator=${SIMULATOR} gpus_per_node=${GPUS_PER_NODE}"

uv run python -m src --multirun \
    ${OVERRIDES} \
    dataset="${DATASET}" \
    N_qubits="${N_QUBITS}" \
    runs_batch_size="${RUNS_BATCH_SIZE}" \
    initial_random_seed="${INIT_SEED}" \
    simulator="${SIMULATOR}" \
    gpus_per_node="${GPUS_PER_NODE}" \
    wandb_mode="${WANDB_MODE}"
