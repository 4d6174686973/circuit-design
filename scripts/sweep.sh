#!/usr/bin/env bash
# ==============================================================================
# QCBM sweep across one or more nodes (SLURM array job).
#
# Unlike a wandb-agent-driven sweep, the search grid here is owned by Hydra
# (--multirun key=v1,v2,...), not by wandb -- wandb is used purely for tracking/organizing (a
# real Sweep object is created and every run attaches to it, see src/setup.py::get_or_create_wandb_sweep).
# Within each array task (= 1 node), the Python entrypoint itself fans out `runs_batch_size`
# auto-seeded runs in parallel (one process per seed, optionally one GPU each) -- no wandb agent,
# no submitit. Across array tasks (nodes), work is split by SEED RANGE (see below), so the full
# Hydra grid runs identically-but-disjointly on every node with no duplicated work and no manual
# override-splitting.
#
# Usage:
#   sbatch [--array=0-N] scripts/sweep.sh 'extension=none,metric_based,all_to_all'
#   bash scripts/sweep.sh 'extension=metric_based extension_threshhold=0.3,0.5,0.7'   # local, no SLURM
#
# Examples:
#   sbatch scripts/sweep.sh 'extension=none,metric_based,all_to_all'                  # 1 node
#   sbatch --array=0-3 scripts/sweep.sh 'extension=none,metric_based,all_to_all'      # 4 nodes,
#       # each running the SAME grid but with a disjoint block of seeds (see NODE_INITIAL_SEED below)
#
# NOTE (SLURM): --output/--error directories must already exist before you `sbatch` this script --
# SLURM creates the log FILE but not its parent directory, and the job fails immediately (with no
# log at all) if the directory is missing. Run `mkdir -p outputs/slurm_logs` first, or edit the
# paths below. Edit the partition (-p), --gres (if your cluster requires explicit GPU requests
# even under --exclusive), --cpus-per-task and --time for your cluster.
# ==============================================================================
#SBATCH -p <partition>                          # EDIT: your SLURM partition
#SBATCH --job-name=qcbm-sweep
#SBATCH --error=outputs/slurm_logs/%A_%a.err    # %A = array job id, %a = array task id
#SBATCH --output=outputs/slurm_logs/%A_%a.out
#SBATCH --array=0                                # 0 = 1 node; 0-3 = 4 nodes (or pass --array on the CLI)
#SBATCH --nodes=1                                # keep at 1 -- each array task gets 1 node
#SBATCH --exclusive                               # whole node, all its GPUs
#SBATCH --mem=0                                   # all available RAM on the node
#SBATCH --cpus-per-task=4                         # EDIT: mostly cosmetic under --exclusive
# #SBATCH --gres=gpu:4                            # EDIT/uncomment: some clusters require an explicit
                                                   # GPU request even with --exclusive
#SBATCH --time=72:00:00

set -euo pipefail

OVERRIDES="${1:?Provide a Hydra multirun override string, e.g. 'extension=none,metric_based'}"

# Run from the repo root regardless of where this was submitted/invoked from.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# uv is often installed outside the default SLURM job PATH -- fail fast with a clear error instead
# of a cryptic "command not found" buried in a log, rather than silently doing nothing.
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if ! command -v uv &> /dev/null; then
    echo "Error: 'uv' not found on PATH (checked \$HOME/.local/bin, \$HOME/.cargo/bin)." >&2
    exit 1
fi

# Sync once, explicitly, before any parallel work starts. uv uses file locks, so concurrent syncs
# across nodes sharing a filesystem are safe -- but doing it once up front avoids many nodes/workers
# independently racing to check the lockfile, and fails fast/loud on a real dependency problem.
echo ">>> [Setup] Syncing environment..."
uv sync

# SLURM array vars are unset when this script is run directly (no sbatch) -- default to a single,
# local "task 0 of 1" so everything below works the same for local testing and for a real array job.
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
TASK_COUNT="${SLURM_ARRAY_TASK_COUNT:-1}"
ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-local$$}"

# Sweep-wide settings (overridable via env).
DATASET="${DATASET:-BAS}"
N_QUBITS="${N_QUBITS:-9}"
RUNS_BATCH_SIZE="${RUNS_BATCH_SIZE:-5}"
BASE_SEED="${INIT_SEED:-42}"
SIMULATOR="${SIMULATOR:-aer_statevec_cpu}"
GPUS_PER_NODE="${GPUS_PER_NODE:-0}"          # >0 pins batch runs round-robin across GPUs (see README:
                                              # GPU is only worthwhile at high qubit counts; leave 0
                                              # for the 9-12 qubit sweeps this repo targets by default)
WANDB_MODE="${WANDB_MODE:-online}"

# --- Cross-node work split: disjoint seed ranges, not manual override-splitting ---
# Every node runs the IDENTICAL Hydra grid (${OVERRIDES}), but each array task gets its own block of
# `runs_batch_size` seeds, so a 4-node array gives 4x the seed coverage with zero duplicated work and
# no manual slicing of the override string.
NODE_INITIAL_SEED=$((BASE_SEED + TASK_ID * RUNS_BATCH_SIZE))

# --- Leader/worker sweep synchronization (only needed for a real multi-node online sweep) ---
# A single-node run (or any offline/disabled run) just lets Python create its own sweep id inline,
# same as always -- no coordination needed. For a genuine multi-node ONLINE sweep, task 0 creates
# the real wandb Sweep and publishes its id to a file on the shared filesystem (SLURM nodes on a
# cluster normally share one); other tasks poll for that file instead of independently creating
# their own (and colliding with) a separate sweep.
SHARED_SWEEP_FILE=".sweep_${ARRAY_JOB_ID}.tmp"

cleanup() {
    if [ "$TASK_ID" == "0" ]; then
        rm -f "$SHARED_SWEEP_FILE"
    fi
}
trap cleanup EXIT INT TERM

if [ "$WANDB_MODE" == "online" ] && [ "$TASK_COUNT" -gt 1 ]; then
    if [ "$TASK_ID" == "0" ]; then
        echo ">>> [W&B] Leader (task 0): creating the shared sweep..."
        export WANDB_SWEEP_ID_FILE="$SHARED_SWEEP_FILE"
    else
        echo ">>> [W&B] Worker (task ${TASK_ID}): waiting for the leader's sweep id..."
        while [ ! -f "$SHARED_SWEEP_FILE" ]; do
            sleep 5
        done
        export WANDB_SWEEP_ID
        WANDB_SWEEP_ID=$(cat "$SHARED_SWEEP_FILE")
    fi
fi

echo ">>> [Run] Node: $(hostname) | task ${TASK_ID}/${TASK_COUNT}"
echo ">>> [Run] Sweeping: ${OVERRIDES}"
echo ">>> [Run] dataset=${DATASET} N_qubits=${N_QUBITS} runs_batch_size=${RUNS_BATCH_SIZE} initial_random_seed=${NODE_INITIAL_SEED} simulator=${SIMULATOR} gpus_per_node=${GPUS_PER_NODE} wandb_mode=${WANDB_MODE}"
echo ">>> [Run] WANDB_SWEEP_ID=${WANDB_SWEEP_ID:-<created by this node>}"

uv run --no-sync python -m src --multirun \
    ${OVERRIDES} \
    dataset="${DATASET}" \
    N_qubits="${N_QUBITS}" \
    runs_batch_size="${RUNS_BATCH_SIZE}" \
    initial_random_seed="${NODE_INITIAL_SEED}" \
    simulator="${SIMULATOR}" \
    gpus_per_node="${GPUS_PER_NODE}" \
    wandb_mode="${WANDB_MODE}"

echo ">>> [Run] Task ${TASK_ID} finished successfully."
