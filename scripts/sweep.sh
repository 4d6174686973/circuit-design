#!/usr/bin/env bash
# ==============================================================================
# QCBM sweep across one or more nodes (SLURM array job). CPU-only (GPU: see bottom).
#
# The Hydra grid (--multirun key=v1,v2,...) is what's swept, not wandb (no wandb agent). Each array
# task (= 1 node) runs its whole local grid concurrently in one pool sized to the node's CPUs
# (sweep.max_parallel_runs x sweep.threads_per_run, 0/0 = auto -- see setup.py::plan_resources).
# Across tasks, work is split by disjoint seed ranges (src/__main__.py, from SLURM_ARRAY_TASK_ID).
#
# All trailing arguments are passed through as Hydra overrides; any config key, swept
# (comma-separated) or fixed, goes there. src/conf/config.yaml supplies everything else.
#
# Usage:
#   sbatch [--array=0-N] [--nodelist=node1,node2,...] scripts/sweep.sh <hydra overrides...>
#   bash scripts/sweep.sh <hydra overrides...>   # local, no SLURM
#
# Examples:
#   sbatch scripts/sweep.sh circuit.extension=none,metric_based,all_to_all                  # 1 node
#   sbatch --array=0-3 scripts/sweep.sh circuit.extension=none,metric_based,all_to_all      # 4 nodes
#   sbatch --array=0-1 --nodelist=pgi14-gpu7,pgi14-gpu8 scripts/sweep.sh circuit.extension=none
#   scripts/sweep.sh circuit.extension=none,metric_based sweep.runs_batch_size=5 dataset=BAS
#   scripts/sweep.sh circuit.extension=none,metric_based sweep.threads_per_run=4
#
# NOTE: mkdir -p outputs/slurm_logs before `sbatch` -- SLURM creates the log file, not its parent.
# Edit the partition (-p), --gres, --cpus-per-task and --time below for your cluster.
# ==============================================================================
#SBATCH -p pgi14                                # EDIT: your SLURM partition
#SBATCH --job-name=qcbm
#SBATCH --error=outputs/slurm_logs/%A_%a.err    # %A = array job id, %a = array task id
#SBATCH --output=outputs/slurm_logs/%A_%a.out
#SBATCH --array=0                                # 0 = 1 node; 0-3 = 4 nodes (or --array on the CLI)
#SBATCH --nodes=1                                # keep at 1 -- one node per array task
#SBATCH --exclusive                              # whole node, all its CPUs
#SBATCH --mem=0                                  # all available RAM
#SBATCH --cpus-per-task=4                        # EDIT: mostly cosmetic under --exclusive
#SBATCH --time=72:00:00

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: scripts/sweep.sh <hydra overrides...>, e.g. circuit.extension=none,metric_based" >&2
    exit 1
fi
OVERRIDES=("$@")

# Run from the repo root. Under sbatch the script runs from a spooled copy on the compute node, so
# BASH_SOURCE doesn't point at the repo -- use SLURM_SUBMIT_DIR, falling back to BASH_SOURCE locally.
cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/..}"

# uv is often outside the default SLURM job PATH -- fail fast instead of "command not found".
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if ! command -v uv &> /dev/null; then
    echo "Error: 'uv' not found on PATH (checked \$HOME/.local/bin, \$HOME/.cargo/bin)." >&2
    exit 1
fi

# Sync once up front (uv's file locks make concurrent syncs safe, but this fails loud and early).
echo ">>> [Setup] Syncing environment..."
uv sync

# SLURM array vars are unset for a direct (non-sbatch) run -- default to "task 0 of 1".
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
TASK_COUNT="${SLURM_ARRAY_TASK_COUNT:-1}"
ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-local$$}"

# Cross-node work split needs nothing here: every node runs the identical grid and src/__main__.py
# shifts each node's seed block by SLURM_ARRAY_TASK_ID to keep them disjoint.

# Multi-node sweep sync: task 0 creates the sweep and publishes its id to the shared filesystem;
# other tasks poll for it instead of creating (and colliding with) their own. Single-node runs just
# let Python create the id inline.
SHARED_SWEEP_FILE=".sweep_${ARRAY_JOB_ID}.tmp"

cleanup() {
    if [ "$TASK_ID" == "0" ]; then
        rm -f "$SHARED_SWEEP_FILE"
    fi
    if [ -n "${WANDB_SERVICE:-}" ]; then
        uv run --no-sync wandb beta core stop > /dev/null 2>&1 || true
    fi
}
trap cleanup EXIT INT TERM

# Optional: one shared wandb-core backend for the node instead of one process per concurrent run
# (at max_parallel_runs ~= core count that adds up). Opt-in -- beta, and a single point of failure
# for LOGGING only, since src/wandb_logging.py keeps training alive without it.
#   QCBM_SHARED_WANDB_CORE=1 sbatch scripts/sweep.sh <overrides...>
if [ "${QCBM_SHARED_WANDB_CORE:-0}" = "1" ]; then
    echo ">>> [W&B] Starting one shared wandb-core service for this node..."
    # --idle-timeout 0 disables idle shutdown; the token goes to stdout, notes to stderr.
    WANDB_SERVICE="$(uv run --no-sync wandb beta core start --idle-timeout 0 | tail -n 1)"
    export WANDB_SERVICE
    echo ">>> [W&B] WANDB_SERVICE=${WANDB_SERVICE}"
fi

if [ "$TASK_COUNT" -gt 1 ]; then
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

echo ">>> [Run] Node: $(hostname) | task ${TASK_ID}/${TASK_COUNT} | cpus=$(nproc 2>/dev/null || echo '?')"
echo ">>> [Run] Sweeping: ${OVERRIDES[*]}"
echo ">>> [Run] WANDB_SWEEP_ID=${WANDB_SWEEP_ID:-<created by this node>}"

uv run --no-sync python -m src --multirun "${OVERRIDES[@]}"

echo ">>> [Run] Task ${TASK_ID} finished successfully."

# ==============================================================================
# GPU support -- FUTURE WORK (this launcher is CPU-only)
# ==============================================================================
# CPU throughput comes from many parallel runs with a few threads each: per iteration it's mostly
# small numpy ops (kernel, gradient, Adam) plus one heavier Aer sampling step. A GPU only pays off at
# higher qubit counts, and "sampling on GPU, rest on CPU" is dominated by per-iteration transfers.
# When tackling it, touch:
#
#   * setup.py::plan_resources    -- branch on cfg.sweep.gpus_per_node: size the pool to GPUs
#                                    (~1 run/GPU, bounded by VRAM) instead of CPU cores.
#   * setup.py::train_worker      -- CUDA_VISIBLE_DEVICES pinning is stubbed; skip the thread cap.
#   * setup.py::setup_qiskit_simulator -- device="GPU" path exists; verify blocking_qubits/VRAM.
#   * src/cost.py, src/qcbm.py    -- move the per-iteration kernel/gradient math onto the GPU (cupy)
#                                    so a run stays device-resident, not just during sampling.
#   * this script                 -- add #SBATCH --gres=gpu:N; pass sweep.gpus_per_node=N and
#                                    ibm.simulator=aer_statevec_gpu as overrides.
# The "GPU NOTE" comments in the Python sources mark each swap point inline.
