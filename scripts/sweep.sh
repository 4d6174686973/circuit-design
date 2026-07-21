#!/usr/bin/env bash
# ==============================================================================
# QCBM sweep across one or more nodes (SLURM array job). CPU-only (GPU is future work, see bottom).
#
# The Hydra grid (--multirun key=v1,v2,...) is what's swept, not wandb (no wandb agent). Each array
# task (= 1 node) runs its ENTIRE local grid concurrently in one process pool sized to the node's
# CPUs (sweep.max_parallel_runs x sweep.threads_per_run, 0/0 = auto -- see src/setup.py::plan_resources),
# instead of one combo at a time. Across array tasks, work is split by disjoint seed ranges (computed
# in src/__main__.py from SLURM_ARRAY_TASK_ID), so every node runs the same grid without duplicating work.
#
# All trailing arguments are passed straight through as Hydra overrides -- same as calling
# `python -m src --multirun <these args>` directly, no quoting needed. ANY config key -- swept
# (comma-separated) or fixed -- goes there; src/conf/config.yaml supplies the value for anything you
# don't mention. There's no separate per-key env-var mechanism here to keep in sync with
# config.yaml's fields.
#
# Usage:
#   sbatch [--array=0-N] [--nodelist=node1,node2,...] scripts/sweep.sh <hydra overrides...>
#   bash scripts/sweep.sh <hydra overrides...>   # local, no SLURM
#
# Examples:
#   sbatch scripts/sweep.sh circuit.extension=none,metric_based,all_to_all                  # 1 node
#   sbatch --array=0-3 scripts/sweep.sh circuit.extension=none,metric_based,all_to_all      # 4 nodes
#   sbatch --array=0-1 --nodelist=pgi14-gpu7,pgi14-gpu8 scripts/sweep.sh circuit.extension=none  # 2 named nodes
#   scripts/sweep.sh circuit.extension=none,metric_based sweep.runs_batch_size=5 dataset=BAS
#   scripts/sweep.sh circuit.extension=none,metric_based sweep.threads_per_run=4   # force 4 threads/run
#
# NOTE: --output/--error directories must exist before `sbatch` (mkdir -p outputs/slurm_logs) --
# SLURM creates the log file but not its parent directory. Edit the partition (-p), --gres,
# --cpus-per-task and --time below for your cluster.
# ==============================================================================
#SBATCH -p pgi14                                # EDIT: your SLURM partition
#SBATCH --job-name=qcbm
#SBATCH --error=outputs/slurm_logs/%A_%a.err    # %A = array job id, %a = array task id
#SBATCH --output=outputs/slurm_logs/%A_%a.out
#SBATCH --array=0                                # 0 = 1 node; 0-3 = 4 nodes (or pass --array on the CLI)
#SBATCH --nodes=1                                # keep at 1 -- each array task gets 1 node
#SBATCH --exclusive                               # whole node, all its CPUs (and GPUs, though unused here)
#SBATCH --mem=0                                   # all available RAM on the node
#SBATCH --cpus-per-task=4                         # EDIT: mostly cosmetic under --exclusive
                                                   # GPU request even with --exclusive, even if unused
#SBATCH --time=72:00:00

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: scripts/sweep.sh <hydra overrides...>, e.g. circuit.extension=none,metric_based" >&2
    exit 1
fi
OVERRIDES=("$@")

# Run from the repo root regardless of where this was submitted/invoked from. Under sbatch, the
# script runs from a spooled copy on the compute node, so BASH_SOURCE doesn't point at the repo --
# use SLURM_SUBMIT_DIR (always set by sbatch, = the directory `sbatch` was run from) instead, and
# fall back to BASH_SOURCE only for local (non-sbatch) runs.
cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/..}"

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
# (Pure SLURM job bookkeeping, not a config.yaml value, so a hardcoded fallback here is fine.)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
TASK_COUNT="${SLURM_ARRAY_TASK_COUNT:-1}"
ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-local$$}"

# --- Cross-node work split: disjoint seed ranges, not manual override-splitting ---
# Every node runs the IDENTICAL Hydra grid (${OVERRIDES}); src/__main__.py reads SLURM_ARRAY_TASK_ID
# itself and shifts each node's seed block to stay disjoint, so a 4-node array gives 4x the seed
# coverage with zero duplicated work -- nothing to compute here.

# --- Leader/worker sweep synchronization (only needed for real multi-node runs) ---
# A single-node run just lets Python create its own sweep id inline, same as always -- no
# coordination needed. For multi-node, task 0 creates the sweep (real if wandb_mode=online, a local
# synthetic id otherwise -- get_or_create_wandb_sweep handles both) and publishes it to a file on the
# shared filesystem (SLURM nodes on a cluster normally share one); other tasks poll for that file
# instead of independently creating (and colliding with) their own.
SHARED_SWEEP_FILE=".sweep_${ARRAY_JOB_ID}.tmp"

cleanup() {
    if [ "$TASK_ID" == "0" ]; then
        rm -f "$SHARED_SWEEP_FILE"
    fi
}
trap cleanup EXIT INT TERM

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
# The current implementation is tuned for CPU: each run is many small numpy/scipy ops (kernel,
# gradient, Adam) plus one heavier Aer statevector sampling step, and throughput comes from running
# MANY runs in parallel with a few threads each. A GPU is only worthwhile at higher qubit counts,
# and a naive "sampling on GPU, everything else on CPU" hybrid is dominated by per-iteration
# CPU<->GPU transfers. Making GPUs pay off is a non-trivial change; when tackling it, touch:
#
#   * src/setup.py::plan_resources    -- branch on cfg.sweep.gpus_per_node: size the pool to GPUs
#                                        (~1 run/GPU, bounded by VRAM) instead of CPU cores.
#   * src/setup.py::train_worker      -- CUDA_VISIBLE_DEVICES pinning is already stubbed; skip the
#                                        CPU thread-capping for GPU runs.
#   * src/setup.py::setup_qiskit_simulator -- device="GPU" path exists (batched_shots_gpu, blocking);
#                                        verify blocking_qubits/VRAM sizing for the target GPUs.
#   * src/cost.py, src/qcbm.py        -- to avoid transfer overhead, move the per-iteration kernel/
#                                        gradient math onto the GPU (e.g. cupy) so a run stays
#                                        device-resident across the whole iteration, not just sampling.
#   * this script                     -- add #SBATCH --gres=gpu:N; pass sweep.gpus_per_node=N and
#                                        ibm.simulator=aer_statevec_gpu in the override string.
# The "GPU NOTE" comments in the Python sources mark each of these swap points inline.
