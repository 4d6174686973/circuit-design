#!/usr/bin/env bash
# ==============================================================================
# QCBM sweep across one or more nodes (SLURM array job). CPU-only (GPU is future work, see bottom).
#
# The Hydra grid (--multirun key=v1,v2,...) is what's swept, not wandb (no wandb agent). Each array
# task (= 1 node) runs its ENTIRE local grid concurrently in one process pool sized to the node's
# CPUs (MAX_PARALLEL_RUNS x THREADS_PER_RUN below), instead of one combo at a time. Across array
# tasks, work is split by disjoint seed ranges (NODE_INITIAL_SEED below), so every node runs the
# same grid without duplicating work.
#
# Usage:
#   sbatch [--array=0-N] [--nodelist=node1,node2,...] scripts/sweep.sh '<hydra overrides>'
#   bash scripts/sweep.sh '<hydra overrides>'   # local, no SLURM
#
# Examples:
#   sbatch scripts/sweep.sh 'circuit.extension=none,metric_based,all_to_all'                  # 1 node
#   sbatch --array=0-3 scripts/sweep.sh 'circuit.extension=none,metric_based,all_to_all'      # 4 nodes
#   sbatch --array=0-1 --nodelist=pgi14-gpu7,pgi14-gpu8 scripts/sweep.sh 'circuit.extension=none'  # 2 named nodes
#   RUNS_BATCH_SIZE=5 scripts/sweep.sh 'circuit.extension=none,metric_based circuit.extension_threshhold=0.3,0.5'
#   THREADS_PER_RUN=4 scripts/sweep.sh 'circuit.extension=none,metric_based'   # force 4 threads/run
#
# NOTE: --output/--error directories must exist before `sbatch` (mkdir -p outputs/slurm_logs) --
# SLURM creates the log file but not its parent directory. Edit the partition (-p), --gres,
# --cpus-per-task and --time below for your cluster.
# ==============================================================================
#SBATCH -p pgi14                                # EDIT: your SLURM partition
#SBATCH --job-name=qcbm-sweep
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
WANDB_MODE="${WANDB_MODE:-online}"

# CPU parallelism budget (0 = auto from the node's core count; see src/setup.py::plan_resources).
# Leave both 0 to saturate the node: the planner picks MAX_PARALLEL_RUNS concurrent runs each with
# THREADS_PER_RUN threads so their product ~= cores. Set one to steer the run-vs-thread trade-off.
MAX_PARALLEL_RUNS="${MAX_PARALLEL_RUNS:-0}"
THREADS_PER_RUN="${THREADS_PER_RUN:-0}"

# GPU note: this launcher targets CPU. GPU_PER_NODE stays 0 -- GPU scheduling is future work (see
# the GPU section in the README and the "GPU NOTE" markers in src/setup.py / src/__main__.py).
GPUS_PER_NODE="${GPUS_PER_NODE:-0}"

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

echo ">>> [Run] Node: $(hostname) | task ${TASK_ID}/${TASK_COUNT} | cpus=$(nproc 2>/dev/null || echo '?')"
echo ">>> [Run] Sweeping: ${OVERRIDES}"
echo ">>> [Run] dataset=${DATASET} N_qubits=${N_QUBITS} runs_batch_size=${RUNS_BATCH_SIZE} initial_random_seed=${NODE_INITIAL_SEED} simulator=${SIMULATOR} wandb_mode=${WANDB_MODE}"
echo ">>> [Run] max_parallel_runs=${MAX_PARALLEL_RUNS} threads_per_run=${THREADS_PER_RUN} (0 = auto from cores)"
echo ">>> [Run] WANDB_SWEEP_ID=${WANDB_SWEEP_ID:-<created by this node>}"

uv run --no-sync python -m src --multirun \
    ${OVERRIDES} \
    dataset="${DATASET}" \
    data.N_qubits="${N_QUBITS}" \
    sweep.runs_batch_size="${RUNS_BATCH_SIZE}" \
    sweep.initial_random_seed="${NODE_INITIAL_SEED}" \
    sweep.max_parallel_runs="${MAX_PARALLEL_RUNS}" \
    sweep.threads_per_run="${THREADS_PER_RUN}" \
    ibm.simulator="${SIMULATOR}" \
    sweep.gpus_per_node="${GPUS_PER_NODE}" \
    logging.wandb_mode="${WANDB_MODE}"

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
#   * this script                     -- add #SBATCH --gres=gpu:N, set GPUS_PER_NODE=N, and expose a
#                                        GPU simulator via SIMULATOR=aer_statevec_gpu.
# The "GPU NOTE" comments in the Python sources mark each of these swap points inline.
