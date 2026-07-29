import numpy as np
import logging
import json
import pandas as pd
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import time

from qiskit.circuit import QuantumCircuit
from qiskit import qpy

from src.utils import array_to_str, sample_info
from src.cost import adam, cost_mmd_pre, cost_grad_mmd_pre, cost_grad_kl_div
from src.data import DataLoader


def resolve_iterations(mode: str, iterations: int, measurement_budget: int,
                       measurements_per_step: int) -> int:
    """How many training iterations this run gets.

    "iterations": fixed iteration count -- connectivities then consume different amounts of
    quantum resources, since a step costs (2P+1)*N_shots and P grows with added connections.

    "measurements": fixed measurement budget instead, so runs are compared at EQUAL quantum cost;
    iterations = budget // per-step cost (a step that would exceed the budget is cut, never
    partially run).

    Either way the step-0 baseline doesn't consume the budget (see _log_baseline_step).
    """
    if mode == "iterations":
        if iterations < 1:
            raise ValueError(f"qcbm.iterations must be >= 1, got {iterations}")
        return int(iterations)

    if mode != "measurements":
        raise ValueError(f"qcbm.mode must be 'iterations' or 'measurements', got {mode!r}")

    if measurement_budget < measurements_per_step:
        raise ValueError(
            f"qcbm.measurement_budget ({measurement_budget:,}) does not cover a single training "
            f"iteration of this circuit ({measurements_per_step:,} measurements = "
            f"(2*P+1)*N_shots). Raise the budget, lower qcbm.N_shots, or use a smaller circuit.")
    return int(measurement_budget // measurements_per_step)


class QCBM:
    def __init__(
            self,
            sampler,
            backend,
            circuit: QuantumCircuit = None,
            parameters: np.ndarray = None,
            adam_learning_rate: float = 0.01,
            finite_diff_epsilon: float = 1.0e-8,  # for KL gradient
            gradient_workers: int = 8,  # per-parameter gradient loop
            use_parameter_binds: bool = True,  # False: SamplerV2 primitive (hardware) instead of Aer fast path
            ) -> None:

        self.sampler = sampler
        self.backend = backend
        self.circuit: QuantumCircuit = circuit
        self.parameters: np.ndarray = parameters  # current parameters
        assert len(parameters) == circuit.num_parameters, "Number of parameters does not match number of circuit parameters"
        self.adam_learning_rate: float = adam_learning_rate
        self.finite_diff_epsilon: float = finite_diff_epsilon
        self.gradient_workers: int = gradient_workers
        self.use_parameter_binds: bool = use_parameter_binds

        # training state and results
        self.weight_grad: np.ndarray = np.zeros(self.circuit.num_parameters)
        self.parameter_hist: list[np.ndarray] = [self.parameters]
        self.losses: dict[str, list] = {"mmd_train": [], "mmd_val": []}  # test tracking is test_bench_hist below
        # per-eval-step held-out test benchmark suite (bench_dist + bench_val + BAS-only bench_BAS),
        # variable-key dicts dumped to test_metrics.parquet in save(); also logged to wandb inline.
        self.test_bench_hist: list[dict] = []

        # best checkpoint (model selection); populated during training
        self.best_params: np.ndarray = self.parameters.copy()
        self.best_iter: int = -1
        self.best_metric: float = np.inf
        self.model_selection_metric: str = "mmd_val"
        self.total_measurements: int = 0
        # Run length actually used + its per-step cost; resolved in stochastic_gradient_descent, since
        # under a measurement budget the iteration count depends on this circuit's parameter count.
        self.iterations_run: int = 0
        self.measurements_per_step: int = 0
        self._last_wandb_s: float = 0.0  # cost of the previous _log_step call (see _log_step)

    def _run_binds(self, stack: np.ndarray, N_shots: int, circuit=None, seed_simulator=None) -> list:
        """Run a (n, P) stack of parameter sets and return a list of n count dicts.

        Uses Aer's parameter_binds fast path when available, else a single 2D-parameter PUB through
        SamplerV2 (hardware). `circuit` defaults to self.circuit. `seed_simulator` overrides the RNG
        seed for this run only (pins the step-0 baseline across connectivities); Aer path only.
        """
        circuit = circuit if circuit is not None else self.circuit
        if self.use_parameter_binds:
            binds = {p: stack[:, p.index] for p in circuit.parameters}
            run_kwargs = {"shots": N_shots}
            if seed_simulator is not None:
                run_kwargs["seed_simulator"] = int(seed_simulator)
            result = self.sampler.run(circuit, parameter_binds=[binds], **run_kwargs).result()
            counts = result.get_counts()
            return counts if isinstance(counts, list) else [counts]
        # primitive path (hardware): one PUB with a 2D parameter array
        result = self.sampler.run([(circuit, stack)], shots=N_shots).result()[0]
        meas = result.data.meas
        n = stack.shape[0]
        flat = meas.reshape(n) if meas.shape else meas
        return [flat[i].get_counts() for i in range(n)] if meas.shape else [meas.get_counts()]

    def sample(self, N_shots: int, circuit=None, params: np.ndarray = None, seed_simulator=None) -> dict:
        '''Samples from a circuit at given parameters. Defaults to self.circuit at self.parameters;
        pass both to sample an arbitrary circuit (e.g. the linear baseline). seed_simulator pins the
        RNG seed for this run (see _run_binds).'''
        circuit = circuit if circuit is not None else self.circuit
        params = params if params is not None else self.parameters
        stack = np.asarray(params).reshape(1, circuit.num_parameters)
        return self._run_binds(stack, N_shots, circuit=circuit, seed_simulator=seed_simulator)[0]

    def param_shift_sampling(self, parameter_values: np.ndarray, shift: float, N_shots: int = 10000) -> tuple:
        """ Parameter-shift sampling for gradient computation.

        Builds a single (2P+1, P) stack (plus-shifted, minus-shifted, centre) and runs it in one
        Aer `parameter_binds` call instead of 2P+1 separate submissions.

        Returns:
            dist_plus:  list of P count dicts for the +shift circuits
            dist_minus: list of P count dicts for the -shift circuits
            dist:       count dict for the unshifted (centre) circuit
        """
        P = self.circuit.num_parameters
        assert isinstance(parameter_values, np.ndarray), "Parameters must be a numpy array"

        offset = np.identity(P)
        plus = parameter_values + shift * offset          # (P, P)
        minus = parameter_values - shift * offset         # (P, P)
        centre = parameter_values.reshape(1, P)           # (1, P)
        stack = np.concatenate([plus, minus, centre], axis=0)  # (2P+1, P)

        counts = self._run_binds(stack, N_shots)
        dist_plus = counts[:P]
        dist_minus = counts[P:2 * P]
        dist = counts[2 * P]
        return dist_plus, dist_minus, dist

    def _mmd_gradient(self, T, T_probs, S, S_probs, dists_plus, dists_minus, sigmas):
        """Compute the full MMD gradient vector, parallelized across parameters.

        T and S arrays/probabilities are extracted once and reused for every parameter; only the
        plus/minus distributions differ per parameter.
        """
        def grad_i(i):
            Pn, P_probs = sample_info(dists_plus[i])
            Mn, M_probs = sample_info(dists_minus[i])
            return cost_grad_mmd_pre(T, T_probs, S, S_probs, Pn, P_probs, Mn, M_probs, sigmas)

        n = self.circuit.num_parameters
        workers = max(1, min(self.gradient_workers, n))
        if workers == 1:
            return np.array([grad_i(i) for i in range(n)])
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return np.array(list(executor.map(grad_i, range(n))))

    def _log_step(self, wandb_run, payload: dict, step: int) -> float:
        """Hand one step's metrics to the (buffering) wandb logger; returns the seconds it took.

        wandb_run is a src.wandb_logging.WandbLogger: rows are buffered locally and pushed in chunks,
        so this normally costs ~0 and only occasionally pays for a push. That cost cannot be reported
        in the row it is measured on (a call can't time itself), so it is carried in
        self._last_wandb_s and logged as time/wandb_s on the FOLLOWING step -- see the time/* keys in
        stochastic_gradient_descent.
        """
        if wandb_run is None:
            return 0.0
        payload["time/wandb_s"] = self._last_wandb_s
        self._last_wandb_s = wandb_run.log(payload, step=step) or 0.0
        return self._last_wandb_s

    def _log_baseline_step(self, baseline_circuit, baseline_params, baseline_seed, N_shots: int,
                           T_train, T_train_probs, T_val, T_val_probs, sigmas: np.ndarray,
                           dataset_kind: str, valid_patterns, X_train_count: dict,
                           X_test_count: dict, wandb_run, logger) -> None:
        """Sample the shared LINEAR, UNEXTENDED circuit and log it as step 0 (see the
        baseline_circuit/baseline_params docstring on stochastic_gradient_descent). Mutates
        self.losses/self.test_bench_hist like a training-iteration eval step, but is not a
        selectable checkpoint.

        Sampled with a FIXED seed (baseline_seed, the sweep's global initial_random_seed) rather
        than the per-run seed, so step 0 is bit-identical across all connectivities."""
        from src.benchmark import evaluate  # local import avoids any import-time cycle with setup

        start_time = time.time()
        B_counts = self.sample(N_shots, circuit=baseline_circuit, params=baseline_params,
                               seed_simulator=baseline_seed)
        sample_time = time.time()
        B, B_probs = sample_info(B_counts)
        b_train = cost_mmd_pre(T_train, T_train_probs, B, B_probs, sigmas)
        b_val = cost_mmd_pre(T_val, T_val_probs, B, B_probs, sigmas) if T_val is not None else np.nan
        b_bench = {}
        if len(X_test_count):
            b_bench = evaluate(B_counts, {"test": X_test_count}, dataset_kind, sigmas=sigmas,
                               valid_patterns=valid_patterns, train_patterns=X_train_count)
            self.test_bench_hist.append({"iteration": 0, **b_bench})
        self.losses["mmd_train"].append(b_train)
        self.losses["mmd_val"].append(b_val)
        eval_time = time.time()
        logger.debug(f"| Step 0 baseline (linear, unextended) | Train MMD = {np.round(b_train, 6)} "
                    f"| Val MMD = {np.round(b_val, 6)}")
        if wandb_run is not None:
            log0 = {"train/step": 0, "train/cumulative_measurements": 0,
                    "train/measurements_per_step": 0, "train/num_parameters": self.circuit.num_parameters,
                    "train/mmd_train": b_train, "train/mmd_val": b_val,
                    "train/mmd_test": b_bench.get("bench_dist/test/mmd", np.nan),
                    "time/sampling_s": sample_time - start_time, "time/gradient_s": 0.0,
                    "time/eval_s": eval_time - sample_time,
                    "time/total_s": eval_time - start_time}
            log0.update(b_bench)
            self._log_step(wandb_run, log0, step=0)

    def stochastic_gradient_descent(
            self, X_train: np.ndarray, X_train_count: dict, X_val_count: dict, X_test_count: dict,
            iterations: int, N_shots: int, mmd_batch_fraction: float = 0.0,
            loss_func: str = 'MMD', sigmas: list = [1.0],
            eval_every: int = 1, model_selection_metric: str = 'mmd_val', wandb_run=None,
            dataset_kind: str = None, valid_patterns: np.ndarray = None,
            mode: str = 'iterations', measurement_budget: int = 0,
            *, baseline_circuit, baseline_params: np.ndarray, baseline_seed: int):
        """ Stochastic Gradient Descent with parameter-shift / finite-difference sampling.

        Splits are supplied pre-computed by the caller; validation MMD drives model selection.

        On every eval step the full benchmark suite (src.benchmark.evaluate: bench_dist/test/{mmd,
        kl,tv,fidelity}, Gili et al. bench_val/* generalization metrics, and -- BAS only --
        bench_BAS/{precision,recall,qbas}, scoped by valid_patterns; None for JGB, where bench_val/*
        treats every bitstring as valid) is computed on the TEST split and logged to wandb, tracking
        the held-out test trajectory rather than just its final value. `train/mmd_test` mirrors
        bench_dist/test/mmd under the "train" tab alongside train/mmd_{train,val} for one
        at-a-glance curve. All of this is skipped when the test split is empty (e.g.
        val_size + train_size == 1).

        baseline_circuit/baseline_params/baseline_seed (REQUIRED, keyword-only): the shared LINEAR,
        UNEXTENDED circuit/params/seed (the sweep's fixed initial_random_seed), evaluated with the
        same metric suite and logged at wandb step 0 as a common pre-training reference across
        connectivities -- bit-identical every run since circuit and seed are fixed. Step 0 is
        always this baseline (training starts at step 1); it's a reference only, never a checkpoint
        of this circuit, and never participates in model selection.

        mode ('iterations' | 'measurements') selects what ends the run: a fixed iteration count, or as
        many iterations as `measurement_budget` affords, so connectivities are compared at equal
        quantum cost (see resolve_iterations). Only the run LENGTH changes -- logging is per training
        iteration in both modes.

        wandb_run is a src.wandb_logging.WandbLogger (or None for no wandb logging). Every iteration
        is logged at its own step, but rows are buffered and pushed in chunks of
        logging.wandb_flush_every so request volume stays flat with hundreds of concurrent runs; the
        logger never raises, so a rate-limited or broken wandb only costs logging, not the run.
        """

        logger = logging.getLogger('QCBM')
        self.model_selection_metric = model_selection_metric
        sigmas = np.array(sigmas)
        from src.benchmark import evaluate  # avoids import-time cycle with setup

        # Parameters
        if loss_func == 'KL':
            shift = self.finite_diff_epsilon
        elif loss_func == 'MMD':
            shift = np.pi / 2
        else:
            raise ValueError("Loss Function not implemented")

        # Variables
        self.weight_grad = np.zeros(self.circuit.num_parameters)
        [m, v] = [np.zeros(self.circuit.num_parameters) for _ in range(2)]  # adam variables

        # pre-extract held-out targets once; the test split is evaluated via benchmark.evaluate()
        # below instead (bench_dist/test/mmd uses the same kernel/sigmas).
        T_train, T_train_probs = sample_info(X_train_count)
        T_val, T_val_probs = sample_info(X_val_count) if len(X_val_count) else (None, None)

        measurements_per_step = (2 * self.circuit.num_parameters + 1) * N_shots

        # Run length: a fixed iteration count, or as many whole iterations as the measurement budget
        # affords (see resolve_iterations). Recorded so save()/the caller can report the length that
        # was actually used, which in 'measurements' mode is circuit-dependent.
        iterations = resolve_iterations(mode, iterations, measurement_budget,
                                        measurements_per_step)
        self.iterations_run = iterations
        self.measurements_per_step = measurements_per_step
        budget_note = (f" (budget {measurement_budget:,} / {measurements_per_step:,} per step, "
                       f"{measurement_budget - iterations * measurements_per_step:,} left unused)"
                       if mode == "measurements" else "")
        logger.info(f"Run length ({mode}): {iterations} iterations x "
                    f"{measurements_per_step:,} measurements = "
                    f"{iterations * measurements_per_step:,} total{budget_note}")

        # Resolve the MMD-target minibatch size once. 0 => full train set. (0,1] => that fraction,
        # rounded and clamped to [1, |X_train|]. Only subsamples the (classical) kernel target data --
        # the dominant cost is the (2P+1)*N_shots quantum sampling below, independent of batch size,
        # so a smaller batch buys ~no speedup and only adds gradient noise.
        n_train = len(X_train)
        if mmd_batch_fraction and mmd_batch_fraction > 0:
            batch_n = int(min(n_train, max(1, round(mmd_batch_fraction * n_train))))
            if batch_n >= n_train:
                batch_n = 0  # fraction covers the whole train set -> full-batch fast path
        else:
            batch_n = 0
        logger.info(f"MMD target batch: {'full train set' if batch_n == 0 else f'{batch_n}/{n_train}'} "
                    f"samples (mmd_batch_fraction={mmd_batch_fraction})")

        # Step 0 is always the shared LINEAR, UNEXTENDED circuit; training iterations are logged at
        # steps 1..iterations. Not a selectable checkpoint, so excluded from model selection.
        self._log_baseline_step(
            baseline_circuit, baseline_params, baseline_seed, N_shots, T_train, T_train_probs,
            T_val, T_val_probs, sigmas, dataset_kind, valid_patterns, X_train_count, X_test_count,
            wandb_run, logger)

        # Training loop
        for it in range(iterations):

            logger.debug(f" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ")
            logger.debug(f"| Iteration {it + 1} / {iterations}")

            # snapshot of the parameters that produce this iteration's samples S
            params_snapshot = self.parameters.copy()

            # target for the gradient step (full train set or a shuffled minibatch)
            if batch_n == 0:
                T_batch, T_batch_probs = T_train, T_train_probs
            else:
                X_shuffled = X_train.copy()
                np.random.shuffle(X_shuffled)
                batch = Counter(array_to_str(X_shuffled[:batch_n, :]))
                T_batch, T_batch_probs = sample_info(batch)

            # Parameter Shift Sampling
            start_time = time.time()
            dists_plus, dists_minus, S_counts = self.param_shift_sampling(self.parameters, shift, N_shots)
            sample_time = time.time()
            self.total_measurements += measurements_per_step

            # Compute the gradient
            S, S_probs = sample_info(S_counts)
            if loss_func == 'MMD':
                self.weight_grad = self._mmd_gradient(T_batch, T_batch_probs, S, S_probs,
                                                      dists_plus, dists_minus, sigmas)
            elif loss_func == 'KL':
                batch_dict = X_train_count if batch_n == 0 else Counter(array_to_str(X_train[:batch_n]))
                self.weight_grad = np.array([
                    cost_grad_kl_div(batch_dict, dists_plus[i], dists_minus[i], self.finite_diff_epsilon)
                    for i in range(self.circuit.num_parameters)])
            grad_time = time.time()

            # Update parameters
            param_update, m, v = adam(self.adam_learning_rate, it, self.weight_grad, m, v)
            self.parameters = self.parameters + param_update
            self.parameter_hist.append(self.parameters.copy())

            # step 0 is the linear-baseline reference, so iteration `it` logs at step it + 1
            step = it + 1

            # Evaluate held-out losses (every eval_every iterations and on the final iteration)
            do_eval = (it % eval_every == 0) or (it == iterations - 1)
            test_bench = {}
            if do_eval:
                mmd_train = cost_mmd_pre(T_train, T_train_probs, S, S_probs, sigmas)
                mmd_val = cost_mmd_pre(T_val, T_val_probs, S, S_probs, sigmas) if T_val is not None else np.nan
                if len(X_test_count):
                    test_bench = evaluate(S_counts, {"test": X_test_count}, dataset_kind,
                                          sigmas=sigmas, valid_patterns=valid_patterns,
                                          train_patterns=X_train_count)
                    self.test_bench_hist.append({"iteration": step, **test_bench})
            else:
                mmd_train = mmd_val = np.nan
            self.losses["mmd_train"].append(mmd_train)
            self.losses["mmd_val"].append(mmd_val)
            eval_time = time.time()

            # Keep the checkpoint (pre-update params) with the best metric. Test-set metrics are
            # deliberately NOT selectable -- using the held-out test split would leak it into
            # selection. best_iter indexes the same step where this checkpoint's metrics appear.
            if do_eval:
                sel = {"mmd_train": mmd_train, "mmd_val": mmd_val}[model_selection_metric]
                if np.isfinite(sel) and sel < self.best_metric:
                    self.best_metric = float(sel)
                    self.best_iter = step
                    self.best_params = params_snapshot

            # Every iteration gets its own row (cumulative_measurements is the MMD-vs-measurements
            # x-axis, so it must not be sparse); the rows are buffered and pushed in chunks by the
            # logger, not sent one by one -- see src/wandb_logging.py. time/total_s is the whole step
            # minus the wandb call, whose cost lands in the next step's time/wandb_s (see _log_step).
            total_s = eval_time - start_time
            if wandb_run is not None:
                log = {
                    "train/step": step,
                    "train/measurements_per_step": measurements_per_step,
                    "train/cumulative_measurements": self.total_measurements,
                    "train/num_parameters": self.circuit.num_parameters,
                    "time/sampling_s": sample_time - start_time,
                    "time/gradient_s": grad_time - sample_time,
                    "time/eval_s": eval_time - grad_time,
                    "time/total_s": total_s,
                }
                if do_eval:
                    log.update({"train/mmd_train": mmd_train, "train/mmd_val": mmd_val,
                               "train/mmd_test": test_bench.get("bench_dist/test/mmd", np.nan)})
                    log.update(test_bench)
                logging_s = self._log_step(wandb_run, log, step)
            else:
                logging_s = 0.0

            # Final logging
            logger.debug(f"| Total = {np.round(total_s, 2)} s | Sampling = {np.round(sample_time - start_time, 2)} s | Gradient = {np.round(grad_time - sample_time, 2)} s | Eval = {np.round(eval_time - grad_time, 2)} s | Logging = {np.round(logging_s, 2)} s")
            test_mmd_str = f"{test_bench['bench_dist/test/mmd']:.6f}" if "bench_dist/test/mmd" in test_bench else "n/a"
            logger.debug(f"| MMD loss | Train = {np.round(mmd_train, 6)} | Val = {np.round(mmd_val, 6)} | Test = {test_mmd_str}")

        # Push whatever is still buffered before the caller moves on to saving/artifact upload, so a
        # partially-filled last chunk isn't held until run.finish().
        if wandb_run is not None:
            wandb_run.flush(force=True)

        logger.info(f"Training finished | best {model_selection_metric} = {np.round(self.best_metric, 6)} @ iter {self.best_iter}")

    def save(self, save_dir: str):
        '''Save the model, including both the best (validation-selected) and final checkpoints.'''

        losses_df = pd.DataFrame(self.losses)
        losses_df.to_parquet(f"{save_dir}/losses.parquet")

        if self.test_bench_hist:
            pd.DataFrame(self.test_bench_hist).to_parquet(f"{save_dir}/test_metrics.parquet")

        np.save(f"{save_dir}/params.npy", np.array(self.parameter_hist, dtype=object), allow_pickle=True)

        # best (lowest model_selection_metric) and final (last iteration) checkpoints can differ
        # whenever training doesn't monotonically improve on the selection metric
        np.save(f"{save_dir}/best_params.npy", np.asarray(self.best_params, dtype=float))
        np.save(f"{save_dir}/final_params.npy", np.asarray(self.parameters, dtype=float))
        with open(f"{save_dir}/checkpoint_meta.json", "w") as f:
            json.dump({
                "best_iter": self.best_iter,
                "best_metric": self.best_metric,
                "final_iter": len(self.parameter_hist) - 1,
                "model_selection_metric": self.model_selection_metric,
                "total_measurements": self.total_measurements,
                "num_parameters": int(self.circuit.num_parameters),
                # length actually run: under a measurement budget this is circuit-dependent, so it is
                # not recoverable from qcbm.iterations in the config alone
                "iterations_run": self.iterations_run,
                "measurements_per_step": self.measurements_per_step,
            }, f, indent=2)

        with open(f"{save_dir}/circuit.qpy", 'wb') as file:
            qpy.dump(self.circuit, file)

    def load(self):
        raise NotImplementedError("Loading not implemented yet")
