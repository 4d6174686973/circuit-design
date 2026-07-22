# other imports
import numpy as np
import logging
import json
import pandas as pd
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import time

# qiskit imports
from qiskit.circuit import QuantumCircuit
from qiskit import qpy  # for saving the circuit as file

# own imports
from src.utils import array_to_str, sample_info
from src.cost import adam, cost_mmd_pre, cost_grad_mmd_pre, cost_grad_kl_div
from src.data import DataLoader


class QCBM:
    def __init__(
            self,
            sampler,
            backend,
            circuit: QuantumCircuit = None,
            parameters: np.ndarray = None,
            adam_learning_rate: float = 0.01,  # initial learning rate for Adam optimizer
            finite_diff_epsilon: float = 1.0e-8,  # finite difference epsilon for KL gradient
            gradient_workers: int = 8,  # threads for the per-parameter gradient loop
            use_parameter_binds: bool = True,  # True: direct AerSimulator.run fast path; False: SamplerV2 primitive (hardware)
            ) -> None:

        # member variables
        self.sampler = sampler
        self.backend = backend
        self.circuit: QuantumCircuit = circuit
        self.parameters: np.ndarray = parameters  # current parameters
        assert len(parameters) == circuit.num_parameters, "Number of parameters does not match number of circuit parameters"
        self.adam_learning_rate: float = adam_learning_rate
        self.finite_diff_epsilon: float = finite_diff_epsilon
        self.gradient_workers: int = gradient_workers
        self.use_parameter_binds: bool = use_parameter_binds

        # for training state and results
        self.weight_grad: np.ndarray = np.zeros(self.circuit.num_parameters)  # current weight gradients
        self.parameter_hist: list[np.ndarray] = [self.parameters]  # store all parameters
        self.losses: dict[str, list] = {
            "mmd_train": [],
            "mmd_val": [],
        }  # store all losses during training; held-out test tracking is test_bench_hist below
        # full held-out test benchmark suite (mmd/kl/tv/fidelity + BAS coverage/spurious/gen*) per
        # eval step; variable-key rows (gen/* only for BAS holdout) accumulated as dicts and dumped
        # to test_metrics.parquet in save(). wandb receives the same dict inline each eval step.
        self.test_bench_hist: list[dict] = []

        # best checkpoint (model selection); populated during training
        self.best_params: np.ndarray = self.parameters.copy()
        self.best_iter: int = -1
        self.best_metric: float = np.inf
        self.model_selection_metric: str = "mmd_val"
        self.total_measurements: int = 0  # cumulative circuit measurements over training

    def _run_binds(self, stack: np.ndarray, N_shots: int, circuit=None) -> list:
        """Run a (n, P) stack of parameter sets and return a list of n count dicts.

        Uses Aer's native parameter_binds fast path when available (simulation), else falls back to
        a single 2D-parameter PUB through the SamplerV2 primitive (hardware). `circuit` defaults to
        self.circuit; pass a different one to sample e.g. the linear baseline circuit.
        """
        circuit = circuit if circuit is not None else self.circuit
        if self.use_parameter_binds:
            binds = {p: stack[:, p.index] for p in circuit.parameters}
            result = self.sampler.run(circuit, parameter_binds=[binds], shots=N_shots).result()
            counts = result.get_counts()
            return counts if isinstance(counts, list) else [counts]
        # primitive path (hardware): one PUB with a 2D parameter array
        result = self.sampler.run([(circuit, stack)], shots=N_shots).result()[0]
        meas = result.data.meas
        n = stack.shape[0]
        flat = meas.reshape(n) if meas.shape else meas
        return [flat[i].get_counts() for i in range(n)] if meas.shape else [meas.get_counts()]

    def sample(self, N_shots: int, circuit=None, params: np.ndarray = None) -> dict:
        '''Generates samples from a circuit at given parameters. Defaults to self.circuit at
        self.parameters; pass both to sample an arbitrary circuit instead (e.g. the linear baseline).'''
        circuit = circuit if circuit is not None else self.circuit
        params = params if params is not None else self.parameters
        stack = np.asarray(params).reshape(1, circuit.num_parameters)
        return self._run_binds(stack, N_shots, circuit=circuit)[0]

    def param_shift_sampling(self, parameter_values: np.ndarray, shift: float, N_shots: int = 10000) -> tuple:
        """ Parameter-shift sampling for gradient computation.

        Builds a single (2P+1, P) stack of parameter sets (plus-shifted, minus-shifted, centre) and
        runs them in one Aer native `parameter_binds` call instead of 2P+1 separate submissions.

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

        The target (T) and current-sample (S) arrays/probabilities are extracted once and reused
        for every parameter; only the plus/minus distributions differ per parameter.
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

    def _log_baseline_step(self, baseline_circuit, baseline_params, N_shots: int,
                           T_train, T_train_probs, T_val, T_val_probs, sigmas: np.ndarray,
                           dataset_kind: str, valid_patterns, X_train_count: dict,
                           X_test_count: dict, wandb_run, logger) -> int:
        """Sample the shared LINEAR, UNEXTENDED circuit and log it as step 0 (see the
        baseline_circuit/baseline_params docstring on stochastic_gradient_descent). Mutates
        self.losses/self.test_bench_hist exactly like a training-iteration eval step, but is not a
        selectable checkpoint. Returns the step_offset (1: training iterations start at step 1;
        0: no baseline given, training starts at step 0)."""
        if baseline_circuit is None or baseline_params is None:
            return 0

        from src.benchmark import evaluate  # local import avoids any import-time cycle with setup

        B_counts = self.sample(N_shots, circuit=baseline_circuit, params=baseline_params)
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
        logger.info(f"| Step 0 baseline (linear, unextended) | Train MMD = {np.round(b_train, 6)} "
                    f"| Val MMD = {np.round(b_val, 6)}")
        if wandb_run is not None:
            log0 = {"iteration": 0, "cumulative_measurements": 0, "measurements_per_step": 0,
                    "num_parameters": self.circuit.num_parameters,
                    "mmd_train": b_train, "mmd_val": b_val}
            log0.update(b_bench)
            wandb_run.log(log0, step=0)
        return 1

    def stochastic_gradient_descent(
            self, X_train: np.ndarray, X_train_count: dict, X_val_count: dict, X_test_count: dict,
            iterations: int, N_shots: int, mmd_batch_size: int = 0,
            loss_func: str = 'MMD', sigmas: list = [1.0],
            eval_every: int = 1, model_selection_metric: str = 'mmd_val', wandb_run=None,
            dataset_kind: str = None, valid_patterns: np.ndarray = None,
            baseline_circuit=None, baseline_params: np.ndarray = None):
        """ Stochastic Gradient Descent with parameter-shift / finite-difference sampling.

        The train/validation/test splits are supplied by the caller (computed once upstream) so the
        split is not recomputed here. Validation MMD drives model selection.

        On every eval step the full benchmark suite (src.benchmark.evaluate: test/mmd, test/kl,
        test/tv, test/fidelity, plus BAS coverage/spurious-mass and the Gili et al. gen/*
        generalization metrics) is computed on the TEST split from the current-parameter samples and
        logged to wandb, so the held-out test trajectory -- not just its final-checkpoint value -- is
        tracked. This supersedes tracking a bare `mmd_test` loss: `test/mmd` uses the identical MMD
        kernel/sigmas and is reported alongside the rest of the suite instead of as a separate metric.
        valid_patterns scopes the BAS-only extras (coverage/spurious_mass/gen/*); it is ignored for
        JGB. This logging is skipped (no test/* keys, no test_bench_hist rows) whenever the test
        split is empty (e.g. val_size + train_size == 1).

        baseline_circuit/baseline_params (optional): the shared LINEAR, UNEXTENDED circuit and its
        parameters. When given, its sampling output is evaluated with the same metric suite and
        logged at wandb step 0 (cumulative_measurements=0) -- a common pre-training reference point
        for every connectivity, since they all start from this same circuit. The first training
        iteration is then step 1. The baseline is a reference only: it is NOT a checkpoint of this
        (extended) circuit and never participates in model selection.
        """

        logger = logging.getLogger('QCBM')
        self.model_selection_metric = model_selection_metric
        sigmas = np.array(sigmas)
        from src.benchmark import evaluate  # local import avoids any import-time cycle with setup

        # Parameters
        if loss_func == 'KL':
            shift = self.finite_diff_epsilon
        elif loss_func == 'MMD':
            shift = np.pi / 2
        else:
            raise ValueError("Loss Function not implemented")

        # Variables
        self.weight_grad = np.zeros(self.circuit.num_parameters)  # current weight gradients
        [m, v] = [np.zeros(self.circuit.num_parameters) for _ in range(2)]  # adam variables

        # pre-extract the held-out targets once (unchanged across iterations); the test split is
        # evaluated via the benchmark.evaluate() suite below instead (test/mmd there uses the same
        # kernel/sigmas), so it needs no separate sample_info extraction here.
        T_train, T_train_probs = sample_info(X_train_count)
        T_val, T_val_probs = sample_info(X_val_count) if len(X_val_count) else (None, None)

        measurements_per_step = (2 * self.circuit.num_parameters + 1) * N_shots

        # Step 0 baseline: sample the shared LINEAR, UNEXTENDED circuit (common to every
        # connectivity) so all runs share a pre-training reference point at step 0. Training
        # iterations are then logged at steps 1..iterations (step_offset=1); with no baseline,
        # training starts at step 0 (step_offset=0). Not a selectable checkpoint (different circuit
        # than self.circuit), so it is excluded from model selection.
        step_offset = self._log_baseline_step(
            baseline_circuit, baseline_params, N_shots, T_train, T_train_probs, T_val, T_val_probs,
            sigmas, dataset_kind, valid_patterns, X_train_count, X_test_count, wandb_run, logger)

        # Training loop
        for it in range(iterations):

            logger.info(f" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ")
            logger.info(f"| Iteration {it + 1} / {iterations}")

            # snapshot of the parameters that produce this iteration's samples S
            params_snapshot = self.parameters.copy()

            # target for the gradient step (full train set or a shuffled minibatch)
            if mmd_batch_size == 0:
                T_batch, T_batch_probs = T_train, T_train_probs
            else:
                X_shuffled = X_train.copy()
                np.random.shuffle(X_shuffled)
                batch = Counter(array_to_str(X_shuffled[:mmd_batch_size, :]))
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
                batch_dict = X_train_count if mmd_batch_size == 0 else Counter(array_to_str(X_train[:mmd_batch_size]))
                self.weight_grad = np.array([
                    cost_grad_kl_div(batch_dict, dists_plus[i], dists_minus[i], self.finite_diff_epsilon)
                    for i in range(self.circuit.num_parameters)])
            grad_time = time.time()

            # Update parameters
            param_update, m, v = adam(self.adam_learning_rate, it, self.weight_grad, m, v)
            self.parameters = self.parameters + param_update
            self.parameter_hist.append(self.parameters.copy())

            # Logged step: step 0 is the linear-baseline reference (when present), so training
            # iteration `it` is step it + step_offset. step 1 == the first training iteration.
            step = it + step_offset

            # Evaluate held-out losses (every eval_every iterations and on the final iteration)
            do_eval = (it % eval_every == 0) or (it == iterations - 1)
            test_bench = {}
            if do_eval:
                mmd_train = cost_mmd_pre(T_train, T_train_probs, S, S_probs, sigmas)
                mmd_val = cost_mmd_pre(T_val, T_val_probs, S, S_probs, sigmas) if T_val is not None else np.nan
                # full held-out test-set benchmark suite on the current-parameter samples S_counts;
                # test/mmd (same kernel/sigmas) replaces the old standalone mmd_test loss.
                if len(X_test_count):
                    test_bench = evaluate(S_counts, {"test": X_test_count}, dataset_kind,
                                          sigmas=sigmas, valid_patterns=valid_patterns,
                                          train_patterns=X_train_count)
                    self.test_bench_hist.append({"iteration": step, **test_bench})
            else:
                mmd_train = mmd_val = np.nan
            self.losses["mmd_train"].append(mmd_train)
            self.losses["mmd_val"].append(mmd_val)
            loss_time = time.time()

            # Model selection: keep the checkpoint (pre-update params) with the best metric. Test-set
            # metrics are deliberately NOT selectable here -- using the held-out test split to pick a
            # checkpoint would leak it into model selection, defeating its purpose as a clean,
            # touched-once evaluation set. best_iter is the logged step (it + step_offset), so it
            # indexes the same step where this checkpoint's metrics appear.
            if do_eval:
                sel = {"mmd_train": mmd_train, "mmd_val": mmd_val}[model_selection_metric]
                if np.isfinite(sel) and sel < self.best_metric:
                    self.best_metric = float(sel)
                    self.best_iter = step
                    self.best_params = params_snapshot

            # wandb logging (cumulative_measurements every step for the MMD-vs-measurements x-axis)
            if wandb_run is not None:
                log = {
                    "iteration": step,
                    "measurements_per_step": measurements_per_step,
                    "cumulative_measurements": self.total_measurements,
                    "num_parameters": self.circuit.num_parameters,
                    "time/total_s": grad_time - start_time,
                    "time/sampling_s": sample_time - start_time,
                    "time/gradient_s": grad_time - sample_time,
                    "time/loss_s": loss_time - grad_time,
                }
                if do_eval:
                    log.update({"mmd_train": mmd_train, "mmd_val": mmd_val})
                    # full test-set benchmark suite: test/mmd, test/kl, test/tv, test/fidelity, plus
                    # BAS coverage/spurious_mass/gen/* when applicable
                    log.update(test_bench)
                wandb_run.log(log, step=step)

            # Final logging
            logger.info(f"| Total = {np.round(grad_time - start_time, 2)} s | Sampling = {np.round(sample_time - start_time, 2)} s | Gradient = {np.round(grad_time - sample_time, 2)} s | Loss = {np.round(loss_time - grad_time, 2)} s")
            test_mmd_str = f"{test_bench['test/mmd']:.6f}" if "test/mmd" in test_bench else "n/a"
            logger.info(f"| MMD loss | Train = {np.round(mmd_train, 6)} | Val = {np.round(mmd_val, 6)} | Test = {test_mmd_str}")

        logger.info(f"Training finished | best {model_selection_metric} = {np.round(self.best_metric, 6)} @ iter {self.best_iter}")

    def save(self, save_dir: str):
        '''Save the model, including the best checkpoint for benchmarking.'''

        # save losses as file
        losses_df = pd.DataFrame(self.losses)
        losses_df.to_parquet(f"{save_dir}/losses.parquet")

        # save the per-eval-step test benchmark suite (variable columns; gen/* only for BAS holdout)
        if self.test_bench_hist:
            pd.DataFrame(self.test_bench_hist).to_parquet(f"{save_dir}/test_metrics.parquet")

        # save full parameter history
        np.save(f"{save_dir}/params.npy", np.array(self.parameter_hist, dtype=object), allow_pickle=True)

        # save best checkpoint parameters + metadata for model selection / benchmarking
        np.save(f"{save_dir}/best_params.npy", np.asarray(self.best_params, dtype=float))
        with open(f"{save_dir}/checkpoint_meta.json", "w") as f:
            json.dump({
                "best_iter": self.best_iter,
                "best_metric": self.best_metric,
                "model_selection_metric": self.model_selection_metric,
                "total_measurements": self.total_measurements,
                "num_parameters": int(self.circuit.num_parameters),
            }, f, indent=2)

        # save circuit as file
        with open(f"{save_dir}/circuit.qpy", 'wb') as file:
            qpy.dump(self.circuit, file)

    def load(self):
        raise NotImplementedError("Loading not implemented yet")
