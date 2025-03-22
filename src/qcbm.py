# other imports
import numpy as np
import logging
import pandas as pd
from collections import Counter
import time

# qiskit imports
from qiskit.circuit import QuantumCircuit
from qiskit_ibm_runtime import SamplerV2
from qiskit import qpy  # for saving the circuit as file
from qiskit_aer import AerSimulator

# own imports
from src.utils import array_to_str
from src.cost import adam, cost_mmd, cost_grad_mmd, cost_grad_kl_div
from src.data import DataLoader


class QCBM:
    def __init__(
            self,
            sampler: SamplerV2,
            backend: AerSimulator,
            circuit: QuantumCircuit = None,
            parameters: np.ndarray = None,
            adam_learning_rate: float = 0.01,  # initial learning rate for Adam optimizer
            finite_diff_epsilon: float = 1.0e-8,  # finite difference epsilon for KL gradient
            kernel_multithreading: bool = True,  # use multithreading for kernel computation
            ) -> None:

        # member variables
        self.sampler: SamplerV2 = sampler
        self.backend: AerSimulator = backend
        self.circuit: QuantumCircuit = circuit
        self.parameters: np.ndarray = parameters  # current parameters
        assert len(parameters) == circuit.num_parameters, "Number of parameters does not match number of circuit parameters"
        self.adam_learning_rate: float = adam_learning_rate
        self.finite_diff_epsilon: float = finite_diff_epsilon
        self.kernel_multithreading: bool = kernel_multithreading

        # for training state and results
        self.weight_grad: np.ndarray = np.zeros(self.circuit.num_parameters)  # current weight gradients
        self.parameter_hist: list[np.ndarray] = [self.parameters]  # store all parameters 
        self.losses: dict[str, list] = {
            "mmd_train": [],
            "mmd_test": [],
        }  # store all losses during training

    def sample(self, N_shots: int) -> tuple:
        '''Generates samples from the quantum circuit'''
        pub = (self.circuit, self.parameters)
        job = self.sampler.run([pub], shots=N_shots)
        samples = job.result()[0].data.meas.get_counts()
        return samples

    def param_shift_sampling(self, parameter_values: np.ndarray, shift: float,  N_shots: int = 10000) -> tuple:
        """ Finite Difference Sampling for KL Divergence gradient computation
            Implementation inspired by qiskit-algorithms (not maintained by IBM anymore)
        Args:
            parameter_values: List of parameter values for which to compute the gradient
            N_shots: Number of shots for each parameter value
            epsilon: Finite difference step size
        Returns:
            dist_plus: List of dictionaries with counts for the circuit with the shifted parameters
            dist_minus: List of dictionaries with counts for the circuit with the shifted parameters
            dist: List of counts for the circuit with the original parameters
        """

        # Setup variables
        circuits = [self.circuit]
        parameters = [self.circuit.parameters]
        parameter_values = [parameter_values]
        job_circuits, job_param_values, metadata = [], [], []
        all_n = []

        for circuit, parameter_values_, parameters_ in zip(circuits, parameter_values, parameters):

            assert isinstance(parameter_values_, np.ndarray), "Parameters must be a numpy array"

            # Indices of parameters to be differentiated
            indices = [circuit.parameters.data.index(p) for p in parameters_]
            metadata.append({"parameters": parameters_})
            
            # Combine inputs into a single job to reduce overhead.
            offset = np.identity(circuit.num_parameters)[indices, :]
            plus = parameter_values_ + shift * offset
            minus = parameter_values_ - shift * offset
            n = 2 * len(indices) + 1
            job_circuits.extend([circuit] * n)
            job_param_values.extend(plus.tolist() + minus.tolist() + parameter_values_.reshape(1,circuit.num_parameters).tolist())
            all_n.append(n)
        
        # Run the job
        pub = zip(job_circuits, job_param_values)
        job = self.sampler.run(pub, shots=N_shots)
        results = job.result()
        
        # Extract the results
        partial_sum_n = 0
        for n in all_n:
            result = results[partial_sum_n : partial_sum_n + n]
            result = [r.data.meas.get_counts() for r in result]
            dist_plus, dist_minus, dist = result[: (n - 1) // 2], result[(n - 1) // 2 : n - 1], result[n - 1]
        
        return dist_plus, dist_minus, dist

    def stochastic_gradient_descent(
            self, dataloader: DataLoader, iterations: int, N_shots: int, batchsize: int = 0,
            loss_func: str = 'MMD', sigmas: list = [1.0], train_test_split: float = 0.8):
        """ Stochastic Gradient Descent with KL Divergence and Finite Difference Sampling """

        logger = logging.getLogger('QCBM')

        # get split from dataloader
        X_train, _, X_train_count, X_test_count = dataloader.train_test_split(train_test_split)

        # Parameters
        if loss_func == 'KL':
            shift = self.finite_diff_epsilon
        elif loss_func == 'MMD':  
            shift = np.pi / 2
        else:
            raise ValueError("Loss Function not implemented")

        # Variables
        self.weight_grad = np.zeros(self.circuit.num_parameters)  # current weight gradients
        [m, v] = [np.zeros(self.circuit.num_parameters) for _ in range(2)] # adam variables
 
        # Training loop
        for it in range(iterations):

            logger.info(f" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ")
            logger.info(f"| Iteration {it + 1} / {iterations}")

            # Shuffle train set and use minibatch size
            if batchsize == 0:
                batch = X_train_count
            else:
                X_shuffled = X_train.copy()
                np.random.shuffle(X_shuffled)
                X_minibatch = X_shuffled[:batchsize, :]
                batch = Counter(array_to_str(X_minibatch))

            # Parameter Shift Sampling
            start_time = time.time()
            dists_plus, dists_minus, S = self.param_shift_sampling(self.parameters, shift, N_shots)
            sample_time = time.time()
            
            # Compute the gradient
            for i, plus, minus in zip(range(self.circuit.num_parameters), dists_plus, dists_minus):
                if loss_func == 'KL':
                    self.weight_grad[i] = cost_grad_kl_div(batch, plus, minus, self.finite_diff_epsilon)
                elif loss_func == 'MMD':
                    self.weight_grad[i] = cost_grad_mmd(batch, S, plus, minus, np.array(sigmas), multithread=self.kernel_multithreading)
            grad_time = time.time()

            # Update parameters
            param_update, m, v = adam(self.adam_learning_rate, it, self.weight_grad, m, v)
            self.parameters += param_update
            self.parameter_hist.append(self.parameters.copy())

            # Compute loss on train and test set during training for logging
            self.losses["mmd_train"].append(cost_mmd(X_train_count, S, np.array(sigmas), multithread=self.kernel_multithreading))
            self.losses["mmd_test"].append(cost_mmd(X_test_count, S, np.array(sigmas), multithread=self.kernel_multithreading))
            loss_time = time.time()
            
            # Final logging
            mmd_train_loss = np.round(self.losses["mmd_train"][-1], 6)
            mmd_test_loss = np.round(self.losses["mmd_test"][-1], 6)

            logger.info(f"| Total = {np.round(grad_time - start_time, 2)} s | Sampling = {np.round(sample_time - start_time, 2)} s | Gradient = {np.round(grad_time - sample_time, 2)} s | Loss = {np.round(loss_time - grad_time, 2)} s")
            logger.info(f"| MMD loss | Train = {mmd_train_loss} | Test = {mmd_test_loss}")

        logger.info("Training finished")

    def save(self, save_dir: str):
        '''Save the model'''
        
        # save losses as file
        losses_df = pd.DataFrame(self.losses)
        losses_df.to_parquet(f"{save_dir}/losses.parquet")

        # save params as file
        np.save(f"{save_dir}/params.npy", np.array(self.parameter_hist, dtype=object), allow_pickle=True)

        # save circuit as file
        with open(f"{save_dir}/circuit.qpy", 'wb') as file:
            qpy.dump(self.circuit, file)
        
    def load(self):
        raise NotImplementedError("Loading not implemented yet")