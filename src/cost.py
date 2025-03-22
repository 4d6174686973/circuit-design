import numpy as np
from src.utils import sample_info
from concurrent.futures import ThreadPoolExecutor
import math

# optimizer
def adam(learning_rate_init, timestep, gradient, m, v, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):
    '''
    Method to compute Adam learning rate which includes momentum
    Parameters, beta1, beta2, epsilon are as recommended in orginal Adam paper
    m, v are the first and second moments of the gradient, to be estimated
    '''
    timestep = timestep +1
    m           = np.multiply(beta1, m) + np.multiply((1-beta1), gradient)
    v           = np.multiply(beta2, v) + np.multiply((1-beta2), gradient**2)
    corrected_m = np.divide(m , (1 - beta1**timestep))
    corrected_v = np.divide(v, (1 - beta2**timestep))
    param_update = learning_rate_init * (np.divide(corrected_m, np.sqrt(corrected_v) + epsilon))
    return param_update, m, v

# this functions is currently not in use
def kernel_matrix_slow(X: np.ndarray, Y: np.ndarray, sigmas: np.ndarray) -> np.ndarray:	
    '''Kernel Matrix between two sets of samples averaged over various bandwidths
    Args:
        x: Sample Set 1
        y: Sample Set 2
        sigmas: Bandwidths for Gaussian kernel
    Returns:
        Kernel Matrix
    ---
    Example:
    >>> x = np.array([0,0,0,0])
    >>> y = np.array([1,1,1,1])
    >>> sigmas = np.array([0.1, 0.5, 1.0, 2, 5, 10])
    >>> gaussian_kernel(x, y, sigmas)
    '''
    K = np.zeros((len(X), len(Y)))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            K[i, j] = np.mean(np.exp( (-1 / (2 * sigmas)) * (np.linalg.norm(np.abs(x - y), 2)**2)))
    return K

# parallelized version of kernel_matrix
def kernel_matrix(X: np.ndarray, Y: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    '''Vectorized computation of kernel matrix between two sets of samples, max about 900 samples per set'''
    X_ext = np.tile(np.expand_dims(X, axis=1), (1, len(Y), 1))
    Y_ext = np.tile(np.expand_dims(Y, axis=0), (len(X), 1, 1))
    l2_norm_matrix = np.linalg.norm(np.abs(X_ext - Y_ext), axis=2)**2
    sigma_factor = -1/(2*sigma)
    l2_norm_matrix_ext = np.tile(np.expand_dims(l2_norm_matrix, axis=2), (1, 1, len(sigma)))
    return np.mean(np.exp(sigma_factor * l2_norm_matrix_ext), axis=2)

# for multithreading
def kernel_matrix_chunk(X: np.ndarray, Y: np.ndarray, sigma: np.ndarray, start: int, end: int) -> np.ndarray:
    X_chunk = X[start:end]
    X_ext = np.tile(np.expand_dims(X_chunk, axis=1), (1, len(Y), 1))
    Y_ext = np.tile(np.expand_dims(Y, axis=0), (end - start, 1, 1))
    l2_norm_matrix = np.linalg.norm(np.abs(X_ext - Y_ext), axis=2)**2
    sigma_factor = -1/(2*sigma)
    l2_norm_matrix_ext = np.tile(np.expand_dims(l2_norm_matrix, axis=2), (1, 1, len(sigma)))
    return np.mean(np.exp(sigma_factor * l2_norm_matrix_ext), axis=2)


def kernel_matrix_multithread(X: np.ndarray, Y: np.ndarray, sigma: np.ndarray, num_threads: int = 4) -> np.ndarray:
    '''Multithreaded computation of kernel matrix between two sets of samples'''
    chunk_size = math.ceil(len(X) / num_threads)
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for i in range(0, len(X), chunk_size):
            future = executor.submit(kernel_matrix_chunk, X, Y, sigma, i, min(i + chunk_size, len(X)))
            futures.append(future)
        
        results = [future.result() for future in futures]
    
    return np.vstack(results)


def cost_mmd(target: dict, samples: dict, sigmas: np.ndarray, multithread: bool) -> float:
    """Maximum Mean Discrepancy cost function
    Args:
        cfg: dictionary containing the configuration parameters
        target: target samples as dictionary: {"0101": 10, "1010": 20, ...}
        samples: samples from the quantum circuit as dictionary: {"0101": 10, "1010": 20, ...}
    """

    # Sample information
    S, S_probs = sample_info(samples)
    T, T_probs = sample_info(target)

    # assert S.shape[0] < 900 and T.shape[0] < 900, "Too many samples for vectorized kernel matrix computation -> use loop version"
    # Kernel matrices
    if multithread:    
        SS = kernel_matrix_multithread(S, S, sigmas, 12)
        TT = kernel_matrix_multithread(T, T, sigmas, 12)
        ST = kernel_matrix_multithread(S, T, sigmas, 12)
    else:
        SS = kernel_matrix(S, S, sigmas)
        TT = kernel_matrix(T, T, sigmas)
        ST = kernel_matrix(S, T, sigmas)

    # Cost
    cost    = np.dot(np.dot(S_probs, SS), S_probs) \
            + np.dot(np.dot(T_probs, TT), T_probs) \
            - 2 * np.dot(np.dot(S_probs, ST), T_probs)
    return cost


def cost_grad_mmd(target: dict, samples: dict, plus_samples: dict, minus_samples: dict, sigmas: np.ndarray, multithread: bool) -> float:

    # Sample information
    T, T_probs = sample_info(target)         # T: target
    S, S_probs = sample_info(samples)        # S: samples
    P, P_probs= sample_info(plus_samples)    # P: plus samples
    M, M_probs = sample_info(minus_samples)  # M: minus samples

    if multithread:
        SP = kernel_matrix_multithread(S, P, sigmas, 12)
        SM = kernel_matrix_multithread(S, M, sigmas, 12)
        TP = kernel_matrix_multithread(T, P, sigmas, 12)
        TM = kernel_matrix_multithread(T, M, sigmas, 12)
    else:
        SP = kernel_matrix(S, P, sigmas)
        SM = kernel_matrix(S, M, sigmas)
        TP = kernel_matrix(T, P, sigmas)
        TM = kernel_matrix(T, M, sigmas)

    cost_grad = np.dot(np.dot(S_probs, SM), M_probs) \
                - np.dot(np.dot(S_probs, SP), P_probs) \
                - np.dot(np.dot(T_probs, TM), M_probs) \
                + np.dot(np.dot(T_probs, TP), P_probs)
    
    return cost_grad

def cost_kl_div(target: dict, samples: dict) -> float:
    """Kullback-Leibler divergence cost function from Synergistic Pretraining paper by @MSRudolph (GitHub)"""
    EPS = 1e-8
    KL = 0
    for bitstring, p_data in target.items():
        if bitstring in samples.keys():
            KL += p_data * np.log(p_data) - p_data * np.log(
                max(EPS, samples[bitstring])
            )
        else:
            KL += p_data * np.log(p_data) - p_data * np.log(EPS)
    return KL

def cost_grad_kl_div(target: dict, plus_samples: dict, minus_samples: dict, epsilon: float) -> float:
    """Finite distance gradient of the Kullback-Leibler divergence cost function"""
    kl_plus = cost_kl_div(target, plus_samples)
    kl_minus = cost_kl_div(target, minus_samples)
    grad = (kl_plus - kl_minus) / (2 * epsilon)
    return grad