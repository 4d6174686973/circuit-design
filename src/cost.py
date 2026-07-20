import numpy as np
from scipy.spatial.distance import cdist
from src.utils import sample_info

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

def kernel_matrix(X: np.ndarray, Y: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    '''Vectorized kernel matrix between two sample sets, averaged over bandwidths.
    Uses scipy.spatial.distance.cdist for the pairwise squared L2 distances, which is
    computed in C without materializing (nX, nY, nqubits) intermediates. For the 0/1
    bitstrings used here this is numerically identical to the previous np.tile version.'''
    d2 = cdist(X, Y, 'sqeuclidean')                      # (nX, nY): ||x - y||_2^2
    sigma = np.asarray(sigma, dtype=float)
    # average the Gaussian kernel over the bandwidths (last axis broadcast)
    return np.mean(np.exp(-d2[..., None] / (2.0 * sigma)), axis=2)


def cost_mmd_pre(T: np.ndarray, T_probs: np.ndarray, S: np.ndarray, S_probs: np.ndarray,
                 sigmas: np.ndarray) -> float:
    """Maximum Mean Discrepancy from pre-extracted sample arrays/probabilities.
    Args:
        T, T_probs: target bitstring array and probabilities (from sample_info)
        S, S_probs: model-sample bitstring array and probabilities (from sample_info)
    """
    SS = kernel_matrix(S, S, sigmas)
    TT = kernel_matrix(T, T, sigmas)
    ST = kernel_matrix(S, T, sigmas)
    cost = np.dot(np.dot(S_probs, SS), S_probs) \
         + np.dot(np.dot(T_probs, TT), T_probs) \
         - 2 * np.dot(np.dot(S_probs, ST), T_probs)
    return cost


def cost_mmd(target: dict, samples: dict, sigmas: np.ndarray) -> float:
    """Maximum Mean Discrepancy cost function (dict interface).
    Args:
        target: target samples as dictionary: {"0101": 10, "1010": 20, ...}
        samples: samples from the quantum circuit as dictionary: {"0101": 10, "1010": 20, ...}
    """
    S, S_probs = sample_info(samples)
    T, T_probs = sample_info(target)
    return cost_mmd_pre(T, T_probs, S, S_probs, sigmas)


def cost_grad_mmd_pre(T: np.ndarray, T_probs: np.ndarray, S: np.ndarray, S_probs: np.ndarray,
                      P: np.ndarray, P_probs: np.ndarray, M: np.ndarray, M_probs: np.ndarray,
                      sigmas: np.ndarray) -> float:
    """MMD cost gradient from pre-extracted arrays/probabilities.
    T: target, S: samples, P: plus-shifted samples, M: minus-shifted samples."""
    SP = kernel_matrix(S, P, sigmas)
    SM = kernel_matrix(S, M, sigmas)
    TP = kernel_matrix(T, P, sigmas)
    TM = kernel_matrix(T, M, sigmas)
    cost_grad = np.dot(np.dot(S_probs, SM), M_probs) \
              - np.dot(np.dot(S_probs, SP), P_probs) \
              - np.dot(np.dot(T_probs, TM), M_probs) \
              + np.dot(np.dot(T_probs, TP), P_probs)
    return cost_grad


def cost_grad_mmd(target: dict, samples: dict, plus_samples: dict, minus_samples: dict,
                  sigmas: np.ndarray) -> float:
    """MMD cost gradient (dict interface)."""
    T, T_probs = sample_info(target)         # T: target
    S, S_probs = sample_info(samples)        # S: samples
    P, P_probs = sample_info(plus_samples)   # P: plus samples
    M, M_probs = sample_info(minus_samples)  # M: minus samples
    return cost_grad_mmd_pre(T, T_probs, S, S_probs, P, P_probs, M, M_probs, sigmas)


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
