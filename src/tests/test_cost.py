import numpy as np
from src.cost import kernel_matrix, cost_mmd, cost_grad_mmd, adam

# Global variable
precision = 14  # 14 decimal places for testing

def test_adam():
    grad = np.array([0.5, 0.5])
    lr = 0.1
    m = np.zeros_like(grad)
    v = np.zeros_like(grad)
    t = 0

    param_update_test = np.array([1/20, 1/20]) / (np.array([1/2, 1/2]) + 1e-8)

    assert (adam(lr, t, grad, m, v)[0] == param_update_test).all() , 'Test 1 for ADAM failed'
    assert (adam(lr, t, grad, m, v, epsilon=0)[0] == np.array([1/10, 1/10])).all() , 'Test 2 for ADAM failed'

def test_kernel_matrix():

    # Calculated by hand
    X = np.array([[0,0], [0,1]])
    Y = np.array([[0,0], [1,1]])
    sigmas = np.array([0.5])
    XX_test = np.array([[1,1/np.exp(1)], [1/np.exp(1),1]])
    XY_test = np.array([[1,1/np.exp(2)], [1/np.exp(1),1/np.exp(1)]])

    # Test kernel_matrix
    assert (kernel_matrix(X,X,sigmas).round(precision) == XX_test.round(precision)).all(), "Test 1 for kernel_matrix failed"
    assert (kernel_matrix(X,Y,sigmas).round(precision) == XY_test.round(precision)).all(), "Test 2 for kernel_matrix failed"


def test_cost_mmd():
    X = {"00": 2, "01": 1}
    Y = {"00": 1, "11": 2}
    sigmas = np.array([0.5])

    XX_test = np.array([[1,1/np.exp(1)], [1/np.exp(1),1]])
    YY_test = np.array([[1,1/np.exp(2)], [1/np.exp(2),1]])
    XY_test = np.array([[1,1/np.exp(2)], [1/np.exp(1),1/np.exp(1)]])
    # X_probs = np.array([2/3, 1/3])
    # Y_probs = np.array([1/3, 2/3])

    XX_exp = 4/9 * XX_test[0,0] + 2/9 * XX_test[0,1] + 2/9 * XX_test[1,0] + 1/9 * XX_test[1,1]
    YY_exp = 1/9 * YY_test[0,0] + 2/9 * YY_test[0,1] + 2/9 * YY_test[1,0] + 4/9 * YY_test[1,1]
    XY_exp = 2/9 * XY_test[0,0] + 4/9 * XY_test[0,1] + 1/9 * XY_test[1,0] + 2/9 * XY_test[1,1]

    mmd_cost_test = XX_exp + YY_exp - 2 * XY_exp

    assert mmd_cost_test.round(precision) == cost_mmd(X, Y, sigmas).round(precision), "Test for MMD Cost failed"


def test_cost_grad_mmd():
    S = {"00": 2, "01": 2}
    S_plus = {"00": 1, "01": 3}
    S_minus = {"00": 3, "01": 1}
    T = {"00": 3, "01": 1}
    sigmas = np.array([0.5])
    cost_grad_test = (1/2) * ((1/(2*np.exp(1))) - (1/2))
    assert cost_grad_mmd(T, S, S_plus, S_minus, sigmas).round(precision) == cost_grad_test.round(precision), "Test for MMD Cost Gradient failed"
