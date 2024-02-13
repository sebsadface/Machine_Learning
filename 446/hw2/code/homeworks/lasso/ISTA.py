from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from utils import problem


@problem.tag("hw2-A")
def step(
    X: np.ndarray, y: np.ndarray, weight: np.ndarray, bias: float, _lambda: float, eta: float
) -> Tuple[np.ndarray, float]:
    """Single step in ISTA algorithm.
    It should update every entry in weight, and then return an updated version of weight along with calculated bias on input weight!

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        weight (np.ndarray): An (d,) array. Weight returned from the step before.
        bias (float): Bias returned from the step before.
        _lambda (float): Regularization constant. Determines when weight is updated to 0, and when to other values.
        eta (float): Step-size. Determines how far the ISTA iteration moves for each step.

    Returns:
        Tuple[np.ndarray, float]: Tuple with 2 entries. First represents updated weight vector, second represents bias.

    """
    err =  X @ weight + bias - y # (n, )
    b = bias - 2 * eta * np.sum(err)
    w = weight - 2 * eta * X.T @ err
    w = np.sign(w) * np.maximum(np.abs(w) - 2 * _lambda * eta, 0)

    return w, b


@problem.tag("hw2-A")
def loss(
    X: np.ndarray, y: np.ndarray, weight: np.ndarray, bias: float, _lambda: float
) -> float:
    """L-1 (Lasso) regularized SSE loss.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        weight (np.ndarray): An (d,) array. Currently predicted weights.
        bias (float): Currently predicted bias.
        _lambda (float): Regularization constant. Should be used along with L1 norm of weight.

    Returns:
        float: value of the loss function
    """
    return np.sum((X @ weight + bias - y) ** 2) + _lambda * np.sum(np.abs(weight))


@problem.tag("hw2-A", start_line=5)
def train(
    X: np.ndarray,
    y: np.ndarray,
    _lambda: float = 0.01,
    eta: float = 0.00001,
    convergence_delta: float = 1e-4,
    start_weight: np.ndarray = None,
    start_bias: float = None
) -> Tuple[np.ndarray, float]:
    """Trains a model and returns predicted weight and bias.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        _lambda (float): Regularization constant. Should be used for both step and loss.
        eta (float): Step size.
        convergence_delta (float, optional): Defines when to stop training algorithm.
            The smaller the value the longer algorithm will train.
            Defaults to 1e-4.
        start_weight (np.ndarray, optional): Weight for hot-starting model.
            If None, defaults to array of zeros. Defaults to None.
            It can be useful when testing for multiple values of lambda.
        start_bias (np.ndarray, optional): Bias for hot-starting model.
            If None, defaults to zero. Defaults to None.
            It can be useful when testing for multiple values of lambda.

    Returns:
        Tuple[np.ndarray, float]: A tuple with first item being array of shape (d,) representing predicted weights,
            and second item being a float representing the bias.

    Note:
        - You will have to keep an old copy of weights for convergence criterion function.
            Please use `np.copy(...)` function, since numpy might sometimes copy by reference,
            instead of by value leading to bugs.
        - You might wonder why do we also return bias here, if we don't need it for this problem.
            There are two reasons for it:
                - Model is fully specified only with bias and weight.
                    Otherwise you would not be able to make predictions.
                    Training function that does not return a fully usable model is just weird.
                - You will use bias in next problem.
    """
    w = np.zeros(X.shape[1]) if start_weight is None else start_weight
    b = 0 if start_bias is None else start_bias
    old_w: Optional[np.ndarray] = None
    old_b: Optional[np.ndarray] = None

    while True:
        old_w = np.copy(w)
        old_b = b
        w, b = step(X, y, w, b, _lambda, eta)
        if convergence_criterion(w, old_w, b, old_b, convergence_delta):
            break

    return w, b


@problem.tag("hw2-A")
def convergence_criterion(
    weight: np.ndarray, old_w: np.ndarray, bias: float, old_b: float, convergence_delta: float
) -> bool:
    """Function determining whether weight and bias has converged or not.
    It should calculate the maximum absolute change between weight and old_w vector, and compare it to convergence delta.
    It should also calculate the maximum absolute change between the bias and old_b, and compare it to convergence delta.

    Args:
        weight (np.ndarray): Weight from current iteration of gradient descent.
        old_w (np.ndarray): Weight from previous iteration of gradient descent.
        bias (float): Bias from current iteration of gradient descent.
        old_b (float): Bias from previous iteration of gradient descent.
        convergence_delta (float): Aggressiveness of the check.

    Returns:
        bool: False, if weight and bias has not converged yet. True otherwise.
    """
    return np.max(np.abs(weight - old_w)) < convergence_delta and np.abs(bias - old_b) < convergence_delta

@problem.tag("hw2-A")
def main():
    """
    Use all of the functions above to make plots.
    """
    n = 500
    d = 1000
    k = 100
    sigma = 1

    epsilon = np.random.randn(n) * sigma
    X = np.random.randn(n, d)
    w = np.array([j/k if j <= k else 0 for j in range(1, d + 1)])
    print(w)
    y = X @ w + epsilon

    # standardize
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    # see Equation (1)
    lambda_max = 2 * np.abs(X.T @ (y - np.mean(y))).max()

    eta = 1e-4
    lam = lambda_max
    convergence_delta = 1e-4

    weights = []
    lambdas = []
    fdr_s = []
    tpr_s = []
    non_zero_counts = []

    while True:
        weight, _ = train(X, y, _lambda=lam, eta=eta, convergence_delta=convergence_delta)

        chosen_w = weight != 0
        actual_w = w != 0
        false_discov = np.logical_and(chosen_w, ~actual_w)
        true_pos = np.logical_and(chosen_w, actual_w)
        fdr = false_discov.sum() / chosen_w.sum() if chosen_w.sum() > 0 else 0
        tpr = true_pos.sum() / k

        fdr_s.append(fdr)
        tpr_s.append(tpr)
        lambdas.append(lam)
        weights.append(weight)
        non_zero_counts.append(chosen_w.sum())

        print ("Lambda:", lam, " Non-zero weights:", non_zero_counts[-1], " FDR:", fdr, " TPR:", tpr)

        if non_zero_counts[-1] == d:
            break

        lam /= 2

    plt.plot(lambdas, non_zero_counts)
    plt.xscale('log')
    plt.xlabel('λ')
    plt.ylabel('Number of non-zero weights')
    plt.title('The number of non-zero weights as a function of λ')
    plt.show()

    plt.plot(fdr_s, tpr_s)
    plt.xlabel('FDR')
    plt.ylabel('TPR')
    plt.title('FDR vs TPR for different values of λ')
    plt.show()

if __name__ == "__main__":
    main()
