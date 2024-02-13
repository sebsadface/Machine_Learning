if __name__ == "__main__":
    from ISTA import train  # type: ignore
else:
    from .ISTA import train

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem


@problem.tag("hw2-A", start_line=3)
def main():
    # df_train and df_test are pandas dataframes.
    # Make sure you split them into observations and targets
    df_train, df_test = load_dataset("crime")

    X_train, y_train = df_train.drop('ViolentCrimesPerPop', axis=1).values, df_train['ViolentCrimesPerPop'].values
    X_test, y_test = df_test.drop('ViolentCrimesPerPop', axis=1).values, df_test['ViolentCrimesPerPop'].values

    # standardize
    X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
    X_test = (X_test - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)

    # see Equation (1)
    lambda_max = 2 * np.abs(X_train.T @ (y_train - np.mean(y_train))).max()


    weight = None
    bias = None
    lam = lambda_max
    eta = 1e-5
    convergence_delta = 1e-4

    lambdas = []
    weights = []
    train_mse = []
    test_mse = []
    non_zero_counts = []

    while lam >= 0.01:
        weight, bias = train(X_train, y_train, _lambda=lam, eta=eta, convergence_delta=convergence_delta, start_weight=weight, start_bias=bias)
        lambdas.append(lam)
        weights.append(weight)
        non_zero_count = np.sum(weight != 0)
        non_zero_counts.append(non_zero_count)
        train_mse.append(np.mean((y_train - (X_train @ weight + bias)) ** 2))
        test_mse.append(np.mean((y_test - (X_test @ weight + bias)) ** 2))

        print(f"Lambda: {lam}, Non-zero weights: {non_zero_count}")
        lam /= 2

    # a6c
    plt.plot(lambdas, non_zero_counts)
    plt.xscale('log')
    plt.xlabel('λ')
    plt.ylabel('Number of Non-Zero Weights')
    plt.title('Number of Non-Zero Weights as a Function of λ')
    plt.show()

    a6d_vars = ['agePct12t29', 'pctWSocSec', 'pctUrban', 'agePct65up', 'householdsize']
    a6d_idx = [list(df_train.columns).index(var) for var in a6d_vars]

    for var, idx in zip(a6d_vars, a6d_idx):
        plt.plot(lambdas, [weight[idx] for weight in weights], label=var)
    # a6d
    plt.xscale('log')
    plt.xlabel('λ')
    plt.ylabel('Coefficient')
    plt.title('Regularization Paths for Selected Variables')
    plt.legend()
    plt.show()

    # a6e
    plt.plot(lambdas, train_mse, label='Train MSE')
    plt.plot(lambdas, test_mse, label='Test MSE')
    plt.xscale('log')
    plt.xlabel('λ')
    plt.ylabel('Mean Squared Error')
    plt.title('Train & Test MSE as a Function of λ')
    plt.legend()
    plt.show()

    # a6f
    large_lam = 30
    weight, _ = train(X_train, y_train, _lambda=large_lam, eta=eta, convergence_delta=convergence_delta)

    max_idx = np.argmax(weight)
    min_idx = np.argmin(weight)

    feat = df_train.drop('ViolentCrimesPerPop', axis=1).columns.to_list()

    print(weight)
    print(f"Most positive: {feat[max_idx]}, Value: {weight[max_idx]}")
    print(f"Most negative: {feat[min_idx]}, Value: {weight[min_idx]}")


if __name__ == "__main__":
    main()
