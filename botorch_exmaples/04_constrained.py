"""
https://dro.deakin.edu.au/articles/conference_contribution/Bayesian_Optimization_with_Discrete_Variables/20723074
"""

from scipy.optimize import minimize
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

from utils import plot_history


# 最適化対象の関数
def objective_function(X):
    return torch.sin(10 * X) * X + torch.cos(2 * X)


def initialize_data(search_space, initial_points=1):
    # 各変数の初期サンプルをランダムに選択
    import random

    samples = []
    for key in search_space:
        values = search_space[key]
        indices = random.sample(range(len(values)), initial_points)
        samples.append(values[indices])

    train_X = torch.stack(samples, dim=-1).float()
    train_Y = objective_function(train_X)
    return train_X, train_Y


def build_and_fit_model(train_X, train_Y):
    gp = SingleTaskGP(train_X, train_Y)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    return gp


def define_acquisition_function(
    model,
    search_space,
    beta,
    observed_points,
    optimize_beta_flag=False,
    beta_h=10,
    l_h=2,
):
    def adjust_parameters():
        """
        Adjust the beta and length scale l to avoid sampling pre-existing observations.
        """

        # Define optimization problem to adjust beta and l
        def optimization_objective(params):
            delta_beta, l = params
            l = max(1e-6, l)  # Ensure l is positive
            adjusted_beta = beta + delta_beta
            ucb = UpperConfidenceBound(model, beta=adjusted_beta)
            candidates, _ = optimize_acqf(
                acq_function=ucb,
                bounds=torch.tensor([[0.0], [1.0]]).repeat(1, 1),
                q=1,
                num_restarts=5,
                raw_samples=20,
            )
            candidate = candidates[0].item()
            rounded_candidate = min(
                search_space["x1"], key=lambda x: abs(x - candidate)
            )
            penalty = 0 if rounded_candidate not in observed_points else 1
            return delta_beta + (candidate - rounded_candidate) ** 2 + penalty

        result = minimize(
            optimization_objective,
            [0, 1],
            bounds=[(0, beta_h), (1e-6, l_h)],
            method="L-BFGS-B",
        )
        return result.x

    if optimize_beta_flag:
        # Adjust beta and l to avoid repetition
        delta_beta, _ = adjust_parameters()
        adjusted_beta = beta + delta_beta
    else:
        adjusted_beta = beta

    # Create UCB acquisition function with adjusted beta
    ucb = UpperConfidenceBound(model, beta=adjusted_beta)
    return ucb, adjusted_beta


def optimize_acquisition_function(acq_func, bounds):
    candidate, acq_value = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=1,
        num_restarts=5,
        raw_samples=20,
    )
    return candidate


def round_to_nearest_search_space_point(new_X, search_space):
    rounded_X = torch.zeros_like(new_X)
    for i, key in enumerate(search_space.keys()):
        rounded_X[0, i] = min(
            search_space[key], key=lambda x: abs(x - new_X[0, i].item())
        )
    return rounded_X


def update_data(train_X, train_Y, new_X, new_Y):
    if torch.any(torch.all(train_X == new_X, dim=1)):
        return train_X, train_Y, False  # No update, already exists
    else:
        train_X = torch.cat([train_X, new_X])
        train_Y = torch.cat([train_Y, new_Y])
        return train_X, train_Y, True  # Successful update


def main():
    search_space = {
        "x1": torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        "x2": torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
    }

    train_X, train_Y = initialize_data(search_space)

    bounds = torch.tensor([[0.0], [1.0]])

    beta = 0.1

    best_values = []

    for _ in range(5):
        print()
        print(
            f"Iteration: {_ + 1} ----------------------------------------------------------"
        )
        model = build_and_fit_model(train_X, train_Y)
        acq_func, beta = define_acquisition_function(
            model, search_space, beta=beta, observed_points=train_X.numpy()
        )
        new_X = optimize_acquisition_function(acq_func, bounds)
        rounded_X = round_to_nearest_search_space_point(new_X, search_space)
        new_Y = objective_function(rounded_X)
        train_X, train_Y, success = update_data(train_X, train_Y, rounded_X, new_Y)

        while not success:
            print()
            print("Re-sampling due to repetition")

            acq_func, updated = define_acquisition_function(
                model,
                search_space,
                beta=0.1,
                observed_points=train_X.numpy(),
                optimize_beta_flag=True,
            )
            new_X = optimize_acquisition_function(acq_func, bounds)
            rounded_X = round_to_nearest_search_space_point(new_X, search_space)
            new_Y = objective_function(rounded_X)
            train_X, train_Y, success = update_data(train_X, train_Y, rounded_X, new_Y)

        best_values.append(train_Y.max().item())

    plot_history(best_values)
    # plot_best_1D(train_X, train_Y, model, objective_function)


if __name__ == "__main__":
    main()
