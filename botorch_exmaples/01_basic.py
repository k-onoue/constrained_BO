import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

from utils import plot_history, plot_best_1D


# 最適化対象の関数
def objective_function(X):
    return torch.sin(10 * X) * X + torch.cos(2 * X)


def initialize_data():
    train_X = torch.linspace(0, 1, 5).unsqueeze(-1)
    train_Y = objective_function(train_X)
    return train_X, train_Y


def build_and_fit_model(train_X, train_Y):
    gp = SingleTaskGP(train_X, train_Y)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    return gp


def define_acquisition_function(model):
    return UpperConfidenceBound(model, beta=0.1)


def optimize_acquisition_function(acq_func, bounds):
    candidate, acq_value = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=1,
        num_restarts=5,
        raw_samples=20,
    )
    return candidate


def update_data(train_X, train_Y, new_X, new_Y):
    train_X = torch.cat([train_X, new_X])
    train_Y = torch.cat([train_Y, new_Y])
    return train_X, train_Y


def main():
    train_X, train_Y = initialize_data()
    bounds = torch.tensor([[0.0], [1.0]])

    best_values = []

    for _ in range(10):
        model = build_and_fit_model(train_X, train_Y)
        acq_func = define_acquisition_function(model)
        new_X = optimize_acquisition_function(acq_func, bounds)
        new_Y = objective_function(new_X)
        train_X, train_Y = update_data(train_X, train_Y, new_X, new_Y)

        best_values.append(train_Y.max().item())

    plot_history(best_values)
    plot_best_1D(train_X, train_Y, model, objective_function)


if __name__ == "__main__":
    main()
