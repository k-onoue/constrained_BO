import logging
from typing import Callable

import torch
import torch.nn.utils as nn_utils
from botorch.acquisition import AcquisitionFunction
from torch import Tensor

from .bnn import BayesianMLPModel


# 1. ユーティリティ関数
def negate_function(func):
    """
    目的関数の符号を反転させる
    """

    def negated_func(X):
        return -func(X)

    return negated_func


def generate_integer_samples(
    bounds, n, device=torch.device("cpu"), dtype=torch.float32
):
    """
    整数をランダムにサンプリングして、n 行 m 列の torch.Tensor を生成します。
    """
    lower_bounds = torch.tensor(bounds[0], device=device, dtype=torch.int)
    upper_bounds = torch.tensor(bounds[1], device=device, dtype=torch.int)

    m = lower_bounds.shape[0]
    samples = set()

    while len(samples) < n:
        new_samples = []
        for _ in range(n):
            sample = []
            for i in range(m):
                sample.append(
                    torch.randint(
                        low=lower_bounds[i].item(),
                        high=upper_bounds[i].item() + 1,
                        size=(1,),
                        device=device,
                    ).item()
                )
            new_samples.append(tuple(sample))

        for sample in new_samples:
            samples.add(sample)

        if len(samples) >= n:
            break

    unique_samples = torch.tensor(list(samples)[:n], device=device, dtype=dtype)
    return unique_samples


# 2. ユーティリティクラス
class InputTransformer:
    """
    計算安定性のために，目的変数を正規化する．
    その他，実験の際に使用する変換処理を提供．
    """

    def __init__(self, search_space, lower_bound=-1, upper_bound=1) -> None:
        self.x_min = search_space[0]
        self.x_max = search_space[1]
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def normalize(self, x: torch.tensor) -> torch.tensor:
        if self.x_max is None:
            self.x_max = x.max()
        if self.x_min is None:
            self.x_min = x.min()
        return (x - self.x_min) / (self.x_max - self.x_min) * (
            self.upper_bound - self.lower_bound
        ) + self.lower_bound

    def denormalize(self, x: torch.tensor) -> torch.tensor:
        return (x - self.lower_bound) / (self.upper_bound - self.lower_bound) * (
            self.x_max - self.x_min
        ) + self.x_min

    def discretize(self, x: torch.tensor) -> torch.tensor:
        return x.round()

    def clipping(self, x: torch.tensor) -> torch.tensor:
        """
        各次元ごとに異なる範囲でのクリッピングを可能にする
        """
        return torch.max(torch.min(x, self.x_max), self.x_min)


# 3. メインロジックに関係する関数
def initialize_model(
    setting_dict: dict,
    device: torch.device,
    X_train_normalized: torch.Tensor,
    y_train: torch.Tensor,
):
    model_settings = setting_dict["model"]
    model = BayesianMLPModel(
        X_train_normalized,
        y_train,
        hidden_unit_size=model_settings["hidden_unit_size"],
        num_hidden_layers=model_settings["num_hidden_layers"],
        activation_fn=model_settings["activation_fn"],
    ).to(device)
    return model


def log_initial_data(
    X_train: torch.Tensor, y_train: torch.Tensor, initial_data_size: int
):
    logging.info("Initial data points and corresponding function values:")
    for i in range(initial_data_size):
        logging.info(f"Candidate: {X_train[i].cpu().numpy()}")
        logging.info(f"Function Value: {y_train[i].item()}")


def evaluate_candidate(model, trans, acqf, candidate, objective_function, device):
    candidate = trans.discretize(candidate)
    candidate = trans.clipping(candidate)
    candidate = candidate.squeeze(0)

    y_new = objective_function(candidate).to(device)

    candidate_normalized = trans.normalize(candidate.unsqueeze(0))
    pred_dist = model(candidate_normalized)
    mean = pred_dist.mean
    covariance = pred_dist.variance

    logging.info(f"Candidate: {candidate.cpu().numpy()}")
    logging.info(f"Suroggate Mean: {mean.item()}")
    logging.info(f"Suroggate Covariance: {covariance.item()}")
    logging.info(f"Acquisition Value: {acqf(candidate.unsqueeze(0)).item()}")
    logging.info(f"Function Value: {y_new.item()}")

    return candidate, y_new


def fit_pytorch_model(
    model: torch.nn.Module, num_epochs: int = 100, learning_rate: float = 0.001
) -> float:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    loss: Tensor = torch.tensor(0.0)
    for _ in range(num_epochs):
        optimizer.zero_grad()
        loss = -model(model.train_inputs).log_prob(model.train_targets).mean()
        loss.backward()
        optimizer.step()

    return loss.item()


def fit_pytorch_model_with_constraint(
    model: torch.nn.Module,
    acqf: AcquisitionFunction,
    g_constraint: Callable[[Tensor], Tensor],
    num_epochs: int = 100,
    learning_rate: float = 0.001,
    lambda1: float = 0.5,
) -> float:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    lambda1: Tensor = torch.tensor(
        lambda1, device=model.train_inputs.device, dtype=model.train_inputs.dtype
    )
    lambda2: Tensor = 1 - lambda1

    X: Tensor = model.train_inputs
    y: Tensor = model.train_targets

    loss: Tensor = torch.tensor(0.0)
    for _ in range(num_epochs):
        optimizer.zero_grad()

        g_eval: Tensor = g_constraint(X)
        acqf_eval: Tensor = acqf(
            X.unsqueeze(0) if X.shape[0] == 1 else X.unsqueeze(1)
        ).unsqueeze(-1)

        loss = compute_constrained_loss(
            model, X, y, g_eval, acqf_eval, lambda1, lambda2
        )

        loss.backward()
        nn_utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    return loss.item()


def compute_constrained_loss(
    model: torch.nn.Module,
    X: Tensor,
    y: Tensor,
    g_eval: Tensor,
    acqf_eval: Tensor,
    lambda1: Tensor,
    lambda2: Tensor,
) -> Tensor:
    ones: Tensor = torch.ones_like(g_eval)
    log_prob_loss: Tensor = -model(X).log_prob(y).T @ g_eval
    constrained_loss: Tensor = (
        lambda1 * (ones - g_eval) * acqf_eval + lambda2 * log_prob_loss
    )

    return constrained_loss.mean()
