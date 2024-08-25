import configparser
import sys
import time
import warnings

from copy import deepcopy

import torch
import torch.nn.utils as nn_utils
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf

# 設定の読み込み
config = configparser.ConfigParser()
config_path = "./config.ini"
config.read(config_path)
PROJECT_DIR = config["paths"]["project_dir"]
EXPT_RESULT_DIR = config["paths"]["results_dir"]
LOG_DIR = config["paths"]["logs_dir"]
sys.path.append(PROJECT_DIR)

from src.bnn import BayesianMLPModel
# from src.bnn import fit_pytorch_model
# from src.objectives_botorch import WarcraftObjectiveBoTorch
# from src.objectives_botorch import generate_initial_data
from src.utils_experiment import negate_function
from src.utils_experiment import generate_integer_samples

import logging
# from src.utils_experiment import log_print
from src.utils_experiment import set_logger


def fit_pytorch_model_with_constraint(
    model, acqf, num_epochs=1000, learning_rate=0.01
):
    def g(X):
        """
        制約：x1 == x2
        """
        X1 = X[:, 0]
        X2 = X[:, 1]

        return (X1 == X2).float().unsqueeze(1)

    # def g(X):
    #     constraint1 = X[:, 5] == -3. 
    #     constraint2 = X[:, 3] == -3. 
    #     constraint = constraint1 & constraint2
    #     return constraint.float().unsqueeze(1)
    
    # def g(X):
    #     constraint = X[:, 5] == -3. 
    #     return constraint.float().unsqueeze(1)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    lambda1 = torch.tensor(
        0.5, device=model.train_inputs.device, dtype=model.train_inputs.dtype
    )
    lambda2 = 1 - lambda1

    X = model.train_inputs
    y = model.train_targets
    m = y.size(0)
    f = model

    for _ in range(num_epochs):
        optimizer.zero_grad()

        g_eval = g(X)

        acqf_eval = []
        for x in X:
            x = x.unsqueeze(1).reshape(1, -1).to(X.device, dtype=X.dtype)
            acqf_eval.append(acqf(x))
        acqf_eval = torch.stack(acqf_eval).reshape(-1, 1)

        ones = torch.ones_like(g_eval)

        loss = (
            lambda1 * (ones - g_eval).T * (-1) @ acqf_eval
            + lambda2 * (-f(X).log_prob(y).T @ g_eval)
        ) / m

        loss.backward()
        nn_utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    return loss.item()


def run_bo(setting_dict):
    logging.info(f"Running optimization with settings: \n{setting_dict}")
    device = setting_dict["device"]
    logging.info(f"Running on device: {device}")
    start_time = time.time()

    # ---------------------------------------------------------------------------------------------
    # Step 1: Define the Objective Function
    def objective_function(X):
        return (X ** 2).sum(dim=-1)
    
    objective_function = negate_function(objective_function)

    bounds = torch.tensor([[-10, 10], [-10, 10], [-10, 10], [-10, 10], [-10, 10]]).to(device)

    # ---------------------------------------------------------------------------------------------
    # Step 2: Generate Initial Data
    initial_data_size = setting_dict["initial_data_size"]
    X_train = generate_integer_samples(bounds, initial_data_size).float()
    y_train = objective_function(X_train)

    logging.info("Initial data points and corresponding function values:")
    for i in range(initial_data_size):
        logging.info(f"Candidate: {X_train[i].cpu().numpy()}, Function Value: {y_train[i].item()}")

    # Flatten X_train and move it to the correct device
    n_samples = X_train.shape[0]
    X_train_flat = X_train.view(n_samples, -1).float().to(device)
    y_train = y_train.to(device)

    # ---------------------------------------------------------------------------------------------
    # Step 3: Train the Bayesian MLP Model
    model_settings = setting_dict["model"]
    model = BayesianMLPModel(
        X_train_flat,
        y_train,
        hidden_unit_size=model_settings["hidden_unit_size"],
        num_hidden_layers=model_settings["num_hidden_layers"],
    ).to(device)

    acq_optim_settings = setting_dict["acquisition_optim"]
    beta = deepcopy(acq_optim_settings["beta"])

    ucb = UpperConfidenceBound(model, beta=beta)
    model_optim_settings = setting_dict["model_optim"]
    final_loss = fit_pytorch_model_with_constraint(
        model,
        ucb,
        num_epochs=model_optim_settings["num_epochs"],
        learning_rate=model_optim_settings["learning_rate"],
    )

    # 最終的なロスのみをログに記録
    # if final_loss:
    #     log_print(f"Final training loss: {final_loss:.6f}")
    # else:
    #     log_print(f"Final training loss: {final_loss}")
    logging.info(f"Final training loss: {final_loss:.6f}")

    # Repeat optimization for a specified number of iterations
    n_iterations = setting_dict["bo_iter"]
    best_value = float('-inf')

    for iteration in range(n_iterations):
        iter_start_time = time.time()
        
        logging.info(f"Iteration {iteration + 1}/{n_iterations}")
        
        # ---------------------------------------------------------------------------------------------
        # Step 4: Define the Acquisition Function
        acq_optim_settings = setting_dict["acquisition_optim"]
        # ucb = UpperConfidenceBound(model, beta=beta)

        candidate_flat, acq_value = optimize_acqf(
            acq_function=ucb,
            bounds=bounds,
            q=1,
            num_restarts=acq_optim_settings["num_restarts"],
            raw_samples=acq_optim_settings["raw_samples"],
        )

        candidate_flat = torch.round(candidate_flat).to(device)
        min_key, max_key = -10, 10

        # ---------------------------------------------------------------------------------------------
        # Step 5: Optimize the Acquisition Function
        candidate_flat = torch.clamp(candidate_flat, min=min_key, max=max_key)
        candidate = candidate_flat
        y_new = objective_function(candidate).to(device)

        logging.info(f"Candidate: {candidate_flat.cpu().numpy()}")
        logging.info(f"Acquisition Value: {acq_value.item()}")
        logging.info(f"Function Value: {y_new.item()}")

        X_train = torch.cat([X_train.to(device), candidate.unsqueeze(0).to(device)])
        y_train = torch.cat(
            [y_train.to(device), y_new.unsqueeze(0).unsqueeze(-1).to(device)]
        )
        X_train_flat = X_train.view(X_train.shape[0], -1).float().to(device)

        # ---------------------------------------------------------------------------------------------
        # Update and refit the Bayesian MLP model
        model.set_train_data(X_train_flat, y_train)
        final_loss = fit_pytorch_model_with_constraint(
            model,
            ucb,
            num_epochs=model_optim_settings["num_epochs"],
            learning_rate=model_optim_settings["learning_rate"],
        )

        # if final_loss:
        #     logging.info(f"Final training loss: {final_loss:.6f}")
        # else:
        #     logging.info(f"Final training loss: {final_loss}")
        logging.info(f"Final training loss: {final_loss:.6f}")

        if y_new.item() > best_value:
            best_value = y_new.item()
            logging.info(f"New best value found: {best_value}")

        iter_end_time = time.time()
        elapsed_time = iter_end_time - iter_start_time
        logging.info(f"Iteration time: {elapsed_time:.4f} seconds")

    # ---------------------------------------------------------------------------------------------
    logging.info("Optimization completed.")
    optim_idx = y_train.argmax()
    logging.info(
        f"Optimal solution: \n{X_train[optim_idx]}, \nFunction value: {y_train[optim_idx].item()}"
    )

    end_time = time.time()
    logging.info(f"Total time on {device}: {end_time - start_time:.4f} seconds")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = __file__.split("/")[-1].strip(".py")

    settings = {
        "name": name,
        "device": device,
        "bo_iter": 10,
        "initial_data_size": 10,
        "model": {
            "hidden_unit_size": 64,
            "num_hidden_layers": 3,
        },
        "model_optim": {
            "num_epochs": 1000,
            "learning_rate": 0.01,
        },
        "acquisition_optim": {
            "beta": 0.1,
            "num_restarts": 5,
            "raw_samples": 20,
        },
        "memo": "",
    }

    set_logger(settings["name"], LOG_DIR)

    run_bo(settings)