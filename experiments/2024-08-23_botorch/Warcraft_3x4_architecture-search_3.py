import configparser
import datetime
import logging
import os
import sys
import time
import warnings

import torch
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

from src.bnn import BayesianMLPModel, fit_pytorch_model
from src.objectives_botorch import (WarcraftObjectiveBoTorch,
                                    generate_initial_data)


def set_logger(log_filename_base):
    # ログの設定
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"{current_time}_{log_filename_base}.log"
    log_filepath = os.path.join(LOG_DIR, log_filename)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_filepath), logging.StreamHandler(sys.stdout)],
    )


def log_print(message):
    print(message)
    logging.info(message)


def run_bo(setting_dict):
    log_print(f"Running optimization with settings: \n{setting_dict}")
    device = setting_dict["device"]
    log_print(f"Running on device: {device}")
    start_time = time.time()

    # ---------------------------------------------------------------------------------------------
    # Step 1: Define the Objective Function
    weight_matrix = torch.Tensor(
        [
            [0.1, 0.4, 0.8, 0.8],
            [0.2, 0.4, 0.4, 0.8],
            [0.8, 0.1, 0.1, 0.2],
        ]
    ).to(device)

    objective_function = WarcraftObjectiveBoTorch(weight_matrix=weight_matrix)

    # ---------------------------------------------------------------------------------------------
    # Step 2: Generate Initial Data
    initial_data_size = setting_dict["initial_data_size"]
    X_train, y_train = generate_initial_data(
        objective_function, initial_data_size, weight_matrix.shape
    )

    log_print("Initial data points and corresponding function values:")
    for i in range(initial_data_size):
        log_print(f"Candidate: {X_train[i].cpu().numpy()}, Function Value: {y_train[i].item()}")

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

    model_optim_settings = setting_dict["model_optim"]
    final_loss = fit_pytorch_model(
        model,
        num_epochs=model_optim_settings["num_epochs"],
        learning_rate=model_optim_settings["learning_rate"],
    )

    # 最終的なロスのみをログに記録
    log_print(f"Final training loss: {final_loss:.6f}")

    # Repeat optimization for a specified number of iterations
    n_iterations = setting_dict["bo_iter"]
    best_value = float('-inf')

    for iteration in range(n_iterations):
        iter_start_time = time.time()
        
        log_print(f"Iteration {iteration + 1}/{n_iterations}")
        
        # ---------------------------------------------------------------------------------------------
        # Step 4: Define the Acquisition Function
        acq_optim_settings = setting_dict["acquisition_optim"]
        ucb = UpperConfidenceBound(model, beta=acq_optim_settings["beta"])

        candidate_flat, acq_value = optimize_acqf(
            acq_function=ucb,
            bounds=torch.tensor(
                [[-3.0] * X_train_flat.shape[1], [3.0] * X_train_flat.shape[1]]
            ).to(device),
            q=1,
            num_restarts=acq_optim_settings["num_restarts"],
            raw_samples=acq_optim_settings["raw_samples"],
        )

        candidate_flat = torch.round(candidate_flat).to(device)
        min_key, max_key = -3, 3

        # ---------------------------------------------------------------------------------------------
        # Step 5: Optimize the Acquisition Function
        candidate_flat = torch.clamp(candidate_flat, min=min_key, max=max_key)
        candidate = candidate_flat.view(weight_matrix.shape).to(device)
        y_new = objective_function(candidate).to(device)

        log_print(f"Candidate: {candidate_flat.cpu().numpy()}")
        log_print(f"Acquisition Value: {acq_value.item()}")
        log_print(f"Function Value: {y_new.item()}")

        X_train = torch.cat([X_train.to(device), candidate.unsqueeze(0).to(device)])
        y_train = torch.cat(
            [y_train.to(device), y_new.unsqueeze(0).unsqueeze(-1).to(device)]
        )
        X_train_flat = X_train.view(X_train.shape[0], -1).float().to(device)

        # ---------------------------------------------------------------------------------------------
        # Update and refit the Bayesian MLP model
        model.set_train_data(X_train_flat, y_train)
        final_loss = fit_pytorch_model(
            model,
            num_epochs=model_optim_settings["num_epochs"],
            learning_rate=model_optim_settings["learning_rate"],
        )

        log_print(f"Final training loss: {final_loss:.6f}")

        if y_new.item() > best_value:
            best_value = y_new.item()
            log_print(f"New best value found: {best_value}")

        iter_end_time = time.time()
        elapsed_time = iter_end_time - iter_start_time
        log_print(f"Iteration time: {elapsed_time:.4f} seconds")

    # ---------------------------------------------------------------------------------------------
    log_print("Optimization completed.")
    optim_idx = y_train.argmax()
    log_print(
        f"Optimal solution: \n{X_train[optim_idx]}, \nFunction value: {y_train[optim_idx].item()}"
    )

    end_time = time.time()
    log_print(f"Total time on {device}: {end_time - start_time:.4f} seconds")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = __file__.split("/")[-1].strip(".py")

    settings = {
        "name": name,
        "device": device,
        "bo_iter": 10000,
        "initial_data_size": 10,
        "model": {
            "hidden_unit_size": 64 * 4,
            "num_hidden_layers": 5,
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

    set_logger(settings["name"])

    run_bo(settings)
