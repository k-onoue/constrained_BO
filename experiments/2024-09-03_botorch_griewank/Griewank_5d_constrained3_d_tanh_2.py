import configparser
import logging
import sys
import time
import warnings
from copy import deepcopy

import torch
from botorch.acquisition import UpperConfidenceBound

# 設定の読み込み
config = configparser.ConfigParser()
config_path = "./config.ini"
config.read(config_path)
PROJECT_DIR = config["paths"]["project_dir"]
EXPT_RESULT_DIR = config["paths"]["results_dir"]
LOG_DIR = config["paths"]["logs_dir"]
sys.path.append(PROJECT_DIR)

from src.utils_benchmark_functions import *
from src.utils_bo import (
    InputTransformer,
    adjust_beta,
    evaluate_candidate,
    fit_pytorch_model_with_constraint,
    generate_integer_samples,
    initialize_model,
    log_initial_data,
    optimize_acquisition,
)
from src.utils_experiment import set_logger


def run_bo(setting_dict: dict):
    try:
        logging.info(f"Running optimization with settings: \n{setting_dict}")
        device = setting_dict["device"]
        logging.info(f"Running on device: {device}")
        start_time = time.time()

        ######################################################################
        # Step 1: Define the Objective Function, Search Space and Constraints
        objective_function = griewank_function

        search_space = torch.tensor([[-50] * 5, [600] * 5]).to(torch.float32).to(device)

        trans = InputTransformer(search_space, lower_bound=0, upper_bound=1)

        def g(X):
            """
            制約：x1 == x2
            """
            X1 = X[:, 0]
            X2 = X[:, 1]
            return (X1 == X2).float().unsqueeze(1)

        ######################################################################
        # Step 2: Generate Initial Data
        initial_data_size = setting_dict["initial_data_size"]
        X_train = (
            generate_integer_samples(search_space, initial_data_size).to(device).float()
        )
        y_train = objective_function(X_train)
        log_initial_data(X_train, y_train, initial_data_size)

        # Flatten X_train and move it to the correct device
        X_train_normalized = trans.normalize(X_train)
        y_train = y_train.to(device)

        ######################################################################
        # Step 3: Train the Bayesian MLP Model
        model = initialize_model(setting_dict, X_train_normalized, y_train)
        acq_optim_settings = setting_dict["acquisition_optim"]
        beta = deepcopy(acq_optim_settings["beta"])
        beta_h = setting_dict["acquisition_optim"].get("beta_h", 10.0)

        ucb = UpperConfidenceBound(model, beta=beta)
        model_optim_settings = setting_dict["model_optim"]

        final_loss = fit_pytorch_model_with_constraint(
            model=model,
            acqf=ucb,
            g_constraint=g,
            num_epochs=model_optim_settings["num_epochs"],
            learning_rate=model_optim_settings["learning_rate"],
            lambda1=model_optim_settings["lambda1"],
        )

        logging.info(f"Final training loss: {final_loss:.6f}")

        ######################################################################
        # Step 4: Optimization Iterations
        n_iterations = setting_dict["bo_iter"]
        best_value = float("-inf")
        start_time = time.time()

        for iteration in range(n_iterations):
            iter_start_time = time.time()

            logging.info(f"Iteration {iteration + 1}/{n_iterations}")

            x_new = None

            # Beta adjustment loop
            while x_new is None:
                candidate = optimize_acquisition(ucb, search_space, acq_optim_settings)

                candidate_temp = trans.discretize(candidate)
                candidate_temp = trans.clipping(candidate_temp)
                candidate_temp = candidate_temp.squeeze(0)

                if (candidate_temp == X_train).all(dim=1).any():
                    # Adjust beta
                    beta = adjust_beta(
                        model, X_train, search_space, beta, beta_h, acq_optim_settings
                    )

                    if beta > beta_h:
                        beta = beta_h
                    ucb = UpperConfidenceBound(model, beta=beta)
                else:
                    # Evaluate Candidate
                    candidate, y_new = evaluate_candidate(
                        model, trans, ucb, candidate, objective_function, device
                    )
                    x_new = candidate

            X_train = torch.cat([X_train.to(device), x_new.unsqueeze(0).to(device)])
            y_train = torch.cat([y_train.to(device), y_new.unsqueeze(-1).to(device)])

            # Update and refit the Bayesian MLP model
            X_train_normalized = trans.normalize(X_train)
            model.set_train_data(X_train_normalized, y_train)
            final_loss = fit_pytorch_model_with_constraint(
                model=model,
                acqf=ucb,
                g_constraint=g,
                num_epochs=model_optim_settings["num_epochs"],
                learning_rate=model_optim_settings["learning_rate"],
                lambda1=model_optim_settings["lambda1"],
            )

            logging.info(f"Final training loss: {final_loss:.6f}")

            if y_new.item() > best_value:
                best_value = y_new.item()
                logging.info(f"New best value found: {best_value}")

            elapsed_time = time.time() - iter_start_time
            logging.info(f"Iteration time: {elapsed_time:.4f} seconds")

    except Exception as e:
        logging.exception(e)
        raise e

    # Final results
    logging.info("Optimization completed.")
    optim_idx = y_train.argmax()
    logging.info(f"Optimal solution: {X_train[optim_idx]}")
    logging.info(f"Function value: {y_train[optim_idx].item()}")

    elapsed_time = time.time() - start_time
    logging.info(f"Total time on {device}: {elapsed_time:.4f} seconds")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = __file__.split("/")[-1].strip(".py")

    settings = {
        "name": name,
        "device": device,
        "bo_iter": 1000,
        "initial_data_size": 1,
        "model": {
            "hidden_unit_size": 64,
            "num_hidden_layers": 3,
            "activation_fn": torch.nn.Tanh(),
            "min_val": None,
            "max_val": None,
        },
        "model_optim": {
            "num_epochs": 100,
            "learning_rate": 0.001,
            "lambda1": 0.2,
        },
        "acquisition_optim": {
            "beta": 0.1,
            "num_restarts": 5,
            "raw_samples": 20,
        },
        "memo": "discrete algorithm to avoid duplicate samples",
    }

    set_logger(settings["name"], LOG_DIR)
    run_bo(settings)
