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

from src.bnn import fit_pytorch_model
# from src.objectives_botorch import WarcraftObjectiveBoTorch
# from src.objectives_botorch import generate_initial_data
from src.utils_experiment import negate_function
from src.utils_experiment import generate_integer_samples
from utils_bo import InputTransformer

import logging

# from src.utils_experiment import log_print
from src.utils_experiment import set_logger



def run_bo(setting_dict):
    try:
        logging.info(f"Running optimization with settings: \n{setting_dict}")
        device = setting_dict["device"]
        logging.info(f"Running on device: {device}")
        start_time = time.time()

        # ---------------------------------------------------------------------------------------------
        # Step 1: Define the Objective Function and Search Space

        def griewank_function(X):
            r"""
            Griewank function implementation in PyTorch.
            
            f(x) = \sum_{i=1}^d \frac{x_i^2}{4000} - \prod_{i=1}^d \cos \left( \frac{x_i}{\sqrt{i}} \right) + 1
            
            Args:
            - X (torch.Tensor): Input tensor of shape (n_samples, n_dimensions)
            
            Returns:
            - torch.Tensor: Output tensor of shape (n_samples,)
            """
            # Ensure X is 2D (n_samples, n_dimensions)
            if X.dim() == 1:
                X = X.unsqueeze(0)
            
            # Sum term
            sum_term = torch.sum(X**2 / 4000, dim=1)
            
            # Product term
            i = torch.arange(1, X.shape[1] + 1, dtype=X.dtype, device=X.device)
            prod_term = torch.prod(torch.cos(X / torch.sqrt(i)), dim=1)
            
            # Griewank function
            return (sum_term - prod_term + 1).unsqueeze(0)

        objective_function = negate_function(griewank_function)

        search_space = torch.tensor([[-10] * 5, [10] * 5]).to(torch.float32).to(device)

        trans = InputTransformer(
            search_space,
            lower_bound=0,
            upper_bound=1
        )

        # ---------------------------------------------------------------------------------------------
        # Step 2: Generate Initial Data
        initial_data_size = setting_dict["initial_data_size"]
        X_train = (
            generate_integer_samples(search_space, initial_data_size).to(device).float()
        )
        y_train = objective_function(X_train)

        logging.info("Initial data points and corresponding function values:")
        for i in range(initial_data_size):
            logging.info(f"Candidate: {X_train[i].cpu().numpy()}")
            logging.info(f"Function Value: {y_train[i].item()}")

        # Flatten X_train and move it to the correct device
        n_samples = X_train.shape[0]
        X_train_normalized = trans.normalize(X_train)
        y_train = y_train.to(device)

        # ---------------------------------------------------------------------------------------------
        # Step 3: Train the Bayesian MLP Model
        model_settings = setting_dict["model"]
        model = BayesianMLPModel(
            X_train_normalized,
            y_train,
            hidden_unit_size=model_settings["hidden_unit_size"],
            num_hidden_layers=model_settings["num_hidden_layers"],
            activation_fn=model_settings["activation_fn"],
            # clipping=True
        ).to(device)

        acq_optim_settings = setting_dict["acquisition_optim"]
        beta = deepcopy(acq_optim_settings["beta"])

        ucb = UpperConfidenceBound(model, beta=beta)
        model_optim_settings = setting_dict["model_optim"]

        final_loss = fit_pytorch_model(
            model,
            num_epochs=model_optim_settings["num_epochs"],
            learning_rate=model_optim_settings["learning_rate"],
        )

        logging.info(f"Final training loss: {final_loss:.6f}")

        # Repeat optimization for a specified number of iterations
        n_iterations = setting_dict["bo_iter"]
        best_value = float("-inf")

        for iteration in range(n_iterations):
            iter_start_time = time.time()

            logging.info(f"Iteration {iteration + 1}/{n_iterations}")

            # ---------------------------------------------------------------------------------------------
            # Step 4: Define and optimize the Acquisition Function
            acq_optim_settings = setting_dict["acquisition_optim"]
            ucb = UpperConfidenceBound(model, beta=beta)
            candidate, acq_value = optimize_acqf(
                acq_function=ucb,
                bounds=search_space,
                q=1,
                num_restarts=acq_optim_settings["num_restarts"],
                raw_samples=acq_optim_settings["raw_samples"],
            )

            candidate = trans.discretize(candidate)
            candidate = trans.clipping(candidate)
            candidate = candidate.squeeze(0)

            y_new = objective_function(candidate).to(device)

            candidate_normaliezed = trans.normalize(candidate.unsqueeze(0))
            pred_dist = model(candidate_normaliezed)
            mean = pred_dist.mean
            covariance = pred_dist.variance

            logging.info(f"Candidate: {candidate.cpu().numpy()}")
            logging.info(f"Suroggate Mean: {mean.item()}")
            logging.info(f"Suroggate Covariance: {covariance.item()}")
            logging.info(f"Acquisition Value: {acq_value.item()}")
            logging.info(f"Function Value: {y_new.item()}")

            X_train = torch.cat([X_train.to(device), candidate.unsqueeze(0).to(device)])
            y_train = torch.cat([y_train.to(device), y_new.unsqueeze(-1).to(device)])

            # ---------------------------------------------------------------------------------------------
            # Update and refit the Bayesian MLP model
            X_train_normalized = trans.normalize(X_train)

            model.set_train_data(X_train_normalized, y_train)
            final_loss = fit_pytorch_model(
                model,
                num_epochs=model_optim_settings["num_epochs"],
                learning_rate=model_optim_settings["learning_rate"],
            )

            logging.info(f"Final training loss: {final_loss:.6f}")

            if y_new.item() > best_value:
                best_value = y_new.item()
                logging.info(f"New best value found: {best_value}")

            iter_end_time = time.time()
            elapsed_time = iter_end_time - iter_start_time
            logging.info(f"Iteration time: {elapsed_time:.4f} seconds")

    except Exception as e:
        logging.exception(e)
        raise e

    # ---------------------------------------------------------------------------------------------
    logging.info("Optimization completed.")
    optim_idx = y_train.argmax()
    logging.info(f"Optimal solution: {X_train[optim_idx]}")
    logging.info(f"Function value: {y_train[optim_idx].item()}")

    end_time = time.time()
    logging.info(f"Total time on {device}: {end_time - start_time:.4f} seconds")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = __file__.split("/")[-1].strip(".py")

    settings = {
        "name": name,
        "device": device,
        "bo_iter": 10000,
        "initial_data_size": 1,
        "model": {
            "hidden_unit_size": 64,
            "num_hidden_layers": 3,
            "activation_fn": torch.nn.Tanh(),
        },
        "model_optim": {
            "num_epochs": 100,
            "learning_rate": 0.001,
        },
        "acquisition_optim": {
            "beta": 0.1,
            "num_restarts": 5,
            "raw_samples": 20,
        },
        "memo": "tanh activation function",
    }

    set_logger(settings["name"], LOG_DIR)

    run_bo(settings)