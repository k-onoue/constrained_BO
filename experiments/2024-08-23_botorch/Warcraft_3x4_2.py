import warnings
import os
import sys
import configparser
import datetime
import logging
import time
import csv

import torch
from botorch.optim import optimize_acqf
from botorch.acquisition import UpperConfidenceBound
from scipy.optimize import minimize


warnings.filterwarnings("ignore")

# 設定の読み込み
config = configparser.ConfigParser()
config_path = "./config.ini"
config.read(config_path)
PROJECT_DIR = config["paths"]["project_dir"]
EXPT_RESULT_DIR = config["paths"]["results_dir"]
LOG_DIR = config["paths"]["logs_dir"]
sys.path.append(PROJECT_DIR)

from src.bnn import BayesianMLPModel, fit_pytorch_model
from src.objectives_botorch import WarcraftObjectiveBoTorch, generate_initial_data


# ログの設定
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename_base = "Warcraft_3x4_botorch_2"
log_filename = f"{current_time}_{log_filename_base}.log"
log_filepath = os.path.join(LOG_DIR, log_filename)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_filepath), logging.StreamHandler(sys.stdout)],
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")
logging.info(f"Running on device: {device}")
start_time = time.time()

# Step 1: Define the Objective Function
weight_matrix = torch.Tensor(
    [
        [0.1, 0.4, 0.8, 0.8],
        [0.2, 0.4, 0.4, 0.8],
        [0.8, 0.1, 0.1, 0.2],
    ]
).to(device)

objective_function = WarcraftObjectiveBoTorch(weight_matrix=weight_matrix)

# Step 2: Generate Initial Data
X_train, y_train = generate_initial_data(objective_function, 10, weight_matrix.shape)

# Flatten X_train and move it to the correct device
n_samples = X_train.shape[0]
X_train_flat = X_train.view(n_samples, -1).float().to(device)
y_train = y_train.to(device)

# Step 3: Train the Bayesian MLP Model
model = BayesianMLPModel(
    X_train_flat, y_train, hidden_unit_size=64, num_hidden_layers=3
).to(device)
fit_pytorch_model(model, num_epochs=1000, learning_rate=0.01)

# Repeat optimization for a specified number of iterations
n_iterations = 1000

for iteration in range(n_iterations):
    print(f"Iteration {iteration + 1}/{n_iterations}")
    logging.info(f"Iteration {iteration + 1}/{n_iterations}")
    # Step 4: Define the Acquisition Function
    ucb = UpperConfidenceBound(model, beta=0.1)

    # Step 5: Optimize the Acquisition Function
    candidate_flat, acq_value = optimize_acqf(
        acq_function=ucb,
        bounds=torch.tensor(
            [[-3.0] * X_train_flat.shape[1], [3.0] * X_train_flat.shape[1]]
        ).to(device),
        q=1,
        num_restarts=5,
        raw_samples=20,
    )

    candidate_flat = torch.round(candidate_flat).to(device)
    min_key, max_key = -3, 3
    candidate_flat = torch.clamp(candidate_flat, min=min_key, max=max_key)
    candidate = candidate_flat.view(weight_matrix.shape).to(device)
    y_new = objective_function(candidate).to(device)

    # Update the Model
    X_train = torch.cat([X_train.to(device), candidate.unsqueeze(0).to(device)])
    y_train = torch.cat(
        [y_train.to(device), y_new.unsqueeze(0).unsqueeze(-1).to(device)]
    )
    X_train_flat = X_train.view(X_train.shape[0], -1).float().to(device)

    # Refit the Bayesian MLP model
    model.set_train_data(X_train_flat, y_train)
    fit_pytorch_model(model, num_epochs=1000, learning_rate=0.01)


print("Optimization completed.")
logging.info("Optimization completed.")
optim_idx = y_train.argmax()
print(
    f"Optimal solution: \n{X_train[optim_idx]}, \nFunction value: {y_train[optim_idx].item()}"
)
logging.info(
    f"Optimal solution: \n{X_train[optim_idx]}, \nFunction value: {y_train[optim_idx].item()}"
)

end_time = time.time()
print(f"Total time on {device}: {end_time - start_time} seconds")
logging.info(f"Total time on {device}: {end_time - start_time} seconds")
