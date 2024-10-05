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

# 設定の読み込み
config = configparser.ConfigParser()
config_path = "./config.ini"
config.read(config_path)
PROJECT_DIR = config["paths"]["project_dir"]
EXPT_RESULT_DIR = config["paths"]["results_dir"]
LOG_DIR = config["paths"]["logs_dir"]
sys.path.append(PROJECT_DIR)

from src.bnn import BayesianMLPModel
from src.utils_experiment import generate_integer_samples, negate_function

# ログの設定
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename_base = "Griewank_3d_strictly_constrained2"
log_filename = f"{current_time}_{log_filename_base}.log"
log_filepath = os.path.join(LOG_DIR, log_filename)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_filepath), logging.StreamHandler(sys.stdout)],
)


def fit_pytorch_model_with_constraint(
    model, bounds, num_epochs=1000, learning_rate=0.01
):
    def g(X, bounds):
        """
        制約：x1 == x2
        """
        X1 = X[:, 0]
        X2 = X[:, 1]

        return (X1 == X2).float().unsqueeze(1)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for _ in range(num_epochs):
        optimizer.zero_grad()

        g_eval = g(model.train_inputs, bounds)
        loss = -(
            model(model.train_inputs).log_prob(model.train_targets).T @ g_eval
        ) / model.train_targets.size(0)

        loss.backward()
        optimizer.step()


class Experiment:
    def __init__(self, config):
        self.config = config
        self.bounds = config["bounds"]
        self.objective_function = config["objective_function"]
        self.train_x, self.train_y = self.generate_initial_data(
            config["initial_points"]
        )
        self.model = self.initialize_model(self.train_x, self.train_y)
        self.best_values = []
        self.best_x_values = []

        self.beta = config["algo_params"].get("beta", 2.0)
        self.beta_h = config["algo_params"].get("beta_h", 10.0)
        self.time_budget = config.get("time_budget", None)  # time_budget (秒) を追加
        self.start_time = None  # 開始時間を保存するための属性を追加

    def generate_initial_data(self, n):
        train_x = generate_integer_samples(self.bounds, n).float()
        train_y = self.objective_function(train_x).unsqueeze(-1)
        return train_x, train_y

    def initialize_model(self, train_x, train_y):
        model = BayesianMLPModel(train_x, train_y)
        return model

    def acquisition_function(self, beta):
        return UpperConfidenceBound(self.model, beta=beta)

    def optimize_acquisition(self, acq_function):
        try:
            candidates, _ = optimize_acqf(
                acq_function,
                bounds=self.bounds,
                q=1,
                num_restarts=self.config["num_restarts"],
                raw_samples=self.config["raw_samples"],
            )
        except RuntimeError as e:
            logging.warning(f"RuntimeError during acquisition optimization: {e}")
            return None

        if torch.isnan(candidates).any() or torch.isinf(candidates).any():
            logging.warning("Candidates contain NaN or Inf values")
            return None

        return candidates.detach()

    def adjust_beta(self):
        def objective(params):
            delta_beta = params[0]
            adjusted_beta = self.beta + delta_beta
            acq_function = self.acquisition_function(adjusted_beta)
            new_x = self.optimize_acquisition(acq_function)
            if new_x is None:
                return float("inf")
            rounded_new_x = torch.round(new_x)

            penalty = float("inf")
            if (rounded_new_x == self.train_x).all(dim=1).any():
                penalty = 1000

            return delta_beta + torch.norm(new_x - rounded_new_x).item() + penalty

        initial_guess = [0.0]
        bounds = [(0.0, self.beta_h)]

        result = minimize(objective, initial_guess, bounds=bounds, method="L-BFGS-B")
        delta_beta = result.x[0]
        self.beta += delta_beta

    def get_new_candidate(self):
        acq_function = self.acquisition_function(self.beta)
        new_x = self.optimize_acquisition(acq_function)
        if new_x is None:
            return None

        new_x = torch.round(new_x)

        cnt = 1

        while (new_x == self.train_x).all(dim=1).any():
            cnt += 1

            elapsed_time = time.time() - self.start_time
            if self.time_budget and elapsed_time > self.time_budget:
                logging.info(
                    f"Stopping candidate search due to time limit during beta adjustment. Elapsed time: {elapsed_time:.2f} seconds"
                )
                return None

            logging.info(f"Trying to get new candidate with beta = {self.beta}")

            self.adjust_beta()
            acq_function = self.acquisition_function(self.beta)
            new_x = self.optimize_acquisition(acq_function)
            if new_x is None:
                return None
            new_x = torch.round(new_x)

        return new_x

    def optimize_acqf_and_get_observation(self):
        new_x = self.get_new_candidate()
        if new_x is None:
            return None, None

        new_y = self.objective_function(new_x).unsqueeze(-1)
        return new_x, new_y

    def run(self):
        self.start_time = time.time()  # 開始時間を記録

        for iteration in range(1, self.config["n_iterations"] + 1):
            # 経過時間のチェック
            elapsed_time = time.time() - self.start_time
            if self.time_budget and elapsed_time > self.time_budget:
                logging.info(
                    f"Stopping optimization due to time limit. Elapsed time: {elapsed_time:.2f} seconds"
                )
                break

            fit_pytorch_model_with_constraint(self.model, self.bounds)

            new_x, new_y = self.optimize_acqf_and_get_observation()
            if new_x is None or new_y is None:
                logging.info("Stopping optimization due to numerical issues.")
                break

            self.train_x = torch.cat([self.train_x, new_x])
            self.train_y = torch.cat([self.train_y, new_y])
            self.model = self.initialize_model(self.train_x, self.train_y)

            best_value = self.train_y.max().item()
            self.best_values.append(best_value)
            best_x = self.train_x[self.train_y.argmax()]
            self.best_x_values.append(best_x)

            logging.info(
                f"Iteration {iteration}/{self.config['n_iterations']}: Best value = {best_value}, Best x = {best_x}, New x = {new_x}"
            )

        logging.info("All done.")

    def save_results(self):
        # 実験結果のディレクトリを設定
        date = current_time.split("_")[0]
        solver = "botorch"
        objective_function = "Griewank_3d"
        sampler_name = "strictly_constrained2"

        results_dir = os.path.join(
            EXPT_RESULT_DIR, date, solver, objective_function, sampler_name
        )
        os.makedirs(results_dir, exist_ok=True)

        # CSVファイルに結果を保存
        csv_filename = os.path.join(
            results_dir, f"experiment_results_{current_time}.csv"
        )
        with open(csv_filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["iter", "best_x", "best_f", "train_x", "train_y"])
            for i, best_x, best_val, train_x, train_y in zip(
                range(1, len(self.best_x_values) + 1),
                self.best_x_values,
                self.best_values,
                self.train_x,
                self.train_y,
            ):
                writer.writerow(
                    [i, best_x.tolist(), best_val, train_x.tolist(), train_y.item()]
                )
        logging.info(f"Results saved to {csv_filename}")

        # モデルの状態を保存
        model_save_path = os.path.join(results_dir, f"model_state_{current_time}.pth")
        torch.save(self.model.state_dict(), model_save_path)
        logging.info(f"Model state saved to {model_save_path}")


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float

    def griewank_function(X):
        r"""
        f(x) = \sum_{i=1}^d \frac{x_i^2}{4000} - \prod_{i=1}^d \cos \left( \frac{x_i}{\sqrt{i}} \right) + 1
        """
        sum_term = torch.sum(X**2 / 4000, dim=1)
        prod_term = torch.prod(
            torch.cos(X / torch.sqrt(torch.arange(1, X.shape[1] + 1).float())), dim=1
        )
        return sum_term - prod_term + 1

    objective_function = griewank_function

    experiment_config = {
        "initial_points": 5,
        "bounds": torch.tensor(
            [[-10.0, -10.0, -10.0], [10.0, 10.0, 10.0]], device=device, dtype=dtype
        ),
        "batch_size": 1,
        "num_restarts": 10,
        "raw_samples": 20,
        "n_iterations": 200,
        "time_budget": 60 * 60 * 10,  # 秒
        "objective_function": negate_function(objective_function),
        "algo_params": {
            "beta": 2.0,
            "beta_h": 10.0,
        },
    }

    experiment = Experiment(experiment_config)
    experiment.run()
    experiment.save_results()
