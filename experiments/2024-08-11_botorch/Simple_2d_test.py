import os
import sys
import configparser
import datetime
import logging
import csv

import torch
from botorch.optim import optimize_acqf
from botorch.acquisition import UpperConfidenceBound

# 設定の読み込み
config = configparser.ConfigParser()
config_path = './config.ini'
config.read(config_path)
PROJECT_DIR = config['paths']['project_dir']
EXPT_RESULT_DIR = config['paths']['results_dir']
LOG_DIR = config['paths']['logs_dir']
sys.path.append(PROJECT_DIR)

from src.bnn import BayesianMLPModel, fit_pytorch_model
from src.utils_experiment import negate_function

# ログの設定
current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
log_filename_base = "simple_2d_test"
log_filename = f"{current_time}_{log_filename_base}.log"
log_filepath = os.path.join(LOG_DIR, log_filename)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(log_filepath), logging.StreamHandler(sys.stdout)])


class Experiment:
    def __init__(self, config):
        self.config = config
        self.bounds = config["bounds"]
        self.objective_function = config["objective_function"]
        self.train_x, self.train_y = self.generate_initial_data(config["initial_points"])
        self.model = self.initialize_model(self.train_x, self.train_y)
        self.best_values = []
        self.best_x_values = []

    def generate_initial_data(self, n):
        train_x = (self.bounds[1] - self.bounds[0]) * torch.rand(n, self.bounds.size(1), device=device, dtype=dtype) + self.bounds[0]
        train_y = self.objective_function(train_x).unsqueeze(-1)
        return train_x, train_y

    def initialize_model(self, train_x, train_y):
        model = BayesianMLPModel(train_x, train_y)
        return model

    def optimize_acqf_and_get_observation(self, acq_func):
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=self.bounds,
            q=1,
            num_restarts=self.config["num_restarts"],
            raw_samples=self.config["raw_samples"],
        )
        new_x = candidates.detach()
        new_y = self.objective_function(new_x).unsqueeze(-1)
        return new_x, new_y

    def run(self):
        for iteration in range(1, self.config["n_iterations"] + 1):

            fit_pytorch_model(self.model, num_epochs=1000)

            UCB = UpperConfidenceBound(model=self.model, beta=2.0)
            new_x, new_y = self.optimize_acqf_and_get_observation(UCB)

            self.train_x = torch.cat([self.train_x, new_x])
            self.train_y = torch.cat([self.train_y, new_y])
            self.model = self.initialize_model(self.train_x, self.train_y)

            best_value = self.train_y.max().item()
            self.best_values.append(best_value)

            best_x = self.train_x[self.train_y.argmax()]
            self.best_x_values.append(best_x)

            logging.info(f"Iteration {iteration}/{self.config['n_iterations']}: Best value = {best_value}, Best x = {best_x}")

        logging.info("All done.")

    def save_results(self):
        # 実験結果のディレクトリを設定
        date = current_time.split('_')[0]
        solver = "botorch"
        objective_function = "Simple_2d"
        sampler_name = "test"

        results_dir = os.path.join(EXPT_RESULT_DIR, date, solver, objective_function, sampler_name)
        os.makedirs(results_dir, exist_ok=True)

        # CSVファイルに結果を保存
        csv_filename = os.path.join(results_dir, f"experiment_results_{current_time}.csv")
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["iter", "best_x", "best_f", "train_x", "train_y"])
            for i, best_x, best_val, train_x, train_y in zip(
                    range(1, len(self.best_x_values) + 1), 
                    self.best_x_values, 
                    self.best_values, 
                    self.train_x, 
                    self.train_y):
                writer.writerow([i, best_x.tolist(), best_val, train_x.tolist(), train_y.item()])
        logging.info(f"Results saved to {csv_filename}")

        # モデルの状態を保存
        model_save_path = os.path.join(results_dir, f"model_state_{current_time}.pth")
        torch.save(self.model.state_dict(), model_save_path)
        logging.info(f"Model state saved to {model_save_path}")


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings('ignore')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float

    # 多変数の目的関数の定義：最大化のみ対応，最小化の場合は，目的関数の出力の符号を反転させる
    def objective_function(X):
        return ((X - 2) ** 2).sum(dim=-1)

    # 実験の設定を辞書で管理
    experiment_config = {
        "initial_points": 5,
        "bounds": torch.tensor([[-4.0, -4.0], [8.0, 8.0]], device=device, dtype=dtype),  # 2次元入力変数の範囲
        "batch_size": 1,
        "num_restarts": 10,
        "raw_samples": 20,
        "n_iterations": 20,
        "objective_function": negate_function(objective_function)
    }

    # 実験の実行
    experiment = Experiment(experiment_config)
    experiment.run()
    experiment.save_results()

    for i, best_x, best_val, train_x in zip(range(1, len(experiment.best_x_values) + 1), experiment.best_x_values, experiment.best_values, experiment.train_x):
        print(f"Iteration {i}: Best value = {best_val}, Best x = {best_x}, train_x = {train_x}")