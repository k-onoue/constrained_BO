import datetime
import logging
import os
import re
import sys

import pandas as pd
import torch


def extract_info_from_filename(filename):
    # Extract date and objective function from the filename
    match = re.match(r".*(\d{4}-\d{2}-\d{2})_(\w+)_v\d+\.py", filename)
    if match:
        date = match.group(1)
        objective_function = match.group(2)
        return date, objective_function
    return None, None


# def generate_integer_samples(bounds, n, device=torch.device("cpu"), framework="botorch"):
def generate_integer_samples(bounds, n, device=torch.device("cpu"), dtype=torch.float32):
    """
    整数をランダムにサンプリングして、n 行 m 列の torch.Tensor を生成します。
    重複のない n サンプルが得られるまでサンプリングを繰り返します。

    Parameters:
    - bounds: list of list, 変数の下限と上限のリスト
    - n: int, サンプル数
    - device: torch.device, テンソルを配置するデバイス

    Returns:
    - torch.Tensor, n 行 m 列のテンソル
    """
    # frameworks = ["botorch", "optuna"]
    # if framework not in frameworks:
    #     raise ValueError(f"Invalid framework: {framework}")
    # elif framework == "botorch":
    #     bounds = bounds.T

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


def negate_function(func):
    """
    目的関数の符号を反転させる
    """

    def negated_func(X):
        return -func(X)

    return negated_func


def set_logger(log_filename_base, save_dir):
    # ログの設定
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"{current_time}_{log_filename_base}.log"
    log_filepath = os.path.join(save_dir, log_filename)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_filepath), logging.StreamHandler(sys.stdout)],
    )


def log_print(message):
    print(message)
    logging.info(message)


class OptimLogParser:
    def __init__(self, log_file):
        self.log_file = log_file
        self.settings = {}
        self.initial_data = {
            "candidate": [],
            "function_value": [],
            "final_training_loss": [],
        }
        self.bo_data = {
            "iteration": [],
            "candidate": [],
            "acquisition_value": [],
            "function_value": [],
            "final_training_loss": [],
            "iteration_time": [],
        }

    def combine_log_entries(self):
        with open(self.log_file, "r") as file:
            lines = file.readlines()

        timestamp_pattern = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - "

        combined_lines = []
        current_entry = ""

        for line in lines:
            if re.match(timestamp_pattern, line):
                if current_entry:
                    combined_lines.append(current_entry.strip())
                current_entry = line.strip()
            else:
                current_entry += " " + line.strip()

        if current_entry:
            combined_lines.append(current_entry.strip())

        return combined_lines

    def parse_log_file(self):
        combined_lines = self.combine_log_entries()

        mode = None

        for line in combined_lines:
            if "Running optimization with settings:" in line:
                mode = "settings"
                self._parse_settings(line)
            elif "Initial data points" in line:
                mode = "init"
            elif "Iteration" in line:
                mode = "bo_loop"
            elif "Optimization completed." in line:
                break  # 終了条件

            if mode == "init":
                self._parse_init_data(line)
            elif mode == "bo_loop":
                self._parse_bo_data(line)

        # Fill in the final training loss for the initial data
        val = self.initial_data["final_training_loss"][-1]
        length = len(self.initial_data["candidate"])
        self.initial_data["final_training_loss"] = [val] * length

        final_iter = self.bo_data["iteration"][-1] if self.bo_data["iteration"] else 0
        for column in self.bo_data:
            while len(self.bo_data[column]) < final_iter:
                self.bo_data[column].append(None)

        self.initial_data = pd.DataFrame(self.initial_data)
        self.bo_data = pd.DataFrame(self.bo_data)

    def _parse_settings(self, line):
        settings_str = line.split("settings:")[1].strip()
        settings_str = re.sub(r"device\(type='[^']+'\)", "'cpu'", settings_str)
        settings_str = re.sub(r"device\(type=\"[^\"]+\"\)", "'cpu'", settings_str)
        try:
            self.settings = eval(settings_str)
        except SyntaxError as e:
            print(f"Failed to parse settings: {e}")
            self.settings = {}

    def _parse_init_data(self, line):
        candidate_match = re.search(r"Candidate: (.*?) Function Value:", line)
        function_value_match = re.search(r"Function Value: ([-+]?\d*\.\d+|\d+)", line)
        final_training_loss_match = re.search(
            r"Final training loss: ([-+]?\d*\.\d+|\d+)", line
        )

        if candidate_match:
            self.initial_data["candidate"].append(candidate_match.group(1).strip())
        if function_value_match:
            self.initial_data["function_value"].append(
                float(function_value_match.group(1))
            )
        if final_training_loss_match:
            self.initial_data["final_training_loss"].append(
                float(final_training_loss_match.group(1))
            )

    def _parse_bo_data(self, line):
        iteration_match = re.search(r"Iteration (\d+)/", line)
        candidate_match = re.search(r"Candidate: (\[.*?\])", line)
        acquisition_value_match = re.search(
            r"Acquisition Value: ([-+]?\d*\.\d+|\d+)", line
        )
        function_value_match = re.search(r"Function Value: ([-+]?\d*\.\d+|\d+)", line)
        final_training_loss_match = re.search(
            r"Final training loss: ([-+]?\d*\.\d+|\d+)", line
        )
        iteration_time_match = re.search(r"Iteration time: ([-+]?\d*\.\d+)", line)

        if iteration_match:
            self.bo_data["iteration"].append(int(iteration_match.group(1)))
        if candidate_match:
            self.bo_data["candidate"].append(candidate_match.group(1).strip())
        if acquisition_value_match:
            self.bo_data["acquisition_value"].append(
                float(acquisition_value_match.group(1))
            )
        if function_value_match:
            self.bo_data["function_value"].append(float(function_value_match.group(1)))
        if final_training_loss_match:
            self.bo_data["final_training_loss"].append(
                float(final_training_loss_match.group(1))
            )
        if iteration_time_match:
            self.bo_data["iteration_time"].append(float(iteration_time_match.group(1)))


# if __name__ == "__main__":

#     # log_file = "logs/2024-08-23_16-24-48_Warcraft_3x4_architecture-search_4.log"
#     log_file = "/Users/keisukeonoue/ws/constrained_BO/logs/2024-08-23_22-28-52_Warcraft_3x4_unconstrained.log"

#     parser = OptimLogParser(log_file)
#     parser.parse_log_file()

#     print("Experimental settings:")
#     print(parser.settings)

#     print("Initial data:")
#     print(parser.initial_data)

#     print("BO data:")
#     print(parser.bo_data)
