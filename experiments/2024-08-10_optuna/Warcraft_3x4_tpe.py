import sys
import os
import json
import pickle
import numpy as np
import optuna
import configparser

# .ini ファイルをプロジェクトルートから読み込む
config = configparser.ConfigParser()
config.read("config.ini")  # プロジェクトルートから実行する場合

# パスを取得
PROJECT_DIR = config["paths"]["project_dir"]
EXPT_RESULT_DIR = config["paths"]["results_dir"]

# PROJECT_DIR を Python のパスに追加
sys.path.append(PROJECT_DIR)

print(PROJECT_DIR)

from src.objectives_optuna import WarcraftObjective


def objective_wrapper(weight_matrix):
    objective = WarcraftObjective(weight_matrix)
    return objective


def run_study(weight_matrix, sampler, results_dir, num_trials, timeout, sampler_name):
    study = optuna.create_study(direction="minimize", sampler=sampler)
    objective = objective_wrapper(weight_matrix)
    study.optimize(objective, n_trials=num_trials, timeout=timeout)

    # Best trialの情報を取得
    best_trial = study.best_trial
    # 保存するデータを作成
    best_trial_data = {
        "params": best_trial.params,
        "value": best_trial.value,
        "number": best_trial.number,
    }
    # JSONファイルに保存
    with open(os.path.join(results_dir, f"best_trial_{sampler_name}.json"), "w") as f:
        json.dump(best_trial_data, f, indent=4)

    # studyオブジェクトをpickleで保存
    with open(os.path.join(results_dir, f"study_{sampler_name}.pkl"), "wb") as f:
        pickle.dump(study, f)

    print()
    print(
        f"Best trial saved in {os.path.join(results_dir, f'best_trial_{sampler_name}.json')}"
    )
    print(
        f"Study object saved in {os.path.join(results_dir, f'study_{sampler_name}.pkl')}"
    )


if __name__ == "__main__":
    date, solver, objective_function, sampler_name = (
        "2024-08-10",
        "optuna",
        "Warcraft_3x4",
        "tpe",
    )

    results_dir = os.path.join(EXPT_RESULT_DIR, date, solver, objective_function)
    os.makedirs(results_dir, exist_ok=True)

    optuna.logging.set_verbosity(optuna.logging.INFO)

    weight_matrix = np.array(
        [
            [0.1, 0.4, 0.8, 0.8],
            [0.2, 0.4, 0.4, 0.8],
            [0.8, 0.1, 0.1, 0.2],
        ]
    )

    sampler = optuna.samplers.TPESampler(seed=1)
    num_trials = 2000  # トライアル数を指定
    timeout = 3600 * 2

    # 最適化を実行
    run_study(weight_matrix, sampler, results_dir, num_trials, timeout, sampler_name)
