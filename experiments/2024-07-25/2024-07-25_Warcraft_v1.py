import sys
import os
import json
import numpy as np
import optuna

from path_info import PROJECT_DIR, EXPT_RESULT_DIR

sys.path.append(PROJECT_DIR)
from src.objectives_optuna import WarcraftObjective
from src.utils_experiment import extract_info_from_filename


def objective_wrapper(weight_matrix):
    objective = WarcraftObjective(weight_matrix)
    return objective


def run_study(weight_matrix, sampler, results_dir):
    study = optuna.create_study(direction="minimize", sampler=sampler)
    objective = objective_wrapper(weight_matrix)
    study.optimize(objective)

    # Best trialの情報を取得
    best_trial = study.best_trial
    # 保存するデータを作成
    best_trial_data = {
        "params": best_trial.params,
        "value": best_trial.value,
        "number": best_trial.number,
    }
    # JSONファイルに保存
    with open(os.path.join(results_dir, "exhaustive_search.json"), "w") as f:
        json.dump(best_trial_data, f, indent=4)
    print(f"Best trial saved in {os.path.join(results_dir, 'exhaustive_search.json')}")


if __name__ == "__main__":
    date, objective_function = extract_info_from_filename(__file__)

    if not date or not objective_function:
        print("Failed to extract date or objective function from the filename.")
        sys.exit(1)

    results_dir = os.path.join(EXPT_RESULT_DIR, date, objective_function)
    os.makedirs(results_dir, exist_ok=True)

    optuna.logging.set_verbosity(optuna.logging.INFO)

    weight_matrix = np.array(
        [
            [0.1, 0.9],
            [0.4, 0.1],
        ]
    )

    sampler = optuna.samplers.BruteForceSampler()

    # 最適化を実行
    run_study(weight_matrix, sampler, results_dir)
