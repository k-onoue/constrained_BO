import sys
import os
import json
import pickle
import numpy as np
import optuna

from path_info import PROJECT_DIR, EXPT_RESULT_DIR
sys.path.append(PROJECT_DIR)

print(PROJECT_DIR)

from src.objectives_optuna import WarcraftObjective
from src.utils_experiment import extract_info_from_filename


def objective_wrapper(weight_matrix):
    objective = WarcraftObjective(weight_matrix)
    return objective


def run_study(weight_matrix, sampler, results_dir, num_trials):
    study = optuna.create_study(
        direction='minimize',
        sampler=sampler
    )
    objective = objective_wrapper(weight_matrix)
    study.optimize(objective, n_trials=num_trials)

    # Best trialの情報を取得
    best_trial = study.best_trial
    # 保存するデータを作成
    best_trial_data = {
        'params': best_trial.params,
        'value': best_trial.value,
        'number': best_trial.number,
    }
    # JSONファイルに保存
    with open(os.path.join(results_dir, 'best_trial.json'), 'w') as f:
        json.dump(best_trial_data, f, indent=4)

    # studyオブジェクトをpickleで保存
    with open(os.path.join(results_dir, 'study.pkl'), 'wb') as f:
        pickle.dump(study, f)

    print(f"Best trial saved in {os.path.join(results_dir, 'best_trial.json')}")
    print(f"Study object saved in {os.path.join(results_dir, 'study.pkl')}")


if __name__ == '__main__':
    date, objective_function = extract_info_from_filename(__file__)
    
    if not date or not objective_function:
        print("Failed to extract date or objective function from the filename.")
        sys.exit(1)

    results_dir = os.path.join(EXPT_RESULT_DIR, date, objective_function)
    os.makedirs(results_dir, exist_ok=True)

    optuna.logging.set_verbosity(optuna.logging.INFO)

    weight_matrix = np.array([
        [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 1.2, 1.2, 1.2, 0.8],
        [0.8, 0.8, 0.8, 0.8, 0.8, 1.2, 1.2, 1.2, 1.2, 9.2, 1.2, 1.2],
        [0.8, 0.8, 0.8, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2],
        [1.2, 0.8, 0.8, 1.2, 1.2, 0.8, 1.2, 1.2, 7.7, 7.7, 1.2, 1.2],
        [1.2, 0.8, 1.2, 1.2, 0.8, 0.8, 0.8, 1.2, 1.2, 7.7, 1.2, 1.2],
        [1.2, 0.8, 1.2, 1.2, 0.8, 0.8, 0.8, 0.8, 1.2, 7.7, 1.2, 1.2],
        [1.2, 0.8, 1.2, 1.2, 1.2, 0.8, 0.8, 1.2, 1.2, 1.2, 7.7, 7.7],
        [0.8, 0.8, 1.2, 1.2, 0.8, 0.8, 1.2, 0.8, 0.8, 1.2, 7.7, 7.7],
        [0.8, 0.8, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 7.7, 7.7],
        [0.8, 0.8, 1.2, 9.2, 9.2, 9.2, 9.2, 1.2, 1.2, 7.7, 7.7, 7.7],
        [0.8, 0.8, 1.2, 9.2, 9.2, 9.2, 9.2, 1.2, 1.2, 7.7, 7.7, 7.7],
        [0.8, 0.8, 1.2, 9.2, 9.2, 9.2, 9.2, 1.2, 1.2, 1.2, 7.7, 7.7]
    ])

    sampler = optuna.samplers.TPESampler(seed=1)
    num_trials = 100  # ここでトライアル数を指定

    # 最適化を実行
    run_study(weight_matrix, sampler, results_dir, num_trials)