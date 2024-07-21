import os
import sys
import re
import pickle
from constants import PATH_INFO
sys.path.append(PATH_INFO.get('project_dir'))

from src.utils_experiment import extract_info_from_filename

import optuna
from src.objectives import AckleyObjective


setting_dict = {
    'n_trials': 100,
    'seed': 42,
    'sampler': 'TPESampler',
    'direction': 'minimize',
    'study_dir': 'study.pkl'
}


if __name__ == '__main__':
    print(__file__)
    
    # Extract date and objective function from the filename
    date, objective_function = extract_info_from_filename(__file__)
    
    if not date or not objective_function:
        print("Failed to extract date or objective function from the filename.")
        sys.exit(1)

    # Create directory for results
    results_dir = os.path.join(PATH_INFO.get('expt_res_dir'), date, objective_function)
    os.makedirs(results_dir, exist_ok=True)
    
    # Define the objective function
    objective = AckleyObjective(dim=2)

    # O
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction='minimize', sampler=sampler)

    # 最適化の実行
    study.optimize(objective, n_trials=n_trials)

    # 結果の表示
    print('Best parameters:', study.best_params)
    print('Best value:', study.best_value)

    # 結果の保存
    study_path = os.path.join(results_dir, 'study.pkl')
    with open(study_path, 'wb') as f:
        pickle.dump(study, f)

    print(f'Study saved to {study_path}')

    # 軌跡のプロット
    objective.plot_optimization_trajectory(study)