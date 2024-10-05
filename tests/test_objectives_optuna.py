import sys
from path_info import PROJECT_DIR, EXPT_RESULT_DIR

sys.path.append(PROJECT_DIR)

import numpy as np
import optuna
from src.objectives_optuna import WarcraftObjective


if __name__ == "__main__":
    # optuna.logging.set_verbosity(optuna.logging.ERROR)
    optuna.logging.set_verbosity(optuna.logging.INFO)

    weight_matrix = np.array([[0.1, 0.4, 0.9], [0.4, 0.1, 0.4], [0.9, 0.4, 0.1]])

    objective = WarcraftObjective(weight_matrix)

    n_trials = 1000
    seed = 42
    sampler = optuna.samplers.TPESampler(seed=seed)
    # sampler = optuna.samplers.BruteForceSampler()

    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    print(study.best_params)
    print(study.best_value)

    import os
    import json

    # Best trialの情報を取得
    best_trial = study.best_trial
    # 保存するデータを作成
    best_trial_data = {
        "params": best_trial.params,
        "value": best_trial.value,
        "number": best_trial.number,
    }
    # JSONファイルに保存
    with open(os.path.join(EXPT_RESULT_DIR, "test_best_trial.json"), "w") as f:
        json.dump(best_trial_data, f, indent=4)


# if __name__ == '__main__':
#     # optuna.logging.set_verbosity(optuna.logging.ERROR)
#     optuna.logging.set_verbosity(optuna.logging.INFO)

#     # Define the objective function
#     # objective = AckleyObjective(dim=2)
#     # objective = DiscreteAckleyObjective(dim=2, n_split=30)
#     # objective = RosenbrockObjective(dim=2)
#     objective = DiscreteRosenbrockObjective(dim=2, n_split=30)

#     n_trials = 100
#     seed = 42
#     sampler = optuna.samplers.TPESampler(seed=seed)

#     study = optuna.create_study(direction='minimize', sampler=sampler)
#     study.optimize(objective, n_trials=n_trials)

#     print(study.best_params)
#     print(study.best_value)

#     # Plot optimization trajectory
#     objective.plot_optimization_trajectory(study)
